import Foundation

enum APIError: Error, LocalizedError {
    case invalidURL
    case networkError(Error)
    case decodingError(Error)
    case serverError(Int)
    case noData
    case emptyResponse
    case maxRetriesExceeded

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .decodingError(let error):
            return "Decoding error: \(error.localizedDescription)"
        case .serverError(let code):
            return "Server error: \(code)"
        case .noData:
            return "No data received"
        case .emptyResponse:
            return "Empty response from server"
        case .maxRetriesExceeded:
            return "Request failed after multiple retries"
        }
    }

    /// Whether this error is retryable
    var isRetryable: Bool {
        switch self {
        case .networkError:
            return true
        case .serverError(let code):
            // Retry on 5xx errors and 429 (rate limit)
            return code >= 500 || code == 429
        default:
            return false
        }
    }
}

/// API environment configuration
enum APIEnvironment {
    case production
    case staging
    case development

    var baseURL: String {
        switch self {
        case .production:
            return "https://web-production-f2de9.up.railway.app"
        case .staging:
            return "https://staging.futbolstats.app"
        case .development:
            return "http://localhost:8000"
        }
    }

    /// Current environment based on build configuration
    static var current: APIEnvironment {
        #if DEBUG
        // Use production for now, can be changed to .development for local testing
        return .production
        #else
        return .production
        #endif
    }
}

/// Retry configuration with exponential backoff
struct RetryConfig {
    let maxRetries: Int
    let baseDelay: TimeInterval
    let maxDelay: TimeInterval
    let jitterFactor: Double

    static let `default` = RetryConfig(
        maxRetries: 3,
        baseDelay: 1.0,
        maxDelay: 30.0,
        jitterFactor: 0.3
    )

    /// Calculate delay for a given retry attempt with exponential backoff and jitter
    func delay(for attempt: Int) -> TimeInterval {
        let exponentialDelay = baseDelay * pow(2.0, Double(attempt))
        let cappedDelay = min(exponentialDelay, maxDelay)

        // Add jitter to prevent thundering herd
        let jitter = cappedDelay * jitterFactor * Double.random(in: -1...1)
        return max(0, cappedDelay + jitter)
    }
}

actor APIClient {
    static let shared = APIClient()

    private let environment: APIEnvironment
    private let retryConfig: RetryConfig

    // Shared session for ALL requests - ensures connection reuse and warm TLS
    // Static to allow nonisolated methods to access without actor hop
    private static let sharedSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        // Enable HTTP cache: 50MB memory, 100MB disk
        config.urlCache = URLCache(memoryCapacity: 50_000_000, diskCapacity: 100_000_000)
        config.requestCachePolicy = .useProtocolCachePolicy
        // Keep connections alive longer for faster subsequent requests
        config.httpShouldSetCookies = true
        config.httpShouldUsePipelining = true
        config.httpMaximumConnectionsPerHost = 6
        // Don't wait for connectivity - fail fast for better UX
        config.waitsForConnectivity = false
        return URLSession(configuration: config)
    }()

    // Instance reference for actor-isolated methods
    private var session: URLSession { Self.sharedSession }

    private init(
        environment: APIEnvironment = .current,
        retryConfig: RetryConfig = .default
    ) {
        self.environment = environment
        self.retryConfig = retryConfig
    }

    // MARK: - Generic Request with Retry

    private func _applyAuthHeaders(_ request: inout URLRequest) {
        // Prefer header token to avoid query-param token leaking into logs.
        if let token = AppConfiguration.dashboardToken {
            request.setValue(token, forHTTPHeaderField: "X-Dashboard-Token")
        }
        request.setValue("application/json", forHTTPHeaderField: "Accept")
    }

    private func performRequest<T: Decodable>(
        url: URL,
        method: String = "GET",
        body: Data? = nil
    ) async throws -> T {
        var lastError: APIError?

        for attempt in 0..<retryConfig.maxRetries {
            do {
                var request = URLRequest(url: url)
                request.httpMethod = method
                _applyAuthHeaders(&request)
                if let body = body {
                    request.httpBody = body
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                }

                let (data, response) = try await session.data(for: request)

                guard let httpResponse = response as? HTTPURLResponse else {
                    throw APIError.noData
                }

                guard (200..<300).contains(httpResponse.statusCode) else {
                    let error = APIError.serverError(httpResponse.statusCode)
                    if error.isRetryable && attempt < retryConfig.maxRetries - 1 {
                        lastError = error
                        let delay = retryConfig.delay(for: attempt)
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                        continue
                    }
                    throw error
                }

                // Defensive: treat whitespace-only as empty response
                let trimmed = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
                guard let trimmed = trimmed, !trimmed.isEmpty else {
                    throw APIError.emptyResponse
                }

                do {
                    return try JSONDecoder().decode(T.self, from: data)
                } catch {
                    throw APIError.decodingError(error)
                }

            } catch let error as APIError {
                if error.isRetryable && attempt < retryConfig.maxRetries - 1 {
                    lastError = error
                    let delay = retryConfig.delay(for: attempt)
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                }
                throw error
            } catch {
                let apiError = APIError.networkError(error)
                if apiError.isRetryable && attempt < retryConfig.maxRetries - 1 {
                    lastError = apiError
                    let delay = retryConfig.delay(for: attempt)
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    continue
                }
                throw apiError
            }
        }

        throw lastError ?? APIError.maxRetriesExceeded
    }

    /// Concurrent request that bypasses actor queue - for fast, idempotent GETs
    /// This prevents head-of-line blocking when other requests are in flight
    /// Uses shared session to reuse TLS connections (avoids cold handshake latency)
    nonisolated private static func performConcurrentRequest<T: Decodable>(
        url: URL,
        endpoint: String = "unknown"
    ) async throws -> T {
        let startTime = CFAbsoluteTimeGetCurrent()

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 10
        // Apply auth headers (static context, read from AppConfiguration)
        if let token = AppConfiguration.dashboardToken {
            request.setValue(token, forHTTPHeaderField: "X-Dashboard-Token")
        }
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let setupMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        let networkStart = CFAbsoluteTimeGetCurrent()

        // Use shared session to reuse warm connections
        let (data, response) = try await sharedSession.data(for: request)
        let networkMs = (CFAbsoluteTimeGetCurrent() - networkStart) * 1000
        let dataSize = data.count

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            print("[APIClient] \(endpoint) | error | status=\(httpResponse.statusCode), network_ms=\(String(format: "%.1f", networkMs))")
            throw APIError.serverError(httpResponse.statusCode)
        }

        // Defensive: treat whitespace-only as empty response
        let trimmed = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed = trimmed, !trimmed.isEmpty else {
            throw APIError.emptyResponse
        }

        let decodeStart = CFAbsoluteTimeGetCurrent()
        do {
            let result = try JSONDecoder().decode(T.self, from: data)
            let decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000
            let totalMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

            // Granular timing to diagnose TLS/DNS latency (per auditor recommendation)
            print("[APIClient] \(endpoint) | ok | setup_ms=\(String(format: "%.1f", setupMs)), network_ms=\(String(format: "%.1f", networkMs)), decode_ms=\(String(format: "%.1f", decodeMs)), total_ms=\(String(format: "%.1f", totalMs)), bytes=\(dataSize)")

            return result
        } catch let decodingError as DecodingError {
            // Log detailed decoding error for debugging
            switch decodingError {
            case .keyNotFound(let key, let context):
                print("[APIClient] \(endpoint) | decode_error | missing key '\(key.stringValue)' at path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
            case .typeMismatch(let type, let context):
                print("[APIClient] \(endpoint) | decode_error | type mismatch for \(type) at path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
            case .valueNotFound(let type, let context):
                print("[APIClient] \(endpoint) | decode_error | value not found for \(type) at path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
            case .dataCorrupted(let context):
                print("[APIClient] \(endpoint) | decode_error | data corrupted at path: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
            @unknown default:
                print("[APIClient] \(endpoint) | decode_error | unknown: \(decodingError)")
            }
            throw APIError.decodingError(decodingError)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    // MARK: - Health Check

    func checkHealth() async throws -> HealthResponse {
        let url = URL(string: "\(environment.baseURL)/health")!
        return try await performRequest(url: url)
    }

    // MARK: - Predictions

    /// Fetch upcoming predictions with explicit date range control
    /// - Parameters:
    ///   - daysBack: Past N days for finished matches (default 7)
    ///   - daysAhead: Future N days for upcoming matches (default 7)
    /// - Priority window: daysBack=1, daysAhead=1 → yesterday/today/tomorrow
    /// - Full window: daysBack=7, daysAhead=7 → 15-day range
    func getUpcomingPredictions(daysBack: Int = 7, daysAhead: Int = 7) async throws -> PredictionResponse {
        var components = URLComponents(string: "\(environment.baseURL)/predictions/upcoming")!
        components.queryItems = [
            URLQueryItem(name: "days_back", value: String(daysBack)),
            URLQueryItem(name: "days_ahead", value: String(daysAhead))
        ]

        guard let url = components.url else {
            throw APIError.invalidURL
        }

        return try await performRequest(url: url)
    }

    // MARK: - Training

    func trainModel() async throws -> TrainingResponse {
        let url = URL(string: "\(environment.baseURL)/model/train")!
        return try await performRequest(url: url, method: "POST")
    }

    // MARK: - Match Details (concurrent - bypasses actor queue for fast response)

    nonisolated func getMatchDetails(matchId: Int) async throws -> MatchDetailsResponse {
        let callStart = CFAbsoluteTimeGetCurrent()
        print("[APIClient] getMatchDetails START match_id=\(matchId)")

        let url = URL(string: "\(APIEnvironment.current.baseURL)/matches/\(matchId)/details")!
        let result: MatchDetailsResponse = try await Self.performConcurrentRequest(url: url, endpoint: "match_details")

        let callMs = (CFAbsoluteTimeGetCurrent() - callStart) * 1000
        print("[APIClient] getMatchDetails END match_id=\(matchId) total_ms=\(String(format: "%.1f", callMs))")

        return result
    }

    // MARK: - Teams

    func getTeams(teamType: String? = nil, limit: Int = 500) async throws -> [TeamItem] {
        var components = URLComponents(string: "\(environment.baseURL)/teams")!
        var queryItems: [URLQueryItem] = [URLQueryItem(name: "limit", value: String(limit))]

        if let teamType = teamType {
            queryItems.append(URLQueryItem(name: "team_type", value: teamType))
        }

        components.queryItems = queryItems

        guard let url = components.url else {
            throw APIError.invalidURL
        }

        return try await performRequest(url: url)
    }

    // MARK: - Competitions

    func getCompetitions() async throws -> [CompetitionItem] {
        let url = URL(string: "\(environment.baseURL)/competitions")!
        return try await performRequest(url: url)
    }

    // MARK: - Dashboards (OPS / PIT)

    func getOpsDashboard() async throws -> OpsDashboardResponse {
        let url = URL(string: "\(environment.baseURL)/dashboard/ops.json")!
        let wrapper: OpsDashboardWrapper = try await performRequest(url: url)
        return wrapper.data
    }

    func getPITDashboard() async throws -> PITDashboardResponse {
        let url = URL(string: "\(environment.baseURL)/dashboard/pit.json")!
        return try await performRequest(url: url)
    }

    func getAlphaProgressSnapshots(limit: Int = 50) async throws -> AlphaProgressSnapshotsResponse {
        var components = URLComponents(string: "\(environment.baseURL)/dashboard/ops/progress_snapshots.json")!
        components.queryItems = [URLQueryItem(name: "limit", value: String(limit))]
        guard let url = components.url else {
            throw APIError.invalidURL
        }
        return try await performRequest(url: url)
    }

    // MARK: - League Standings

    func getStandings(leagueId: Int, season: Int? = nil) async throws -> StandingsResponse {
        var components = URLComponents(string: "\(environment.baseURL)/standings/\(leagueId)")!

        if let season = season {
            components.queryItems = [URLQueryItem(name: "season", value: String(season))]
        }

        guard let url = components.url else {
            throw APIError.invalidURL
        }

        return try await performRequest(url: url)
    }

    // MARK: - Match Timeline (concurrent - bypasses actor queue)

    nonisolated func getMatchTimeline(matchId: Int) async throws -> MatchTimelineResponse {
        let url = URL(string: "\(APIEnvironment.current.baseURL)/matches/\(matchId)/timeline")!
        return try await Self.performConcurrentRequest(url: url, endpoint: "match_timeline")
    }

    // MARK: - Match Insights (concurrent - bypasses actor queue)

    nonisolated func getMatchInsights(matchId: Int) async throws -> MatchInsightsResponse {
        let url = URL(string: "\(APIEnvironment.current.baseURL)/matches/\(matchId)/insights")!
        return try await Self.performConcurrentRequest(url: url, endpoint: "match_insights")
    }

    // MARK: - ETL Sync (requires API key in production)

    func syncData(leagueIds: [Int], season: Int) async throws -> [String: Any] {
        let url = URL(string: "\(environment.baseURL)/etl/sync")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        _applyAuthHeaders(&request)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "league_ids": leagueIds,
            "season": season
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw APIError.decodingError(NSError(domain: "", code: 0))
        }

        return json
    }
}
