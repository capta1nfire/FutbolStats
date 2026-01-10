import Foundation

enum APIError: Error, LocalizedError {
    case invalidURL
    case networkError(Error)
    case decodingError(Error)
    case serverError(Int)
    case noData
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
    private let session: URLSession
    private let retryConfig: RetryConfig

    private init(
        environment: APIEnvironment = .current,
        retryConfig: RetryConfig = .default
    ) {
        self.environment = environment
        self.retryConfig = retryConfig

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)
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

                guard !data.isEmpty else {
                    throw APIError.noData
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

    // MARK: - Health Check

    func checkHealth() async throws -> HealthResponse {
        let url = URL(string: "\(environment.baseURL)/health")!
        return try await performRequest(url: url)
    }

    // MARK: - Predictions

    func getUpcomingPredictions(days: Int = 7) async throws -> PredictionResponse {
        var components = URLComponents(string: "\(environment.baseURL)/predictions/upcoming")!
        components.queryItems = [URLQueryItem(name: "days", value: String(days))]

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

    // MARK: - Match Details

    func getMatchDetails(matchId: Int) async throws -> MatchDetailsResponse {
        let url = URL(string: "\(environment.baseURL)/matches/\(matchId)/details")!
        return try await performRequest(url: url)
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

    // MARK: - Match Timeline

    func getMatchTimeline(matchId: Int) async throws -> MatchTimelineResponse {
        let url = URL(string: "\(environment.baseURL)/matches/\(matchId)/timeline")!
        return try await performRequest(url: url)
    }

    // MARK: - Match Insights (Narrative Analysis)

    func getMatchInsights(matchId: Int) async throws -> MatchInsightsResponse {
        let url = URL(string: "\(environment.baseURL)/matches/\(matchId)/insights")!
        return try await performRequest(url: url)
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
