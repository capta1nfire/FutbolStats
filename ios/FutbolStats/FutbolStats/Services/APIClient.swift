import Foundation

enum APIError: Error, LocalizedError {
    case invalidURL
    case networkError(Error)
    case decodingError(Error)
    case serverError(Int)
    case noData

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
        }
    }
}

actor APIClient {
    static let shared = APIClient()

    private let baseURL = "https://web-production-f2de9.up.railway.app"
    private let session: URLSession

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)
    }

    // MARK: - Health Check

    func checkHealth() async throws -> HealthResponse {
        let url = URL(string: "\(baseURL)/health")!
        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        do {
            return try JSONDecoder().decode(HealthResponse.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    // MARK: - Predictions

    func getUpcomingPredictions(days: Int = 7) async throws -> PredictionResponse {
        var components = URLComponents(string: "\(baseURL)/predictions/upcoming")!
        components.queryItems = [URLQueryItem(name: "days", value: String(days))]

        guard let url = components.url else {
            throw APIError.invalidURL
        }

        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        do {
            return try JSONDecoder().decode(PredictionResponse.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    // MARK: - Training

    func trainModel() async throws -> TrainingResponse {
        let url = URL(string: "\(baseURL)/model/train")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        do {
            return try JSONDecoder().decode(TrainingResponse.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    // MARK: - Match Details

    func getMatchDetails(matchId: Int) async throws -> MatchDetailsResponse {
        let url = URL(string: "\(baseURL)/matches/\(matchId)/details")!
        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        do {
            return try JSONDecoder().decode(MatchDetailsResponse.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    // MARK: - Teams

    func getTeams(teamType: String? = nil, limit: Int = 500) async throws -> [TeamItem] {
        var components = URLComponents(string: "\(baseURL)/teams")!
        var queryItems: [URLQueryItem] = [URLQueryItem(name: "limit", value: String(limit))]

        if let teamType = teamType {
            queryItems.append(URLQueryItem(name: "team_type", value: teamType))
        }

        components.queryItems = queryItems

        guard let url = components.url else {
            throw APIError.invalidURL
        }

        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        do {
            return try JSONDecoder().decode([TeamItem].self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    // MARK: - Competitions

    func getCompetitions() async throws -> [CompetitionItem] {
        let url = URL(string: "\(baseURL)/competitions")!
        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        do {
            return try JSONDecoder().decode([CompetitionItem].self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    // MARK: - League Standings

    func getStandings(leagueId: Int, season: Int? = nil) async throws -> StandingsResponse {
        var components = URLComponents(string: "\(baseURL)/standings/\(leagueId)")!

        if let season = season {
            components.queryItems = [URLQueryItem(name: "season", value: String(season))]
        }

        guard let url = components.url else {
            throw APIError.invalidURL
        }

        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.noData
        }

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        do {
            return try JSONDecoder().decode(StandingsResponse.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    // MARK: - ETL Sync

    func syncData(leagueIds: [Int], season: Int) async throws -> [String: Any] {
        let url = URL(string: "\(baseURL)/etl/sync")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
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

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(httpResponse.statusCode)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw APIError.decodingError(NSError(domain: "", code: 0))
        }

        return json
    }
}
