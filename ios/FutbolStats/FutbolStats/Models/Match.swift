import Foundation
import SwiftUI

// MARK: - API Response Models

struct PredictionResponse: Codable {
    let predictions: [MatchPrediction]
    let modelVersion: String
    let generatedAt: String?

    enum CodingKeys: String, CodingKey {
        case predictions
        case modelVersion = "model_version"
        case generatedAt = "generated_at"
    }
}

struct MatchPrediction: Codable, Identifiable {
    let matchId: Int?
    let matchExternalId: Int?
    let homeTeam: String
    let awayTeam: String
    let homeTeamLogo: String?
    let awayTeamLogo: String?
    let date: String?
    let status: String?           // Match status: NS, FT, 1H, 2H, HT, etc.
    let homeGoals: Int?           // Final score (nil if not played)
    let awayGoals: Int?           // Final score (nil if not played)
    let leagueId: Int?            // League ID for grouping
    let probabilities: Probabilities
    let fairOdds: FairOdds
    let marketOdds: MarketOdds?
    let valueBets: [ValueBet]?
    let hasValueBet: Bool?
    let bestValueBet: ValueBet?
    let confidenceTier: String?   // gold, silver, copper

    var id: Int {
        matchId ?? matchExternalId ?? UUID().hashValue
    }

    /// League logo URL (derived from leagueId)
    var leagueLogo: String? {
        guard let id = leagueId else { return nil }
        return "https://media.api-sports.io/football/leagues/\(id).png"
    }

    /// League name (fallback mapping for common leagues)
    var leagueName: String {
        guard let id = leagueId else { return "Other" }
        return Self.leagueNames[id] ?? "League \(id)"
    }

    /// Common league names mapping
    private static let leagueNames: [Int: String] = [
        // International
        1: "World Cup",
        4: "Euro",
        5: "UEFA Nations League",
        6: "Africa Cup of Nations",
        9: "Copa America",
        10: "Friendlies",
        // South America
        128: "Liga Argentina",
        129: "Copa Argentina",
        239: "Liga Colombia",
        71: "Brasileirão",
        73: "Copa Brasil",
        13: "Libertadores",
        11: "Sudamericana",
        // Europe - Top 5 Leagues
        39: "Premier League",
        40: "Championship",
        45: "FA Cup",
        140: "La Liga",
        143: "Copa del Rey",
        135: "Serie A",
        78: "Bundesliga",
        88: "Eredivisie",
        61: "Ligue 1",
        94: "Primeira Liga",
        2: "Champions League",
        3: "Europa League",
        848: "Conference League",
        // Mexico & CONCACAF
        262: "Liga MX",
        16: "Concacaf Champions",
    ]

    /// Check if match is finished
    var isFinished: Bool {
        status == "FT" || status == "AET" || status == "PEN"
    }

    /// Check if match is live
    var isLive: Bool {
        guard let s = status else { return false }
        return ["1H", "2H", "HT", "ET", "BT", "P", "LIVE"].contains(s)
    }

    /// Check if match has a score to display (finished or live)
    var hasScore: Bool {
        (isFinished || isLive) && homeGoals != nil && awayGoals != nil
    }

    /// Score display for finished matches
    var scoreDisplay: String? {
        guard let home = homeGoals, let away = awayGoals else { return nil }
        return "\(home) - \(away)"
    }

    /// Quick check if this prediction has a value betting opportunity
    var isValueBet: Bool {
        hasValueBet ?? (valueBets?.isEmpty == false)
    }

    /// What the model predicted (home, draw, away)
    var predictedOutcome: String {
        if probabilities.home > probabilities.draw && probabilities.home > probabilities.away {
            return "home"
        } else if probabilities.away > probabilities.draw && probabilities.away > probabilities.home {
            return "away"
        }
        return "draw"
    }

    /// Actual result based on final score
    var actualOutcome: String? {
        guard isFinished, let home = homeGoals, let away = awayGoals else { return nil }
        if home > away { return "home" }
        if away > home { return "away" }
        return "draw"
    }

    /// Did the prediction match the actual result?
    var predictionCorrect: Bool? {
        guard let actual = actualOutcome else { return nil }
        return predictedOutcome == actual
    }

    /// Tier emoji for display
    var tierEmoji: String {
        switch confidenceTier?.lowercased() {
        case "gold": return "🥇"
        case "silver": return "🥈"
        case "copper": return "🥉"
        default: return ""
        }
    }

    /// Best EV percentage for display (e.g., "+12.5%")
    var bestEvDisplay: String? {
        guard let best = bestValueBet else { return nil }
        let ev = best.evPercentage ?? (best.edge.map { $0 * 100 }) ?? 0
        return String(format: "+%.1f%% EV", ev)
    }

    var matchDate: Date? {
        guard let dateString = date else { return nil }

        // Try ISO8601 without fractional seconds first (API format)
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime]
        if let date = isoFormatter.date(from: dateString) {
            return date
        }

        // Try with fractional seconds
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = isoFormatter.date(from: dateString) {
            return date
        }

        // Try simple format without timezone
        let simpleFormatter = DateFormatter()
        simpleFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        simpleFormatter.timeZone = TimeZone(identifier: "UTC")
        return simpleFormatter.date(from: dateString)
    }

    var formattedDate: String {
        guard let date = matchDate else { return "TBD" }
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    enum CodingKeys: String, CodingKey {
        case matchId = "match_id"
        case matchExternalId = "match_external_id"
        case homeTeam = "home_team"
        case awayTeam = "away_team"
        case homeTeamLogo = "home_team_logo"
        case awayTeamLogo = "away_team_logo"
        case date
        case status
        case homeGoals = "home_goals"
        case awayGoals = "away_goals"
        case leagueId = "league_id"
        case probabilities
        case fairOdds = "fair_odds"
        case marketOdds = "market_odds"
        case valueBets = "value_bets"
        case hasValueBet = "has_value_bet"
        case bestValueBet = "best_value_bet"
        case confidenceTier = "confidence_tier"
    }
}

struct Probabilities: Codable {
    let home: Double
    let draw: Double
    let away: Double

    var homePercent: String { String(format: "%.1f%%", home * 100) }
    var drawPercent: String { String(format: "%.1f%%", draw * 100) }
    var awayPercent: String { String(format: "%.1f%%", away * 100) }

    var predictedOutcome: String {
        if home > draw && home > away { return "Home Win" }
        if away > draw && away > home { return "Away Win" }
        return "Draw"
    }
}

struct FairOdds: Codable {
    let home: Double?
    let draw: Double?
    let away: Double?

    var homeFormatted: String { home.map { String(format: "%.2f", $0) } ?? "-" }
    var drawFormatted: String { draw.map { String(format: "%.2f", $0) } ?? "-" }
    var awayFormatted: String { away.map { String(format: "%.2f", $0) } ?? "-" }
}

struct MarketOdds: Codable {
    let home: Double?
    let draw: Double?
    let away: Double?
}

struct ValueBet: Codable, Identifiable {
    let outcome: String
    let ourProbability: Double?
    let impliedProbability: Double?
    let edge: Double?
    let edgePercentage: Double?
    let expectedValue: Double?
    let evPercentage: Double?
    let marketOdds: Double?
    let fairOdds: Double?
    let isValueBet: Bool?

    var id: String { outcome }

    /// Edge display (e.g., "+5.2%")
    var edgePercent: String {
        if let pct = edgePercentage {
            return String(format: "+%.1f%%", pct)
        }
        if let e = edge {
            return String(format: "+%.1f%%", e * 100)
        }
        return "—"
    }

    /// EV display (e.g., "+12.5% EV")
    var evDisplay: String {
        if let pct = evPercentage {
            return String(format: "+%.1f%% EV", pct)
        }
        if let ev = expectedValue {
            return String(format: "+%.1f%% EV", ev * 100)
        }
        return edgePercent
    }

    /// Color based on EV strength
    var strengthColor: String {
        let evValue = evPercentage ?? (expectedValue.map { $0 * 100 }) ?? (edge.map { $0 * 100 }) ?? 0
        if evValue >= 15 { return "gold" }      // Strong value
        if evValue >= 10 { return "green" }     // Good value
        return "yellow"                          // Moderate value
    }

    enum CodingKeys: String, CodingKey {
        case outcome
        case ourProbability = "our_probability"
        case impliedProbability = "implied_probability"
        case edge
        case edgePercentage = "edge_percentage"
        case expectedValue = "expected_value"
        case evPercentage = "ev_percentage"
        case marketOdds = "market_odds"
        case fairOdds = "fair_odds"
        case isValueBet = "is_value_bet"
    }
}

// MARK: - Match Details Response

struct MatchDetailsResponse: Codable {
    let match: MatchInfo
    let homeTeam: TeamWithHistory
    let awayTeam: TeamWithHistory
    let prediction: MatchPrediction?
    let standingsStatus: String?  // "hit" | "miss" | "skipped"

    enum CodingKeys: String, CodingKey {
        case match
        case homeTeam = "home_team"
        case awayTeam = "away_team"
        case prediction
        case standingsStatus = "standings_status"
    }

    /// Whether standings data is available
    var hasStandings: Bool {
        standingsStatus == "hit"
    }
}

struct MatchInfo: Codable {
    let id: Int
    let date: String?
    let leagueId: Int
    let status: String
    let homeGoals: Int?
    let awayGoals: Int?

    enum CodingKeys: String, CodingKey {
        case id
        case date
        case leagueId = "league_id"
        case status
        case homeGoals = "home_goals"
        case awayGoals = "away_goals"
    }
}

struct TeamWithHistory: Codable {
    let id: Int?
    let name: String
    let logo: String?
    let history: [MatchHistoryItem]
    let position: Int?
    let leaguePoints: Int?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case logo
        case history
        case position
        case leaguePoints = "league_points"
    }
}

struct MatchHistoryItem: Codable, Identifiable {
    let matchId: Int
    let date: String?
    let opponent: String
    let opponentLogo: String?
    let isHome: Bool
    let teamGoals: Int?
    let opponentGoals: Int?
    let result: String  // "W", "L", "D"
    let leagueId: Int

    var id: Int { matchId }

    var resultColor: String {
        switch result {
        case "W": return "green"
        case "L": return "red"
        default: return "gray"
        }
    }

    var scoreDisplay: String {
        let tg = teamGoals ?? 0
        let og = opponentGoals ?? 0
        return "\(tg) - \(og)"
    }

    var formattedDate: String {
        guard let dateString = date else { return "" }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        if let date = formatter.date(from: dateString) {
            let displayFormatter = DateFormatter()
            displayFormatter.dateFormat = "dd/MM"
            return displayFormatter.string(from: date)
        }
        return ""
    }

    enum CodingKeys: String, CodingKey {
        case matchId = "match_id"
        case date
        case opponent
        case opponentLogo = "opponent_logo"
        case isHome = "is_home"
        case teamGoals = "team_goals"
        case opponentGoals = "opponent_goals"
        case result
        case leagueId = "league_id"
    }
}

// MARK: - Team Item (for Teams list)

struct TeamItem: Codable, Identifiable {
    let id: Int
    let externalId: Int?
    let name: String
    let country: String?
    let teamType: String
    let logoUrl: String?

    enum CodingKeys: String, CodingKey {
        case id
        case externalId = "external_id"
        case name
        case country
        case teamType = "team_type"
        case logoUrl = "logo_url"
    }
}

// MARK: - Competition Item (for Competitions list)

struct CompetitionItem: Codable, Identifiable {
    let leagueId: Int
    let name: String
    let matchType: String
    let priority: String
    let matchWeight: Double

    var id: Int { leagueId }

    // Map league IDs to confederations
    var confederation: String {
        switch leagueId {
        // FIFA World Cup & Qualifiers
        case 1: return "FIFA"        // World Cup
        case 32: return "CONMEBOL"   // WC Qualifiers South America
        case 33: return "CAF"        // WC Qualifiers Africa
        case 34: return "UEFA"       // WC Qualifiers Europe
        case 35: return "AFC"        // WC Qualifiers Asia
        case 31: return "CONCACAF"   // WC Qualifiers CONCACAF

        // Continental
        case 9: return "CONMEBOL"    // Copa America
        case 4: return "UEFA"        // Euro
        case 6: return "CAF"         // Africa Cup
        case 7: return "AFC"         // Asian Cup
        case 8: return "CONCACAF"    // Gold Cup

        // Club Competitions - CONMEBOL
        case 13: return "CONMEBOL"   // Libertadores
        case 14: return "CONMEBOL"   // Sudamericana

        // Club Competitions - UEFA
        case 2: return "UEFA"        // Champions League
        case 3: return "UEFA"        // Europa League
        case 848: return "UEFA"      // Conference League

        // Club Competitions - CONCACAF
        case 16: return "CONCACAF"   // Champions Cup

        // Domestic Leagues - South America
        case 128: return "CONMEBOL"  // Argentina
        case 71: return "CONMEBOL"   // Brazil

        // Domestic Leagues - Europe
        case 39: return "UEFA"       // Premier League
        case 140: return "UEFA"      // La Liga
        case 135: return "UEFA"      // Serie A
        case 78: return "UEFA"       // Bundesliga
        case 61: return "UEFA"       // Ligue 1

        // Youth
        case 438: return "CONMEBOL"  // U17 South America
        case 440: return "AFC"       // U17 Asia
        case 506: return "FIFA"      // U17 World Cup
        case 507: return "FIFA"      // U20 World Cup

        default: return "Other"
        }
    }

    enum CodingKeys: String, CodingKey {
        case leagueId = "league_id"
        case name
        case matchType = "match_type"
        case priority
        case matchWeight = "match_weight"
    }
}

// MARK: - League Standings

struct StandingsResponse: Codable {
    let leagueId: Int
    let season: Int
    let standings: [StandingsEntry]

    enum CodingKeys: String, CodingKey {
        case leagueId = "league_id"
        case season
        case standings
    }
}

struct StandingsEntry: Codable, Identifiable {
    let position: Int
    let teamId: Int
    let teamName: String
    let teamLogo: String?
    let points: Int
    let played: Int
    let won: Int
    let drawn: Int
    let lost: Int
    let goalsFor: Int
    let goalsAgainst: Int
    let goalDiff: Int
    let form: String

    var id: Int { teamId }

    /// Form as array of results (W, D, L)
    var formArray: [String] {
        form.map { String($0) }
    }

    enum CodingKeys: String, CodingKey {
        case position
        case teamId = "team_id"
        case teamName = "team_name"
        case teamLogo = "team_logo"
        case points
        case played
        case won
        case drawn
        case lost
        case goalsFor = "goals_for"
        case goalsAgainst = "goals_against"
        case goalDiff = "goal_diff"
        case form
    }
}

// MARK: - Match Timeline

struct MatchTimelineResponse: Codable {
    let matchId: Int
    let status: String
    let finalScore: TimelineScore
    let prediction: TimelinePrediction
    let totalMinutes: Int
    let goals: [TimelineGoal]
    let segments: [TimelineSegment]
    let summary: TimelineSummary
    let meta: TimelineMeta?  // Optional telemetry metadata

    enum CodingKeys: String, CodingKey {
        case matchId = "match_id"
        case status
        case finalScore = "final_score"
        case prediction
        case totalMinutes = "total_minutes"
        case goals
        case segments
        case summary
        case meta = "_meta"
    }
}

struct TimelineMeta: Codable {
    let eventsSource: String?
    let eventsCount: Int?

    enum CodingKeys: String, CodingKey {
        case eventsSource = "events_source"
        case eventsCount = "events_count"
    }
}

struct TimelineScore: Codable {
    let home: Int?  // Optional: may be null for incomplete match data
    let away: Int?  // Optional: may be null for incomplete match data
}

struct TimelinePrediction: Codable {
    let outcome: String  // "home", "draw", "away"
    let homeProb: Double
    let drawProb: Double
    let awayProb: Double
    let correct: Bool?  // Optional: may be null for incomplete data

    enum CodingKeys: String, CodingKey {
        case outcome
        case homeProb = "home_prob"
        case drawProb = "draw_prob"
        case awayProb = "away_prob"
        case correct
    }
}

struct TimelineGoal: Codable, Identifiable {
    let minute: Int
    let extraMinute: Int?
    let team: String  // "home" or "away"
    let teamName: String?  // Optional: may be null in legacy events
    let player: String?
    let isOwnGoal: Bool?  // Optional: may be null in legacy events
    let isPenalty: Bool?  // Optional: may be null in legacy events

    var id: String {
        "\(minute)-\(team)-\(player ?? "unknown")"
    }

    /// Display minute (e.g., "45+2" or "67")
    var displayMinute: String {
        if let extra = extraMinute, extra > 0 {
            return "\(minute)+\(extra)"
        }
        return "\(minute)"
    }

    /// Effective minute for positioning on timeline
    var effectiveMinute: Int {
        minute + (extraMinute ?? 0)
    }

    enum CodingKeys: String, CodingKey {
        case minute
        case extraMinute = "extra_minute"
        case team
        case teamName = "team_name"
        case player
        case isOwnGoal = "is_own_goal"
        case isPenalty = "is_penalty"
    }
}

struct TimelineSegment: Codable, Identifiable {
    let startMinute: Int
    let endMinute: Int
    let homeGoals: Int
    let awayGoals: Int
    let status: String  // "correct", "neutral", "wrong"

    var id: String {
        "\(startMinute)-\(endMinute)"
    }

    /// Duration of this segment in minutes
    var duration: Int {
        endMinute - startMinute
    }

    enum CodingKeys: String, CodingKey {
        case startMinute = "start_minute"
        case endMinute = "end_minute"
        case homeGoals = "home_goals"
        case awayGoals = "away_goals"
        case status
    }
}

struct TimelineSummary: Codable {
    let correctMinutes: Double
    let correctPercentage: Double

    enum CodingKeys: String, CodingKey {
        case correctMinutes = "correct_minutes"
        case correctPercentage = "correct_percentage"
    }
}

// MARK: - Match Insights Response

struct MatchInsightsResponse: Codable {
    let matchId: Int
    let predictionCorrect: Bool
    let predictedResult: String
    let actualResult: String
    let confidence: Double
    let deviationType: String
    let insights: [MatchInsightItem]
    let momentumAnalysis: MatchMomentumAnalysis?

    // LLM Narrative (new system - replaces heuristic insights)
    let llmNarrative: LLMNarrativePayload?
    let llmNarrativeStatus: String?

    // Match stats for UI table (independent of narrative)
    let matchStats: MatchStats?
    let matchEvents: [MatchEvent]?

    enum CodingKeys: String, CodingKey {
        case matchId = "match_id"
        case predictionCorrect = "prediction_correct"
        case predictedResult = "predicted_result"
        case actualResult = "actual_result"
        case confidence
        case deviationType = "deviation_type"
        case insights
        case momentumAnalysis = "momentum_analysis"
        case llmNarrative = "llm_narrative"
        case llmNarrativeStatus = "llm_narrative_status"
        case matchStats = "match_stats"
        case matchEvents = "match_events"
    }
}

// MARK: - Match Stats (from API-Football)

struct MatchStats: Codable {
    let home: TeamStats?
    let away: TeamStats?
}

struct TeamStats: Codable {
    let ballPossession: Double?  // Backend sends as float or string (e.g., 43.0 or "43%")
    let totalShots: Int?
    let shotsOnGoal: Int?
    let shotsOffGoal: Int?
    let blockedShots: Int?
    let shotsInsidebox: Int?
    let shotsOutsidebox: Int?
    let fouls: Int?
    let cornerKicks: Int?
    let offsides: Int?
    let yellowCards: Int?
    let redCards: Int?
    let goalkeeperSaves: Int?
    let totalPasses: Int?
    let passesAccurate: Int?
    let passesPct: Double?  // Backend sends as float or string (e.g., 82.0 or "82%")
    let expectedGoals: String?

    /// Formatted possession for display (e.g., "43%")
    var possessionDisplay: String {
        guard let poss = ballPossession else { return "-" }
        return "\(Int(poss))%"
    }

    /// Formatted pass accuracy for display (e.g., "82%")
    var passAccuracyDisplay: String {
        guard let pct = passesPct else { return "-" }
        return "\(Int(pct))%"
    }

    enum CodingKeys: String, CodingKey {
        case ballPossession = "ball_possession"
        case totalShots = "total_shots"
        case shotsOnGoal = "shots_on_goal"
        case shotsOffGoal = "shots_off_goal"
        case blockedShots = "blocked_shots"
        case shotsInsidebox = "shots_insidebox"
        case shotsOutsidebox = "shots_outsidebox"
        case fouls
        case cornerKicks = "corner_kicks"
        case offsides
        case yellowCards = "yellow_cards"
        case redCards = "red_cards"
        case goalkeeperSaves = "goalkeeper_saves"
        case totalPasses = "total_passes"
        case passesAccurate = "passes_accurate"
        case passesPct = "passes_pct"
        case expectedGoals = "expected_goals"
    }

    // Memberwise initializer for previews and tests
    init(
        ballPossession: Double? = nil,
        totalShots: Int? = nil,
        shotsOnGoal: Int? = nil,
        shotsOffGoal: Int? = nil,
        blockedShots: Int? = nil,
        shotsInsidebox: Int? = nil,
        shotsOutsidebox: Int? = nil,
        fouls: Int? = nil,
        cornerKicks: Int? = nil,
        offsides: Int? = nil,
        yellowCards: Int? = nil,
        redCards: Int? = nil,
        goalkeeperSaves: Int? = nil,
        totalPasses: Int? = nil,
        passesAccurate: Int? = nil,
        passesPct: Double? = nil,
        expectedGoals: String? = nil
    ) {
        self.ballPossession = ballPossession
        self.totalShots = totalShots
        self.shotsOnGoal = shotsOnGoal
        self.shotsOffGoal = shotsOffGoal
        self.blockedShots = blockedShots
        self.shotsInsidebox = shotsInsidebox
        self.shotsOutsidebox = shotsOutsidebox
        self.fouls = fouls
        self.cornerKicks = cornerKicks
        self.offsides = offsides
        self.yellowCards = yellowCards
        self.redCards = redCards
        self.goalkeeperSaves = goalkeeperSaves
        self.totalPasses = totalPasses
        self.passesAccurate = passesAccurate
        self.passesPct = passesPct
        self.expectedGoals = expectedGoals
    }

    // Custom decoder to handle String or Double for possession/passesPct
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // ballPossession: try Double first, then String
        if let doubleValue = try? container.decode(Double.self, forKey: .ballPossession) {
            ballPossession = doubleValue
        } else if let stringValue = try? container.decode(String.self, forKey: .ballPossession) {
            // Parse "43%" or "43" to Double
            let cleaned = stringValue.replacingOccurrences(of: "%", with: "").trimmingCharacters(in: .whitespaces)
            ballPossession = Double(cleaned)
        } else {
            ballPossession = nil
        }

        // passesPct: try Double first, then String
        if let doubleValue = try? container.decode(Double.self, forKey: .passesPct) {
            passesPct = doubleValue
        } else if let stringValue = try? container.decode(String.self, forKey: .passesPct) {
            let cleaned = stringValue.replacingOccurrences(of: "%", with: "").trimmingCharacters(in: .whitespaces)
            passesPct = Double(cleaned)
        } else {
            passesPct = nil
        }

        // Standard decoding for the rest
        totalShots = try? container.decode(Int.self, forKey: .totalShots)
        shotsOnGoal = try? container.decode(Int.self, forKey: .shotsOnGoal)
        shotsOffGoal = try? container.decode(Int.self, forKey: .shotsOffGoal)
        blockedShots = try? container.decode(Int.self, forKey: .blockedShots)
        shotsInsidebox = try? container.decode(Int.self, forKey: .shotsInsidebox)
        shotsOutsidebox = try? container.decode(Int.self, forKey: .shotsOutsidebox)
        fouls = try? container.decode(Int.self, forKey: .fouls)
        cornerKicks = try? container.decode(Int.self, forKey: .cornerKicks)
        offsides = try? container.decode(Int.self, forKey: .offsides)
        yellowCards = try? container.decode(Int.self, forKey: .yellowCards)
        redCards = try? container.decode(Int.self, forKey: .redCards)
        goalkeeperSaves = try? container.decode(Int.self, forKey: .goalkeeperSaves)
        totalPasses = try? container.decode(Int.self, forKey: .totalPasses)
        passesAccurate = try? container.decode(Int.self, forKey: .passesAccurate)
        expectedGoals = try? container.decode(String.self, forKey: .expectedGoals)
    }
}

// MARK: - Match Events

struct MatchEvent: Codable, Identifiable {
    var id: String { "\(minute ?? 0)-\(type ?? "")-\(player ?? "")" }
    let minute: Int?
    let extraMinute: Int?
    let type: String?
    let detail: String?
    let team: String?
    let player: String?
    let assist: String?

    enum CodingKeys: String, CodingKey {
        case minute
        case extraMinute = "extra_minute"
        case type
        case detail
        case team
        case player
        case assist
    }

    var displayMinute: String {
        if let extra = extraMinute, extra > 0 {
            return "\(minute ?? 0)+\(extra)'"
        }
        return "\(minute ?? 0)'"
    }

    var typeIcon: String {
        switch type {
        case "Goal": return "soccerball"
        case "Card":
            if detail == "Red Card" { return "rectangle.fill" }
            return "rectangle.fill"
        case "subst": return "arrow.left.arrow.right"
        case "Var": return "tv"
        default: return "circle.fill"
        }
    }

    var typeColor: Color {
        switch type {
        case "Goal": return .green
        case "Card":
            if detail == "Red Card" { return .red }
            return .yellow
        case "subst": return .blue
        case "Var": return .purple
        default: return .gray
        }
    }
}

struct MatchInsightItem: Codable, Identifiable {
    var id: UUID { UUID() }
    let type: String
    let icon: String
    let message: String
    let priority: Int

    enum CodingKeys: String, CodingKey {
        case type, icon, message, priority
    }
}

struct MatchMomentumAnalysis: Codable {
    let type: String
    let icon: String
    let message: String
}

// MARK: - LLM Narrative Models (Schema v3.2)

/// Top-level LLM narrative payload from backend
struct LLMNarrativePayload: Codable {
    let matchId: Int?
    let lang: String?
    let result: LLMResult?
    let prediction: LLMPrediction?
    let narrative: LLMNarrative?
    let marketOdds: LLMMarketOdds?  // Market odds at prediction time

    enum CodingKeys: String, CodingKey {
        case matchId = "match_id"
        case lang
        case result
        case prediction
        case narrative
        case marketOdds = "market_odds"
    }
}

/// Match result info
struct LLMResult: Codable {
    let ftScore: String?
    let outcome: String?  // "home", "draw", "away"
    let betWon: Bool?

    enum CodingKeys: String, CodingKey {
        case ftScore = "ft_score"
        case outcome
        case betWon = "bet_won"
    }
}

/// Prediction info
struct LLMPrediction: Codable {
    let selection: String?  // "HOME", "DRAW", "AWAY"
    let confidence: Double?
    let probabilities: LLMProbabilities?
    let marketOdds: LLMMarketOdds?  // Note: backend uses "market_odds" at narrative level, not here

    enum CodingKeys: String, CodingKey {
        case selection
        case confidence
        case probabilities
        case marketOdds = "market_odds"
    }
}

/// Probabilities from LLM prediction
struct LLMProbabilities: Codable {
    let home: Double?
    let draw: Double?
    let away: Double?
}

/// Market odds at prediction time
struct LLMMarketOdds: Codable {
    let home: Double?
    let draw: Double?
    let away: Double?
}

/// The narrative content
struct LLMNarrative: Codable {
    let title: String?
    let body: String?
    let keyFactors: [LLMKeyFactor]?
    let tone: String?  // "reinforce_win" or "mitigate_loss"
    let responsibleNote: String?

    enum CodingKeys: String, CodingKey {
        case title
        case body
        case keyFactors = "key_factors"
        case tone
        case responsibleNote = "responsible_note"
    }
}

/// Key factor in narrative
struct LLMKeyFactor: Codable, Identifiable {
    var id: String { label ?? UUID().uuidString }
    let label: String?
    let evidence: String?
    let direction: String?  // "pro-pick", "anti-pick", "neutral"
}

// MARK: - Health Check

struct HealthResponse: Codable {
    let status: String
    let modelLoaded: Bool

    enum CodingKeys: String, CodingKey {
        case status
        case modelLoaded = "model_loaded"
    }
}

// MARK: - Training Response

struct TrainingResponse: Codable {
    let modelVersion: String
    let brierScore: Double
    let samplesTrained: Int
    let featureImportance: [String: Double]

    enum CodingKeys: String, CodingKey {
        case modelVersion = "model_version"
        case brierScore = "brier_score"
        case samplesTrained = "samples_trained"
        case featureImportance = "feature_importance"
    }
}
