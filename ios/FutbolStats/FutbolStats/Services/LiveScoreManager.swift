import Foundation

/// Response model for /live-summary endpoint (v2 - FASE 1: includes events)
struct LiveSummary: Codable {
    let ts: Int
    let matches: [String: LiveScore]  // Key is match_id as String (JSON keys are strings)

    struct LiveScore: Codable {
        let s: String      // status
        let e: Int         // elapsed
        let ex: Int        // elapsed_extra
        let h: Int         // home goals
        let a: Int         // away goals
        let ev: [LiveEvent]?  // FASE 1: live events (goals, cards)

        var status: String { s }
        var elapsed: Int { e }
        var elapsedExtra: Int { ex }
        var homeGoals: Int { h }
        var awayGoals: Int { a }
        var events: [LiveEvent]? { ev }
    }

    /// Compact live event from /live-summary (FASE 1)
    struct LiveEvent: Codable, Identifiable {
        let m: Int?        // minute
        let x: Int?        // extra minute (injury time)
        let t: String?     // type: Goal, Card
        let d: String?     // detail: Normal Goal, Yellow Card, Red Card, Penalty, Own Goal
        let tm: Int?       // team_id
        let p: String?     // player name
        let a: String?     // assist name (goals only)

        var id: String {
            "\(m ?? 0)-\(t ?? "")-\(p ?? "")"
        }

        var minute: Int { m ?? 0 }
        var extraMinute: Int? { x }
        var type: String { t ?? "" }
        var detail: String { d ?? "" }
        var teamId: Int? { tm }
        var playerName: String { p ?? "Unknown" }
        var assistName: String? { a }

        /// Display minute (e.g., "45+2'" or "67'")
        var displayMinute: String {
            if let extra = x, extra > 0 {
                return "\(m ?? 0)+\(extra)'"
            }
            return "\(m ?? 0)'"
        }

        /// Is this a goal event?
        var isGoal: Bool { t == "Goal" }

        /// Is this a card event?
        var isCard: Bool { t == "Card" }

        /// Is this a red card?
        var isRedCard: Bool { t == "Card" && d == "Red Card" }
    }
}

/// Centralized manager for live score polling.
///
/// Implements smart polling per Auditor requirements:
/// - Only polls when there are live matches OR user is viewing a live match
/// - scenePhase-aware: stops polling when app goes to background
/// - Backoff to 60s when no live matches
/// - Updates MatchCache for list view overlay
///
/// Usage:
/// ```swift
/// // In App.swift onChange of scenePhase:
/// .onChange(of: scenePhase) { _, newPhase in
///     switch newPhase {
///     case .active:
///         LiveScoreManager.shared.onAppBecameActive()
///     case .inactive, .background:
///         LiveScoreManager.shared.onAppBecameInactive()
///     }
/// }
///
/// // In PredictionsViewModel after loading predictions:
/// LiveScoreManager.shared.updateLiveMatchIds(liveMatchIds)
/// ```
@MainActor
final class LiveScoreManager: ObservableObject {
    static let shared = LiveScoreManager()

    // MARK: - Configuration

    /// Normal polling interval when there are live matches (10s per Auditor v2)
    private let normalPollInterval: TimeInterval = 10.0

    /// Burst polling interval after score change (3s x 3 times)
    private let burstPollInterval: TimeInterval = 3.0

    /// Backoff interval when no live matches (60s per Auditor)
    private let backoffPollInterval: TimeInterval = 60.0

    /// Current polling interval (adjusts based on live match count and burst mode)
    private var currentPollInterval: TimeInterval = 10.0

    /// Burst mode: rapid polling after score change
    private var burstRemaining: Int = 0

    /// Last known scores to detect changes (matchId -> "home-away")
    private var lastKnownScores: [Int: String] = [:]

    // MARK: - State

    /// Task handle for the polling loop
    private var pollTask: Task<Void, Never>?

    /// Whether the app is in foreground (scenePhase == .active)
    private var isAppActive = false

    /// Set of match IDs that are currently LIVE in the predictions list
    private var liveMatchIdsFromList: Set<Int> = []

    /// Match ID being viewed in MatchDetailView (if any)
    private var viewingMatchId: Int?

    /// Whether polling is currently running
    @Published private(set) var isPolling = false

    /// Last poll timestamp (for debugging)
    @Published private(set) var lastPollAt: Date?

    /// Number of live matches from last response
    @Published private(set) var liveMatchCount: Int = 0

    // MARK: - Init

    private init() {}

    // MARK: - Lifecycle (scenePhase aware)

    /// Call when app becomes active (scenePhase == .active)
    func onAppBecameActive() {
        isAppActive = true
        evaluatePollingState()
    }

    /// Call when app becomes inactive/background
    func onAppBecameInactive() {
        isAppActive = false
        stopPolling()
    }

    // MARK: - Gating (Auditor requirement: only poll when needed)

    /// Update the set of live match IDs from PredictionsListView
    /// Called after predictions are loaded/refreshed
    func updateLiveMatchIds(_ matchIds: Set<Int>) {
        liveMatchIdsFromList = matchIds
        evaluatePollingState()
    }

    /// Set the match ID being viewed in MatchDetailView
    /// Pass nil when leaving the detail view
    func setViewingMatch(_ matchId: Int?) {
        viewingMatchId = matchId
        evaluatePollingState()
    }

    /// Determine if we should be polling based on current state
    /// FIX (Auditor v2): viewingMatchId should trigger polling independently of liveMatchIdsFromList
    private var shouldPoll: Bool {
        guard isAppActive else { return false }

        // Poll if viewing ANY match in detail (user expects live updates)
        // This fixes the bug where detail wouldn't poll if list didn't mark it as live
        if viewingMatchId != nil {
            return true
        }

        // Poll if there are live matches in the list
        if !liveMatchIdsFromList.isEmpty {
            return true
        }

        // No live matches and not viewing one - don't poll
        return false
    }

    /// Re-evaluate whether we should be polling
    private func evaluatePollingState() {
        if shouldPoll {
            startPolling()
        } else {
            stopPolling()
        }
    }

    // MARK: - Polling Control

    private func startPolling() {
        guard pollTask == nil else { return }
        isPolling = true

        print("[LiveScoreManager] Starting polling (interval: \(currentPollInterval)s, liveMatches: \(liveMatchIdsFromList.count))")

        pollTask = Task {
            while !Task.isCancelled {
                let scoreChanged = await fetchLiveSummary()

                // Determine next interval
                let interval: TimeInterval
                if scoreChanged {
                    // Score changed - enter burst mode (3s x 3)
                    burstRemaining = 3
                    interval = burstPollInterval
                    print("[LiveScoreManager] BURST MODE: Score changed, polling every \(burstPollInterval)s for \(burstRemaining) cycles")
                } else if burstRemaining > 0 {
                    // Continue burst mode
                    burstRemaining -= 1
                    interval = burstPollInterval
                    print("[LiveScoreManager] BURST MODE: \(burstRemaining) cycles remaining")
                } else if liveMatchCount > 0 {
                    // Normal live polling
                    interval = normalPollInterval
                } else {
                    // No live matches - backoff
                    interval = backoffPollInterval
                }

                if interval != currentPollInterval {
                    currentPollInterval = interval
                    print("[LiveScoreManager] Adjusted poll interval to \(interval)s (liveMatches: \(liveMatchCount))")
                }

                try? await Task.sleep(for: .seconds(currentPollInterval))
            }
        }
    }

    private func stopPolling() {
        guard pollTask != nil else { return }

        print("[LiveScoreManager] Stopping polling")
        pollTask?.cancel()
        pollTask = nil
        isPolling = false
    }

    // MARK: - Fetch

    /// Fetch live summary and update cache
    /// - Returns: true if any score changed (triggers burst mode)
    @discardableResult
    private func fetchLiveSummary() async -> Bool {
        let baseURL = APIEnvironment.current.baseURL
        guard let url = URL(string: "\(baseURL)/live-summary") else {
            print("[LiveScoreManager] Invalid URL")
            return false
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        // Add API key header (required by backend)
        if let apiKey = AppConfiguration.apiKey {
            request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        } else {
            print("[LiveScoreManager] Warning: No API key configured")
        }

        do {
            let startTime = Date()
            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                print("[LiveScoreManager] Invalid response type")
                return false
            }

            guard httpResponse.statusCode == 200 else {
                print("[LiveScoreManager] HTTP \(httpResponse.statusCode)")
                return false
            }

            let summary = try JSONDecoder().decode(LiveSummary.self, from: data)
            let latencyMs = Date().timeIntervalSince(startTime) * 1000

            // Update state
            lastPollAt = Date()
            liveMatchCount = summary.matches.count

            // Detect score changes and update cache
            var anyScoreChanged = false
            for (matchIdStr, score) in summary.matches {
                guard let matchId = Int(matchIdStr) else { continue }

                // Check for score change
                let currentScore = "\(score.homeGoals)-\(score.awayGoals)"
                if let lastScore = lastKnownScores[matchId], lastScore != currentScore {
                    print("[LiveScoreManager] GOAL DETECTED: Match \(matchId) score changed \(lastScore) -> \(currentScore)")
                    anyScoreChanged = true
                }
                lastKnownScores[matchId] = currentScore

                // Update cache (FASE 1: now includes events)
                MatchCache.shared.update(
                    matchId: matchId,
                    status: score.status,
                    elapsed: score.elapsed,
                    elapsedExtra: score.elapsedExtra,
                    homeGoals: score.homeGoals,
                    awayGoals: score.awayGoals,
                    events: score.events
                )
            }

            print("[LiveScoreManager] Fetched \(summary.matches.count) live matches in \(String(format: "%.1f", latencyMs))ms")
            return anyScoreChanged

        } catch {
            print("[LiveScoreManager] Error: \(error.localizedDescription)")
            return false
        }
    }
}
