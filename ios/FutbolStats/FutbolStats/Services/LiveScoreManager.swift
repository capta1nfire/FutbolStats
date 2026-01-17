import Foundation

/// Response model for /live-summary endpoint
struct LiveSummary: Codable {
    let ts: Int
    let matches: [String: LiveScore]  // Key is match_id as String (JSON keys are strings)

    struct LiveScore: Codable {
        let s: String      // status
        let e: Int         // elapsed
        let ex: Int        // elapsed_extra
        let h: Int         // home goals
        let a: Int         // away goals

        var status: String { s }
        var elapsed: Int { e }
        var elapsedExtra: Int { ex }
        var homeGoals: Int { h }
        var awayGoals: Int { a }
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

    /// Normal polling interval when there are live matches (15s per Auditor)
    private let normalPollInterval: TimeInterval = 15.0

    /// Backoff interval when no live matches (60s per Auditor)
    private let backoffPollInterval: TimeInterval = 60.0

    /// Current polling interval (adjusts based on live match count)
    private var currentPollInterval: TimeInterval = 15.0

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
    private var shouldPoll: Bool {
        guard isAppActive else { return false }

        // Poll if there are live matches in the list
        if !liveMatchIdsFromList.isEmpty {
            return true
        }

        // Poll if viewing a live match in detail
        if let viewingId = viewingMatchId, liveMatchIdsFromList.contains(viewingId) {
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
                await fetchLiveSummary()

                // Adjust interval based on results
                let interval = liveMatchCount > 0 ? normalPollInterval : backoffPollInterval
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

    private func fetchLiveSummary() async {
        let baseURL = APIEnvironment.current.baseURL
        guard let url = URL(string: "\(baseURL)/live-summary") else {
            print("[LiveScoreManager] Invalid URL")
            return
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
                return
            }

            guard httpResponse.statusCode == 200 else {
                print("[LiveScoreManager] HTTP \(httpResponse.statusCode)")
                return
            }

            let summary = try JSONDecoder().decode(LiveSummary.self, from: data)
            let latencyMs = Date().timeIntervalSince(startTime) * 1000

            // Update state
            lastPollAt = Date()
            liveMatchCount = summary.matches.count

            // Update MatchCache with fresh data
            for (matchIdStr, score) in summary.matches {
                guard let matchId = Int(matchIdStr) else { continue }
                MatchCache.shared.update(
                    matchId: matchId,
                    status: score.status,
                    elapsed: score.elapsed,
                    elapsedExtra: score.elapsedExtra,
                    homeGoals: score.homeGoals,
                    awayGoals: score.awayGoals
                )
            }

            print("[LiveScoreManager] Fetched \(summary.matches.count) live matches in \(String(format: "%.1f", latencyMs))ms")

        } catch {
            print("[LiveScoreManager] Error: \(error.localizedDescription)")
        }
    }
}
