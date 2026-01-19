import Foundation
import SwiftUI

@MainActor
class PredictionsViewModel: ObservableObject {
    @Published var predictions: [MatchPrediction] = [] {
        didSet {
            rebuildMatchCountCache()
            updateCachedPredictions()
            predictionsLoadedAt = Date()  // Track when predictions were loaded for local clock
        }
    }
    @Published var isLoading = false
    @Published var error: String?
    @Published var modelLoaded = false
    @Published var lastUpdated: Date?

    // MARK: - Live Match Clock
    /// Timestamp when predictions were loaded (for calculating elapsed time locally)
    @Published private(set) var predictionsLoadedAt: Date = Date()
    /// Current time - updated every 60s to trigger UI refresh for live matches
    @Published private(set) var clockTick: Date = Date()
    private var clockTimer: Timer?

    // MARK: - Match Cache Observer
    private var cacheObserver: NSObjectProtocol?
    @Published var selectedDate: Date {
        didSet {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withFullDate]
            // Immediate print (sync) to see timing
            print("[DateChange] \(formatter.string(from: oldValue)) -> \(formatter.string(from: selectedDate))")
            PerfLogger.shared.log(
                endpoint: "selectedDateChange",
                message: "date_changed",
                data: [
                    "old_date": formatter.string(from: oldValue),
                    "new_date": formatter.string(from: selectedDate)
                ]
            )
            updateCachedPredictions()
        }
    }
    @Published var opsProgress: OpsProgress?
    @Published var isLoadingMore = false  // Progressive loading indicator

    // Request tracking for race condition prevention
    private var currentRequestId: UUID?

    // Throttle refreshes to avoid duplicate calls on navigation
    // 30s is enough to navigate to details and back without re-fetching
    private var lastRefreshTime: Date?
    private let refreshThrottleInterval: TimeInterval = 30.0

    // Cancelable task for full refresh (allows cancellation on view disappear)
    private var fullRefreshTask: Task<Void, Never>?

    // MARK: - Cached Filtered Predictions (avoid recomputation on every body eval)
    @Published private(set) var cachedPredictionsForDate: [MatchPrediction] = []
    @Published private(set) var cachedValueBets: [MatchPrediction] = []
    @Published private(set) var cachedRegularMatches: [MatchPrediction] = []

    private let apiClient = APIClient.shared

    // Local calendar for all date operations (user sees dates in their timezone)
    // Matches are grouped by their kickoff time in user's local timezone
    private var localCalendar: Calendar {
        Calendar.current
    }

    init() {
        // Initialize selectedDate to today in LOCAL timezone
        // User expects "Today" to mean their local day, not UTC
        self.selectedDate = Calendar.current.startOfDay(for: Date())
        startClockTimer()
        setupCacheObserver()
    }

    deinit {
        clockTimer?.invalidate()
        if let observer = cacheObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    // MARK: - Cache Observer

    /// Subscribe to MatchCache updates to trigger UI refresh for live match overlays
    private func setupCacheObserver() {
        cacheObserver = NotificationCenter.default.addObserver(
            forName: MatchCache.didUpdateNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let matchId = notification.object as? Int else { return }
            // Dispatch to MainActor for thread safety (Swift 6 strict concurrency)
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                // Only trigger refresh if this match is in the current view
                if self.cachedPredictionsForDate.contains(where: { $0.matchId == matchId }) {
                    // Trigger clockTick to force UI refresh (minimal re-render)
                    self.clockTick = Date()
                    print("[CacheObserver] Match \(matchId) updated, triggering refresh")
                }
            }
        }
    }

    // MARK: - Live Clock Timer

    /// Start the 60-second timer for updating live match minutes
    private func startClockTimer() {
        clockTimer?.invalidate()
        clockTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.clockTick = Date()
            }
        }
    }

    /// Calculate the current elapsed minute for a live match
    /// Uses cached data if fresher, otherwise falls back to API elapsed + local time,
    /// or calculates from kickoff time if no elapsed data available
    /// - Parameter prediction: The match prediction
    /// - Returns: Formatted elapsed string (e.g., "32'", "45+2'", "90+3'", or status like "HT")
    func calculatedElapsedDisplay(for prediction: MatchPrediction) -> String {
        // Check cache overlay first (fresher data from MatchDetailView polling)
        let (status, elapsed, elapsedExtra, loadedAt): (String?, Int?, Int?, Date) = {
            if let matchId = prediction.matchId,
               let cached = MatchCache.shared.get(matchId: matchId) {
                return (cached.status, cached.elapsed, cached.elapsedExtra, cached.updatedAt)
            }
            return (prediction.status, prediction.elapsed, prediction.elapsedExtra, predictionsLoadedAt)
        }()

        guard let status = status else { return "LIVE" }

        // Only calculate for active play statuses
        let activeStatuses = ["1H", "2H", "LIVE"]
        guard activeStatuses.contains(status) else {
            // For HT, ET, BT, P, etc. - show status code
            return status
        }

        // If no elapsed from backend, calculate from kickoff time
        guard let baseElapsed = elapsed else {
            // Use kickoff-based calculation from MatchPrediction
            if let calculatedMins = prediction.calculatedElapsed(at: clockTick) {
                return "\(calculatedMins)'"
            }
            return status
        }

        // If we have injury/added time from API, show it directly (e.g., "90+3'")
        if let extra = elapsedExtra, extra > 0 {
            return "\(baseElapsed)+\(extra)'"
        }

        // At regulation time limits, don't calculate locally - wait for API injury time
        if status == "1H" && baseElapsed >= 45 {
            return "45'"
        } else if status == "2H" && baseElapsed >= 90 {
            return "90'"
        }

        // Calculate time passed since data was loaded (local clock estimation)
        let secondsPassed = clockTick.timeIntervalSince(loadedAt)
        let totalSeconds = (baseElapsed * 60) + Int(secondsPassed)
        let displayMinutes = totalSeconds / 60

        // Apply caps - stop local calculation at regulation time
        if status == "1H" && displayMinutes >= 45 {
            return "45'"
        } else if status == "2H" && displayMinutes >= 90 {
            return "90'"
        }

        return "\(displayMinutes)'"
    }

    /// Get overlayed score for a prediction (uses cache if fresher)
    /// - Parameter prediction: The match prediction
    /// - Returns: Tuple of (homeGoals, awayGoals) with cache overlay applied
    func overlayedScore(for prediction: MatchPrediction) -> (home: Int?, away: Int?) {
        if let matchId = prediction.matchId,
           let cached = MatchCache.shared.get(matchId: matchId) {
            return (cached.homeGoals, cached.awayGoals)
        }
        return (prediction.homeGoals, prediction.awayGoals)
    }

    /// Get overlayed status for a prediction (uses cache if fresher)
    /// - Parameter prediction: The match prediction
    /// - Returns: Status string with cache overlay applied
    func overlayedStatus(for prediction: MatchPrediction) -> String? {
        if let matchId = prediction.matchId,
           let cached = MatchCache.shared.get(matchId: matchId) {
            return cached.status
        }
        return prediction.status
    }

    /// Check if prediction is live (uses cache if fresher)
    func isLive(for prediction: MatchPrediction) -> Bool {
        let status = overlayedStatus(for: prediction)
        guard let s = status else { return false }
        return ["1H", "2H", "HT", "ET", "BT", "P", "LIVE"].contains(s)
    }

    /// Check if prediction is finished (uses cache if fresher)
    func isFinished(for prediction: MatchPrediction) -> Bool {
        let status = overlayedStatus(for: prediction)
        return status == "FT" || status == "AET" || status == "PEN"
    }

    /// Check if prediction has score to display (uses cache if fresher)
    func hasScore(for prediction: MatchPrediction) -> Bool {
        let live = isLive(for: prediction)
        let finished = isFinished(for: prediction)
        let (home, away) = overlayedScore(for: prediction)
        return (live || finished) && home != nil && away != nil
    }

    // MARK: - LiveScoreManager Integration

    /// Extract live match IDs and update LiveScoreManager for gating
    /// Called after predictions are loaded to enable/disable live polling
    private func updateLiveScoreManagerGating() {
        let liveMatchIds = Set(
            predictions
                .filter { isLive(for: $0) }
                .compactMap { $0.matchId }
        )
        LiveScoreManager.shared.updateLiveMatchIds(liveMatchIds)
        print("[LiveScore] Updated gating with \(liveMatchIds.count) live matches")
    }

    // MARK: - Load Predictions

    /// Load predictions with progressive loading support
    /// - Parameters:
    ///   - daysBack: Past N days for finished matches
    ///   - daysAhead: Future N days for upcoming matches
    ///   - mode: "priority" for initial fast load, "full" for complete dataset
    ///   - requestId: UUID to track request validity (prevents race conditions)
    func loadPredictions(daysBack: Int = 7, daysAhead: Int = 7, mode: String = "full", requestId: UUID? = nil) async {
        // For priority mode, set loading state
        if mode == "priority" {
            isLoading = true
            error = nil
        }

        let totalTimer = PerfTimer()
        print("[Perf] loadPredictions(\(mode), back=\(daysBack), ahead=\(daysAhead)) START")

        do {
            let networkTimer = PerfTimer()
            let response = try await apiClient.getUpcomingPredictions(daysBack: daysBack, daysAhead: daysAhead)
            let networkMs = networkTimer.elapsedMs
            print("[Perf] loadPredictions(\(mode)) NETWORK done: \(String(format: "%.0f", networkMs))ms")

            // Check if this request is still valid (prevents race conditions)
            if let reqId = requestId, reqId != currentRequestId {
                print("[Perf] loadPredictions(\(mode)) STALE - discarding (requestId mismatch)")
                return
            }

            let assignTimer = PerfTimer()

            if mode == "full" {
                // Full mode: replace all predictions
                predictions = response.predictions
            } else {
                // Priority mode: set initial predictions
                predictions = response.predictions
            }

            let assignMs = assignTimer.elapsedMs
            print("[Perf] loadPredictions(\(mode)) ASSIGN done: \(String(format: "%.0f", assignMs))ms (\(predictions.count) items)")

            lastUpdated = Date()

            // Update LiveScoreManager with current live match IDs (gating requirement)
            updateLiveScoreManagerGating()

            let logMode = mode == "priority" ? "TTFC_priority" : "TTFC_full"
            print("[Perf] loadPredictions \(logMode) END - network: \(String(format: "%.0f", networkMs))ms, assign: \(String(format: "%.0f", assignMs))ms, total: \(String(format: "%.0f", totalTimer.elapsedMs))ms")
            PerfLogger.shared.log(
                endpoint: "loadPredictions",
                message: logMode,
                data: [
                    "network_ms": networkMs,
                    "assign_ms": assignMs,
                    "total_ms": totalTimer.elapsedMs,
                    "count": predictions.count,
                    "days_back": daysBack,
                    "days_ahead": daysAhead
                ]
            )
        } catch {
            // Only set error for priority mode (user-facing)
            if mode == "priority" {
                self.error = error.localizedDescription
            }
            print("[Perf] loadPredictions(\(mode)) ERROR: \(error.localizedDescription)")
            PerfLogger.shared.log(
                endpoint: "loadPredictions",
                message: "error",
                data: ["error": error.localizedDescription, "total_ms": totalTimer.elapsedMs, "mode": mode]
            )
        }

        // Only clear loading for priority mode
        if mode == "priority" {
            isLoading = false
        } else {
            isLoadingMore = false
        }
    }

    // MARK: - Check Health

    func checkHealth() async {
        let timer = PerfTimer()
        do {
            let response = try await apiClient.checkHealth()
            modelLoaded = response.modelLoaded
            PerfLogger.shared.log(
                endpoint: "checkHealth",
                message: "end",
                data: ["total_ms": timer.elapsedMs, "model_loaded": modelLoaded]
            )
        } catch {
            // Don't overwrite prediction errors with health check errors
            modelLoaded = false
            PerfLogger.shared.log(
                endpoint: "checkHealth",
                message: "error",
                data: ["error": error.localizedDescription, "total_ms": timer.elapsedMs]
            )
        }
    }

    // MARK: - Refresh

    /// Tracks refresh start time for total duration measurement
    private var refreshStartTime: Date?

    /// Progressive loading refresh:
    /// 1. Load priority data (yesterday/today/tomorrow) - blocking for fast TTFC
    /// 2. Load full data (7 days back + 7 ahead) - fire-and-forget in background
    func refresh() async {
        // Throttle: skip if we refreshed recently (avoids duplicate calls on navigation)
        if let lastRefresh = lastRefreshTime,
           Date().timeIntervalSince(lastRefresh) < refreshThrottleInterval,
           !predictions.isEmpty {
            print("[Perf] refresh() THROTTLED - skipping (last refresh \(String(format: "%.1f", Date().timeIntervalSince(lastRefresh)))s ago)")
            return
        }
        lastRefreshTime = Date()

        refreshStartTime = Date()
        let requestId = UUID()
        currentRequestId = requestId

        print("[Perf] refresh() START (requestId: \(requestId.uuidString.prefix(8)))")

        // Fire-and-forget: warmup connections in parallel (doesn't block UI)
        // This pre-warms both the actor and concurrent connection pools
        Task { await checkHealth() }
        Task { await apiClient.warmupConcurrentConnection() }

        // Phase 1: Load priority data - BLOCKING for fast TTFC
        // daysBack=2, daysAhead=2 → 5 calendar days to cover timezone edge cases
        // (e.g., LA user at 6pm Jan 13 = UTC 2am Jan 14, needs Jan 13 local matches)
        async let opsTask: () = loadOpsProgress()
        await loadPredictions(daysBack: 2, daysAhead: 2, mode: "priority", requestId: requestId)

        // Wait for ops (non-blocking on error)
        _ = await opsTask

        let priorityMs = refreshStartTime.map { Date().timeIntervalSince($0) * 1000 } ?? 0
        print("[Perf] refresh() PRIORITY DONE - \(String(format: "%.0f", priorityMs))ms (\(predictions.count) predictions)")
        PerfLogger.shared.log(
            endpoint: "refresh",
            message: "priority_done",
            data: ["total_ms": priorityMs, "predictions_count": predictions.count]
        )

        // Phase 2: Load full data (7 days back + 7 ahead) - cancelable background task
        // This matches competitor's 15-day window (Mon-Mon)
        // Cancel previous full refresh if still running (prevents connection competition)
        fullRefreshTask?.cancel()
        isLoadingMore = true

        fullRefreshTask = Task { [weak self] in
            guard let self = self else { return }

            // Check for cancellation before starting network request
            if Task.isCancelled {
                print("[Perf] refresh() FULL CANCELLED before start")
                return
            }

            await self.loadPredictions(daysBack: 7, daysAhead: 7, mode: "full", requestId: requestId)

            // Check if cancelled during load
            if Task.isCancelled {
                print("[Perf] refresh() FULL CANCELLED after load")
                return
            }

            let totalMs = self.refreshStartTime.map { Date().timeIntervalSince($0) * 1000 } ?? 0
            let predictionsCount = self.predictions.count
            print("[Perf] refresh() FULL DONE - \(String(format: "%.0f", totalMs))ms")
            PerfLogger.shared.log(
                endpoint: "refresh",
                message: "full_done",
                data: ["total_ms": totalMs, "predictions_count": predictionsCount]
            )
        }
    }

    /// Cancel background refresh tasks (call from view's onDisappear)
    func cancelBackgroundTasks() {
        if fullRefreshTask != nil {
            fullRefreshTask?.cancel()
            fullRefreshTask = nil
            print("[Perf] Background refresh cancelled")
        }
    }

    // MARK: - Ops Progress (Alpha readiness)

    func loadOpsProgress() async {
        do {
            let ops = try await apiClient.getOpsDashboard()
            opsProgress = ops.progress
        } catch {
            // Non-blocking: predictions can still work even if dashboard token is missing
            opsProgress = nil
        }
    }

    // MARK: - Filtered Predictions (cached)

    /// Updates cached predictions - called when predictions or selectedDate changes
    private func updateCachedPredictions() {
        let timer = PerfTimer()

        // O(1) lookup from pre-built index (already sorted)
        let key = dayKey(from: selectedDate)
        let forDate = _predictionsByDay[key] ?? []

        // Split into value bets and regular (single pass)
        var valueBets: [MatchPrediction] = []
        var regular: [MatchPrediction] = []
        valueBets.reserveCapacity(forDate.count / 10)
        regular.reserveCapacity(forDate.count)

        for prediction in forDate {
            if (prediction.valueBets?.count ?? 0) > 0 {
                valueBets.append(prediction)
            } else {
                regular.append(prediction)
            }
        }

        cachedPredictionsForDate = forDate
        cachedValueBets = valueBets
        cachedRegularMatches = regular

        print("[Cache] updateCachedPredictions: \(forDate.count) matches (\(valueBets.count) value, \(regular.count) regular) in \(String(format: "%.1f", timer.elapsedMs))ms")
    }

    /// All predictions for selected date (cached)
    var predictionsForSelectedDate: [MatchPrediction] {
        cachedPredictionsForDate
    }

    /// Value bet predictions for selected date (cached)
    var valueBetPredictionsForSelectedDate: [MatchPrediction] {
        cachedValueBets
    }

    /// Regular predictions for selected date (cached)
    var regularPredictionsForSelectedDate: [MatchPrediction] {
        cachedRegularMatches
    }

    // Legacy - all predictions
    var valueBetPredictions: [MatchPrediction] {
        predictions.filter { ($0.valueBets?.count ?? 0) > 0 }
    }

    var upcomingPredictions: [MatchPrediction] {
        predictions.sorted { (p1, p2) in
            guard let d1 = p1.matchDate, let d2 = p2.matchDate else { return false }
            return d1 < d2
        }
    }

    // MARK: - Date Helpers

    // Pre-computed data structures (rebuilt when predictions change)
    private var _matchCountByDay: [Int: Int] = [:]           // dayKey -> count
    private var _predictionsByDay: [Int: [MatchPrediction]] = [:]  // dayKey -> predictions

    /// Compute day key from Date (YYYYMMDD as Int)
    /// Uses LOCAL calendar so matches are grouped by their kickoff in user's timezone.
    /// Example: A match at 2am UTC on Jan 14 appears on Jan 13 for LA user (6pm local).
    private func dayKey(from date: Date) -> Int {
        let components = localCalendar.dateComponents([.year, .month, .day], from: date)
        return components.year! * 10000 + components.month! * 100 + components.day!
    }

    /// Rebuild all date-indexed caches in a single pass
    private func rebuildMatchCountCache() {
        let timer = PerfTimer()
        _matchCountByDay.removeAll()
        _predictionsByDay.removeAll()

        // Single pass: group predictions by day and count
        for prediction in predictions {
            guard let matchDate = prediction.matchDate else { continue }
            let key = dayKey(from: matchDate)
            _matchCountByDay[key, default: 0] += 1
            _predictionsByDay[key, default: []].append(prediction)
        }

        // Sort each day's predictions by time
        for key in _predictionsByDay.keys {
            _predictionsByDay[key]?.sort { (p1, p2) in
                guard let d1 = p1.matchDate, let d2 = p2.matchDate else { return false }
                return d1 < d2
            }
        }

        print("[Cache] rebuildMatchCountCache: \(_matchCountByDay.count) days, \(predictions.count) predictions in \(String(format: "%.1f", timer.elapsedMs))ms")
    }

    func matchCount(for date: Date) -> Int {
        return _matchCountByDay[dayKey(from: date)] ?? 0
    }
}
