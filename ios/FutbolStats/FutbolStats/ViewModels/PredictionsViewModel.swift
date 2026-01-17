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
    }

    deinit {
        clockTimer?.invalidate()
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
    /// Uses the API elapsed as base and adds local time passed since load
    /// - Parameter prediction: The match prediction
    /// - Returns: Formatted elapsed string (e.g., "32'", "45+", "90+", or status like "HT")
    func calculatedElapsedDisplay(for prediction: MatchPrediction) -> String {
        guard let status = prediction.status else { return "LIVE" }

        // Only calculate for active play statuses
        let activeStatuses = ["1H", "2H", "LIVE"]
        guard activeStatuses.contains(status) else {
            // For HT, ET, BT, P, etc. - show status code
            return status
        }

        guard let baseElapsed = prediction.elapsed else {
            return status
        }

        // Calculate minutes passed since predictions were loaded
        let minutesPassed = Int(clockTick.timeIntervalSince(predictionsLoadedAt) / 60)
        var calculatedElapsed = baseElapsed + minutesPassed

        // Apply caps based on status
        if status == "1H" {
            // Cap at 45+ for first half
            if calculatedElapsed > 45 {
                return "45+"
            }
        } else if status == "2H" {
            // Cap at 90+ for second half
            if calculatedElapsed > 90 {
                return "90+"
            }
        }

        return "\(calculatedElapsed)'"
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
