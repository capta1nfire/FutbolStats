import Foundation

/// State for tracking live clock per match
struct LiveClockState {
    let matchId: Int
    var observedKickoffAt: Date?      // Device time when we detected kickoff
    var lastAFElapsed: Int?           // Last elapsed from API-Football
    var lastAFStatus: String?         // Last status from AF (only valid statuses)
    var lastAFExtra: Int?             // Last elapsed_extra from AF
    var computedOffsetSeconds: Int = 0 // Offset calculated, clamped 0-180
    var lastDisplayedElapsed: Int = 0  // For smoothing (avoid jumps)
    var halfStartedAt: Date?          // Timestamp when current half started
    var scheduledKickoff: Date?       // Scheduled kickoff time
    var firstHalfTriggered: Bool = false  // Whether 1H clock was armed
    var secondHalfTriggered: Bool = false // Whether 2H clock was armed
    var kickoffDetectionMethod: String? // For diagnostics: "transition", "fallback", "late-join"
}

/// Adaptive local clock for live matches.
///
/// Compensates for API-Football delay by:
/// 1. Detecting kickoff transition (NS→1H or HT→2H)
/// 2. Computing delay offset dynamically as drift vs AF (clamped 0-180s)
/// 3. Running local clock with offset compensation
/// 4. Smooth correction: interpolate if drift ≤60s, snap with cap if >60s
///
/// IMPORTANT: This only affects display, never business logic.
@MainActor
final class LiveClockEstimator {
    static let shared = LiveClockEstimator()

    // MARK: - Configuration Constants

    /// Maximum offset to compensate (3 minutes)
    private let maxOffsetSeconds: Int = 180

    /// Threshold for snap vs interpolate (60 seconds)
    private let snapThresholdSeconds: Int = 60

    /// Maximum correction per poll to avoid jumps >2 min
    private let maxCorrectionPerPollSeconds: Int = 120

    /// Empirical AF delay: when AF reports elapsed=1, typically ~100s have passed since real kickoff
    /// Tuned to be ~1 second ahead of Google's clock for perceived "real-time" experience
    private let afReportingDelaySeconds: Int = 32

    /// Late join threshold for 1H: if elapsed > this, use conservative offset (no AF delay compensation)
    private let lateJoin1HThreshold: Int = 10

    /// Late join threshold for 2H: if elapsed > this (from start of 2H), use conservative offset
    private let lateJoin2HThreshold: Int = 60

    /// Valid match statuses (ignore glitched nil or unexpected values)
    private let validStatuses: Set<String> = ["NS", "1H", "HT", "2H", "ET", "BT", "P", "FT", "AET", "PEN", "SUSP", "INT", "PST", "CANC", "ABD", "AWD", "WO", "LIVE"]

    // MARK: - State

    /// Clock state per match_id
    private var clockStates: [Int: LiveClockState] = [:]

    /// Logging enabled for diagnostics
    var enableLogging: Bool = true

    private init() {}

    // MARK: - Public API

    /// Check if local clock compensation is enabled (reads from AppConfiguration)
    var enableLocalClockCompensation: Bool {
        AppConfiguration.shared.enableLocalClockCompensation
    }

    /// Update clock state when new data arrives from API-Football
    /// - Parameters:
    ///   - matchId: Internal match ID
    ///   - status: Current match status (1H, HT, 2H, FT, etc.)
    ///   - elapsed: Current elapsed minute from AF
    ///   - extra: Elapsed extra (injury time)
    ///   - scheduledKickoff: Originally scheduled kickoff time
    func update(matchId: Int, status: String, elapsed: Int, extra: Int, scheduledKickoff: Date) {
        var state = clockStates[matchId] ?? LiveClockState(matchId: matchId)
        state.scheduledKickoff = scheduledKickoff

        let now = Date()

        // GUARDRAIL: Ignore glitched/invalid statuses - keep last valid state
        guard validStatuses.contains(status) else {
            logDiagnostic(matchId: matchId, event: "GLITCH_IGNORED", details: "invalid status='\(status)', keeping lastStatus=\(state.lastAFStatus ?? "nil")")
            clockStates[matchId] = state
            return
        }

        // GUARDRAIL: Ignore momentary NS regression (AF glitch)
        // If we've already triggered 1H and AF sends NS again, ignore it
        if status == "NS" && state.firstHalfTriggered {
            logDiagnostic(matchId: matchId, event: "NS_REGRESSION_IGNORED", details: "already in 1H, ignoring NS glitch")
            clockStates[matchId] = state
            return
        }

        // Capture previous status BEFORE updating for transition detection
        let previousStatus = state.lastAFStatus

        // Update AF values (only after validation)
        state.lastAFElapsed = elapsed
        state.lastAFStatus = status
        state.lastAFExtra = extra

        // ========== 1H KICKOFF DETECTION ==========
        // GUARDRAIL: Only arm once (firstHalfTriggered flag prevents double-arming)
        if !state.firstHalfTriggered && status == "1H" {
            let isTransition = (previousStatus == "NS" || previousStatus == nil)
            let isLateJoin = elapsed > lateJoin1HThreshold

            state.firstHalfTriggered = true
            state.observedKickoffAt = now
            state.halfStartedAt = now

            let afElapsedSeconds = elapsed * 60

            // computedOffsetSeconds = estimated seconds of play at the moment we armed the clock
            // This is added to secondsSinceHalfStart to get total elapsed
            let detectionMethod: String

            if isLateJoin {
                // GUARDRAIL: Late join (elapsed > 10) - trust AF elapsed directly
                // NO clamp here - we need the full elapsed time for late joins
                detectionMethod = "late-join-1H"
                state.computedOffsetSeconds = afElapsedSeconds
            } else if elapsed <= 1 && isTransition {
                // Best case: caught NS→1H transition early
                // AF says 1 min but ~90s have really passed, add compensation
                detectionMethod = "transition"
                // Clamp the COMPENSATION only, not the base elapsed
                let compensation = clamp(afReportingDelaySeconds, min: 0, max: maxOffsetSeconds)
                state.computedOffsetSeconds = afElapsedSeconds + compensation
            } else {
                // Fallback: first time seeing 1H with elapsed 2-10
                detectionMethod = "fallback"
                let compensation = clamp(afReportingDelaySeconds / 2, min: 0, max: maxOffsetSeconds)
                state.computedOffsetSeconds = afElapsedSeconds + compensation
            }

            state.kickoffDetectionMethod = detectionMethod

            logDiagnostic(matchId: matchId, event: "1H_KICKOFF", details: "method=\(detectionMethod), prevStatus=\(previousStatus ?? "nil"), elapsed=\(elapsed), offset=\(state.computedOffsetSeconds)s")
        }

        // ========== 2H KICKOFF DETECTION ==========
        // GUARDRAIL: Only arm once (secondHalfTriggered flag prevents double-arming)
        if !state.secondHalfTriggered && status == "2H" && elapsed >= 46 {
            let isTransition = previousStatus == "HT"
            let minutesInto2H = elapsed - 45
            let isLateJoin = minutesInto2H > (lateJoin2HThreshold - 45) // elapsed > 60

            state.secondHalfTriggered = true
            state.halfStartedAt = now

            // For 2H, offset is seconds INTO the second half (not total match time)
            // calculateLocalElapsedSeconds adds baseSeconds (45*60) for 2H
            let afElapsedSecondsInto2H = minutesInto2H * 60
            let detectionMethod: String

            if isLateJoin {
                // GUARDRAIL: Late join 2H (elapsed > 60) - trust AF directly
                // NO clamp here - we need the full elapsed time for late joins
                detectionMethod = "late-join-2H"
                state.computedOffsetSeconds = afElapsedSecondsInto2H
            } else if minutesInto2H <= 1 && isTransition {
                // Best case: caught HT→2H transition early
                detectionMethod = "transition"
                // Clamp the COMPENSATION only, not the base elapsed
                let compensation = clamp(afReportingDelaySeconds, min: 0, max: maxOffsetSeconds)
                state.computedOffsetSeconds = afElapsedSecondsInto2H + compensation
            } else {
                // Fallback: first time seeing 2H
                detectionMethod = "fallback"
                let compensation = clamp(afReportingDelaySeconds / 2, min: 0, max: maxOffsetSeconds)
                state.computedOffsetSeconds = afElapsedSecondsInto2H + compensation
            }

            state.kickoffDetectionMethod = detectionMethod

            logDiagnostic(matchId: matchId, event: "2H_KICKOFF", details: "method=\(detectionMethod), prevStatus=\(previousStatus ?? "nil"), elapsed=\(elapsed), minutesInto2H=\(minutesInto2H), offset=\(state.computedOffsetSeconds)s")
        }

        // ========== DRIFT CORRECTION ==========
        // Only apply corrections during active play, and be conservative for late joins
        // IMPORTANT: We only correct when LOCAL is BEHIND AF (diffSeconds < 0)
        // If local is ahead, we let it naturally slow down via the "never go backwards" guardrail in displayElapsed
        if state.halfStartedAt != nil && (status == "1H" || status == "2H") {
            let localElapsedSeconds = calculateLocalElapsedSeconds(state: state, at: now)
            let afElapsedSeconds = elapsed * 60
            let diffSeconds = localElapsedSeconds - afElapsedSeconds  // positive = local ahead, negative = local behind

            // Only correct if local is BEHIND AF (negative diff)
            // If local is ahead, the "never go backwards" guardrail in displayElapsed handles it
            if diffSeconds < -10 {
                // Local is behind AF - need to speed up
                let isLateJoinCorrection = state.kickoffDetectionMethod?.contains("late-join") ?? false
                let effectiveSnapThreshold = isLateJoinCorrection ? snapThresholdSeconds * 2 : snapThresholdSeconds
                let effectiveMaxCorrection = isLateJoinCorrection ? maxCorrectionPerPollSeconds / 2 : maxCorrectionPerPollSeconds

                if abs(diffSeconds) > effectiveSnapThreshold {
                    // Large drift: snap with bounded correction (move halfStartedAt to past = increase elapsed)
                    let correction = clamp(diffSeconds, min: -effectiveMaxCorrection, max: 0)

                    if let currentHalfStart = state.halfStartedAt {
                        state.halfStartedAt = currentHalfStart.addingTimeInterval(Double(correction))
                    }

                    logDiagnostic(matchId: matchId, event: "SNAP_CATCHUP", details: "local=\(localElapsedSeconds/60)', af=\(elapsed)', diff=\(diffSeconds)s, correction=\(correction)s")
                } else {
                    // Small drift (10-60s behind): interpolate gradually
                    let interpolationFactor = isLateJoinCorrection ? 8 : 4
                    let correction = diffSeconds / interpolationFactor  // negative, moves halfStartedAt to past

                    if let currentHalfStart = state.halfStartedAt {
                        state.halfStartedAt = currentHalfStart.addingTimeInterval(Double(correction))
                    }

                    logDiagnostic(matchId: matchId, event: "INTERPOLATE_CATCHUP", details: "diff=\(diffSeconds)s, correction=\(correction)s")
                }
            }
            // If local is ahead (diffSeconds > 0): let it ride, displayElapsed guardrail prevents going backwards
            // If within ±10s: no correction needed, we're in sync
        }

        // Log match end
        if status == "FT" || status == "AET" || status == "PEN" {
            logDiagnostic(matchId: matchId, event: "MATCH_END", details: "status=\(status)")
        }

        clockStates[matchId] = state
    }

    /// Get display elapsed string for a match
    /// - Parameters:
    ///   - matchId: Internal match ID
    ///   - now: Current time (usually Date())
    /// - Returns: Formatted elapsed string (e.g., "32'", "45+2'", "90+", "HT")
    func displayElapsed(for matchId: Int, at now: Date = Date()) -> String {
        guard enableLocalClockCompensation else {
            // Fallback: return raw AF elapsed
            return fallbackDisplay(for: matchId)
        }

        guard let state = clockStates[matchId] else {
            return fallbackDisplay(for: matchId)
        }

        guard let status = state.lastAFStatus else {
            return "LIVE"
        }

        // For non-playing statuses, return the status
        let activeStatuses = ["1H", "2H"]
        guard activeStatuses.contains(status) else {
            // HT, ET, BT, P, FT, etc.
            if status == "HT" {
                return "HT"
            }
            return status
        }

        // Calculate local elapsed in minutes
        let localElapsedSeconds = calculateLocalElapsedSeconds(state: state, at: now)
        var localElapsed = localElapsedSeconds / 60

        // GUARDRAIL: Never let displayed minute go backwards
        // This prevents jarring UX when drift correction pulls the clock back
        if localElapsed < state.lastDisplayedElapsed {
            localElapsed = state.lastDisplayedElapsed
        }

        // Handle injury time - ONLY show "+" format when API has sent elapsedExtra > 0
        if let extra = state.lastAFExtra, extra > 0 {
            if status == "2H" {
                return "90+\(extra)'"
            } else if status == "1H" {
                return "45+\(extra)'"
            }
        }

        // Cap display at 45' for 1H and 90' for 2H until API confirms injury time
        if status == "1H" && localElapsed > 45 {
            return "45'"
        }
        if status == "2H" && localElapsed > 90 {
            return "90'"
        }

        // Update lastDisplayedElapsed for monotonic progression
        clockStates[matchId]?.lastDisplayedElapsed = localElapsed
        return "\(localElapsed)'"
    }

    /// Reset state for a match (e.g., when match ends or user leaves)
    func reset(matchId: Int) {
        clockStates.removeValue(forKey: matchId)
        log("[Clock] Match \(matchId): State reset")
    }

    /// Clear all states
    func resetAll() {
        clockStates.removeAll()
        log("[Clock] All states cleared")
    }

    // MARK: - Private Helpers

    /// Calculate local elapsed time in SECONDS (for more precise drift calculation)
    private func calculateLocalElapsedSeconds(state: LiveClockState, at now: Date) -> Int {
        guard let halfStart = state.halfStartedAt,
              let status = state.lastAFStatus else {
            return (state.lastAFElapsed ?? 0) * 60
        }

        let secondsSinceHalfStart = Int(now.timeIntervalSince(halfStart))
        let totalSeconds = secondsSinceHalfStart + state.computedOffsetSeconds

        let baseSeconds: Int
        let maxSeconds: Int

        if status == "1H" {
            baseSeconds = 0
            maxSeconds = 45 * 60  // 45 minutes in seconds
        } else if status == "2H" {
            baseSeconds = 45 * 60  // Start at 45 minutes
            maxSeconds = 90 * 60  // Cap at 90 minutes
        } else {
            return (state.lastAFElapsed ?? 0) * 60
        }

        let displaySeconds = baseSeconds + min(totalSeconds, maxSeconds - baseSeconds)
        return displaySeconds
    }

    private func fallbackDisplay(for matchId: Int) -> String {
        guard let state = clockStates[matchId],
              let elapsed = state.lastAFElapsed,
              let status = state.lastAFStatus else {
            return "LIVE"
        }

        if status == "HT" {
            return "HT"
        }

        // CORRECTED: Consistent injury time format
        if let extra = state.lastAFExtra, extra > 0 {
            if status == "2H" && elapsed >= 90 {
                return "90+\(extra)'"
            } else if status == "1H" && elapsed >= 45 {
                return "45+\(extra)'"
            }
            return "\(elapsed)+\(extra)'"
        }

        return "\(elapsed)'"
    }

    private func clamp(_ value: Int, min minVal: Int, max maxVal: Int) -> Int {
        return max(minVal, min(maxVal, value))
    }

    private func log(_ message: String) {
        if enableLogging {
            print(message)
        }
    }

    /// Structured diagnostic logging for clock events
    /// Format: [Clock] Match {id}: {EVENT} | {details}
    private func logDiagnostic(matchId: Int, event: String, details: String) {
        guard enableLogging else { return }

        let timestamp = ISO8601DateFormatter().string(from: Date())
        print("[Clock] \(timestamp) | Match \(matchId) | \(event) | \(details)")
    }

    // MARK: - Diagnostics API

    /// Get current clock state for a match (for debugging/testing)
    func diagnosticState(for matchId: Int) -> [String: Any]? {
        guard let state = clockStates[matchId] else { return nil }

        return [
            "matchId": state.matchId,
            "lastAFStatus": state.lastAFStatus ?? "nil",
            "lastAFElapsed": state.lastAFElapsed ?? -1,
            "lastAFExtra": state.lastAFExtra ?? 0,
            "computedOffsetSeconds": state.computedOffsetSeconds,
            "firstHalfTriggered": state.firstHalfTriggered,
            "secondHalfTriggered": state.secondHalfTriggered,
            "kickoffDetectionMethod": state.kickoffDetectionMethod ?? "none",
            "halfStartedAt": state.halfStartedAt?.description ?? "nil",
            "observedKickoffAt": state.observedKickoffAt?.description ?? "nil"
        ]
    }

    /// Get summary of all active clock states (for debugging)
    func diagnosticSummary() -> [[String: Any]] {
        return clockStates.values.compactMap { state -> [String: Any]? in
            guard let status = state.lastAFStatus,
                  ["1H", "2H", "HT"].contains(status) else { return nil }

            return [
                "matchId": state.matchId,
                "status": status,
                "elapsed": state.lastAFElapsed ?? 0,
                "offset": state.computedOffsetSeconds,
                "method": state.kickoffDetectionMethod ?? "none"
            ]
        }
    }
}
