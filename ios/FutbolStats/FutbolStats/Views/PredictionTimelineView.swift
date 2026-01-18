import SwiftUI

/// Timeline visualization showing when the prediction was correct during the match.
/// Uses SF Symbols 5 for native iOS premium aesthetics.
/// Green segments = prediction aligned with current score
/// Gray segments = prediction misaligned (wrong or neutral)
///
/// Supports two modes:
/// - **Finished matches**: Uses MatchTimelineResponse from backend
/// - **Live matches**: Builds timeline dynamically from LiveTimelineData
struct PredictionTimelineView: View {
    // Backend timeline data (for finished matches)
    let timeline: MatchTimelineResponse?

    // Live match data (for in-progress matches)
    let liveData: LiveTimelineData?

    // Convenience initializer for finished matches (existing API)
    init(timeline: MatchTimelineResponse) {
        self.timeline = timeline
        self.liveData = nil
    }

    // Initializer for live matches
    init(liveData: LiveTimelineData) {
        self.timeline = nil
        self.liveData = liveData
    }

    private let correctColor = Color.green
    private let neutralColor = Color.gray.opacity(0.4)
    private let barHeight: CGFloat = 24

    // MARK: - Computed Properties (unified for both modes)

    private var isLive: Bool {
        liveData != nil && timeline == nil
    }

    private var isHalfTime: Bool {
        liveData?.status == "HT"
    }

    /// At regulation time limit (45' in 1H or 90' in 2H) - pulse should stop
    private var isAtRegulationLimit: Bool {
        guard let status = liveData?.status else { return false }
        let elapsed = liveData?.elapsed ?? 0

        // During HT, elapsed might still be 45 from 1H
        if status == "1H" && elapsed >= 45 {
            return true
        } else if status == "2H" && elapsed >= 90 {
            return true
        }
        // Also stop if elapsed is exactly at regulation time (handles edge cases)
        if elapsed == 45 || elapsed == 90 {
            return true
        }
        return false
    }

    /// Pulse should stop during HT, at regulation limits, or when finished
    private var shouldStopPulsing: Bool {
        isHalfTime || isAtRegulationLimit || !isLive
    }

    private var totalMinutes: Int {
        if let timeline = timeline {
            return timeline.totalMinutes
        }
        return liveData?.totalMinutes ?? 90
    }

    private var currentElapsed: Int {
        liveData?.elapsed ?? totalMinutes
    }

    private var segments: [TimelineSegment] {
        if let timeline = timeline {
            return timeline.segments
        }
        return liveData?.computedSegments ?? []
    }

    private var goals: [TimelineGoal] {
        if let timeline = timeline {
            return timeline.goals
        }
        return liveData?.goals ?? []
    }

    private var predictedOutcome: String {
        if let timeline = timeline {
            return timeline.prediction.outcome
        }
        return liveData?.predictedOutcome ?? "home"
    }

    private var predictionCorrect: Bool? {
        if let timeline = timeline {
            return timeline.prediction.correct
        }
        return liveData?.isPredictionCurrentlyCorrect
    }

    private var homeScore: Int {
        timeline?.finalScore.home ?? liveData?.homeGoals ?? 0
    }

    private var awayScore: Int {
        timeline?.finalScore.away ?? liveData?.awayGoals ?? 0
    }

    private var correctMinutes: Double {
        if let timeline = timeline {
            return timeline.summary.correctMinutes
        }
        return liveData?.correctMinutes ?? 0
    }

    private var correctPercentage: Double {
        if let timeline = timeline {
            return timeline.summary.correctPercentage
        }
        return liveData?.correctPercentage ?? 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Timeline bar (full width) - same structure as probability bar
            GeometryReader { geo in
                ZStack(alignment: .topLeading) {
                    // Background bar (full width as base)
                    RoundedRectangle(cornerRadius: 6)
                        .fill(neutralColor)
                        .frame(width: geo.size.width, height: barHeight)

                    // Segments overlay (played portion) - aligned left
                    // Each segment handles its own corner radius internally
                    timelineBar(width: geo.size.width)
                        .frame(height: barHeight, alignment: .leading)

                    // Live progress indicator (animated edge)
                    if isLive {
                        liveProgressIndicator(width: geo.size.width)
                    }

                    // Goal markers (soccer balls centered on bar)
                    goalMarkers(width: geo.size.width)
                }
            }
            .frame(height: barHeight)

            // Minute markers - positioned to align with bar
            GeometryReader { geo in
                let progressPosition = geo.size.width * CGFloat(currentElapsed) / CGFloat(totalMinutes)

                ZStack(alignment: .top) {
                    // Start minute (1')
                    Text("1'")
                        .font(.custom("BarlowCondensed-SemiBold", size: 12))
                        .foregroundStyle(Color(.systemGray))
                        .position(x: 10, y: 12)

                    // Current minute for live matches - aligned with progress bar
                    if isLive {
                        Text("\(currentElapsed)'")
                            .font(.custom("BarlowCondensed-SemiBold", size: 14))
                            .foregroundStyle(.white)
                            .modifier(PulseModifier())
                            .position(x: min(progressPosition, geo.size.width - 20), y: 12)
                    }

                    // End minute (90' or 120') - hide when live minute is close to avoid overlap
                    if !isLive || currentElapsed < (totalMinutes - 8) {
                        Text("\(totalMinutes)'")
                            .font(.custom("BarlowCondensed-SemiBold", size: 12))
                            .foregroundStyle(isLive ? Color(.systemGray3) : Color(.systemGray))
                            .position(x: geo.size.width - 15, y: 12)
                    }
                }
            }
            .frame(height: 24)

            // Summary
            HStack(spacing: 16) {
                HStack(spacing: 4) {
                    if isLive {
                        // Pulsing green dot for live
                        Circle()
                            .fill(correctColor)
                            .frame(width: 8, height: 8)
                            .modifier(PulseModifier())
                        // Live: show "Timeline" - time only counts after first goal
                        Text("Timeline")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.8))
                    } else {
                        // Static green dot for finished
                        Circle()
                            .fill(correctColor)
                            .frame(width: 8, height: 8)
                        // Finished: show full stats
                        Text("In Line: \(Int(correctMinutes)) min (\(Int(correctPercentage))%)")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.8))
                    }
                }

                Spacer()

                // Result indicator for finished matches
                if !isLive {
                    if let correct = predictionCorrect {
                        if correct {
                            Label("Correct", systemImage: "checkmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.green)
                        } else {
                            Label("Wrong", systemImage: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                    }
                }
            }
        }
        .padding(16)
        .modifier(GlassEffectModifier())
    }

    // MARK: - Live Progress Indicator

    @ViewBuilder
    private func liveProgressIndicator(width: CGFloat) -> some View {
        let progressWidth = width * CGFloat(currentElapsed) / CGFloat(totalMinutes)

        // Thin vertical white line that extends beyond bar (like probability bar indicator)
        // Pulse only when actively playing (not during HT, regulation limits, or finished)
        if shouldStopPulsing {
            Rectangle()
                .fill(Color.white.opacity(0.8))
                .frame(width: 1, height: barHeight + 6)
                .position(x: min(progressWidth, width - 1), y: barHeight / 2)
        } else {
            Rectangle()
                .fill(Color.white.opacity(0.8))
                .frame(width: 1, height: barHeight + 6)
                .modifier(PulseModifier())
                .position(x: min(progressWidth, width - 1), y: barHeight / 2)
        }
    }

    // MARK: - Timeline Bar

    private func timelineBar(width: CGFloat) -> some View {
        // For live matches, show colored segments up to elapsed time
        // For finished matches, show colored segments for full width
        let effectiveWidth = isLive
            ? width * CGFloat(currentElapsed) / CGFloat(totalMinutes)
            : width

        // Use segments for both live and finished matches
        // computedSegments already calculates correct/neutral based on prediction vs score at each point
        let displaySegments = segments
        let segmentTotalMinutes = isLive ? currentElapsed : totalMinutes

        return AnyView(
            HStack(spacing: 0) {
                ForEach(displaySegments) { segment in
                    let segmentWidth = effectiveWidth * CGFloat(segment.duration) / CGFloat(max(1, segmentTotalMinutes))
                    let isCorrect = segment.status == "correct"

                    Rectangle()
                        .fill(isCorrect ? correctColor : neutralColor)
                        .frame(width: max(1, segmentWidth), height: barHeight)
                }
            }
            .frame(width: effectiveWidth, height: barHeight, alignment: .leading)
            .clipShape(
                UnevenRoundedRectangle(
                    topLeadingRadius: 6,
                    bottomLeadingRadius: 6,
                    bottomTrailingRadius: 0,
                    topTrailingRadius: 0
                )
            )
        )
    }

    // MARK: - Goal Markers (SF Symbol: soccerball.inverse)

    private func goalMarkers(width: CGFloat) -> some View {
        let groupedGoals = groupNearbyGoals(goals)

        return ZStack {
            ForEach(groupedGoals) { group in
                let xPosition = CGFloat(group.effectiveMinute) / CGFloat(totalMinutes) * width

                // Goal icon(s) - centered on bar
                if group.count > 1 {
                    // Grouped goals badge
                    HStack(spacing: 2) {
                        Image(systemName: "soccerball.inverse")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(.white)
                        Text("×\(group.count)")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.white)
                    }
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(Color.black.opacity(0.7))
                    .clipShape(Capsule())
                    .position(x: clampPosition(xPosition, width: width), y: barHeight / 2)
                } else {
                    // Single goal - centered on bar
                    Image(systemName: "soccerball.inverse")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white)
                        .shadow(color: .black.opacity(0.5), radius: 1, x: 0, y: 1)
                        .position(x: clampPosition(xPosition, width: width), y: barHeight / 2)
                }
            }
        }
    }

    /// Clamp position to keep markers visible within bounds
    private func clampPosition(_ x: CGFloat, width: CGFloat) -> CGFloat {
        let padding: CGFloat = 16
        return max(padding, min(width - padding, x))
    }

    /// Group goals that are within 2 minutes of each other
    private func groupNearbyGoals(_ goals: [TimelineGoal]) -> [GoalGroup] {
        guard !goals.isEmpty else { return [] }

        let sorted = goals.sorted { $0.effectiveMinute < $1.effectiveMinute }
        var groups: [GoalGroup] = []
        var currentGroup: [TimelineGoal] = [sorted[0]]

        for i in 1..<sorted.count {
            let current = sorted[i]
            let last = currentGroup.last!

            if current.effectiveMinute - last.effectiveMinute <= 2 {
                // Add to current group
                currentGroup.append(current)
            } else {
                // Save current group and start new one
                groups.append(GoalGroup(goals: currentGroup))
                currentGroup = [current]
            }
        }

        // Add final group
        groups.append(GoalGroup(goals: currentGroup))

        return groups
    }
}

// MARK: - Goal Group

struct GoalGroup: Identifiable {
    let goals: [TimelineGoal]

    var id: String {
        goals.map { $0.id }.joined(separator: "-")
    }

    var count: Int {
        goals.count
    }

    /// Average minute for positioning
    var effectiveMinute: Int {
        guard !goals.isEmpty else { return 0 }
        let total = goals.reduce(0) { $0 + $1.effectiveMinute }
        return total / goals.count
    }

    /// Display minute range (e.g., "88-90'" or "45'")
    var displayMinute: String {
        guard !goals.isEmpty else { return "" }

        if goals.count == 1 {
            return "\(goals[0].displayMinute)'"
        }

        let minutes = goals.map { $0.effectiveMinute }.sorted()
        let first = minutes.first!
        let last = minutes.last!

        if first == last {
            return "\(first)'"
        }
        return "\(first)-\(last)'"
    }
}

// MARK: - Preview

#Preview {
    VStack {
        PredictionTimelineView(
            timeline: MatchTimelineResponse(
                matchId: 1,
                status: "FT",
                finalScore: TimelineScore(home: 3, away: 1),
                prediction: TimelinePrediction(
                    outcome: "home",
                    homeProb: 0.55,
                    drawProb: 0.25,
                    awayProb: 0.20,
                    correct: true
                ),
                totalMinutes: 90,
                goals: [
                    TimelineGoal(minute: 23, extraMinute: nil, team: "home", teamName: "Real Madrid", player: "Vinicius Jr", isOwnGoal: false, isPenalty: false),
                    TimelineGoal(minute: 45, extraMinute: 2, team: "away", teamName: "Barcelona", player: "Yamal", isOwnGoal: false, isPenalty: false),
                    TimelineGoal(minute: 67, extraMinute: nil, team: "home", teamName: "Real Madrid", player: "Bellingham", isOwnGoal: false, isPenalty: false),
                    TimelineGoal(minute: 89, extraMinute: nil, team: "home", teamName: "Real Madrid", player: "Rodrygo", isOwnGoal: false, isPenalty: false),
                    TimelineGoal(minute: 90, extraMinute: 1, team: "home", teamName: "Real Madrid", player: "Mbappe", isOwnGoal: false, isPenalty: true)
                ],
                segments: [
                    TimelineSegment(startMinute: 0, endMinute: 23, homeGoals: 0, awayGoals: 0, status: "neutral"),
                    TimelineSegment(startMinute: 23, endMinute: 47, homeGoals: 1, awayGoals: 0, status: "correct"),
                    TimelineSegment(startMinute: 47, endMinute: 67, homeGoals: 1, awayGoals: 1, status: "neutral"),
                    TimelineSegment(startMinute: 67, endMinute: 90, homeGoals: 3, awayGoals: 1, status: "correct")
                ],
                summary: TimelineSummary(correctMinutes: 47, correctPercentage: 52.2),
                meta: nil
            )
        )
    }
    .padding()
    .background(Color.black)
    .preferredColorScheme(.dark)
}

// MARK: - Glass Effect Modifier (iOS 26+)

private struct GlassEffectModifier: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 26.0, *) {
            content
                .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 16))
        } else {
            content
                .background(Color(white: 0.1), in: RoundedRectangle(cornerRadius: 16))
        }
    }
}

// MARK: - Pulse Animation Modifier

private struct PulseModifier: ViewModifier {
    @State private var isAnimating = false

    func body(content: Content) -> some View {
        content
            .opacity(isAnimating ? 0.3 : 1.0)
            .animation(
                .easeInOut(duration: 0.8)
                .repeatForever(autoreverses: true),
                value: isAnimating
            )
            .onAppear {
                isAnimating = true
            }
    }
}

// MARK: - Live Timeline Data Model

/// Model for building timeline data from live match information.
/// Used when backend timeline isn't available (match in progress).
struct LiveTimelineData {
    let matchId: Int
    let predictedOutcome: String  // "home", "draw", "away"
    let homeGoals: Int
    let awayGoals: Int
    let elapsed: Int
    let goals: [TimelineGoal]
    let status: String  // "1H", "2H", "HT", etc.

    /// Total minutes for the timeline (90 for regular, 120 for extra time)
    var totalMinutes: Int {
        switch status {
        case "ET", "P", "AET", "PEN":
            return 120
        default:
            return 90
        }
    }

    /// Current outcome based on score
    var currentOutcome: String {
        if homeGoals > awayGoals { return "home" }
        if awayGoals > homeGoals { return "away" }
        return "draw"
    }

    /// Is the prediction currently correct?
    var isPredictionCurrentlyCorrect: Bool {
        return predictedOutcome == currentOutcome
    }

    /// Computed segments based on goals
    var computedSegments: [TimelineSegment] {
        guard elapsed > 0 else {
            return [TimelineSegment(startMinute: 0, endMinute: 0, homeGoals: 0, awayGoals: 0, status: "neutral")]
        }

        // If no goals, single segment from start to now
        if goals.isEmpty {
            let status = segmentStatus(homeGoals: 0, awayGoals: 0)
            return [TimelineSegment(startMinute: 0, endMinute: elapsed, homeGoals: 0, awayGoals: 0, status: status)]
        }

        // Build segments from goals
        var segments: [TimelineSegment] = []
        var currentHome = 0
        var currentAway = 0
        var lastMinute = 0

        let sortedGoals = goals.sorted { $0.effectiveMinute < $1.effectiveMinute }

        for goal in sortedGoals {
            let goalMinute = goal.effectiveMinute

            // Add segment before this goal (if there's time gap)
            if goalMinute > lastMinute {
                let status = segmentStatus(homeGoals: currentHome, awayGoals: currentAway)
                segments.append(TimelineSegment(
                    startMinute: lastMinute,
                    endMinute: goalMinute,
                    homeGoals: currentHome,
                    awayGoals: currentAway,
                    status: status
                ))
            }

            // Update score
            if goal.team == "home" {
                currentHome += 1
            } else {
                currentAway += 1
            }

            lastMinute = goalMinute
        }

        // Add final segment from last goal to current elapsed
        if elapsed > lastMinute {
            let status = segmentStatus(homeGoals: currentHome, awayGoals: currentAway)
            segments.append(TimelineSegment(
                startMinute: lastMinute,
                endMinute: elapsed,
                homeGoals: currentHome,
                awayGoals: currentAway,
                status: status
            ))
        }

        return segments
    }

    /// Calculate segment status based on score and prediction
    private func segmentStatus(homeGoals: Int, awayGoals: Int) -> String {
        let scoreOutcome: String
        if homeGoals > awayGoals {
            scoreOutcome = "home"
        } else if awayGoals > homeGoals {
            scoreOutcome = "away"
        } else {
            scoreOutcome = "draw"
        }

        return scoreOutcome == predictedOutcome ? "correct" : "neutral"
    }

    /// Minutes where prediction was correct
    var correctMinutes: Double {
        Double(computedSegments.filter { $0.status == "correct" }.reduce(0) { $0 + $1.duration })
    }

    /// Percentage of time prediction was correct
    var correctPercentage: Double {
        guard elapsed > 0 else { return 0 }
        return (correctMinutes / Double(elapsed)) * 100
    }

    /// Create from MatchPrediction, extracting goals from events
    static func from(prediction: MatchPrediction) -> LiveTimelineData {
        // Convert FootballMatchEvents to TimelineGoals (only Goal events)
        let goals: [TimelineGoal] = (prediction.events ?? []).compactMap { event in
            // Only process Goal events
            guard event.type == "Goal", let minute = event.minute else {
                return nil
            }

            // Get team name from either teamName (predictions API) or team (insights API)
            let eventTeamName = event.teamName ?? event.team

            // Determine team side (home/away) by comparing team name
            let team: String
            if let eventTeamName = eventTeamName {
                // Match by team name (case-insensitive contains)
                let isHome = prediction.homeTeam.lowercased().contains(eventTeamName.lowercased()) ||
                             eventTeamName.lowercased().contains(prediction.homeTeam.lowercased())
                team = isHome ? "home" : "away"
            } else {
                // Fallback: assume home if we can't determine
                team = "home"
            }

            // Check for own goal or penalty from detail
            let isOwnGoal = event.detail?.lowercased().contains("own goal") ?? false
            let isPenalty = event.detail?.lowercased().contains("penalty") ?? false

            // Get player name from either playerName (predictions API) or player (insights API)
            let playerName = event.playerName ?? event.player

            return TimelineGoal(
                minute: minute,
                extraMinute: event.extraMinute,
                team: team,
                teamName: eventTeamName,
                player: playerName,
                isOwnGoal: isOwnGoal,
                isPenalty: isPenalty
            )
        }

        return LiveTimelineData(
            matchId: prediction.matchId ?? 0,
            predictedOutcome: prediction.predictedOutcome,
            homeGoals: prediction.homeGoals ?? 0,
            awayGoals: prediction.awayGoals ?? 0,
            elapsed: prediction.elapsed ?? 0,
            goals: goals,
            status: prediction.status ?? "1H"
        )
    }

    /// Create from MatchPrediction with inferred goals from live polling
    /// Inferred goals are detected by observing score changes during polling
    static func from(prediction: MatchPrediction, inferredGoals: [InferredGoal]) -> LiveTimelineData {
        // First try to use API events if available
        var goals: [TimelineGoal] = (prediction.events ?? []).compactMap { event in
            guard event.type == "Goal", let minute = event.minute else {
                return nil
            }

            let eventTeamName = event.teamName ?? event.team
            let team: String
            if let eventTeamName = eventTeamName {
                let isHome = prediction.homeTeam.lowercased().contains(eventTeamName.lowercased()) ||
                             eventTeamName.lowercased().contains(prediction.homeTeam.lowercased())
                team = isHome ? "home" : "away"
            } else {
                team = "home"
            }

            let isOwnGoal = event.detail?.lowercased().contains("own goal") ?? false
            let isPenalty = event.detail?.lowercased().contains("penalty") ?? false
            let playerName = event.playerName ?? event.player

            return TimelineGoal(
                minute: minute,
                extraMinute: event.extraMinute,
                team: team,
                teamName: eventTeamName,
                player: playerName,
                isOwnGoal: isOwnGoal,
                isPenalty: isPenalty
            )
        }

        // If no API events, use inferred goals from polling
        if goals.isEmpty && !inferredGoals.isEmpty {
            goals = inferredGoals.map { inferred in
                TimelineGoal(
                    minute: inferred.minute,
                    extraMinute: nil,
                    team: inferred.team,
                    teamName: inferred.teamName,
                    player: nil,  // No player info from polling
                    isOwnGoal: false,
                    isPenalty: false
                )
            }
        }

        return LiveTimelineData(
            matchId: prediction.matchId ?? 0,
            predictedOutcome: prediction.predictedOutcome,
            homeGoals: prediction.homeGoals ?? 0,
            awayGoals: prediction.awayGoals ?? 0,
            elapsed: prediction.elapsed ?? 0,
            goals: goals,
            status: prediction.status ?? "1H"
        )
    }
}

// MARK: - Live Preview

#Preview("Live Match") {
    VStack(spacing: 20) {
        // Live match example - 35 minutes, 1-0
        PredictionTimelineView(
            liveData: LiveTimelineData(
                matchId: 12345,
                predictedOutcome: "home",
                homeGoals: 1,
                awayGoals: 0,
                elapsed: 35,
                goals: [
                    TimelineGoal(minute: 23, extraMinute: nil, team: "home", teamName: "Barcelona", player: "Lewandowski", isOwnGoal: false, isPenalty: false)
                ],
                status: "1H"
            )
        )

        // Live match - prediction wrong (predicted home, currently draw)
        PredictionTimelineView(
            liveData: LiveTimelineData(
                matchId: 12346,
                predictedOutcome: "home",
                homeGoals: 1,
                awayGoals: 1,
                elapsed: 67,
                goals: [
                    TimelineGoal(minute: 15, extraMinute: nil, team: "home", teamName: "Real Madrid", player: "Vinicius", isOwnGoal: false, isPenalty: false),
                    TimelineGoal(minute: 55, extraMinute: nil, team: "away", teamName: "Atletico", player: "Griezmann", isOwnGoal: false, isPenalty: false)
                ],
                status: "2H"
            )
        )
    }
    .padding()
    .background(Color.black)
    .preferredColorScheme(.dark)
}
