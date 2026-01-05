import SwiftUI

/// Timeline visualization showing when the prediction was correct during the match.
/// Green segments = prediction aligned with current score
/// Gray segments = prediction misaligned (wrong or neutral)
struct PredictionTimelineView: View {
    let timeline: MatchTimelineResponse

    private let correctColor = Color.green
    private let neutralColor = Color.gray.opacity(0.4)
    private let barHeight: CGFloat = 24

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Section header
            HStack {
                Image(systemName: "clock.arrow.circlepath")
                    .foregroundStyle(.gray)
                Text("Prediction Timeline")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(.white)
                Spacer()
                // Result indicator
                if timeline.prediction.correct {
                    Label("Correct", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                } else {
                    Label("Wrong", systemImage: "xmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }

            // Timeline bar with goals
            ZStack(alignment: .leading) {
                // Background segments
                GeometryReader { geo in
                    timelineBar(width: geo.size.width)
                }
                .frame(height: barHeight)
                .clipShape(RoundedRectangle(cornerRadius: 6))

                // Goal markers overlay
                GeometryReader { geo in
                    goalMarkers(width: geo.size.width)
                }
                .frame(height: barHeight + 24)
            }
            .frame(height: barHeight + 24)

            // Minute labels
            HStack {
                Text("0'")
                    .font(.caption2)
                    .foregroundStyle(.gray)
                Spacer()
                Text("45'")
                    .font(.caption2)
                    .foregroundStyle(.gray)
                Spacer()
                Text("\(timeline.totalMinutes)'")
                    .font(.caption2)
                    .foregroundStyle(.gray)
            }

            // Summary
            HStack(spacing: 16) {
                HStack(spacing: 4) {
                    Circle()
                        .fill(correctColor)
                        .frame(width: 8, height: 8)
                    Text("In Line: \(Int(timeline.summary.correctMinutes)) min (\(Int(timeline.summary.correctPercentage))%)")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.8))
                }

                Spacer()

                // Final score
                Text("\(timeline.finalScore.home) - \(timeline.finalScore.away)")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.1))
                    .clipShape(Capsule())
            }
        }
        .padding(16)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Timeline Bar

    private func timelineBar(width: CGFloat) -> some View {
        HStack(spacing: 0) {
            ForEach(timeline.segments) { segment in
                let segmentWidth = width * CGFloat(segment.duration) / CGFloat(timeline.totalMinutes)
                Rectangle()
                    .fill(segment.status == "correct" ? correctColor : neutralColor)
                    .frame(width: max(1, segmentWidth))
            }
        }
    }

    // MARK: - Goal Markers

    private func goalMarkers(width: CGFloat) -> some View {
        let groupedGoals = groupNearbyGoals(timeline.goals)

        return ZStack {
            ForEach(groupedGoals) { group in
                let position = CGFloat(group.effectiveMinute) / CGFloat(timeline.totalMinutes) * width

                VStack(spacing: 2) {
                    // Goal icon(s)
                    if group.count > 1 {
                        // Grouped goals
                        HStack(spacing: 2) {
                            Text("⚽")
                                .font(.system(size: 12))
                            Text("×\(group.count)")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundStyle(.white)
                        }
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(Color.black.opacity(0.7))
                        .clipShape(Capsule())
                    } else {
                        // Single goal
                        Text("⚽")
                            .font(.system(size: 14))
                    }

                    // Minute label
                    Text(group.displayMinute)
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.gray)
                }
                .position(x: clampPosition(position, width: width), y: barHeight / 2 + 12)
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
                summary: TimelineSummary(correctMinutes: 47, correctPercentage: 52.2)
            )
        )
    }
    .padding()
    .background(Color.black)
    .preferredColorScheme(.dark)
}
