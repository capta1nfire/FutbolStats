import SwiftUI

// MARK: - Data Models

/// Priority levels for narrative insights
enum NarrativeInsightPriority: Int, Codable, Comparable {
    case caution = 0      // Urgent warnings
    case heroic = 1       // Heroic/collapse moments
    case admission = 2    // Model admissions
    case analysis = 3     // Statistical analysis
    case context = 4      // Context (relegation, rivalry)
    case summary = 5      // Match summary

    static func < (lhs: NarrativeInsightPriority, rhs: NarrativeInsightPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// A single narrative insight from the reasoning engine
struct NarrativeInsight: Identifiable, Codable {
    let id: UUID
    let type: String
    let icon: String
    let message: String
    let priority: Int

    init(id: UUID = UUID(), type: String, icon: String, message: String, priority: Int) {
        self.id = id
        self.type = type
        self.icon = icon
        self.message = message
        self.priority = priority
    }

    enum CodingKeys: String, CodingKey {
        case type, icon, message, priority
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = UUID()
        self.type = try container.decode(String.self, forKey: .type)
        self.icon = try container.decode(String.self, forKey: .icon)
        self.message = try container.decode(String.self, forKey: .message)
        self.priority = try container.decode(Int.self, forKey: .priority)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(type, forKey: .type)
        try container.encode(icon, forKey: .icon)
        try container.encode(message, forKey: .message)
        try container.encode(priority, forKey: .priority)
    }

    /// Computed color based on insight type
    var accentColor: Color {
        switch type {
        case "caution", "goalkeeper_heroic":
            return .yellow
        case "admission", "big_team_collapse":
            return .orange
        case "sterile_favorite", "clinical_underdog":
            return .cyan
        case "urgency_relegation":
            return .red
        case "summary":
            return .green
        default:
            return .blue
        }
    }
}

/// Momentum analysis from the reasoning engine
struct MomentumAnalysis: Codable {
    let type: String      // "collapse", "overwhelmed", "unlucky"
    let icon: String
    let message: String

    var accentColor: Color {
        switch type {
        case "collapse":
            return .orange
        case "overwhelmed":
            return .red
        case "unlucky":
            return .purple
        default:
            return .gray
        }
    }
}

/// Container for all narrative insights from API
struct NarrativeInsightsResponse: Codable {
    let insights: [NarrativeInsight]
    let momentumAnalysis: MomentumAnalysis?

    enum CodingKeys: String, CodingKey {
        case insights
        case momentumAnalysis = "momentum_analysis"
    }
}

// MARK: - Single Insight Card

struct NarrativeInsightCard: View {
    let insight: NarrativeInsight

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Icon circle
            ZStack {
                Circle()
                    .fill(insight.accentColor.opacity(0.2))
                    .frame(width: 36, height: 36)

                Image(systemName: insight.icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(insight.accentColor)
            }

            // Message
            Text(insight.message)
                .font(.subheadline)
                .foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)
                .multilineTextAlignment(.leading)

            Spacer(minLength: 0)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(white: 0.12))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(insight.accentColor.opacity(0.3), lineWidth: 1)
                )
        )
    }
}

// MARK: - Momentum Analysis Card

struct MomentumAnalysisCard: View {
    let momentum: MomentumAnalysis

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: momentum.icon)
                .font(.system(size: 20, weight: .bold))
                .foregroundStyle(momentum.accentColor)

            VStack(alignment: .leading, spacing: 4) {
                Text("Momentum")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(momentum.accentColor)
                    .textCase(.uppercase)

                Text(momentum.message)
                    .font(.subheadline)
                    .foregroundStyle(.white)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(momentum.accentColor.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(momentum.accentColor.opacity(0.4), lineWidth: 1.5)
                )
        )
    }
}

// MARK: - Main Insights Container

struct MatchNarrativeInsightsView: View {
    let insights: [NarrativeInsight]
    let momentumAnalysis: MomentumAnalysis?

    /// Sorted insights by priority (lowest first = most important)
    private var sortedInsights: [NarrativeInsight] {
        insights.sorted { $0.priority < $1.priority }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Section header
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundStyle(.cyan)
                Text("Análisis del Partido")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
            }
            .padding(.bottom, 4)

            // Momentum card (if present) - shown first
            if let momentum = momentumAnalysis {
                MomentumAnalysisCard(momentum: momentum)
            }

            // Insight cards
            ForEach(sortedInsights) { insight in
                NarrativeInsightCard(insight: insight)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(white: 0.08))
        )
    }
}

// MARK: - Preview

struct NarrativeInsightsView_Previews: PreviewProvider {
    static var previews: some View {
        // Sevilla 0-3 Levante example
        let sevillaLevanteInsights = [
            NarrativeInsight(
                type: "admission",
                icon: "exclamationmark.triangle.fill",
                message: "Nos equivocamos. Apostamos por victoria local con 58% de confianza, pero ganó el visitante.",
                priority: 2
            ),
            NarrativeInsight(
                type: "sterile_favorite",
                icon: "target",
                message: "Sevilla llegó mucho pero sin peligro real. De 14 intentos, solo 3 fueron al arco.",
                priority: 3
            ),
            NarrativeInsight(
                type: "big_team_collapse",
                icon: "house.fill",
                message: "Sevilla se bloqueó ante su gente. Perdió en casa contra Levante por 0-3. Una derrota difícil de explicar.",
                priority: 1
            ),
            NarrativeInsight(
                type: "clinical_underdog",
                icon: "scope",
                message: "Levante aprovechó cada oportunidad: 3 goles con solo 5 tiros al arco.",
                priority: 3
            ),
            NarrativeInsight(
                type: "summary",
                icon: "checkmark.circle.fill",
                message: "Levante ganó 0-3. Los números no explican todo.",
                priority: 5
            )
        ]

        let momentum = MomentumAnalysis(
            type: "collapse",
            icon: "arrow.down.right.circle.fill",
            message: "Sevilla no reaccionó después de ir abajo. Solo 3 tiro(s) al arco en todo el partido."
        )

        ScrollView {
            MatchNarrativeInsightsView(
                insights: sevillaLevanteInsights,
                momentumAnalysis: momentum
            )
            .padding()
        }
        .background(Color.black)
        .preferredColorScheme(.dark)
    }
}
