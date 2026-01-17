import SwiftUI

// MARK: - LLM Narrative View (Schema v3.2)

/// Main view for displaying LLM-generated match narrative
struct LLMNarrativeView: View {
    let narrative: LLMNarrativePayload

    private var tone: String {
        narrative.narrative?.tone ?? "neutral"
    }

    private var toneColor: Color {
        switch tone {
        case "reinforce_win":
            return .green
        case "mitigate_loss":
            return .orange
        default:
            return .gray
        }
    }

    private var toneBadgeText: String {
        switch tone {
        case "reinforce_win":
            return "Victoria confirmada"
        case "mitigate_loss":
            return "Derrota analizada"
        default:
            return "Análisis"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Section header
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundStyle(.cyan)
                Text("Análisis del Partido")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
            }

            // Title (headline)
            if let title = narrative.narrative?.title, !title.isEmpty {
                Text(title)
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
            }

            // Body text (preserving line breaks)
            if let body = narrative.narrative?.body, !body.isEmpty {
                Text(body)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.9))
                    .fixedSize(horizontal: false, vertical: true)
                    .multilineTextAlignment(.leading)
            }

            // Key factors
            if let keyFactors = narrative.narrative?.keyFactors, !keyFactors.isEmpty {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Factores Clave")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.gray)
                        .textCase(.uppercase)

                    ForEach(keyFactors) { factor in
                        LLMKeyFactorRow(factor: factor)
                    }
                }
                .padding(.top, 4)
            }

            // Responsible note (caption)
            if let note = narrative.narrative?.responsibleNote, !note.isEmpty {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.gray)
                    .italic()
                    .padding(.top, 8)
            }
        }
        .padding(16)
        .modifier(GlassEffectModifier())
    }
}

// MARK: - Key Factor Row

struct LLMKeyFactorRow: View {
    let factor: LLMKeyFactor

    private var directionColor: Color {
        switch factor.direction {
        case "pro-pick":
            return .green
        case "anti-pick":
            return .red
        default:
            return .gray
        }
    }

    private var directionIcon: String {
        switch factor.direction {
        case "pro-pick":
            return "arrow.up.circle.fill"
        case "anti-pick":
            return "arrow.down.circle.fill"
        default:
            return "circle.fill"
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: directionIcon)
                .font(.system(size: 14))
                .foregroundStyle(directionColor)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                if let label = factor.label {
                    Text(label)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundStyle(.white)
                }

                if let evidence = factor.evidence {
                    Text(evidence)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.7))
                }
            }

            Spacer(minLength: 0)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(directionColor.opacity(0.1))
        )
    }
}

// MARK: - Unavailable State View

struct LLMNarrativeUnavailableView: View {
    let status: String?

    private var message: String {
        switch status {
        case "pending":
            return "Generando análisis del partido..."
        case "skipped":
            return "Análisis no disponible para este partido"
        case "error":
            return "El análisis estará disponible próximamente"
        case "no_prediction":
            return "Análisis no disponible"
        default:
            return "Análisis no disponible"
        }
    }

    private var icon: String {
        switch status {
        case "pending":
            return "clock.fill"
        case "skipped":
            return "minus.circle.fill"
        case "error":
            return "clock.arrow.circlepath"  // Retry icon - less alarming
        case "no_prediction":
            return "doc.text.magnifyingglass"
        default:
            return "doc.text.magnifyingglass"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundStyle(.cyan)
                Text("Análisis del Partido")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
            }

            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 24))
                    .foregroundStyle(.gray)

                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.gray)
            }
            .padding(.vertical, 8)
        }
        .padding(16)
        .modifier(GlassEffectModifier())
    }
}

// MARK: - Preview

struct LLMNarrativeView_Previews: PreviewProvider {
    static var previews: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Example with full narrative
                LLMNarrativeView(
                    narrative: LLMNarrativePayload(
                        matchId: 12345,
                        lang: "es",
                        result: LLMResult(
                            ftScore: "2-1",
                            outcome: "home",
                            betWon: true
                        ),
                        prediction: LLMPrediction(
                            selection: "HOME",
                            confidence: 0.65,
                            probabilities: LLMProbabilities(home: 0.65, draw: 0.20, away: 0.15),
                            marketOdds: LLMMarketOdds(home: 1.80, draw: 3.50, away: 4.50)
                        ),
                        narrative: LLMNarrative(
                            title: "El Barça cumple con lo esperado",
                            body: "Victoria trabajada del equipo local.\n\nEl partido se decidió en los últimos 20 minutos cuando el Barcelona encontró espacios ante un rival cansado.",
                            keyFactors: [
                                LLMKeyFactor(
                                    label: "Dominio territorial",
                                    evidence: "65% de posesión y 18 tiros totales",
                                    direction: "pro-pick"
                                ),
                                LLMKeyFactor(
                                    label: "Eficacia visitante",
                                    evidence: "1 gol con solo 3 tiros a puerta",
                                    direction: "anti-pick"
                                ),
                                LLMKeyFactor(
                                    label: "Factor campo",
                                    evidence: "4 victorias seguidas en casa",
                                    direction: "neutral"
                                )
                            ],
                            tone: "reinforce_win",
                            responsibleNote: "Las cuotas pueden variar. Apuesta con responsabilidad."
                        ),
                        marketOdds: LLMMarketOdds(home: 1.80, draw: 3.50, away: 4.50)
                    )
                )

                // Unavailable state
                LLMNarrativeUnavailableView(status: "pending")
            }
            .padding()
        }
        .background(Color.black)
        .preferredColorScheme(.dark)
    }
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
