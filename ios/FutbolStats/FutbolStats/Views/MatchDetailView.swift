import SwiftUI

struct MatchDetailView: View {
    let prediction: MatchPrediction

    @State private var matchDetails: MatchDetailsResponse?
    @State private var isLoading = true
    @State private var error: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                matchHeader

                // Probabilities Card
                probabilitiesCard

                // Fair Odds Card
                fairOddsCard

                // Team History Section
                if isLoading {
                    ProgressView("Loading team history...")
                        .padding()
                } else if let details = matchDetails {
                    // Home Team History
                    TeamHistoryCard(
                        teamName: details.homeTeam.name,
                        history: details.homeTeam.history,
                        isHome: true
                    )

                    // Away Team History
                    TeamHistoryCard(
                        teamName: details.awayTeam.name,
                        history: details.awayTeam.history,
                        isHome: false
                    )
                } else if let error = error {
                    Text("Could not load history: \(error)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding()
                }

                // Value Bets (if available)
                if let valueBets = prediction.valueBets, !valueBets.isEmpty {
                    valueBetsCard(valueBets)
                }

                // Market Odds (if available)
                if let marketOdds = prediction.marketOdds {
                    marketOddsCard(marketOdds)
                }
            }
            .padding()
        }
        .navigationTitle("Match Prediction")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await loadMatchDetails()
        }
    }

    // MARK: - Load Match Details

    private func loadMatchDetails() async {
        guard let matchId = prediction.matchId else {
            isLoading = false
            error = "No match ID"
            return
        }

        do {
            matchDetails = try await APIClient.shared.getMatchDetails(matchId: matchId)
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    // MARK: - Match Header

    private var matchHeader: some View {
        VStack(spacing: 16) {
            HStack {
                VStack {
                    Text(prediction.homeTeam)
                        .font(.title2)
                        .fontWeight(.bold)
                        .multilineTextAlignment(.center)
                    Text("HOME")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)

                Text("vs")
                    .font(.title3)
                    .foregroundStyle(.secondary)

                VStack {
                    Text(prediction.awayTeam)
                        .font(.title2)
                        .fontWeight(.bold)
                        .multilineTextAlignment(.center)
                    Text("AWAY")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
            }

            Text(prediction.formattedDate)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            // Predicted Outcome
            Text(prediction.probabilities.predictedOutcome)
                .font(.headline)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.blue.opacity(0.1))
                .foregroundStyle(.blue)
                .clipShape(Capsule())
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.05), radius: 8, y: 4)
    }

    // MARK: - Probabilities Card

    private var probabilitiesCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Win Probabilities", systemImage: "chart.pie")
                .font(.headline)

            HStack(spacing: 0) {
                ProbabilityBar(
                    label: "Home",
                    value: prediction.probabilities.home,
                    color: .blue
                )
                ProbabilityBar(
                    label: "Draw",
                    value: prediction.probabilities.draw,
                    color: .gray
                )
                ProbabilityBar(
                    label: "Away",
                    value: prediction.probabilities.away,
                    color: .red
                )
            }
            .frame(height: 40)
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack {
                ProbabilityLabel(label: "Home", value: prediction.probabilities.homePercent, color: .blue)
                Spacer()
                ProbabilityLabel(label: "Draw", value: prediction.probabilities.drawPercent, color: .gray)
                Spacer()
                ProbabilityLabel(label: "Away", value: prediction.probabilities.awayPercent, color: .red)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.05), radius: 8, y: 4)
    }

    // MARK: - Fair Odds Card

    private var fairOddsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Fair Odds", systemImage: "dollarsign.circle")
                .font(.headline)

            HStack {
                OddsBox(label: "Home", value: prediction.fairOdds.homeFormatted, color: .blue)
                OddsBox(label: "Draw", value: prediction.fairOdds.drawFormatted, color: .gray)
                OddsBox(label: "Away", value: prediction.fairOdds.awayFormatted, color: .red)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.05), radius: 8, y: 4)
    }

    // MARK: - Value Bets Card

    private func valueBetsCard(_ valueBets: [ValueBet]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Value Bets Found!", systemImage: "star.fill")
                .font(.headline)
                .foregroundStyle(.green)

            ForEach(valueBets) { bet in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(bet.outcome.capitalized)
                            .font(.subheadline)
                            .fontWeight(.semibold)

                        Spacer()

                        Text(bet.edgePercent)
                            .font(.headline)
                            .foregroundStyle(.green)
                    }

                    HStack {
                        VStack(alignment: .leading) {
                            Text("Our Prob")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(String(format: "%.1f%%", bet.ourProbability * 100))
                                .font(.subheadline)
                        }

                        Spacer()

                        VStack(alignment: .center) {
                            Text("Market Odds")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(String(format: "%.2f", bet.marketOdds))
                                .font(.subheadline)
                        }

                        Spacer()

                        VStack(alignment: .trailing) {
                            Text("Fair Odds")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(String(format: "%.2f", bet.fairOdds))
                                .font(.subheadline)
                        }
                    }
                }
                .padding()
                .background(.green.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.05), radius: 8, y: 4)
    }

    // MARK: - Market Odds Card

    private func marketOddsCard(_ odds: MarketOdds) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Market Odds", systemImage: "building.columns")
                .font(.headline)

            HStack {
                OddsBox(label: "Home", value: String(format: "%.2f", odds.home), color: .blue)
                OddsBox(label: "Draw", value: String(format: "%.2f", odds.draw), color: .gray)
                OddsBox(label: "Away", value: String(format: "%.2f", odds.away), color: .red)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.05), radius: 8, y: 4)
    }
}

// MARK: - Team History Card

struct TeamHistoryCard: View {
    let teamName: String
    let history: [MatchHistoryItem]
    let isHome: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(teamName, systemImage: isHome ? "house.fill" : "airplane")
                    .font(.headline)
                Spacer()
                Text("Last 5 matches")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if history.isEmpty {
                Text("No match history available")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 8)
            } else {
                // Form indicator (W W L D W)
                HStack(spacing: 8) {
                    ForEach(history.prefix(5)) { match in
                        ResultBadge(result: match.result)
                    }
                    Spacer()
                }

                Divider()

                // Match list
                ForEach(history.prefix(5)) { match in
                    MatchHistoryRow(match: match)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.05), radius: 8, y: 4)
    }
}

// MARK: - Result Badge

struct ResultBadge: View {
    let result: String

    var backgroundColor: Color {
        switch result {
        case "W": return .green
        case "L": return .red
        default: return .gray
        }
    }

    var body: some View {
        Text(result)
            .font(.caption)
            .fontWeight(.bold)
            .foregroundStyle(.white)
            .frame(width: 28, height: 28)
            .background(backgroundColor)
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

// MARK: - Match History Row

struct MatchHistoryRow: View {
    let match: MatchHistoryItem

    var resultColor: Color {
        switch match.result {
        case "W": return .green
        case "L": return .red
        default: return .gray
        }
    }

    var body: some View {
        HStack {
            // Result
            Text(match.result)
                .font(.caption)
                .fontWeight(.bold)
                .foregroundStyle(.white)
                .frame(width: 24, height: 24)
                .background(resultColor)
                .clipShape(RoundedRectangle(cornerRadius: 4))

            // Score
            Text(match.scoreDisplay)
                .font(.subheadline)
                .fontWeight(.medium)
                .frame(width: 50)

            // Opponent
            VStack(alignment: .leading, spacing: 2) {
                Text(match.isHome ? "vs" : "@")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(match.opponent)
                    .font(.subheadline)
                    .lineLimit(1)
            }

            Spacer()

            // Date
            Text(match.formattedDate)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Supporting Views

struct ProbabilityBar: View {
    let label: String
    let value: Double
    let color: Color

    var body: some View {
        Rectangle()
            .fill(color)
            .frame(width: nil)
            .frame(maxWidth: .infinity)
            .scaleEffect(x: value, y: 1, anchor: .leading)
    }
}

struct ProbabilityLabel: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 12, height: 12)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
        }
    }
}

struct OddsBox: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundStyle(color)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(color.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

#Preview {
    NavigationStack {
        MatchDetailView(prediction: MatchPrediction(
            matchId: 1,
            matchExternalId: 12345,
            homeTeam: "Argentina",
            awayTeam: "Brazil",
            date: "2025-01-15T20:00:00",
            probabilities: Probabilities(home: 0.45, draw: 0.28, away: 0.27),
            fairOdds: FairOdds(home: 2.22, draw: 3.57, away: 3.70),
            marketOdds: MarketOdds(home: 2.10, draw: 3.40, away: 3.50),
            valueBets: [
                ValueBet(
                    outcome: "home",
                    ourProbability: 0.45,
                    impliedProbability: 0.40,
                    edge: 0.05,
                    marketOdds: 2.50,
                    fairOdds: 2.22
                )
            ]
        ))
    }
}
