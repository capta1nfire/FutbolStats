import SwiftUI

struct PredictionsListView: View {
    @StateObject private var viewModel = PredictionsViewModel()
    @State private var selectedDays = 7

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading && viewModel.predictions.isEmpty {
                    LoadingView()
                } else if let error = viewModel.error, viewModel.predictions.isEmpty {
                    ErrorView(message: error) {
                        Task { await viewModel.refresh() }
                    }
                } else if viewModel.predictions.isEmpty {
                    EmptyStateView()
                } else {
                    predictionsList
                }
            }
            .navigationTitle("FutbolStats")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Picker("Days", selection: $selectedDays) {
                            Text("3 days").tag(3)
                            Text("7 days").tag(7)
                            Text("14 days").tag(14)
                            Text("30 days").tag(30)
                        }
                    } label: {
                        Label("Filter", systemImage: "line.3.horizontal.decrease.circle")
                    }
                }

                ToolbarItem(placement: .topBarLeading) {
                    HStack {
                        Circle()
                            .fill(viewModel.modelLoaded ? .green : .red)
                            .frame(width: 8, height: 8)
                        Text(viewModel.modelLoaded ? "Model Ready" : "Model Offline")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .refreshable {
                await viewModel.refresh()
            }
            .onChange(of: selectedDays) { _, newValue in
                Task { await viewModel.loadPredictions(days: newValue) }
            }
            .task {
                await viewModel.refresh()
            }
        }
    }

    private var predictionsList: some View {
        List {
            if !viewModel.valueBetPredictions.isEmpty {
                Section {
                    ForEach(viewModel.valueBetPredictions) { prediction in
                        NavigationLink(destination: MatchDetailView(prediction: prediction)) {
                            MatchRowView(prediction: prediction, showValueBadge: true)
                        }
                    }
                } header: {
                    Label("Value Bets", systemImage: "star.fill")
                        .foregroundStyle(.yellow)
                }
            }

            Section {
                ForEach(viewModel.upcomingPredictions) { prediction in
                    NavigationLink(destination: MatchDetailView(prediction: prediction)) {
                        MatchRowView(prediction: prediction, showValueBadge: false)
                    }
                }
            } header: {
                Text("Upcoming Matches")
            }

            if let lastUpdated = viewModel.lastUpdated {
                Section {
                    Text("Last updated: \(lastUpdated.formatted(.relative(presentation: .named)))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .listStyle(.insetGrouped)
    }
}

// MARK: - Match Row View

struct MatchRowView: View {
    let prediction: MatchPrediction
    let showValueBadge: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Value Bet Banner (if applicable)
            if showValueBadge && prediction.isValueBet {
                ValueBetBanner(prediction: prediction)
            }

            // Teams
            HStack {
                VStack(alignment: .leading) {
                    Text(prediction.homeTeam)
                        .font(.headline)
                    Text("vs")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(prediction.awayTeam)
                        .font(.headline)
                }

                Spacer()

                // Probabilities with market comparison
                VStack(alignment: .trailing, spacing: 4) {
                    OddsComparisonRow(
                        label: "H",
                        probability: prediction.probabilities.home,
                        marketOdds: prediction.marketOdds?.home,
                        fairOdds: prediction.fairOdds.home
                    )
                    OddsComparisonRow(
                        label: "D",
                        probability: prediction.probabilities.draw,
                        marketOdds: prediction.marketOdds?.draw,
                        fairOdds: prediction.fairOdds.draw
                    )
                    OddsComparisonRow(
                        label: "A",
                        probability: prediction.probabilities.away,
                        marketOdds: prediction.marketOdds?.away,
                        fairOdds: prediction.fairOdds.away
                    )
                }
            }

            // Date
            HStack {
                Text(prediction.formattedDate)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                // Best Value Bet indicator
                if let best = prediction.bestValueBet {
                    Text("\(best.outcome.uppercased()) \(best.evDisplay)")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(evColor(for: best).opacity(0.2))
                        .foregroundStyle(evColor(for: best))
                        .clipShape(Capsule())
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func evColor(for bet: ValueBet) -> Color {
        let ev = bet.evPercentage ?? (bet.expectedValue.map { $0 * 100 }) ?? (bet.edge * 100)
        if ev >= 15 { return .yellow }
        if ev >= 10 { return .green }
        return .mint
    }
}

// MARK: - Value Bet Banner

struct ValueBetBanner: View {
    let prediction: MatchPrediction

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "dollarsign.circle.fill")
                .foregroundStyle(.green)

            Text("VALUE BET")
                .font(.caption2)
                .fontWeight(.heavy)
                .foregroundStyle(.green)

            if let best = prediction.bestValueBet {
                Text(best.evDisplay)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.green)
                    .clipShape(Capsule())
            }

            Spacer()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(.green.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Odds Comparison Row

struct OddsComparisonRow: View {
    let label: String
    let probability: Double
    let marketOdds: Double?
    let fairOdds: Double?

    /// Check if this is a value opportunity (market odds > fair odds)
    var isValue: Bool {
        guard let market = marketOdds, let fair = fairOdds else { return false }
        return market > fair
    }

    var body: some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundStyle(.secondary)
                .frame(width: 14)

            Text(String(format: "%.0f%%", probability * 100))
                .font(.caption)
                .fontWeight(.semibold)
                .frame(width: 32, alignment: .trailing)

            // Show market vs fair odds comparison if available
            if let market = marketOdds, let _ = fairOdds {
                HStack(spacing: 2) {
                    Text(String(format: "%.2f", market))
                        .font(.caption2)
                        .foregroundStyle(isValue ? .green : .secondary)

                    if isValue {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.green)
                    }
                }
                .frame(width: 44, alignment: .trailing)
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(isValue ? Color.green.opacity(0.1) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }
}

// MARK: - Probability Badge

struct ProbabilityBadge: View {
    let label: String
    let value: Double
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundStyle(.secondary)
            Text(String(format: "%.0f%%", value * 100))
                .font(.caption)
                .fontWeight(.semibold)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 2)
        .background(color.opacity(0.1))
        .clipShape(Capsule())
    }
}

// MARK: - Supporting Views

struct LoadingView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.5)
            Text("Loading predictions...")
                .foregroundStyle(.secondary)
        }
    }
}

struct ErrorView: View {
    let message: String
    let retryAction: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.orange)
            Text(message)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("Retry", action: retryAction)
                .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "sportscourt")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)
            Text("No upcoming matches")
                .font(.headline)
            Text("Check back later for predictions")
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    PredictionsListView()
}
