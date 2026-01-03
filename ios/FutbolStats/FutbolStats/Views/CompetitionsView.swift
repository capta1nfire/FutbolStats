import SwiftUI

struct CompetitionsView: View {
    @State private var competitions: [CompetitionItem] = []
    @State private var isLoading = true
    @State private var error: String?

    // Group competitions by confederation
    var groupedCompetitions: [String: [CompetitionItem]] {
        Dictionary(grouping: competitions) { $0.confederation }
    }

    // Confederation order
    let confederationOrder = ["FIFA", "CONMEBOL", "UEFA", "CONCACAF", "CAF", "AFC", "OFC", "Other"]

    var sortedConfederations: [String] {
        groupedCompetitions.keys.sorted { first, second in
            let firstIndex = confederationOrder.firstIndex(of: first) ?? confederationOrder.count
            let secondIndex = confederationOrder.firstIndex(of: second) ?? confederationOrder.count
            return firstIndex < secondIndex
        }
    }

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    VStack {
                        Spacer()
                        ProgressView("Loading competitions...")
                        Spacer()
                    }
                } else if let error = error {
                    VStack(spacing: 16) {
                        Spacer()
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundStyle(.orange)
                        Text(error)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.secondary)
                        Button("Retry") {
                            Task { await loadCompetitions() }
                        }
                        .buttonStyle(.borderedProminent)
                        Spacer()
                    }
                    .padding()
                } else if competitions.isEmpty {
                    VStack(spacing: 16) {
                        Spacer()
                        Image(systemName: "trophy")
                            .font(.system(size: 60))
                            .foregroundStyle(.secondary)
                        Text("No competitions found")
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                } else {
                    competitionsList
                }
            }
            .navigationTitle("Competitions")
            .task {
                await loadCompetitions()
            }
            .refreshable {
                await loadCompetitions()
            }
        }
    }

    private var competitionsList: some View {
        List {
            ForEach(sortedConfederations, id: \.self) { confederation in
                Section {
                    ForEach(groupedCompetitions[confederation] ?? []) { competition in
                        CompetitionRowView(competition: competition)
                    }
                } header: {
                    HStack {
                        confederationIcon(for: confederation)
                        Text(confederation)
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
    }

    @ViewBuilder
    private func confederationIcon(for confederation: String) -> some View {
        switch confederation {
        case "FIFA":
            Image(systemName: "globe")
                .foregroundStyle(.blue)
        case "CONMEBOL":
            Image(systemName: "globe.americas.fill")
                .foregroundStyle(.yellow)
        case "UEFA":
            Image(systemName: "globe.europe.africa.fill")
                .foregroundStyle(.blue)
        case "CONCACAF":
            Image(systemName: "globe.americas.fill")
                .foregroundStyle(.green)
        case "CAF":
            Image(systemName: "globe.europe.africa.fill")
                .foregroundStyle(.orange)
        case "AFC":
            Image(systemName: "globe.asia.australia.fill")
                .foregroundStyle(.red)
        case "OFC":
            Image(systemName: "globe.asia.australia.fill")
                .foregroundStyle(.cyan)
        default:
            Image(systemName: "sportscourt")
                .foregroundStyle(.secondary)
        }
    }

    private func loadCompetitions() async {
        isLoading = true
        error = nil

        do {
            competitions = try await APIClient.shared.getCompetitions()
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }
}

// MARK: - Competition Row View

struct CompetitionRowView: View {
    let competition: CompetitionItem

    var priorityColor: Color {
        switch competition.priority {
        case "critical": return .red
        case "high": return .orange
        case "medium": return .yellow
        default: return .gray
        }
    }

    var body: some View {
        HStack(spacing: 12) {
            // Competition icon
            Image(systemName: competition.matchType == "international" ? "flag.2.crossed.fill" : "soccerball")
                .font(.title2)
                .foregroundStyle(priorityColor)
                .frame(width: 40, height: 40)
                .background(priorityColor.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 4) {
                Text(competition.name)
                    .font(.headline)

                HStack(spacing: 8) {
                    Text(competition.matchType.capitalized)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("•")
                        .foregroundStyle(.secondary)

                    Text("Weight: \(String(format: "%.1f", competition.matchWeight))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            // Priority badge
            Text(competition.priority.capitalized)
                .font(.caption2)
                .fontWeight(.medium)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(priorityColor.opacity(0.2))
                .foregroundStyle(priorityColor)
                .clipShape(Capsule())
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    CompetitionsView()
}
