import SwiftUI

struct TeamsView: View {
    @State private var teams: [TeamItem] = []
    @State private var isLoading = true
    @State private var error: String?
    @State private var selectedCategory: TeamCategory = .national

    enum TeamCategory: String, CaseIterable {
        case national = "National"
        case club = "Clubs"

        var apiValue: String {
            switch self {
            case .national: return "national"
            case .club: return "club"
            }
        }
    }

    var filteredTeams: [TeamItem] {
        teams.filter { $0.teamType == selectedCategory.apiValue }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Category Picker
                Picker("Category", selection: $selectedCategory) {
                    ForEach(TeamCategory.allCases, id: \.self) { category in
                        Text(category.rawValue).tag(category)
                    }
                }
                .pickerStyle(.segmented)
                .padding()

                // Content
                if isLoading {
                    Spacer()
                    ProgressView("Loading teams...")
                    Spacer()
                } else if let error = error {
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundStyle(.orange)
                        Text(error)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.secondary)
                        Button("Retry") {
                            Task { await loadTeams() }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding()
                    Spacer()
                } else if filteredTeams.isEmpty {
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "person.3")
                            .font(.system(size: 60))
                            .foregroundStyle(.secondary)
                        Text("No \(selectedCategory.rawValue.lowercased()) teams found")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                } else {
                    teamsList
                }
            }
            .navigationTitle("Teams")
            .task {
                await loadTeams()
            }
            .refreshable {
                await loadTeams()
            }
        }
    }

    private var teamsList: some View {
        List {
            ForEach(filteredTeams) { team in
                TeamRowView(team: team)
            }
        }
        .listStyle(.insetGrouped)
    }

    private func loadTeams() async {
        isLoading = true
        error = nil

        do {
            teams = try await APIClient.shared.getTeams()
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }
}

// MARK: - Team Row View

struct TeamRowView: View {
    let team: TeamItem

    var body: some View {
        HStack(spacing: 12) {
            // Team Logo placeholder
            if let logoUrl = team.logoUrl, let url = URL(string: logoUrl) {
                AsyncImage(url: url) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } placeholder: {
                    teamPlaceholder
                }
                .frame(width: 40, height: 40)
            } else {
                teamPlaceholder
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(team.name)
                    .font(.headline)

                if let country = team.country {
                    Text(country)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
    }

    private var teamPlaceholder: some View {
        Image(systemName: team.teamType == "national" ? "flag.fill" : "shield.fill")
            .font(.title2)
            .foregroundStyle(.secondary)
            .frame(width: 40, height: 40)
            .background(Color(.systemGray5))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

#Preview {
    TeamsView()
}
