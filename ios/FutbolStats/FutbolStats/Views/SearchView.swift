import SwiftUI

// MARK: - Recent Search Item

struct RecentSearchItem: Codable, Identifiable, Equatable {
    let id: String
    let name: String
    let logoUrl: String?
    let type: SearchItemType
    let timestamp: Date

    enum SearchItemType: String, Codable {
        case team
        case league
    }

    static func == (lhs: RecentSearchItem, rhs: RecentSearchItem) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - Team Search Result

struct TeamSearchResult: Identifiable, Equatable {
    var id: String { name }
    let name: String
    let logoUrl: String?

    static func == (lhs: TeamSearchResult, rhs: TeamSearchResult) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - Search View

struct SearchView: View {
    @StateObject private var viewModel = PredictionsViewModel()
    @State private var searchText = ""
    @AppStorage("recentSearches") private var recentSearchesData: Data = Data()
    private let maxRecentSearches = 10

    var body: some View {
        NavigationStack {
            mainContent
                .background(AppColors.backgroundGradient)
                .navigationTitle("Search")
                .navigationBarTitleDisplayMode(.inline)
                .toolbarBackground(.visible, for: .navigationBar)
                .toolbarBackground(Color(red: 0.02, green: 0.02, blue: 0.06), for: .navigationBar)
                .searchable(text: $searchText, prompt: "Teams, leagues...")
                .task {
                    await viewModel.refresh()
                }
        }
    }

    // MARK: - Main Content

    @ViewBuilder
    private var mainContent: some View {
        if viewModel.isLoading && viewModel.predictions.isEmpty {
            VStack {
                Spacer()
                ProgressView()
                    .tint(.white)
                Text("Loading matches...")
                    .font(.caption)
                    .foregroundStyle(AppColors.textTertiary)
                    .padding(.top, 8)
                Spacer()
            }
        } else if !searchText.isEmpty {
            // Active search with text - show filtered results
            if matchingTeams.isEmpty && filteredPredictions.isEmpty {
                noResultsView
            } else {
                searchResultsView
            }
        } else {
            // No search text - show recents
            recentSearchesView
        }
    }

    // MARK: - No Results View

    private var noResultsView: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "magnifyingglass")
                .font(.system(size: 48))
                .foregroundStyle(AppColors.textMuted)

            Text("No results for \"\(searchText)\"")
                .font(.subheadline)
                .foregroundStyle(AppColors.textTertiary)
            Spacer()
        }
    }

    // MARK: - Recent Searches View

    @ViewBuilder
    private var recentSearchesView: some View {
        if recentSearches.isEmpty {
            VStack(spacing: 12) {
                Spacer()
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 48))
                    .foregroundStyle(AppColors.textMuted)

                Text("Search for teams or leagues")
                    .font(.subheadline)
                    .foregroundStyle(AppColors.textTertiary)

                Text("\(viewModel.predictions.count) matches available")
                    .font(.caption)
                    .foregroundStyle(AppColors.textMuted)
                Spacer()
            }
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        Text("Recent")
                            .font(.headline)
                            .foregroundStyle(AppColors.textPrimary)

                        Spacer()

                        Button {
                            clearRecentSearches()
                        } label: {
                            Text("Clear")
                                .font(.subheadline)
                                .foregroundStyle(AppColors.accent)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 16)

                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 16) {
                            ForEach(recentSearches) { item in
                                Button {
                                    searchText = item.name
                                } label: {
                                    recentSearchCircle(item: item)
                                }
                            }
                        }
                        .padding(.horizontal, 16)
                    }

                    Spacer()
                }
            }
        }
    }

    private func recentSearchCircle(item: RecentSearchItem) -> some View {
        VStack(spacing: 6) {
            Group {
                if let logoUrl = item.logoUrl, let url = URL(string: logoUrl) {
                    CachedAsyncImage(url: url) { image in
                        image.resizable().aspectRatio(contentMode: .fit)
                    } placeholder: {
                        Image(systemName: "shield.fill")
                            .font(.title)
                            .foregroundStyle(AppColors.textTertiary)
                    }
                    .frame(width: 40, height: 40)
                } else {
                    Image(systemName: "shield.fill")
                        .font(.title)
                        .foregroundStyle(AppColors.textTertiary)
                }
            }
            .frame(width: 64, height: 64)
            .modifier(GlassCircleModifier())

            Text(item.name)
                .font(.caption2)
                .foregroundStyle(AppColors.textSecondary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(width: 72, height: 28)
        }
    }

    // MARK: - Search Results View

    private var searchResultsView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if !matchingTeams.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Teams")
                            .font(.headline)
                            .foregroundStyle(AppColors.textPrimary)
                            .padding(.horizontal, 16)

                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 16) {
                                ForEach(matchingTeams) { team in
                                    teamCircle(team: team)
                                }
                            }
                            .padding(.horizontal, 16)
                        }
                    }
                }

                if !filteredPredictions.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Matches")
                            .font(.headline)
                            .foregroundStyle(AppColors.textPrimary)
                            .padding(.horizontal, 16)

                        VStack(spacing: 8) {
                            ForEach(filteredPredictions) { prediction in
                                NavigationLink(destination: MatchDetailView(prediction: prediction)) {
                                    MatchCard(
                                        prediction: prediction,
                                        showValueBadge: prediction.hasValueBet ?? false
                                    )
                                    .padding(.horizontal, 12)
                                    .modifier(GlassCardModifier())
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(.horizontal, 8)
                    }
                }
            }
            .padding(.top, 16)
            .padding(.bottom, 32)
        }
    }

    private func teamCircle(team: TeamSearchResult) -> some View {
        Button {
            addToRecentSearches(team: team)
            searchText = team.name
        } label: {
            VStack(spacing: 6) {
                Group {
                    if let logoUrl = team.logoUrl, let url = URL(string: logoUrl) {
                        CachedAsyncImage(url: url) { image in
                            image.resizable().aspectRatio(contentMode: .fit)
                        } placeholder: {
                            Image(systemName: "shield.fill")
                                .font(.title)
                                .foregroundStyle(AppColors.textTertiary)
                        }
                        .frame(width: 40, height: 40)
                    } else {
                        Image(systemName: "shield.fill")
                            .font(.title)
                            .foregroundStyle(AppColors.textTertiary)
                    }
                }
                .frame(width: 64, height: 64)
                .modifier(GlassCircleModifier())

                Text(team.name)
                    .font(.caption2)
                    .foregroundStyle(AppColors.textPrimary)
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .frame(width: 72, height: 28)
            }
        }
    }

    // MARK: - Search Helpers

    private var recentSearches: [RecentSearchItem] {
        (try? JSONDecoder().decode([RecentSearchItem].self, from: recentSearchesData)) ?? []
    }

    private func saveRecentSearches(_ items: [RecentSearchItem]) {
        if let data = try? JSONEncoder().encode(items) {
            recentSearchesData = data
        }
    }

    private func addToRecentSearches(team: TeamSearchResult) {
        var searches = recentSearches
        searches.removeAll { $0.id == team.id }
        let item = RecentSearchItem(
            id: team.id,
            name: team.name,
            logoUrl: team.logoUrl,
            type: .team,
            timestamp: Date()
        )
        searches.insert(item, at: 0)
        if searches.count > maxRecentSearches {
            searches = Array(searches.prefix(maxRecentSearches))
        }
        saveRecentSearches(searches)
    }

    private func clearRecentSearches() {
        saveRecentSearches([])
    }

    private var matchingTeams: [TeamSearchResult] {
        guard !searchText.isEmpty else { return [] }
        let query = searchText.lowercased()

        var seen = Set<String>()
        var teams: [TeamSearchResult] = []

        for prediction in viewModel.predictions {
            if prediction.homeTeam.lowercased().contains(query) {
                let team = TeamSearchResult(name: prediction.homeTeam, logoUrl: prediction.homeTeamLogo)
                if !seen.contains(team.id) {
                    seen.insert(team.id)
                    teams.append(team)
                }
            }
            if prediction.awayTeam.lowercased().contains(query) {
                let team = TeamSearchResult(name: prediction.awayTeam, logoUrl: prediction.awayTeamLogo)
                if !seen.contains(team.id) {
                    seen.insert(team.id)
                    teams.append(team)
                }
            }
        }
        return teams
    }

    private var filteredPredictions: [MatchPrediction] {
        guard !searchText.isEmpty else { return [] }
        let query = searchText.lowercased()
        return viewModel.predictions.filter { prediction in
            prediction.homeTeam.lowercased().contains(query) ||
            prediction.awayTeam.lowercased().contains(query) ||
            prediction.leagueName.lowercased().contains(query)
        }
    }
}

#Preview {
    SearchView()
}
