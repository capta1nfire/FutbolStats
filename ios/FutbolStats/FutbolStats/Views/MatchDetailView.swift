import SwiftUI

// MARK: - EV Calculation Result

struct EVResult {
    let outcome: String
    let fairOdds: Double
    let bookieOdds: Double
    let ev: Double
    let evPercentage: Double
    let isValue: Bool

    var evDisplay: String {
        String(format: "+%.1f%% EV", evPercentage)
    }
}

// MARK: - Team Form Data

struct TeamFormData: Identifiable {
    let id = UUID()
    let position: Int
    let teamName: String
    let logoUrl: String?
    let form: [String]
    let points: Int
}

// MARK: - Insight Type

enum InsightType: String {
    case h2hDominance = "h2h_dominance"
    case homeSpecialist = "home_specialist"
    case awayScorer = "away_scorer"
    case btts = "btts"
    case timingGoals = "timing_goals"
    case cleanSheets = "clean_sheets"
    case highScoring = "high_scoring"
    case formStreak = "form_streak"
}

struct MatchInsight {
    let type: InsightType
    let message: String
    let confidence: Double
}

// MARK: - View Model

@MainActor
class MatchDetailViewModel: ObservableObject {
    let prediction: MatchPrediction

    @Published var matchDetails: MatchDetailsResponse?
    @Published var isLoading = true
    @Published var error: String?
    @Published var homeEV: EVResult?
    @Published var drawEV: EVResult?
    @Published var awayEV: EVResult?
    @Published var primaryInsight: MatchInsight?
    @Published var homeTeamForm: TeamFormData?
    @Published var awayTeamForm: TeamFormData?
    @Published var timeline: MatchTimelineResponse?
    @Published var timelineError: String?

    // LLM Narrative (replaces old heuristic insights)
    @Published var llmNarrative: LLMNarrativePayload?
    @Published var llmNarrativeStatus: String?
    @Published var llmNarrativeError: String?

    // Match stats for stats table (independent of narrative)
    @Published var matchStats: MatchStats?
    @Published var matchEvents: [MatchEvent]?

    init(prediction: MatchPrediction) {
        self.prediction = prediction
        calculateEV()
    }

    func loadDetails() async {
        guard let matchId = prediction.matchId else {
            isLoading = false
            error = "No match ID available"
            return
        }

        do {
            matchDetails = try await APIClient.shared.getMatchDetails(matchId: matchId)
            processTeamHistory()
            generateInsights()

            // Load timeline and narrative insights for finished matches (in parallel)
            if prediction.isFinished {
                print("Match \(matchId) is finished, loading timeline and insights in parallel...")
                async let timelineTask: () = loadTimeline(matchId: matchId)
                async let narrativeTask: () = loadNarrativeInsights(matchId: matchId)
                _ = await (timelineTask, narrativeTask)
            } else {
                print("Match \(matchId) status: \(prediction.status ?? "nil") - not loading timeline/insights")
            }
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    private func loadTimeline(matchId: Int) async {
        do {
            timeline = try await APIClient.shared.getMatchTimeline(matchId: matchId)
            print("Timeline loaded successfully: \(timeline?.summary.correctPercentage ?? 0)% correct")
        } catch {
            // Timeline is optional - don't fail the whole view
            timelineError = error.localizedDescription
            print("Timeline error: \(error.localizedDescription)")
        }
    }

    private func loadNarrativeInsights(matchId: Int) async {
        do {
            let response = try await APIClient.shared.getMatchInsights(matchId: matchId)

            // Use LLM narrative if available (new system)
            if let narrative = response.llmNarrative {
                llmNarrative = narrative
                llmNarrativeStatus = response.llmNarrativeStatus
                print("LLM narrative loaded: \(narrative.narrative?.title ?? "no title")")
            } else {
                // No LLM narrative available
                llmNarrativeStatus = response.llmNarrativeStatus ?? "pending"
                print("LLM narrative not available, status: \(llmNarrativeStatus ?? "unknown")")
            }

            // Load match stats for stats table (always, independent of narrative)
            matchStats = response.matchStats
            matchEvents = response.matchEvents
            print("DEBUG: matchStats = \(String(describing: response.matchStats))")
            print("DEBUG: matchEvents count = \(response.matchEvents?.count ?? 0)")
            if let stats = matchStats {
                print("Match stats loaded: possession=\(stats.home?.ballPossession ?? "nil")")
            } else {
                print("Match stats is nil")
            }
        } catch {
            // Insights are optional - don't fail the whole view
            llmNarrativeError = error.localizedDescription
            print("LLM narrative error: \(error.localizedDescription)")
        }
    }

    private func calculateEV() {
        guard let marketOdds = prediction.marketOdds else { return }

        // EV formula: (probability * bookieOdds) - 1
        // Since fairOdds = 1/probability, this is: (bookieOdds / fairOdds) - 1
        // Positive EV = profitable bet (bookie pays more than fair value)

        if let fairHome = prediction.fairOdds.home, let bookieHome = marketOdds.home, bookieHome > 0, fairHome > 0 {
            let ev = (bookieHome / fairHome) - 1
            homeEV = EVResult(
                outcome: "home",
                fairOdds: fairHome,
                bookieOdds: bookieHome,
                ev: ev,
                evPercentage: ev * 100,
                isValue: ev > 0
            )
        }

        if let fairDraw = prediction.fairOdds.draw, let bookieDraw = marketOdds.draw, bookieDraw > 0, fairDraw > 0 {
            let ev = (bookieDraw / fairDraw) - 1
            drawEV = EVResult(
                outcome: "draw",
                fairOdds: fairDraw,
                bookieOdds: bookieDraw,
                ev: ev,
                evPercentage: ev * 100,
                isValue: ev > 0
            )
        }

        if let fairAway = prediction.fairOdds.away, let bookieAway = marketOdds.away, bookieAway > 0, fairAway > 0 {
            let ev = (bookieAway / fairAway) - 1
            awayEV = EVResult(
                outcome: "away",
                fairOdds: fairAway,
                bookieOdds: bookieAway,
                ev: ev,
                evPercentage: ev * 100,
                isValue: ev > 0
            )
        }
    }

    private func processTeamHistory() {
        guard let details = matchDetails else { return }

        let homeHistory = details.homeTeam.history.prefix(5)
        let homeForm = homeHistory.map { $0.result }
        // Use real league points from API, fallback to calculated form points
        let homePoints = details.homeTeam.leaguePoints ?? calculatePoints(from: Array(homeHistory))

        homeTeamForm = TeamFormData(
            position: details.homeTeam.position ?? 0,
            teamName: details.homeTeam.name,
            logoUrl: details.homeTeam.logo,
            form: homeForm,
            points: homePoints
        )

        let awayHistory = details.awayTeam.history.prefix(5)
        let awayForm = awayHistory.map { $0.result }
        // Use real league points from API, fallback to calculated form points
        let awayPoints = details.awayTeam.leaguePoints ?? calculatePoints(from: Array(awayHistory))

        awayTeamForm = TeamFormData(
            position: details.awayTeam.position ?? 0,
            teamName: details.awayTeam.name,
            logoUrl: details.awayTeam.logo,
            form: awayForm,
            points: awayPoints
        )
    }

    private func calculatePoints(from history: [MatchHistoryItem]) -> Int {
        history.reduce(0) { total, match in
            switch match.result {
            case "W": return total + 3
            case "D": return total + 1
            default: return total
            }
        }
    }

    private func generateInsights() {
        guard let details = matchDetails else { return }

        var insights: [MatchInsight] = []
        let homeTeamHistory = details.homeTeam.history
        let awayTeamHistory = details.awayTeam.history

        // Home Specialist
        let homeMatches = homeTeamHistory.filter { $0.isHome }
        if homeMatches.count >= 3 {
            let homeWins = homeMatches.filter { $0.result == "W" }.count
            let homeWinRate = Double(homeWins) / Double(homeMatches.count)
            if homeWinRate >= 0.7 {
                insights.append(MatchInsight(
                    type: .homeSpecialist,
                    message: "\(details.homeTeam.name) wins \(Int(homeWinRate * 100))% of home matches",
                    confidence: homeWinRate
                ))
            }
        }

        // Away Scorer
        let awayMatches = awayTeamHistory.filter { !$0.isHome }.prefix(3)
        if awayMatches.count >= 3 {
            let scoredInAll = awayMatches.allSatisfy { ($0.teamGoals ?? 0) > 0 }
            if scoredInAll {
                insights.append(MatchInsight(
                    type: .awayScorer,
                    message: "\(details.awayTeam.name) scored in last 3 away matches",
                    confidence: 0.85
                ))
            }
        }

        // BTTS
        let last5Home = Array(homeTeamHistory.prefix(5))
        let bttsCount = last5Home.filter { ($0.teamGoals ?? 0) > 0 && ($0.opponentGoals ?? 0) > 0 }.count
        if last5Home.count >= 5 && bttsCount >= 4 {
            let bttsRate = Double(bttsCount) / Double(last5Home.count)
            insights.append(MatchInsight(
                type: .btts,
                message: "Both teams scored in \(Int(bttsRate * 100))% of \(details.homeTeam.name)'s matches",
                confidence: bttsRate
            ))
        }

        // High Scoring
        let totalGoalsHome = last5Home.reduce(0) { $0 + ($1.teamGoals ?? 0) + ($1.opponentGoals ?? 0) }
        let avgGoals = Double(totalGoalsHome) / max(1, Double(last5Home.count))
        if avgGoals >= 3.0 {
            insights.append(MatchInsight(
                type: .highScoring,
                message: "\(details.homeTeam.name) games average \(String(format: "%.1f", avgGoals)) goals",
                confidence: min(1.0, avgGoals / 4.0)
            ))
        }

        // Form Streak
        let homeForm = homeTeamHistory.prefix(3).map { $0.result }
        if homeForm.count == 3 && homeForm.allSatisfy({ $0 == "W" }) {
            insights.append(MatchInsight(
                type: .formStreak,
                message: "\(details.homeTeam.name) on a 3-match winning streak",
                confidence: 0.9
            ))
        }

        primaryInsight = insights.max(by: { $0.confidence < $1.confidence })

        // Fallback
        if primaryInsight == nil {
            let homeScoreRate = homeMatches.filter { ($0.teamGoals ?? 0) > 0 }.count
            let rate = Double(homeScoreRate) / max(1, Double(homeMatches.count))
            if rate > 0 {
                primaryInsight = MatchInsight(
                    type: .homeSpecialist,
                    message: "\(details.homeTeam.name) scores in \(Int(rate * 100))% of home matches",
                    confidence: rate
                )
            }
        }
    }

    var formattedMatchDate: String {
        guard let date = prediction.matchDate else { return "TBD" }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d • h:mma"
        return formatter.string(from: date)
    }
}

// MARK: - Main View

struct MatchDetailView: View {
    let prediction: MatchPrediction
    @StateObject private var viewModel: MatchDetailViewModel
    @State private var isFavoriteHome = false
    @State private var isFavoriteAway = false
    @State private var showStandings = false

    init(prediction: MatchPrediction) {
        self.prediction = prediction
        _viewModel = StateObject(wrappedValue: MatchDetailViewModel(prediction: prediction))
    }

    private let valueColor = Color(red: 0.19, green: 0.82, blue: 0.35)  // #30D158

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                matchHeader
                probabilityBar

                // Timeline for finished matches
                if let timeline = viewModel.timeline {
                    PredictionTimelineView(timeline: timeline)
                }

                oddsCards

                if viewModel.homeTeamForm != nil || viewModel.awayTeamForm != nil {
                    formTable
                }

                // LLM Narrative (post-match analysis)
                if let narrative = viewModel.llmNarrative {
                    LLMNarrativeView(narrative: narrative)
                } else if prediction.isFinished && viewModel.llmNarrativeStatus != nil {
                    // Show unavailable state for finished matches without narrative
                    LLMNarrativeUnavailableView(status: viewModel.llmNarrativeStatus)
                }

                // Match Stats Table (below narrative, for finished matches)
                if let stats = viewModel.matchStats {
                    MatchStatsTableView(
                        stats: stats,
                        homeTeam: prediction.homeTeam,
                        awayTeam: prediction.awayTeam
                    )
                }

                // Only show insight footer if no narrative and match not finished
                if let insight = viewModel.primaryInsight,
                   viewModel.llmNarrative == nil,
                   !prediction.isFinished {
                    insightFooter(insight)
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 32)
        }
        .background(Color.black)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Text("Match Prediction")
                    .font(.headline)
                    .foregroundStyle(.white)
            }
        }
        .toolbarBackground(Color.black, for: .navigationBar)
        .toolbarBackground(.visible, for: .navigationBar)
        .task {
            await viewModel.loadDetails()
        }
        .sheet(isPresented: $showStandings) {
            if let leagueId = viewModel.matchDetails?.match.leagueId {
                LeagueStandingsView(
                    leagueId: leagueId,
                    homeTeamId: viewModel.matchDetails?.homeTeam.id,
                    awayTeamId: viewModel.matchDetails?.awayTeam.id
                )
            }
        }
    }

    // MARK: - Match Header

    private var matchHeader: some View {
        VStack(spacing: 16) {
            HStack(alignment: .top, spacing: 0) {
                teamColumn(
                    name: prediction.homeTeam,
                    logo: viewModel.matchDetails?.homeTeam.logo,
                    position: viewModel.homeTeamForm?.position ?? 1,
                    role: "Home",
                    isFavorite: $isFavoriteHome
                )

                VStack(spacing: 4) {
                    Spacer()
                    // Show score if match is finished, otherwise show VS
                    if prediction.isFinished, let score = prediction.scoreDisplay {
                        Text(score)
                            .font(.system(size: 28, weight: .bold))
                            .foregroundStyle(.white)
                        HStack(spacing: 6) {
                            // Prediction result indicator
                            if let correct = prediction.predictionCorrect {
                                Image(systemName: correct ? "checkmark.circle.fill" : "xmark.circle.fill")
                                    .font(.caption)
                                    .foregroundStyle(correct ? .green : .red)
                            }
                            Text("FT")
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundStyle(.green)
                            // Tier emoji
                            if !prediction.tierEmoji.isEmpty {
                                Text(prediction.tierEmoji)
                                    .font(.caption)
                            }
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.2))
                        .clipShape(Capsule())
                    } else if prediction.isLive {
                        if let score = prediction.scoreDisplay {
                            Text(score)
                                .font(.system(size: 28, weight: .bold))
                                .foregroundStyle(.white)
                        }
                        Text(prediction.status ?? "LIVE")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundStyle(.red)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(Color.red.opacity(0.2))
                            .clipShape(Capsule())
                    } else {
                        // Tier emoji for upcoming matches
                        if !prediction.tierEmoji.isEmpty {
                            Text(prediction.tierEmoji)
                                .font(.title2)
                        }
                        Text("VS")
                            .font(.headline)
                            .fontWeight(.medium)
                            .foregroundStyle(.gray)
                        Text(viewModel.formattedMatchDate)
                            .font(.caption2)
                            .foregroundStyle(.gray.opacity(0.8))
                            .multilineTextAlignment(.center)
                    }
                    Spacer()
                }
                .frame(width: 100)

                teamColumn(
                    name: prediction.awayTeam,
                    logo: viewModel.matchDetails?.awayTeam.logo,
                    position: viewModel.awayTeamForm?.position ?? 4,
                    role: "Away",
                    isFavorite: $isFavoriteAway
                )
            }
            .frame(height: 160)

            predictionBadge
        }
        .padding(.top, 8)
    }

    private func teamColumn(name: String, logo: String?, position: Int, role: String, isFavorite: Binding<Bool>) -> some View {
        VStack(spacing: 6) {
            // Favorite star
            Button {
                isFavorite.wrappedValue.toggle()
            } label: {
                Image(systemName: isFavorite.wrappedValue ? "star.fill" : "star")
                    .font(.title2)
                    .foregroundStyle(isFavorite.wrappedValue ? .yellow : .gray.opacity(0.6))
            }

            // Team logo - larger
            if let logoUrl = logo, let url = URL(string: logoUrl) {
                AsyncImage(url: url) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } placeholder: {
                    teamPlaceholder
                }
                .frame(width: 72, height: 72)
            } else {
                teamPlaceholder
            }

            // Team name - larger font
            Text(name)
                .font(.callout)
                .fontWeight(.bold)
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)

            // Position & Role - centered (hide position if 0/unavailable)
            Text(position > 0 ? "#\(position) • \(role)" : role)
                .font(.caption)
                .foregroundStyle(.gray)
        }
        .frame(maxWidth: .infinity)
    }

    private var teamPlaceholder: some View {
        Image(systemName: "shield.fill")
            .font(.system(size: 56))
            .foregroundStyle(.gray.opacity(0.5))
            .frame(width: 72, height: 72)
    }

    private var predictionBadge: some View {
        Text(prediction.probabilities.predictedOutcome)
            .font(.subheadline)
            .fontWeight(.semibold)
            .foregroundStyle(.blue)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.blue.opacity(0.15))
            .clipShape(Capsule())
    }

    // MARK: - Probability Bar

    private var probabilityBar: some View {
        VStack(spacing: 16) {
            // Segmented probability bar - thicker
            GeometryReader { geo in
                HStack(spacing: 0) {
                    Rectangle()
                        .fill(Color.blue)
                        .frame(width: geo.size.width * prediction.probabilities.home)

                    Rectangle()
                        .fill(Color.gray.opacity(0.6))
                        .frame(width: geo.size.width * prediction.probabilities.draw)

                    Rectangle()
                        .fill(Color.red)
                        .frame(width: geo.size.width * prediction.probabilities.away)
                }
                .frame(height: 16)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            .frame(height: 16)

            // Labels below only
            HStack {
                probabilityLabel("Home", value: prediction.probabilities.homePercent, color: .blue)
                Spacer()
                probabilityLabel("Draw", value: prediction.probabilities.drawPercent, color: .gray)
                Spacer()
                probabilityLabel("Away", value: prediction.probabilities.awayPercent, color: .red)
            }
        }
    }

    private func probabilityLabel(_ label: String, value: String, color: Color) -> some View {
        VStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 10, height: 10)
            Text(label)
                .font(.caption)
                .foregroundStyle(.gray)
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .foregroundStyle(.white)
        }
    }

    // MARK: - Odds Cards

    private var oddsCards: some View {
        HStack(spacing: 12) {
            oddsCard(
                label: "Home",
                fairOdds: prediction.fairOdds.homeFormatted,
                bookieOdds: viewModel.homeEV?.bookieOdds,
                evResult: viewModel.homeEV,
                color: .blue
            )

            oddsCard(
                label: "Draw",
                fairOdds: prediction.fairOdds.drawFormatted,
                bookieOdds: viewModel.drawEV?.bookieOdds,
                evResult: viewModel.drawEV,
                color: .gray
            )

            oddsCard(
                label: "Away",
                fairOdds: prediction.fairOdds.awayFormatted,
                bookieOdds: viewModel.awayEV?.bookieOdds,
                evResult: viewModel.awayEV,
                color: .red
            )
        }
    }

    private func oddsCard(label: String, fairOdds: String, bookieOdds: Double?, evResult: EVResult?, color: Color) -> some View {
        let isValue = evResult?.isValue ?? false
        // Check if ANY card has value to reserve space for EV text
        let anyHasValue = (viewModel.homeEV?.isValue ?? false) || (viewModel.drawEV?.isValue ?? false) || (viewModel.awayEV?.isValue ?? false)

        return VStack(spacing: 10) {
            // Label
            Text(label)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundStyle(color)

            // Fair Odds section
            VStack(spacing: 2) {
                Text("Fair Odds")
                    .font(.caption2)
                    .foregroundStyle(.gray)
                Text(fairOdds)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundStyle(color)
            }

            // Bookie section
            VStack(spacing: 2) {
                Text("Bookie")
                    .font(.caption2)
                    .foregroundStyle(.gray)
                Text(bookieOdds.map { String(format: "%.2f", $0) } ?? "-")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundStyle(isValue ? valueColor : .white)

                // EV display - reserve space for all cards if any has value
                if anyHasValue {
                    if isValue, let ev = evResult {
                        Text(ev.evDisplay)
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundStyle(valueColor)
                    } else {
                        Text(" ")
                            .font(.caption)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(isValue ? valueColor.opacity(0.5) : Color.clear, lineWidth: 1.5)
        )
    }

    // MARK: - Form Table

    private var formTable: some View {
        Button {
            showStandings = true
        } label: {
            VStack(spacing: 0) {
                if let homeForm = viewModel.homeTeamForm {
                    formRow(data: homeForm)
                }

                Divider()
                    .background(Color.gray.opacity(0.3))

                if let awayForm = viewModel.awayTeamForm {
                    formRow(data: awayForm)
                }

                // Hint to tap for full standings
                Image(systemName: "chevron.down")
                    .font(.caption)
                    .foregroundStyle(.gray)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            }
            .background(Color(white: 0.1))
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
    }

    private func formRow(data: TeamFormData) -> some View {
        HStack(spacing: 0) {
            // Position
            Text(data.position > 0 ? "#\(data.position)" : "-")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.white.opacity(0.8))
                .frame(width: 36, alignment: .leading)

            // Team logo
            if let logoUrl = data.logoUrl, let url = URL(string: logoUrl) {
                AsyncImage(url: url) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } placeholder: {
                    Image(systemName: "shield.fill")
                        .foregroundStyle(.gray)
                }
                .frame(width: 32, height: 32)
            } else {
                Image(systemName: "shield.fill")
                    .font(.title3)
                    .foregroundStyle(.gray)
                    .frame(width: 32, height: 32)
            }

            Spacer()

            // Form pills - always show 5, pad with empty if needed
            HStack(spacing: 6) {
                let formResults = Array(data.form.prefix(5))
                ForEach(0..<5, id: \.self) { index in
                    if index < formResults.count {
                        formPill(result: formResults[index])
                    } else {
                        formPill(result: "-")
                    }
                }
            }

            Spacer()

            // Points
            HStack(spacing: 4) {
                Text("\(data.points)")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
                Text("Pts")
                    .font(.caption)
                    .foregroundStyle(.gray)
            }
            .frame(width: 56, alignment: .trailing)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    private func formPill(result: String) -> some View {
        let color: Color = {
            switch result {
            case "W": return .green
            case "L": return .red
            case "D": return .gray
            default: return Color(white: 0.2)
            }
        }()

        return Text(result == "-" ? "" : result)
            .font(.caption)
            .fontWeight(.bold)
            .foregroundStyle(.white)
            .frame(width: 28, height: 28)
            .background(color)
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Insight Footer

    private func insightFooter(_ insight: MatchInsight) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "lightbulb.fill")
                .font(.body)
                .foregroundStyle(valueColor)

            Text(insight.message)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundStyle(valueColor)
        }
        .padding(.top, 12)
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        MatchDetailView(prediction: MatchPrediction(
            matchId: 1,
            matchExternalId: 12345,
            homeTeam: "Real Madrid",
            awayTeam: "FC Barcelona",
            homeTeamLogo: nil,
            awayTeamLogo: nil,
            date: "2026-01-03T07:00:00",
            status: "FT",
            homeGoals: 2,
            awayGoals: 1,
            probabilities: Probabilities(home: 0.45, draw: 0.30, away: 0.25),
            fairOdds: FairOdds(home: 2.47, draw: 3.39, away: 3.33),
            marketOdds: MarketOdds(home: 1.45, draw: 4.50, away: 7.00),
            valueBets: nil,
            hasValueBet: true,
            bestValueBet: nil,
            confidenceTier: "gold"
        ))
    }
    .preferredColorScheme(.dark)
}
