import SwiftUI
import UIKit

// MARK: - League Group Model

/// League group for display - groups matches by league
struct LeagueGroup: Identifiable {
    let id: String  // Use leagueName as stable ID
    let leagueId: Int?
    let leagueName: String
    let leagueLogo: String?
    let countryFlag: String?
    let predictions: [MatchPrediction]
}

// MARK: - SF Pro Condensed Font Helper

extension Font {
    /// SF Pro Condensed - uses variable font width axis
    /// Width values: 30 (ultra compressed) to 100 (normal) to 150 (expanded)
    static func sfProCondensed(size: CGFloat, weight: UIFont.Weight = .light, width: CGFloat = 75) -> Font {
        // Use system font with width trait
        let systemFont = UIFont.systemFont(ofSize: size, weight: weight)
        let traits: [UIFontDescriptor.TraitKey: Any] = [
            .width: (width - 100) / 100  // Convert 30-150 scale to -0.7 to 0.5
        ]
        let descriptor = systemFont.fontDescriptor.addingAttributes([
            .traits: traits
        ])
        let uiFont = UIFont(descriptor: descriptor, size: size)
        return Font(uiFont)
    }
}


struct PredictionsListView: View {
    @StateObject private var viewModel = PredictionsViewModel()

    // Pagination: limit initial render to reduce main thread work
    private let initialDisplayLimit = 15
    @State private var displayLimit = 15

    // Performance tracking
    @State private var viewAppearTime: Date?
    @State private var firstContentTime: Date?
    @State private var hasLoggedFirstContent = false
    @State private var dateChangeTapTime: Date?

    var body: some View {
        NavigationStack {
            contentView
                .safeAreaInset(edge: .top) {
                    dateSelector
                        .background(
                            LinearGradient(
                                stops: [
                                    .init(color: Color(red: 0.02, green: 0.02, blue: 0.06), location: 0),
                                    .init(color: Color(red: 0.02, green: 0.02, blue: 0.06), location: 0.75),
                                    .init(color: Color(red: 0.02, green: 0.02, blue: 0.06).opacity(0), location: 1.0)
                                ],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                }
                .background(
                    LinearGradient(
                        stops: [
                            .init(color: Color(red: 0.02, green: 0.02, blue: 0.06), location: 0),
                            .init(color: Color(red: 0.02, green: 0.02, blue: 0.06), location: 0.7),
                            .init(color: Color(red: 0.034, green: 0.034, blue: 0.10), location: 1.0)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .navigationBarHidden(true)
                .refreshable {
                    await viewModel.refresh()
                }
                .task {
                    await viewModel.refresh()
                }
                .onChange(of: viewModel.selectedDate) { oldDate, newDate in
                    // Reset pagination when date changes for fast initial render
                    displayLimit = initialDisplayLimit

                    // Measure time from tap to state update
                    if let tapTime = dateChangeTapTime {
                        let stateUpdateMs = Date().timeIntervalSince(tapTime) * 1000
                        let matchCount = viewModel.predictionsForSelectedDate.count
                        print("[Perf] DATE_CHANGE_COMPLETE: \(String(format: "%.0f", stateUpdateMs))ms, \(matchCount) matches for new date")
                        PerfLogger.shared.log(
                            endpoint: "dateChange",
                            message: "complete",
                            data: ["tap_to_update_ms": stateUpdateMs, "match_count": matchCount]
                        )
                        dateChangeTapTime = nil
                    }
                }
                .onAppear {
                    viewAppearTime = Date()
                    print("[Perf] PredictionsListView.onAppear")
                }
                .onDisappear {
                    // Cancel background refresh when navigating away to free up connections
                    viewModel.cancelBackgroundTasks()
                }
                .onChange(of: viewModel.predictions.count) { oldCount, newCount in
                    // Log when predictions first arrive (transition from 0 to N)
                    if oldCount == 0 && newCount > 0 && !hasLoggedFirstContent {
                        hasLoggedFirstContent = true
                        firstContentTime = Date()
                        if let appear = viewAppearTime, let content = firstContentTime {
                            let ttfc = content.timeIntervalSince(appear) * 1000
                            print("[Perf] TIME_TO_FIRST_CONTENT: \(String(format: "%.0f", ttfc))ms (\(newCount) predictions)")
                            PerfLogger.shared.log(
                                endpoint: "PredictionsListView",
                                message: "first_content",
                                data: ["ttfc_ms": ttfc, "predictions_count": newCount]
                            )
                        }
                    }
                }
        }
    }

    // MARK: - Content View

    @ViewBuilder
    private var contentView: some View {
        if viewModel.isLoading && viewModel.predictions.isEmpty {
            VStack {
                Spacer()
                LoadingView()
                Spacer()
            }
        } else if let error = viewModel.error, viewModel.predictions.isEmpty {
            VStack {
                Spacer()
                ErrorView(message: error) {
                    Task { await viewModel.refresh() }
                }
                Spacer()
            }
        } else if viewModel.predictionsForSelectedDate.isEmpty {
            VStack {
                Spacer()
                noMatchesForDate
                Spacer()
            }
        } else {
            predictionsList
        }
    }

    // MARK: - Date Selector

    // Local calendar for all date operations (user's timezone)
    private var localCalendar: Calendar {
        Calendar.current
    }

    private var dateSelector: some View {
        let _ = print("[Render] dateSelector body evaluated")
        return ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(dateRange, id: \.self) { date in
                        DateCell(
                            date: date,
                            isSelected: localCalendar.isDate(date, inSameDayAs: viewModel.selectedDate),
                            matchCount: viewModel.matchCount(for: date)
                        )
                        .id(date)
                        .onTapGesture {
                            dateChangeTapTime = Date()
                            print("[Perf] DATE_TAP: \(date)")
                            withAnimation(.easeInOut(duration: 0.2)) {
                                viewModel.selectedDate = date
                            }
                        }
                    }
                }
                .padding(.horizontal, 16)
            }
            .onAppear {
                // Scroll to today (local timezone)
                proxy.scrollTo(localCalendar.startOfDay(for: Date()), anchor: .center)
            }
        }
        .padding(.bottom, 2)
        .frame(height: 110) // Fixed height for safeAreaInset
    }

    // 7 days before + today + 7 days ahead = 15 days total
    // Uses LOCAL calendar so "Today" matches user's timezone
    private var dateRange: [Date] {
        let today = localCalendar.startOfDay(for: Date())
        return (-7..<8).compactMap { localCalendar.date(byAdding: .day, value: $0, to: today) }
    }

    // MARK: - No Matches View

    private var noMatchesForDate: some View {
        VStack(spacing: 12) {
            Image(systemName: "calendar.badge.exclamationmark")
                .font(.system(size: 48))
                .foregroundStyle(.gray)

            Text("No matches")
                .font(.headline)
                .foregroundStyle(.white)

            Text(formattedSelectedDate)
                .font(.subheadline)
                .foregroundStyle(.gray)
        }
    }

    private var formattedSelectedDate: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEEE, MMM d"
        // Local timezone for user display
        return formatter.string(from: viewModel.selectedDate)
    }

    // MARK: - Predictions List

    /// Group predictions by league
    private func groupedByLeague(_ predictions: [MatchPrediction]) -> [LeagueGroup] {
        let grouped = Dictionary(grouping: predictions) { $0.leagueId }
        let result = grouped.map { key, value in
            LeagueGroup(
                id: value.first?.leagueName ?? "Other_\(key ?? -1)",
                leagueId: key,
                leagueName: value.first?.leagueName ?? "Other",
                leagueLogo: value.first?.leagueLogo,
                countryFlag: value.first?.leagueCountryFlag,
                predictions: value
            )
        }
        .sorted { ($0.leagueId ?? Int.max) < ($1.leagueId ?? Int.max) }

        print("[Group] groupedByLeague: \(predictions.count) matches -> \(result.count) groups: \(result.map { "\($0.leagueName)(\($0.predictions.count))" }.joined(separator: ", "))")
        return result
    }

    private var predictionsList: some View {
        let allPredictions = viewModel.predictionsForSelectedDate
        let totalCount = allPredictions.count
        let leagueIds = Set(allPredictions.compactMap { $0.leagueId })
        let _ = print("[Render] predictionsList body - \(totalCount) matches, \(leagueIds.count) leagues: \(leagueIds.sorted())")

        // Limit matches to displayLimit
        let limitedMatches = Array(allPredictions.prefix(displayLimit))
        let hasMore = allPredictions.count > displayLimit

        // Group all matches by league (including value bets)
        let groupedMatches = groupedByLeague(limitedMatches)

        // Observe clockTick to trigger re-render for live match minutes
        let _ = viewModel.clockTick

        return ScrollView(showsIndicators: false) {
            LazyVStack(spacing: 16) {
                // All matches grouped by league - each league in one card
                ForEach(groupedMatches) { group in
                    LeagueCard(
                        group: group,
                        viewModel: viewModel
                    )
                }

                // Load More button
                if hasMore {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            displayLimit += 15
                        }
                    } label: {
                        HStack {
                            Text("Load more (\(totalCount - min(displayLimit, totalCount)) remaining)")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            Image(systemName: "chevron.down")
                                .font(.caption)
                        }
                        .foregroundStyle(.blue)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color(white: 0.12))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    .padding(.top, 8)
                }

                // Progressive loading indicator (loading more dates in background)
                if viewModel.isLoadingMore {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("Loading more dates...")
                            .font(.caption)
                            .foregroundStyle(.gray)
                    }
                    .padding(.vertical, 8)
                }

            }
            .padding(.horizontal, 8)
            .padding(.top, -10)
            .padding(.bottom, 20)
        }
    }
}

// MARK: - Date Cell

struct DateCell: View {
    let date: Date
    let isSelected: Bool
    let matchCount: Int

    // Use local calendar for user-facing display (Today/Tomorrow in user's timezone)
    private var isToday: Bool {
        Calendar.current.isDateInToday(date)
    }

    private var isYesterday: Bool {
        Calendar.current.isDateInYesterday(date)
    }

    private var isPast: Bool {
        date < Calendar.current.startOfDay(for: Date())
    }

    private var dayName: String {
        if isToday { return "Today" }
        if isYesterday { return "Yest." }
        let formatter = DateFormatter()
        formatter.dateFormat = "EEE"
        // Local timezone for display
        return formatter.string(from: date)
    }

    private var dayNumber: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "d"
        // Local timezone for display
        return formatter.string(from: date)
    }

    private var textColor: Color {
        if isSelected { return .white }
        if isPast { return .gray.opacity(0.6) }
        return .gray
    }

    private var backgroundColor: Color {
        if isSelected { return Color(red: 0.034, green: 0.034, blue: 0.10) }
        if isPast { return Color(white: 0.08) }
        return Color(white: 0.12)
    }

    var body: some View {
        VStack(spacing: 4) {
            Text(dayName)
                .font(.caption2)
                .fontWeight(isToday ? .bold : .regular)
                .foregroundStyle(textColor)

            Text(dayNumber)
                .font(.title3)
                .fontWeight(.bold)
                .foregroundStyle(textColor)

            // Match count indicator
            if matchCount > 0 {
                Text("\(matchCount)")
                    .font(.caption2)
                    .fontWeight(.medium)
                    .foregroundStyle(isSelected ? .black : (isPast ? .gray.opacity(0.6) : .gray))
                    .frame(width: 20, height: 16)
                    .background(isSelected ? Color.white : Color.gray.opacity(isPast ? 0.2 : 0.3))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            } else {
                Text("-")
                    .font(.caption2)
                    .foregroundStyle(.gray.opacity(0.5))
                    .frame(width: 20, height: 16)
            }
        }
        .frame(width: 52, height: 72)
        .modifier(DateCellBackgroundModifier(isSelected: isSelected, backgroundColor: backgroundColor))
    }
}

struct DateCellBackgroundModifier: ViewModifier {
    let isSelected: Bool
    let backgroundColor: Color

    func body(content: Content) -> some View {
        if #available(iOS 26.0, *) {
            if isSelected {
                content
                    .background(backgroundColor, in: RoundedRectangle(cornerRadius: 12))
                    .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 12))
            } else {
                content
                    .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 12))
            }
        } else {
            content
                .background(backgroundColor, in: RoundedRectangle(cornerRadius: 12))
        }
    }
}

// MARK: - Match Card

struct MatchCard: View {
    let prediction: MatchPrediction
    let showValueBadge: Bool
    /// Calculated elapsed display from ViewModel (for live clock)
    var calculatedElapsed: String?

    // Overlay values from cache (take precedence over prediction values)
    var overlayedHomeGoals: Int?
    var overlayedAwayGoals: Int?
    var overlayedHasScore: Bool?
    var overlayedIsLive: Bool?
    var overlayedIsFinished: Bool?

    /// Effective home goals (overlay or prediction)
    private var homeGoals: Int? { overlayedHomeGoals ?? prediction.homeGoals }
    /// Effective away goals (overlay or prediction)
    private var awayGoals: Int? { overlayedAwayGoals ?? prediction.awayGoals }
    /// Effective hasScore (overlay or prediction)
    private var hasScore: Bool { overlayedHasScore ?? prediction.hasScore }
    /// Effective isLive (overlay or prediction)
    private var isLive: Bool { overlayedIsLive ?? prediction.isLive }
    /// Effective isFinished (overlay or prediction)
    private var isFinished: Bool { overlayedIsFinished ?? prediction.isFinished }

    var body: some View {
        VStack(spacing: 5) {
            // Teams side by side: Logo | Score ... Center ... Score | Logo
            HStack(spacing: 0) {
                // Home team (left side) - logo near edge, score near center
                HStack(spacing: 28) {
                    teamLogo(url: prediction.homeTeamLogo)

                    if hasScore {
                        Text("\(homeGoals ?? 0)")
                            .font(.custom("BarlowCondensed-SemiBold", size: 42))
                            .foregroundStyle(.white)
                    }
                }

                Spacer(minLength: 12)

                // Center: Final/Live status or Time
                centerStatusView

                Spacer(minLength: 12)

                // Away team (right side) - score near center, logo near edge
                HStack(spacing: 28) {
                    if hasScore {
                        Text("\(awayGoals ?? 0)")
                            .font(.custom("BarlowCondensed-SemiBold", size: 42))
                            .foregroundStyle(.white)
                    }

                    teamLogo(url: prediction.awayTeamLogo)
                }
            }

            // Team names below
            HStack {
                Text(prediction.homeTeam)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.white.opacity(0.5))
                    .lineLimit(1)

                Spacer()

                Text(prediction.awayTeam)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.white.opacity(0.5))
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 10)
    }

    /// Team logo only
    private func teamLogo(url: String?) -> some View {
        Group {
            if let logoUrl = url, let url = URL(string: logoUrl) {
                CachedAsyncImage(url: url) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } placeholder: {
                    Circle()
                        .fill(Color(white: 0.2))
                }
                .frame(width: 40, height: 40)
            } else {
                Circle()
                    .fill(Color(white: 0.2))
                    .frame(width: 40, height: 40)
            }
        }
    }

    /// Center status view (Final, Live badge, or Time for upcoming)
    private var centerStatusView: some View {
        VStack(spacing: 4) {
            // Value bet badge (if applicable)
            if showValueBadge {
                HStack(spacing: 3) {
                    Image(systemName: "star.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(neonGreen)

                    Text("Value Bet")
                        .font(.system(size: 9))
                        .fontWeight(.semibold)
                        .foregroundStyle(neonGreen)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(neonGreen.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            }

            // Status (uses effective isFinished/isLive from overlay)
            if isFinished {
                Text("Final")
                    .font(.system(size: 12))
                    .foregroundStyle(.white)
            } else if isLive {
                Text(liveStatusDisplay)
                    .font(.custom("BarlowCondensed-SemiBold", size: 18))
                    .foregroundStyle(.white)
            } else if let date = prediction.matchDate {
                Text(formatTime(date))
                    .font(.custom("BarlowCondensed-SemiBold", size: 24))
                    .foregroundStyle(.white)

            }
        }
    }

    /// Live status display with elapsed minute when appropriate
    /// Uses calculatedElapsed from ViewModel if available (local clock),
    /// otherwise falls back to static elapsed from API
    /// Format: "32'", "45+2'", "90+3'", "Half Time", etc.
    private var liveStatusDisplay: String {
        // Use calculated elapsed if provided (includes local clock adjustment)
        if let calculated = calculatedElapsed {
            // Transform HT to Half Time
            if calculated == "HT" {
                return "Half Time"
            }
            return calculated
        }

        // Fallback to static elapsed from API
        let status = prediction.status ?? "LIVE"
        let activeStatuses = ["1H", "2H", "LIVE"]

        // Only calculate for active play statuses
        guard activeStatuses.contains(status) else {
            // Transform HT to Half Time
            if status == "HT" {
                return "Half Time"
            }
            return status
        }

        guard let baseElapsed = prediction.elapsed else {
            return status
        }

        // If we have injury/added time from API, show it (e.g., "90+3'")
        if let extra = prediction.elapsedExtra, extra > 0 {
            return "\(baseElapsed)+\(extra)'"
        }

        // At regulation time limits, show capped value
        if status == "1H" && baseElapsed >= 45 {
            return "45'"
        } else if status == "2H" && baseElapsed >= 90 {
            return "90'"
        }

        return "\(baseElapsed)'"
    }

    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter.string(from: date)
    }

    private let neonGreen = Color(red: 0.2, green: 1.0, blue: 0.4)
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
        let ev = bet.evPercentage ?? (bet.expectedValue.map { $0 * 100 }) ?? (bet.edge.map { $0 * 100 }) ?? 0
        if ev >= 15 { return .yellow }
        if ev >= 10 { return .green }
        return .mint
    }
}

// MARK: - Value Bet Banner

struct ValueBetBanner: View {
    let prediction: MatchPrediction
    private let neonGreen = Color(red: 0.2, green: 1.0, blue: 0.4)

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "dollarsign.circle.fill")
                .foregroundStyle(neonGreen)

            Text("VALUE BET")
                .font(.caption2)
                .fontWeight(.heavy)
                .foregroundStyle(neonGreen)

            if let best = prediction.bestValueBet {
                Text(best.evDisplay)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundStyle(.black)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(neonGreen)
                    .clipShape(Capsule())
            }

            Spacer()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(neonGreen.opacity(0.1))
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

// MARK: - League Card (groups all matches of a league in one card)

struct LeagueCard: View {
    let group: LeagueGroup
    let viewModel: PredictionsViewModel

    var body: some View {
        VStack(spacing: 0) {
            // League header inside the card
            leagueHeader
                .padding(.horizontal, 14)
                .padding(.top, 12)
                .padding(.bottom, 8)

            // Matches with separators
            ForEach(Array(group.predictions.enumerated()), id: \.element.id) { index, prediction in
                if index > 0 {
                    Divider()
                        .background(Color.white.opacity(0.1))
                        .padding(.horizontal, 14)
                }

                let isValueBet = (prediction.valueBets?.count ?? 0) > 0
                let isLive = viewModel.isLive(for: prediction)
                let elapsed = isLive ? viewModel.calculatedElapsedDisplay(for: prediction) : nil
                let score = viewModel.overlayedScore(for: prediction)
                let hasScore = viewModel.hasScore(for: prediction)
                let isFinished = viewModel.isFinished(for: prediction)

                NavigationLink(destination: MatchDetailView(prediction: prediction)) {
                    MatchCard(
                        prediction: prediction,
                        showValueBadge: isValueBet,
                        calculatedElapsed: elapsed,
                        overlayedHomeGoals: score.home,
                        overlayedAwayGoals: score.away,
                        overlayedHasScore: hasScore,
                        overlayedIsLive: isLive,
                        overlayedIsFinished: isFinished
                    )
                    .padding(.horizontal, 14)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.bottom, 8)
        .modifier(GlassCardModifier())
    }

    private var leagueHeader: some View {
        HStack(spacing: 10) {
            if let logoUrl = group.leagueLogo, let url = URL(string: logoUrl) {
                CachedAsyncImage(url: url) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } placeholder: {
                    Image(systemName: "trophy.fill")
                        .foregroundStyle(.gray)
                }
                .frame(width: 24, height: 24)
            } else {
                Image(systemName: "trophy.fill")
                    .font(.system(size: 18))
                    .foregroundStyle(.gray)
                    .frame(width: 24, height: 24)
            }

            Text(group.leagueName)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.white)

            Spacer()

            // Country name badge
            if let country = group.countryFlag {
                Text(country)
                    .font(.caption)
                    .foregroundStyle(.gray)
            }
        }
    }
}

// MARK: - Glass Card Modifier (iOS 26+)

struct GlassCardModifier: ViewModifier {
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

// MARK: - Glass Circle Modifier (iOS 26+)

struct GlassCircleModifier: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 26.0, *) {
            content
                .glassEffect(.regular, in: Circle())
        } else {
            content
                .background(AppColors.surfaceHighlight, in: Circle())
        }
    }
}

#Preview {
    PredictionsListView()
}
