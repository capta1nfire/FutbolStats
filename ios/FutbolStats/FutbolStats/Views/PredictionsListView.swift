import SwiftUI

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
            ZStack {
                Color.black.ignoresSafeArea()

                VStack(spacing: 0) {
                    // Date selector
                    dateSelector
                        .padding(.top, 8)

                    // Alpha readiness (non-blocking, only show if data available)
                    if viewModel.opsProgress != nil {
                        AlphaStatusBadge(progress: viewModel.opsProgress)
                            .padding(.horizontal, 16)
                            .padding(.bottom, 10)
                    }

                    // Content
                    if viewModel.isLoading && viewModel.predictions.isEmpty {
                        Spacer()
                        LoadingView()
                        Spacer()
                    } else if let error = viewModel.error, viewModel.predictions.isEmpty {
                        Spacer()
                        ErrorView(message: error) {
                            Task { await viewModel.refresh() }
                        }
                        Spacer()
                    } else if viewModel.predictionsForSelectedDate.isEmpty {
                        Spacer()
                        noMatchesForDate
                        Spacer()
                    } else {
                        predictionsList
                    }
                }
            }
            .navigationTitle("Predictions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Label("Leagues filter coming soon", systemImage: "slider.horizontal.3")
                    } label: {
                        Image(systemName: "line.3.horizontal.decrease.circle")
                            .foregroundStyle(.gray)
                    }
                }

                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(viewModel.modelLoaded ? .green : .red)
                            .frame(width: 6, height: 6)
                        Text(viewModel.modelLoaded ? "Ready" : "Offline")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .toolbarBackground(Color.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
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

    // MARK: - Date Selector

    // Local calendar for UI display (day names, formatting)
    private var localCalendar: Calendar {
        Calendar.current
    }

    // UTC calendar for date operations (must match backend's UTC filtering)
    private static var utcCalendar: Calendar = {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        return cal
    }()

    private var dateSelector: some View {
        let _ = print("[Render] dateSelector body evaluated")
        return ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(dateRange, id: \.self) { date in
                        DateCell(
                            date: date,
                            isSelected: Self.utcCalendar.isDate(date, inSameDayAs: viewModel.selectedDate),
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
                // Scroll to today (UTC)
                proxy.scrollTo(Self.utcCalendar.startOfDay(for: Date()), anchor: .center)
            }
        }
        .padding(.bottom, 12)
    }

    // 7 days before + today + 7 days ahead = 15 days total (matches competitor)
    // Uses UTC calendar to match backend's date filtering
    private var dateRange: [Date] {
        let today = Self.utcCalendar.startOfDay(for: Date())
        return (-7..<8).compactMap { Self.utcCalendar.date(byAdding: .day, value: $0, to: today) }
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
        formatter.timeZone = TimeZone(identifier: "UTC")  // Match data grouping
        return formatter.string(from: viewModel.selectedDate)
    }

    // MARK: - Predictions List

    private var predictionsList: some View {
        let allPredictions = viewModel.predictionsForSelectedDate
        let valueBets = viewModel.valueBetPredictionsForSelectedDate
        let regularMatches = viewModel.regularPredictionsForSelectedDate
        let totalCount = allPredictions.count
        let _ = print("[Render] predictionsList body - \(totalCount) matches, showing \(min(displayLimit, totalCount))")

        // Limit regular matches to displayLimit (value bets always shown)
        let limitedRegular = Array(regularMatches.prefix(max(0, displayLimit - valueBets.count)))
        let hasMore = (valueBets.count + regularMatches.count) > displayLimit

        return ScrollView {
            LazyVStack(spacing: 12) {
                // Value Bets Section (always show all - typically few)
                if !valueBets.isEmpty {
                    sectionHeader(title: "Value Bets", icon: "star.fill", color: Color(red: 0.2, green: 1.0, blue: 0.4))

                    ForEach(valueBets) { prediction in
                        NavigationLink(destination: MatchDetailView(prediction: prediction)) {
                            MatchCard(prediction: prediction, showValueBadge: true)
                        }
                        .buttonStyle(.plain)
                    }
                }

                // All Matches Section (paginated)
                if !limitedRegular.isEmpty {
                    sectionHeader(title: "Matches", icon: "sportscourt", color: .gray)
                        .padding(.top, valueBets.isEmpty ? 0 : 8)

                    ForEach(limitedRegular) { prediction in
                        NavigationLink(destination: MatchDetailView(prediction: prediction)) {
                            MatchCard(prediction: prediction, showValueBadge: false)
                        }
                        .buttonStyle(.plain)
                    }
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
            .padding(.horizontal, 16)
            .padding(.bottom, 20)
        }
    }

    private func sectionHeader(title: String, icon: String, color: Color) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(color)

            Text(title)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.white)

            Spacer()
        }
        .padding(.top, 8)
    }
}

// MARK: - Date Cell

struct DateCell: View {
    let date: Date
    let isSelected: Bool
    let matchCount: Int

    // UTC calendar to match backend's date grouping
    private static var utcCalendar: Calendar = {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        return cal
    }()

    // "Today" in UTC (matches how data is grouped)
    private var isToday: Bool {
        Self.utcCalendar.isDateInToday(date)
    }

    // "Yesterday" in UTC
    private var isYesterday: Bool {
        Self.utcCalendar.isDateInYesterday(date)
    }

    // Past dates use UTC for consistency
    private var isPast: Bool {
        date < Self.utcCalendar.startOfDay(for: Date())
    }

    private var dayName: String {
        if isToday { return "Today" }
        if isYesterday { return "Yest." }
        // Use UTC timezone for day name display to match data grouping
        let formatter = DateFormatter()
        formatter.dateFormat = "EEE"
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: date)
    }

    private var dayNumber: String {
        // Use UTC timezone for day number display to match data grouping
        let formatter = DateFormatter()
        formatter.dateFormat = "d"
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: date)
    }

    private var textColor: Color {
        if isSelected { return .white }
        if isPast { return .gray.opacity(0.6) }
        return .gray
    }

    private var backgroundColor: Color {
        if isSelected { return .blue }
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
        .background(backgroundColor)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Match Card

struct MatchCard: View {
    let prediction: MatchPrediction
    let showValueBadge: Bool

    var body: some View {
        VStack(spacing: 12) {
            // Value bet banner
            if showValueBadge, let best = prediction.bestValueBet {
                let neonGreen = Color(red: 0.2, green: 1.0, blue: 0.4)
                HStack {
                    Image(systemName: "star.fill")
                        .font(.caption2)
                        .foregroundStyle(neonGreen)

                    Text("VALUE BET")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundStyle(neonGreen)

                    Spacer()

                    Text("\(best.outcome.uppercased()) \(best.evDisplay)")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundStyle(neonGreen)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(neonGreen.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            // Teams and odds
            HStack {
                // Teams with logos and optional scores
                VStack(alignment: .leading, spacing: 6) {
                    teamRowWithScore(
                        name: prediction.homeTeam,
                        logoUrl: prediction.homeTeamLogo,
                        goals: prediction.isFinished ? prediction.homeGoals : nil
                    )
                    teamRowWithScore(
                        name: prediction.awayTeam,
                        logoUrl: prediction.awayTeamLogo,
                        goals: prediction.isFinished ? prediction.awayGoals : nil
                    )
                }

                Spacer()

                // Match status/time badge
                matchStatusBadge
            }

            // Probabilities bar
            HStack(spacing: 4) {
                probabilityPill(label: "H", value: prediction.probabilities.home, odds: prediction.marketOdds?.home, isValueBet: isValueBet(for: "home"))
                probabilityPill(label: "D", value: prediction.probabilities.draw, odds: prediction.marketOdds?.draw, isValueBet: isValueBet(for: "draw"))
                probabilityPill(label: "A", value: prediction.probabilities.away, odds: prediction.marketOdds?.away, isValueBet: isValueBet(for: "away"))
            }
        }
        .padding(14)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func teamRowWithScore(name: String, logoUrl: String?, goals: Int?) -> some View {
        HStack(spacing: 8) {
            if let logoUrl = logoUrl, let url = URL(string: logoUrl) {
                CachedAsyncImage(url: url) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } placeholder: {
                    Circle()
                        .fill(Color(white: 0.2))
                }
                .frame(width: 22, height: 22)
            } else {
                Circle()
                    .fill(Color(white: 0.2))
                    .frame(width: 22, height: 22)
            }

            Text(name)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.white)
                .lineLimit(1)

            Spacer()

            // Show goals if match is finished
            if let goals = goals {
                Text("\(goals)")
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
                    .frame(width: 24)
            }
        }
    }

    /// Match status badge - shows FT with result indicator, LIVE, or time with tier
    private var matchStatusBadge: some View {
        VStack(alignment: .trailing, spacing: 4) {
            // Status badge
            if prediction.isFinished {
                HStack(spacing: 4) {
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
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.green.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            } else if prediction.isLive {
                Text(prediction.status ?? "LIVE")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.red)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.red.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else if let date = prediction.matchDate {
                Text(formatTime(date))
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.gray)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(white: 0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            // Confidence tier badge
            if prediction.confidenceTier != nil {
                Text(prediction.tierEmoji)
                    .font(.caption)
            }
        }
    }

    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: date)
    }

    private func isValueBet(for outcome: String) -> Bool {
        prediction.bestValueBet?.outcome.lowercased() == outcome.lowercased()
    }

    private let neonGreen = Color(red: 0.2, green: 1.0, blue: 0.4)

    private func probabilityPill(label: String, value: Double, odds: Double?, isValueBet: Bool) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundStyle(isValueBet ? neonGreen : .gray)

            Text(String(format: "%.0f%%", value * 100))
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(isValueBet ? neonGreen : .white)

            if let odds = odds {
                Text(String(format: "%.2f", odds))
                    .font(.caption2)
                    .foregroundStyle(isValueBet ? neonGreen.opacity(0.8) : .gray)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(isValueBet ? neonGreen.opacity(0.15) : Color(white: 0.15))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isValueBet ? neonGreen.opacity(0.5) : Color.clear, lineWidth: 1)
        )
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

#Preview {
    PredictionsListView()
}
