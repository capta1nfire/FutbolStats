import SwiftUI

// MARK: - Inferred Goal (from live polling score changes)

/// Represents a goal inferred from detecting score changes during live polling.
/// Since we don't have exact event data, we record the minute when we detected the change.
struct InferredGoal: Identifiable {
    let id = UUID()
    let minute: Int           // Minute when we detected the goal
    let team: String          // "home" or "away"
    let homeTeamName: String
    let awayTeamName: String

    var teamName: String {
        team == "home" ? homeTeamName : awayTeamName
    }
}

// MARK: - Pulsing Live Minute (MatchDetail only)

/// Subtle pulsing animation for live match minute display
struct PulsingLiveMinute: View {
    let text: String
    @State private var isPulsing = false

    var body: some View {
        Text(text)
            .font(.custom("BarlowCondensed-SemiBold", size: 22))
            .foregroundStyle(.gray)
            .opacity(isPulsing ? 0.5 : 1.0)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                    isPulsing = true
                }
            }
    }
}

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
    let teamId: Int?
    let position: Int
    let teamName: String
    let logoUrl: String?
    let form: [String]
    let points: Int
}

// MARK: - Section Loading State

enum SectionLoadState: Equatable {
    case idle
    case loading
    case loaded
    case error(String)

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }

    var isLoaded: Bool {
        if case .loaded = self { return true }
        return false
    }
}

// MARK: - View Model

@MainActor
class MatchDetailViewModel: ObservableObject {
    let prediction: MatchPrediction

    // Section-specific loading states for progressive UI
    @Published var detailsState: SectionLoadState = .idle
    @Published var timelineState: SectionLoadState = .idle
    @Published var narrativeState: SectionLoadState = .idle
    @Published var statsState: SectionLoadState = .idle

    // Legacy compatibility (computed from detailsState)
    var isLoading: Bool { detailsState == .idle || detailsState.isLoading }
    var error: String? {
        if case .error(let msg) = detailsState { return msg }
        return nil
    }

    @Published var matchDetails: MatchDetailsResponse?
    @Published var homeEV: EVResult?
    @Published var drawEV: EVResult?
    @Published var awayEV: EVResult?
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
    @Published var matchEvents: [FootballMatchEvent]?

    // MARK: - Live Match Polling & Clock
    /// Updated prediction from polling (for live matches)
    @Published var livePrediction: MatchPrediction?
    /// Timestamp when live data was last fetched (for local clock calculation)
    @Published private(set) var liveDataLoadedAt: Date = Date()
    /// Current time - updated every 60s to trigger UI refresh for live matches
    @Published private(set) var clockTick: Date = Date()
    /// Goals inferred from score changes during live polling (minute when detected)
    @Published private(set) var inferredGoals: [InferredGoal] = []
    private var livePollingTimer: Timer?
    private var clockTimer: Timer?
    /// Track previous score to detect changes
    private var previousHomeGoals: Int = 0
    private var previousAwayGoals: Int = 0

    // Task cancellation support
    private var loadTask: Task<Void, Never>?

    /// The current prediction to display (live-updated or original)
    var currentPrediction: MatchPrediction {
        livePrediction ?? prediction
    }

    init(prediction: MatchPrediction) {
        self.prediction = prediction
        // Initialize previous goals from initial prediction
        self.previousHomeGoals = prediction.homeGoals ?? 0
        self.previousAwayGoals = prediction.awayGoals ?? 0
        calculateEV()
    }

    deinit {
        loadTask?.cancel()
        livePollingTimer?.invalidate()
        clockTimer?.invalidate()
    }

    // MARK: - Live Polling (30s)

    func startLivePollingIfNeeded() {
        guard currentPrediction.isLive else { return }

        // Seed existing goals on first load (distribute evenly across elapsed time)
        seedExistingGoals()

        // Start 60s clock timer for local elapsed updates
        clockTimer?.invalidate()
        clockTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.clockTick = Date()
            }
        }

        // Start 30s polling timer for status/score updates
        livePollingTimer?.invalidate()
        livePollingTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.pollLiveData()
            }
        }
        print("[LivePolling] Started for match \(prediction.matchId ?? 0)")
    }

    func stopLivePolling() {
        livePollingTimer?.invalidate()
        livePollingTimer = nil
        clockTimer?.invalidate()
        clockTimer = nil
        print("[LivePolling] Stopped")
    }

    /// Seed inferred goals from existing score when user opens a live match
    /// Since we don't know the exact minute of past goals, we distribute them
    /// evenly across the elapsed time for a reasonable approximation
    private func seedExistingGoals() {
        let pred = currentPrediction
        let homeGoals = pred.homeGoals ?? 0
        let awayGoals = pred.awayGoals ?? 0
        let elapsed = pred.elapsed ?? 1
        let totalGoals = homeGoals + awayGoals

        // Skip if no goals or goals already seeded
        guard totalGoals > 0 && inferredGoals.isEmpty else { return }

        // Distribute goals evenly across elapsed time
        // For 2 goals in 60 minutes: place at ~20' and ~40'
        var seededGoals: [InferredGoal] = []

        // Calculate interval between goals
        let interval = max(1, elapsed / (totalGoals + 1))

        // Interleave home and away goals chronologically
        var homeRemaining = homeGoals
        var awayRemaining = awayGoals
        var currentMinute = interval

        for i in 0..<totalGoals {
            // Alternate between home and away, but weighted by remaining goals
            let team: String
            if homeRemaining > 0 && (awayRemaining == 0 || i % 2 == 0) {
                team = "home"
                homeRemaining -= 1
            } else if awayRemaining > 0 {
                team = "away"
                awayRemaining -= 1
            } else {
                continue
            }

            let goal = InferredGoal(
                minute: min(currentMinute, elapsed - 1),
                team: team,
                homeTeamName: pred.homeTeam,
                awayTeamName: pred.awayTeam
            )
            seededGoals.append(goal)
            currentMinute += interval
        }

        inferredGoals = seededGoals
        print("[LivePolling] Seeded \(seededGoals.count) existing goals for \(pred.homeTeam) vs \(pred.awayTeam)")
    }

    private func pollLiveData() async {
        guard let matchId = prediction.matchId else { return }

        do {
            // Fetch fresh prediction data from API
            let response = try await APIClient.shared.getUpcomingPredictions(daysBack: 1, daysAhead: 1)
            if let updated = response.predictions.first(where: { $0.matchId == matchId }) {
                // Detect goal changes before updating livePrediction
                let newHomeGoals = updated.homeGoals ?? 0
                let newAwayGoals = updated.awayGoals ?? 0
                let currentMinute = updated.elapsed ?? 0

                // Check for new home goals
                if newHomeGoals > previousHomeGoals {
                    let goalsScored = newHomeGoals - previousHomeGoals
                    for _ in 0..<goalsScored {
                        let goal = InferredGoal(
                            minute: currentMinute,
                            team: "home",
                            homeTeamName: updated.homeTeam,
                            awayTeamName: updated.awayTeam
                        )
                        inferredGoals.append(goal)
                        print("[LivePolling] ⚽ Home goal detected at \(currentMinute)' - \(updated.homeTeam)")
                    }
                }

                // Check for new away goals
                if newAwayGoals > previousAwayGoals {
                    let goalsScored = newAwayGoals - previousAwayGoals
                    for _ in 0..<goalsScored {
                        let goal = InferredGoal(
                            minute: currentMinute,
                            team: "away",
                            homeTeamName: updated.homeTeam,
                            awayTeamName: updated.awayTeam
                        )
                        inferredGoals.append(goal)
                        print("[LivePolling] ⚽ Away goal detected at \(currentMinute)' - \(updated.awayTeam)")
                    }
                }

                // Update previous goals for next comparison
                previousHomeGoals = newHomeGoals
                previousAwayGoals = newAwayGoals

                livePrediction = updated
                liveDataLoadedAt = Date()
                print("[LivePolling] Updated match \(matchId): status=\(updated.status ?? "nil"), elapsed=\(updated.elapsed ?? -1), goals=\(inferredGoals.count)")

                // Write to shared cache so PredictionsListView can overlay
                MatchCache.shared.update(from: updated)

                // Stop polling if match finished
                if updated.isFinished {
                    stopLivePolling()
                }
            }
        } catch {
            print("[LivePolling] Error: \(error.localizedDescription)")
        }
    }

    /// Calculate the current elapsed minute for display (with local clock)
    /// Returns formatted string like "32'", "45+2'", "90+3'", or "Half Time"
    func calculatedElapsedDisplay() -> String {
        let pred = currentPrediction
        guard let status = pred.status else { return "LIVE" }

        // Only calculate for active play statuses
        let activeStatuses = ["1H", "2H", "LIVE"]
        guard activeStatuses.contains(status) else {
            // Transform HT to Half Time
            if status == "HT" {
                return "Half Time"
            }
            return status
        }

        // If no elapsed from backend, calculate from kickoff time
        guard let baseElapsed = pred.elapsed else {
            // Use kickoff-based calculation from MatchPrediction
            if let calculatedMins = pred.calculatedElapsed(at: clockTick) {
                return "\(calculatedMins)'"
            }
            return status
        }

        // If we have injury/added time from API, show it directly (e.g., "90+3'")
        if let extra = pred.elapsedExtra, extra > 0 {
            return "\(baseElapsed)+\(extra)'"
        }

        // At regulation time limits, don't calculate locally - wait for API injury time
        if status == "1H" && baseElapsed >= 45 {
            return "45'"
        } else if status == "2H" && baseElapsed >= 90 {
            return "90'"
        }

        // Calculate time passed since data was loaded (local clock estimation)
        let secondsPassed = clockTick.timeIntervalSince(liveDataLoadedAt)
        let totalSeconds = (baseElapsed * 60) + Int(secondsPassed)
        let displayMinutes = totalSeconds / 60

        // Apply caps - stop local calculation at regulation time
        if status == "1H" && displayMinutes >= 45 {
            return "45'"
        } else if status == "2H" && displayMinutes >= 90 {
            return "90'"
        }

        // Update cache with calculated elapsed so parrilla inherits it
        if let matchId = pred.matchId, displayMinutes != baseElapsed {
            MatchCache.shared.update(
                matchId: matchId,
                status: status,
                elapsed: displayMinutes,
                elapsedExtra: pred.elapsedExtra,
                homeGoals: pred.homeGoals,
                awayGoals: pred.awayGoals
            )
        }

        return "\(displayMinutes)'"
    }

    func cancelLoading() {
        loadTask?.cancel()
        loadTask = nil
    }

    func loadDetails() async {
        // Cancel any existing load task
        loadTask?.cancel()

        guard let matchId = prediction.matchId else {
            detailsState = .error("No match ID available")
            return
        }

        // Start loading immediately
        detailsState = .loading

        // For finished matches, start loading timeline/narrative in parallel with details
        // This ensures timeline doesn't appear "last" after other sections
        if prediction.isFinished {
            timelineState = .loading
            narrativeState = .loading
            statsState = .loading
        }

        let totalTimer = PerfTimer()

        do {
            // Check for cancellation
            try Task.checkCancellation()

            let detailsTimer = PerfTimer()
            let timelineTimer = PerfTimer()
            let insightsTimer = PerfTimer()

            // For finished matches, load everything in parallel
            // This ensures timeline appears at the same time as other sections
            if prediction.isFinished {
                async let detailsTask = APIClient.shared.getMatchDetails(matchId: matchId)
                async let timelineTask: () = loadTimeline(matchId: matchId)
                async let narrativeTask: () = loadNarrativeInsights(matchId: matchId)

                // Await details first (needed for UI)
                matchDetails = try await detailsTask
                let detailsMs = detailsTimer.elapsedMs

                // Check for cancellation before processing
                try Task.checkCancellation()

                processTeamHistory()

                // Mark details as loaded (triggers UI transition)
                detailsState = .loaded

                PerfLogger.shared.log(
                    endpoint: "MatchDetail.loadDetails",
                    message: "basic_data_ready",
                    data: [
                        "details_ms": detailsMs,
                        "is_finished": true
                    ]
                )

                // Wait for background tasks to complete
                _ = await (timelineTask, narrativeTask)

                PerfLogger.shared.log(
                    endpoint: "MatchDetail.loadDetails",
                    message: "background_complete",
                    data: [
                        "timeline_ms": timelineTimer.elapsedMs,
                        "insights_ms": insightsTimer.elapsedMs,
                        "total_ms": totalTimer.elapsedMs
                    ]
                )
            } else {
                // For upcoming matches, just load details
                matchDetails = try await APIClient.shared.getMatchDetails(matchId: matchId)
                let detailsMs = detailsTimer.elapsedMs

                // Check for cancellation before processing
                try Task.checkCancellation()

                processTeamHistory()

                // Mark details as loaded
                detailsState = .loaded

                PerfLogger.shared.log(
                    endpoint: "MatchDetail.loadDetails",
                    message: "basic_data_ready",
                    data: [
                        "details_ms": detailsMs,
                        "is_finished": false
                    ]
                )

                print("Match \(matchId) status: \(prediction.status ?? "nil") - not loading timeline")
            }
        } catch is CancellationError {
            // Task was cancelled (user navigated away), don't update state
            print("MatchDetail load cancelled for match \(matchId)")
        } catch {
            detailsState = .error(error.localizedDescription)
            PerfLogger.shared.log(
                endpoint: "MatchDetail.loadDetails",
                message: "error",
                data: [
                    "error": error.localizedDescription,
                    "total_ms": totalTimer.elapsedMs
                ]
            )
        }
    }

    private func loadTimeline(matchId: Int) async {
        do {
            try Task.checkCancellation()
            timeline = try await APIClient.shared.getMatchTimeline(matchId: matchId)
            timelineState = .loaded
            print("Timeline loaded successfully: \(timeline?.summary.correctPercentage ?? 0)% correct")
        } catch is CancellationError {
            // Don't update state on cancellation
        } catch {
            // Timeline is optional - don't fail the whole view
            timelineError = error.localizedDescription
            timelineState = .error(error.localizedDescription)
            print("Timeline error: \(error.localizedDescription)")
        }
    }

    private func loadNarrativeInsights(matchId: Int) async {
        do {
            try Task.checkCancellation()
            let response = try await APIClient.shared.getMatchInsights(matchId: matchId)

            try Task.checkCancellation()

            // Use LLM narrative if available (new system)
            if let narrative = response.llmNarrative {
                llmNarrative = narrative
                llmNarrativeStatus = response.llmNarrativeStatus
                narrativeState = .loaded
                print("LLM narrative loaded: \(narrative.narrative?.title ?? "no title")")
            } else {
                // No LLM narrative available - use "no_prediction" if backend didn't provide a status
                // "no_prediction" means there was no pre-match prediction, so no narrative will ever exist
                llmNarrativeStatus = response.llmNarrativeStatus ?? "no_prediction"
                narrativeState = .loaded  // Still "loaded" even if no narrative
                print("LLM narrative not available, status: \(llmNarrativeStatus ?? "unknown")")
            }

            // Load match stats for stats table (always, independent of narrative)
            matchStats = response.matchStats
            matchEvents = response.matchEvents
            statsState = .loaded
            print("DEBUG: matchStats = \(String(describing: response.matchStats))")
            print("DEBUG: matchEvents count = \(response.matchEvents?.count ?? 0)")
            if let stats = matchStats {
                print("Match stats loaded: possession=\(stats.home?.ballPossession.map { "\($0)%" } ?? "nil")")
            } else {
                print("Match stats is nil")
            }
        } catch is CancellationError {
            // Don't update state on cancellation
        } catch {
            // Insights are optional - don't fail the whole view
            llmNarrativeError = error.localizedDescription
            narrativeState = .error(error.localizedDescription)
            statsState = .error(error.localizedDescription)
            // Log full error for debugging schema mismatches
            print("LLM narrative error: \(error)")
            if let decodingError = error as? DecodingError {
                switch decodingError {
                case .keyNotFound(let key, let context):
                    print("  DecodingError: keyNotFound '\(key.stringValue)' at \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                case .typeMismatch(let type, let context):
                    print("  DecodingError: typeMismatch expected \(type) at \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                case .valueNotFound(let type, let context):
                    print("  DecodingError: valueNotFound \(type) at \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                case .dataCorrupted(let context):
                    print("  DecodingError: dataCorrupted at \(context.codingPath.map { $0.stringValue }.joined(separator: "."))")
                @unknown default:
                    print("  DecodingError: unknown")
                }
            }
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
            teamId: details.homeTeam.id,
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
            teamId: details.awayTeam.id,
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

    var formattedMatchDate: String {
        guard let date = prediction.matchDate else { return "TBD" }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d • h:mma"
        return formatter.string(from: date)
    }

    /// Date display for upcoming match - "Today" if same day, otherwise "Jan 18"
    var matchDateDisplay: String {
        guard let date = prediction.matchDate else { return "TBD" }
        if Calendar.current.isDateInToday(date) {
            return "Today"
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        return formatter.string(from: date)
    }

    /// Time display for upcoming match - "3:30 PM"
    var matchTimeDisplay: String {
        guard let date = prediction.matchDate else { return "" }
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter.string(from: date)
    }
}

// MARK: - Main View

// MARK: - Detail Card Tab

enum DetailCardTab: String, CaseIterable {
    case prediction = "Prediction"
    case stats = "Stats"
    case table = "Table"
}

struct MatchDetailView: View {
    let prediction: MatchPrediction
    @StateObject private var viewModel: MatchDetailViewModel
    @State private var isFavoriteHome = false
    @State private var isFavoriteAway = false
    @State private var showStandings = false
    @State private var selectedTab: DetailCardTab = .prediction

    init(prediction: MatchPrediction) {
        self.prediction = prediction
        _viewModel = StateObject(wrappedValue: MatchDetailViewModel(prediction: prediction))
    }

    private let valueColor = Color(red: 0.19, green: 0.82, blue: 0.35)  // #30D158

    /// Live status display with elapsed minute (uses local clock from viewModel)
    private var liveStatusDisplay: String {
        viewModel.calculatedElapsedDisplay()
    }

    /// Check if at regulation time limit (45' in 1H or 90' in 2H) - pulse should stop
    private func isAtRegulationTimeLimit() -> Bool {
        let pred = viewModel.currentPrediction
        guard let status = pred.status, let elapsed = pred.elapsed else { return false }

        // At 45' in first half or 90' in second half, stop pulsing
        if status == "1H" && elapsed >= 45 {
            return true
        } else if status == "2H" && elapsed >= 90 {
            return true
        }
        return false
    }

    // Animation for section transitions - fade only, no slide
    private let sectionTransition: AnyTransition = .opacity
    private let sectionAnimation: Animation = .easeOut(duration: 0.3)

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header section - shows skeleton until details loaded
                if viewModel.detailsState.isLoaded {
                    matchHeader
                        .transition(sectionTransition)
                } else if viewModel.detailsState.isLoading || viewModel.detailsState == .idle {
                    MatchHeaderSkeleton()
                } else if case .error(let msg) = viewModel.detailsState {
                    errorView(message: msg)
                }

                // Probability bar - shows skeleton until details loaded
                if viewModel.detailsState.isLoaded {
                    probabilityBar
                        .transition(sectionTransition)
                } else if viewModel.detailsState.isLoading || viewModel.detailsState == .idle {
                    ProbabilityBarSkeleton()
                }

                // Timeline for finished AND live matches
                // Always build locally - we have prediction + score data
                // Backend timeline is no longer needed (narrative/stats come separately)
                if prediction.isFinished || prediction.isLive {
                    PredictionTimelineView(
                        liveData: LiveTimelineData.from(
                            prediction: viewModel.currentPrediction,
                            inferredGoals: viewModel.inferredGoals
                        )
                    )
                    .transition(sectionTransition)
                }

                // Odds cards - shows skeleton until details loaded
                if viewModel.detailsState.isLoaded {
                    oddsCards
                        .transition(sectionTransition)
                } else if viewModel.detailsState.isLoading || viewModel.detailsState == .idle {
                    OddsCardsSkeleton()
                }

                // LLM Narrative (post-match analysis) - shows skeleton from idle through loading
                if prediction.isFinished {
                    if let narrative = viewModel.llmNarrative {
                        LLMNarrativeView(narrative: narrative)
                            .transition(sectionTransition)
                    } else if viewModel.narrativeState == .idle || viewModel.narrativeState.isLoading {
                        // Show skeleton immediately for finished matches to reserve space
                        NarrativeSkeleton()
                    } else if viewModel.narrativeState.isLoaded && viewModel.llmNarrativeStatus != nil {
                        // Show unavailable state for finished matches without narrative
                        LLMNarrativeUnavailableView(status: viewModel.llmNarrativeStatus)
                            .transition(sectionTransition)
                    }
                }

            }
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
            .animation(sectionAnimation, value: viewModel.detailsState)
            .animation(sectionAnimation, value: viewModel.timelineState)
            .animation(sectionAnimation, value: viewModel.narrativeState)
            .animation(sectionAnimation, value: viewModel.statsState)
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
            viewModel.startLivePollingIfNeeded()
        }
        .onDisappear {
            viewModel.cancelLoading()
            viewModel.stopLivePolling()
        }
        .sheet(isPresented: $showStandings) {
            if let leagueId = viewModel.matchDetails?.match.leagueId {
                let homeId = viewModel.matchDetails?.homeTeam.id
                let awayId = viewModel.matchDetails?.awayTeam.id
                let _ = print("[Standings] Opening with leagueId=\(leagueId), homeTeamId=\(String(describing: homeId)), awayTeamId=\(String(describing: awayId))")
                LeagueStandingsView(
                    leagueId: leagueId,
                    homeTeamId: homeId,
                    awayTeamId: awayId
                )
            }
        }
    }

    // MARK: - Error View

    private func errorView(message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.largeTitle)
                .foregroundStyle(.orange)

            Text("Error loading match")
                .font(.headline)
                .foregroundStyle(.white)

            Text(message)
                .font(.caption)
                .foregroundStyle(.gray)
                .multilineTextAlignment(.center)

            Button {
                Task {
                    await viewModel.loadDetails()
                }
            } label: {
                Label("Retry", systemImage: "arrow.clockwise")
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(32)
    }

    // MARK: - Match Header

    private var matchHeader: some View {
        VStack(spacing: 16) {
            ZStack {
                // Team columns at fixed positions (top-aligned, full height)
                HStack(spacing: 0) {
                    teamColumn(
                        name: prediction.homeTeam,
                        logo: viewModel.matchDetails?.homeTeam.logo,
                        position: viewModel.homeTeamForm?.position ?? 1,
                        role: "Home",
                        isFavorite: $isFavoriteHome
                    )
                    Spacer()
                    teamColumn(
                        name: prediction.awayTeam,
                        logo: viewModel.matchDetails?.awayTeam.logo,
                        position: viewModel.awayTeamForm?.position ?? 4,
                        role: "Away",
                        isFavorite: $isFavoriteAway
                    )
                }

                // Score row: centered vertically, scores + status aligned together
                if viewModel.currentPrediction.isLive || prediction.isFinished {
                    HStack(spacing: 0) {
                        // Spacer to push past home team column
                        Spacer()
                            .frame(width: 120)

                        // Home score
                        Text("\(viewModel.currentPrediction.homeGoals ?? 0)")
                            .font(.custom("BarlowCondensed-SemiBold", size: 68))
                            .foregroundStyle(.white)
                            .frame(minWidth: 40)

                        Spacer()

                        // Center: status indicator (same baseline as scores)
                        if prediction.isFinished {
                            HStack(spacing: 6) {
                                if let correct = prediction.predictionCorrect {
                                    Image(systemName: correct ? "checkmark.circle.fill" : "xmark.circle.fill")
                                        .font(.caption)
                                        .foregroundStyle(correct ? .green : .red)
                                }
                                Text("Final")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .foregroundStyle(.gray)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .background(Color.gray.opacity(0.2))
                            .clipShape(Capsule())
                        } else if viewModel.currentPrediction.status == "HT" {
                            // Half Time: same style as Final badge
                            Text(liveStatusDisplay)
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundStyle(.gray)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(Color.gray.opacity(0.2))
                                .clipShape(Capsule())
                        } else {
                            // Check if at regulation time limit (45' or 90') - stop pulsing
                            let isAtRegulationLimit = isAtRegulationTimeLimit()
                            if isAtRegulationLimit {
                                // Static text at regulation limit (waiting for injury time or half time)
                                Text(liveStatusDisplay)
                                    .font(.custom("BarlowCondensed-SemiBold", size: 22))
                                    .foregroundStyle(.gray)
                            } else {
                                PulsingLiveMinute(text: liveStatusDisplay)
                            }
                        }

                        Spacer()

                        // Away score
                        Text("\(viewModel.currentPrediction.awayGoals ?? 0)")
                            .font(.custom("BarlowCondensed-SemiBold", size: 68))
                            .foregroundStyle(.white)
                            .frame(minWidth: 40)

                        // Spacer to balance away team column
                        Spacer()
                            .frame(width: 120)
                    }
                } else {
                    // For upcoming matches: date and time centered
                    VStack(spacing: 2) {
                        Text(viewModel.matchDateDisplay)
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundStyle(.gray)
                        Text(viewModel.matchTimeDisplay)
                            .font(.custom("BarlowCondensed-SemiBold", size: 28))
                            .foregroundStyle(.white)
                    }
                }
            }
            .frame(height: 160)
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

            // Team logo - larger (cached for fast reload)
            if let logoUrl = logo, let url = URL(string: logoUrl) {
                CachedAsyncImage(url: url) { image in
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

            // Team name
            Text(name)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)

            // Position & Role - centered (hide position if 0/unavailable)
            if position > 0 {
                HStack(spacing: 2) {
                    Text("#\(position)")
                        .font(.custom("BarlowCondensed-SemiBold", size: 14))
                        .foregroundStyle(.gray)
                    Text("• \(role)")
                        .font(.caption)
                        .foregroundStyle(.gray)
                }
            } else {
                Text(role)
                    .font(.caption)
                    .foregroundStyle(.gray)
            }
        }
        .frame(width: 120)
    }

    private var teamPlaceholder: some View {
        Image(systemName: "shield.fill")
            .font(.system(size: 56))
            .foregroundStyle(.gray.opacity(0.5))
            .frame(width: 72, height: 72)
    }

    // MARK: - Probability Bar

    /// Returns the predicted outcome: "home", "draw", or "away"
    private var predictedOutcome: String {
        let home = prediction.probabilities.home
        let draw = prediction.probabilities.draw
        let away = prediction.probabilities.away

        if home >= draw && home >= away {
            return "home"
        } else if away >= home && away >= draw {
            return "away"
        } else {
            return "draw"
        }
    }

    /// Color for the prediction indicator based on predicted outcome
    private var predictionIndicatorColor: Color {
        switch predictedOutcome {
        case "home": return .blue
        case "away": return .red
        default: return .gray
        }
    }

    /// Calculate indicator position based on the predicted outcome and confidence
    /// - Low confidence (e.g., 36% vs 34%): indicator near the border with other zones
    /// - High confidence (e.g., 70%): indicator deeper into its zone
    private var predictionIndicatorPosition: CGFloat {
        let home = prediction.probabilities.home
        let draw = prediction.probabilities.draw
        let away = prediction.probabilities.away

        // Find the winning prediction and its margin over the second place
        let sorted = [(home, "home"), (draw, "draw"), (away, "away")].sorted { $0.0 > $1.0 }
        let winner = sorted[0]
        let runnerUp = sorted[1]

        // Confidence factor: how much the winner beats the runner-up
        // Range: 0 (tied) to ~0.7 (dominant 70% vs 15% vs 15%)
        // Normalized to 0...1 for positioning within the zone
        let margin = winner.0 - runnerUp.0
        let confidenceFactor = min(margin / 0.4, 1.0)  // 40% margin = max confidence

        switch winner.1 {
        case "home":
            // Blue zone: 0 to home
            // Low confidence → near border (closer to home)
            // High confidence → deeper into zone (closer to 0)
            let zoneBorder = home
            let zoneCenter = home / 2.0
            return zoneBorder - (confidenceFactor * (zoneBorder - zoneCenter))

        case "away":
            // Red zone: home+draw to 1.0
            // Low confidence → near border (closer to zoneStart)
            // High confidence → deeper into zone (closer to 1.0)
            let zoneStart = home + draw
            let zoneCenter = zoneStart + (away / 2.0)
            return zoneStart + (confidenceFactor * (zoneCenter - zoneStart))

        default: // draw
            // Gray zone: home to home+draw (center is fine for draw predictions)
            return home + (draw / 2.0)
        }
    }

    private var probabilityBar: some View {
        VStack(spacing: 12) {
            // Tab selector
            HStack(spacing: 0) {
                ForEach(DetailCardTab.allCases, id: \.self) { tab in
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selectedTab = tab
                        }
                    } label: {
                        Text(tab.rawValue)
                            .font(.subheadline)
                            .fontWeight(selectedTab == tab ? .semibold : .regular)
                            .foregroundStyle(selectedTab == tab ? .white : .gray)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                    }
                    .background(
                        selectedTab == tab ?
                        Color.white.opacity(0.1) : Color.clear
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
            .padding(.horizontal, 4)

            // Separator
            Divider()
                .background(Color.gray.opacity(0.3))

            // Tab content
            switch selectedTab {
            case .prediction:
                predictionTabContent

            case .stats:
                statsTabContent

            case .table:
                tableTabContent
            }
        }
        .padding(16)
        .modifier(GlassEffectModifier())
    }

    // MARK: - Prediction Tab Content

    private var predictionTabContent: some View {
        VStack(spacing: 0) {
            // Prediction indicator (arrow pointing down)
            GeometryReader { geo in
                let position = predictionIndicatorPosition
                let xOffset = geo.size.width * position

                Image(systemName: "arrowtriangle.down.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(predictionIndicatorColor)
                    .position(x: xOffset, y: 6)
            }
            .frame(height: 12)

            // Segmented probability bar with gradient transitions + vertical indicator line
            // Matches timeline bar height (24) and cornerRadius (6)
            GeometryReader { geo in
                let home = prediction.probabilities.home
                let draw = prediction.probabilities.draw
                let blendWidth: CGFloat = 0.04
                let indicatorX = geo.size.width * predictionIndicatorPosition

                ZStack {
                    Rectangle()
                        .fill(
                            LinearGradient(
                                stops: [
                                    .init(color: .blue, location: 0),
                                    .init(color: .blue, location: max(0, home - blendWidth)),
                                    .init(color: Color.gray.opacity(0.6), location: home + blendWidth),
                                    .init(color: Color.gray.opacity(0.6), location: max(home + blendWidth, home + draw - blendWidth)),
                                    .init(color: .red, location: min(1, home + draw + blendWidth)),
                                    .init(color: .red, location: 1)
                                ],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(height: 24)
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    Rectangle()
                        .fill(Color.white.opacity(0.8))
                        .frame(width: 1, height: 30)
                        .position(x: indicatorX, y: 12)
                }
            }
            .frame(height: 24)

            // Labels (with top padding to match timeline minute markers)
            HStack {
                probabilityLabel("Home", value: prediction.probabilities.homePercent, color: .blue)
                Spacer()
                probabilityLabel("Draw", value: prediction.probabilities.drawPercent, color: .gray)
                Spacer()
                probabilityLabel("Away", value: prediction.probabilities.awayPercent, color: .red)
            }
            .padding(.top, 8)
        }
    }

    // MARK: - Stats Tab Content

    private var statsTabContent: some View {
        Group {
            if let stats = viewModel.matchStats {
                statsTabRows(stats: stats)
            } else if prediction.isFinished {
                if viewModel.statsState.isLoading || viewModel.statsState == .idle {
                    ProgressView()
                        .frame(height: 100)
                } else {
                    Text("Estadísticas no disponibles")
                        .font(.subheadline)
                        .foregroundStyle(.gray)
                        .frame(height: 100)
                }
            } else {
                Text("Disponible post-partido")
                    .font(.subheadline)
                    .foregroundStyle(.gray)
                    .frame(height: 100)
            }
        }
    }

    private func statsTabRows(stats: MatchStats) -> some View {
        VStack(spacing: 10) {
            // Possession
            if let homePoss = stats.home?.ballPossession,
               let awayPoss = stats.away?.ballPossession {
                statRow(label: "Posesión", homeValue: "\(Int(homePoss))%", awayValue: "\(Int(awayPoss))%")
            }

            // xG
            if let homeXG = stats.home?.expectedGoals,
               let awayXG = stats.away?.expectedGoals {
                statRow(label: "xG", homeValue: homeXG, awayValue: awayXG)
            }

            // Shots on Target
            if let homeSoT = stats.home?.shotsOnGoal,
               let awaySoT = stats.away?.shotsOnGoal {
                statRow(label: "Tiros a puerta", homeValue: "\(homeSoT)", awayValue: "\(awaySoT)")
            }

            // Total Shots
            if let homeShots = stats.home?.totalShots,
               let awayShots = stats.away?.totalShots {
                statRow(label: "Tiros totales", homeValue: "\(homeShots)", awayValue: "\(awayShots)")
            }

            // Corners
            if let homeCorners = stats.home?.cornerKicks,
               let awayCorners = stats.away?.cornerKicks {
                statRow(label: "Corners", homeValue: "\(homeCorners)", awayValue: "\(awayCorners)")
            }

            // Fouls
            if let homeFouls = stats.home?.fouls,
               let awayFouls = stats.away?.fouls {
                statRow(label: "Faltas", homeValue: "\(homeFouls)", awayValue: "\(awayFouls)")
            }
        }
    }

    private func statRow(label: String, homeValue: String, awayValue: String) -> some View {
        HStack {
            Text(homeValue)
                .font(.custom("BarlowCondensed-SemiBold", size: 16))
                .foregroundStyle(.white)
                .frame(width: 50, alignment: .leading)

            Spacer()

            Text(label)
                .font(.caption)
                .foregroundStyle(.gray)

            Spacer()

            Text(awayValue)
                .font(.custom("BarlowCondensed-SemiBold", size: 16))
                .foregroundStyle(.white)
                .frame(width: 50, alignment: .trailing)
        }
    }

    // MARK: - Table Tab Content

    private var tableTabContent: some View {
        VStack(spacing: 0) {
            if let homeForm = viewModel.homeTeamForm {
                compactFormRow(data: homeForm)
            }

            Divider()
                .background(Color.gray.opacity(0.3))

            if let awayForm = viewModel.awayTeamForm {
                compactFormRow(data: awayForm)
            }

            // Tap hint
            Button {
                showStandings = true
            } label: {
                HStack {
                    Text("Ver clasificación completa")
                        .font(.caption)
                        .foregroundStyle(.cyan)
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundStyle(.cyan)
                }
                .padding(.top, 12)
            }
        }
    }

    private func compactFormRow(data: TeamFormData) -> some View {
        HStack(spacing: 8) {
            // Position
            Text(data.position > 0 ? "#\(data.position)" : "-")
                .font(.custom("BarlowCondensed-SemiBold", size: 16))
                .foregroundStyle(.white.opacity(0.8))
                .frame(width: 30, alignment: .leading)

            // Team logo
            if let logoUrl = data.logoUrl, let url = URL(string: logoUrl) {
                CachedAsyncImage(url: url) { image in
                    image.resizable().aspectRatio(contentMode: .fit)
                } placeholder: {
                    Image(systemName: "shield.fill").foregroundStyle(.gray)
                }
                .frame(width: 24, height: 24)
            } else {
                Image(systemName: "shield.fill")
                    .font(.caption)
                    .foregroundStyle(.gray)
                    .frame(width: 24, height: 24)
            }

            Spacer()

            // Form pills (last 5)
            HStack(spacing: 4) {
                let formResults = Array(data.form.prefix(5))
                ForEach(0..<5, id: \.self) { index in
                    if index < formResults.count {
                        compactFormPill(result: formResults[index])
                    } else {
                        compactFormPill(result: "-")
                    }
                }
            }

            Spacer()

            // Points
            HStack(spacing: 2) {
                Text("\(data.points)")
                    .font(.custom("BarlowCondensed-SemiBold", size: 18))
                    .foregroundStyle(.white)
                Text("Pts")
                    .font(.caption2)
                    .foregroundStyle(.gray)
            }
            .frame(width: 45, alignment: .trailing)
        }
        .padding(.vertical, 10)
    }

    private func compactFormPill(result: String) -> some View {
        let color: Color = {
            switch result {
            case "W": return .green
            case "L": return .red
            case "D": return .gray
            default: return Color(white: 0.2)
            }
        }()

        return Text(result == "-" ? "" : result)
            .font(.caption2)
            .fontWeight(.bold)
            .foregroundStyle(.white)
            .frame(width: 20, height: 20)
            .background(color)
            .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    private func probabilityLabel(_ label: String, value: String, color: Color) -> some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(color)
            Text(value)
                .font(.custom("BarlowCondensed-SemiBold", size: 22))
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
                    .font(.custom("BarlowCondensed-SemiBold", size: 26))
                    .foregroundStyle(color)
            }

            // Bookie section
            VStack(spacing: 2) {
                Text("Bookie")
                    .font(.caption2)
                    .foregroundStyle(.gray)
                Text(bookieOdds.map { String(format: "%.2f", $0) } ?? "-")
                    .font(.custom("BarlowCondensed-SemiBold", size: 26))
                    .foregroundStyle(isValue ? valueColor : .white)

                // EV display - reserve space for all cards if any has value
                if anyHasValue {
                    if isValue, let ev = evResult {
                        Text(ev.evDisplay)
                            .font(.custom("BarlowCondensed-SemiBold", size: 14))
                            .foregroundStyle(valueColor)
                    } else {
                        Text(" ")
                            .font(.custom("BarlowCondensed-SemiBold", size: 14))
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .modifier(GlassEffectModifier())
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(isValue ? valueColor.opacity(0.5) : Color.clear, lineWidth: 1.5)
        )
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
            elapsed: nil,
            elapsedExtra: nil,
            homeGoals: 2,
            awayGoals: 1,
            leagueId: 140,
            events: nil,
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
