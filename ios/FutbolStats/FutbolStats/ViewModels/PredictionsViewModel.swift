import Foundation
import SwiftUI

@MainActor
class PredictionsViewModel: ObservableObject {
    @Published var predictions: [MatchPrediction] = []
    @Published var isLoading = false
    @Published var error: String?
    @Published var modelLoaded = false
    @Published var lastUpdated: Date?
    @Published var selectedDate: Date
    @Published var opsProgress: OpsProgress?

    private let apiClient = APIClient.shared

    // UTC calendar for consistent date comparisons (backend sends UTC dates)
    private var utcCalendar: Calendar {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        return cal
    }

    init() {
        // Initialize selectedDate to today in UTC to match dateRange and backend dates
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        self.selectedDate = cal.startOfDay(for: Date())
    }

    // MARK: - Load Predictions

    func loadPredictions(days: Int = 7) async {
        isLoading = true
        error = nil

        do {
            let response = try await apiClient.getUpcomingPredictions(days: days)
            predictions = response.predictions
            lastUpdated = Date()
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    // MARK: - Check Health

    func checkHealth() async {
        do {
            let response = try await apiClient.checkHealth()
            modelLoaded = response.modelLoaded
        } catch {
            self.error = error.localizedDescription
            modelLoaded = false
        }
    }

    // MARK: - Refresh

    func refresh() async {
        // Parallel: health + ops don't block predictions
        async let healthTask: () = checkHealth()
        async let opsTask: () = loadOpsProgress()
        async let predictionsTask: () = loadPredictions()

        _ = await (healthTask, opsTask, predictionsTask)
    }

    // MARK: - Ops Progress (Alpha readiness)

    func loadOpsProgress() async {
        do {
            let ops = try await apiClient.getOpsDashboard()
            opsProgress = ops.progress
        } catch {
            // Non-blocking: predictions can still work even if dashboard token is missing
            opsProgress = nil
        }
    }

    // MARK: - Filtered Predictions

    var predictionsForSelectedDate: [MatchPrediction] {
        return predictions.filter { prediction in
            guard let matchDate = prediction.matchDate else { return false }
            return utcCalendar.isDate(matchDate, inSameDayAs: selectedDate)
        }.sorted { (p1, p2) in
            guard let d1 = p1.matchDate, let d2 = p2.matchDate else { return false }
            return d1 < d2
        }
    }

    var valueBetPredictionsForSelectedDate: [MatchPrediction] {
        predictionsForSelectedDate.filter { ($0.valueBets?.count ?? 0) > 0 }
    }

    var regularPredictionsForSelectedDate: [MatchPrediction] {
        predictionsForSelectedDate.filter { ($0.valueBets?.count ?? 0) == 0 }
    }

    // Legacy - all predictions
    var valueBetPredictions: [MatchPrediction] {
        predictions.filter { ($0.valueBets?.count ?? 0) > 0 }
    }

    var upcomingPredictions: [MatchPrediction] {
        predictions.sorted { (p1, p2) in
            guard let d1 = p1.matchDate, let d2 = p2.matchDate else { return false }
            return d1 < d2
        }
    }

    // MARK: - Date Helpers

    func matchCount(for date: Date) -> Int {
        return predictions.filter { prediction in
            guard let matchDate = prediction.matchDate else { return false }
            return utcCalendar.isDate(matchDate, inSameDayAs: date)
        }.count
    }
}
