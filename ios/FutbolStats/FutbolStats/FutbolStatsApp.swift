import SwiftUI

@main
struct FutbolStatsApp: App {
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background {
                // Log telemetry when app goes to background
                ImageCache.shared.logStats()
            }
        }
    }
}

// MARK: - Main Tab View

struct MainTabView: View {
    var body: some View {
        TabView {
            PredictionsListView()
                .tabItem {
                    Label("Predictions", systemImage: "chart.bar.fill")
                }

            DashboardView()
                .tabItem {
                    Label("Dashboard", systemImage: "gauge.with.dots.needle.33percent")
                }

            TeamsView()
                .tabItem {
                    Label("Teams", systemImage: "person.3.fill")
                }

            CompetitionsView()
                .tabItem {
                    Label("Competitions", systemImage: "trophy.fill")
                }
        }
    }
}
