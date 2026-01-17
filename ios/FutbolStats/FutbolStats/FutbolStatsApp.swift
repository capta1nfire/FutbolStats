import SwiftUI

@main
struct FutbolStatsApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        // Set tab bar appearance - white for selected items
        UITabBar.appearance().tintColor = .white
        UITabBar.appearance().unselectedItemTintColor = .gray
    }

    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
        .onChange(of: scenePhase) { _, newPhase in
            switch newPhase {
            case .active:
                // Resume live score polling when app becomes active
                LiveScoreManager.shared.onAppBecameActive()
            case .inactive, .background:
                // Stop polling and log telemetry when app goes to background
                LiveScoreManager.shared.onAppBecameInactive()
                ImageCache.shared.logStats()
            @unknown default:
                break
            }
        }
    }
}

// MARK: - Main Tab View

struct MainTabView: View {
    @State private var showingHomeMenu = false
    @State private var navigateToDashboard = false
    @State private var selectedTab = 1  // Default to Match tab

    var body: some View {
        TabView(selection: $selectedTab) {
            // Home - shows menu
            HomeMenuView(showingMenu: $showingHomeMenu, navigateToDashboard: $navigateToDashboard)
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }
                .tag(0)

            PredictionsListView()
                .tabItem {
                    Label("Match", systemImage: "sportscourt.fill")
                }
                .tag(1)

            TeamsView()
                .tabItem {
                    Label("Teams", systemImage: "person.3.fill")
                }
                .tag(2)

            CompetitionsView()
                .tabItem {
                    Label("Competitions", systemImage: "trophy.fill")
                }
                .tag(3)

            SearchView()
                .tabItem {
                    Label("Search", systemImage: "magnifyingglass")
                }
                .tag(4)
        }
        .tint(.white)
    }
}

// MARK: - Home Menu View

struct HomeMenuView: View {
    @Binding var showingMenu: Bool
    @Binding var navigateToDashboard: Bool

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                VStack(spacing: 20) {
                    // User avatar
                    Image("UserAvatar")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 80, height: 80)
                        .clipShape(Circle())
                        .padding(.top, 40)

                    Text("FutbolStats")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)

                    // Menu options
                    VStack(spacing: 0) {
                        NavigationLink(destination: DashboardView()) {
                            menuRow(icon: "gauge.with.dots.needle.33percent", title: "Dashboard")
                        }

                        Divider().background(Color.gray.opacity(0.3))

                        NavigationLink(destination: Text("My Teams").foregroundStyle(.white)) {
                            menuRow(icon: "heart.fill", title: "My Teams")
                        }

                        Divider().background(Color.gray.opacity(0.3))

                        NavigationLink(destination: Text("Settings").foregroundStyle(.white)) {
                            menuRow(icon: "gear", title: "Settings")
                        }
                    }
                    .background(Color(white: 0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal, 20)

                    Spacer()
                }
            }
            .navigationBarHidden(true)
        }
    }

    private func menuRow(icon: String, title: String) -> some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundStyle(.gray)
                .frame(width: 28)

            Text(title)
                .font(.body)
                .foregroundStyle(.white)

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.gray)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }
}

// MARK: - Search View (Placeholder)

struct SearchView: View {
    @State private var searchText = ""

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                VStack(spacing: 20) {
                    // Search bar
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(.gray)

                        TextField("Search teams, matches...", text: $searchText)
                            .foregroundStyle(.white)
                    }
                    .padding(12)
                    .background(Color(white: 0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal, 16)
                    .padding(.top, 16)

                    Spacer()

                    if searchText.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "magnifyingglass")
                                .font(.system(size: 48))
                                .foregroundStyle(.gray)

                            Text("Search for teams or matches")
                                .font(.headline)
                                .foregroundStyle(.gray)
                        }
                        Spacer()
                    }
                }
            }
            .navigationBarHidden(true)
        }
    }
}
