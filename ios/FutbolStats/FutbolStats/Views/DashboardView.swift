import SwiftUI

struct DashboardView: View {
    @State private var ops: OpsDashboardResponse?
    @State private var pit: PITDashboardResponse?
    @State private var snapshots: AlphaProgressSnapshotsResponse?
    @State private var isLoading = true
    @State private var error: String?

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                if isLoading {
                    ProgressView("Loading dashboard…")
                        .tint(.white)
                } else if let error = error {
                    VStack(spacing: 12) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundStyle(.orange)
                        Text(error)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.gray)
                        if AppConfiguration.dashboardToken == nil {
                            Text("Missing DASHBOARD_TOKEN. Set Info.plist key `DASHBOARD_TOKEN` or UserDefaults `dashboard_token` (dev).")
                                .font(.caption)
                                .foregroundStyle(.gray.opacity(0.8))
                                .multilineTextAlignment(.center)
                                .padding(.top, 6)
                        }
                        Button("Retry") { Task { await load() } }
                            .buttonStyle(.borderedProminent)
                    }
                    .padding()
                } else {
                    content
                }
            }
            .navigationTitle("Dashboard")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Color.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .refreshable { await load() }
            .task { await load() }
        }
    }

    private var content: some View {
        ScrollView {
            VStack(spacing: 16) {
                if let ops = ops {
                    opsCard(ops)
                }
                if let pit = pit {
                    pitCard(pit)
                }
                if let snapshots = snapshots {
                    snapshotsCard(snapshots)
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
        }
    }

    private func opsCard(_ ops: OpsDashboardResponse) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            headerRow(title: "OPS", subtitle: ops.generatedAt)

            let progress = ops.progress
            if let progress = progress {
                VStack(spacing: 10) {
                    progressRow(
                        label: "PIT Snapshots (30d)",
                        current: progress.pitSnapshots30d ?? 0,
                        target: progress.targetPitSnapshots30d ?? 100
                    )
                    progressRow(
                        label: "Bets Evaluables (30d)",
                        current: progress.pitBets30d ?? 0,
                        target: progress.targetPitBets30d ?? 100
                    )
                    progressPctRow(
                        label: "Baseline Coverage",
                        pct: progress.baselineCoveragePct ?? 0,
                        targetPct: Double(progress.targetBaselineCoveragePct ?? 60),
                        fracText: "\(progress.pitWithBaseline ?? 0)/\(progress.pitTotalForBaseline ?? 0)"
                    )

                    Text((progress.readyForRetest ?? false) ? "✅ Listo para re-test" : "⏳ Aún no listo para re-test")
                        .font(.caption)
                        .foregroundStyle((progress.readyForRetest ?? false) ? .green : .yellow)
                        .padding(.top, 2)
                }
            }

            Divider().background(Color.gray.opacity(0.25))

            HStack {
                metricPill(label: "PIT live 60m", value: "\(ops.pit?.live60m ?? 0)")
                metricPill(label: "PIT live 24h", value: "\(ops.pit?.live24h ?? 0)")
            }

            HStack {
                metricPill(label: "League mode", value: ops.leagueMode ?? "—")
                metricPill(label: "Tracked leagues", value: "\(ops.trackedLeaguesCount ?? 0)")
            }

            if let budget = ops.budget {
                metricPill(label: "Budget", value: budget.status ?? "unavailable")
                if let resetTime = budget.tokensResetLocalTime, let resetTz = budget.tokensResetTz {
                    metricPill(label: "Budget resets", value: "\(resetTime) (\(resetTz))")
                }
            }
        }
        .padding(14)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func pitCard(_ pit: PITDashboardResponse) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            headerRow(title: "PIT", subtitle: pit.source ?? "—")

            if let weekly = pit.weekly, let s = weekly.summary {
                HStack {
                    metricPill(label: "Principal N", value: "\(s.principalN ?? 0)")
                    metricPill(label: "Ideal N", value: "\(s.idealN ?? 0)")
                }
                HStack {
                    metricPill(label: "Edge", value: s.edgeDiagnostic ?? "—")
                    metricPill(label: "Quality", value: (s.qualityScore != nil) ? "\(s.qualityScore!)%" : "—")
                }
            } else if let daily = pit.daily {
                HStack {
                    metricPill(label: "Phase", value: daily.phase ?? "—")
                    metricPill(label: "N", value: "\(daily.counts?.nPitValid1090 ?? 0)")
                }
                if let interpretation = daily.interpretation {
                    metricPill(label: "Verdict", value: interpretation.verdict ?? "—")
                }
            } else {
                Text("No PIT report available yet.")
                    .font(.caption)
                    .foregroundStyle(.gray)
            }
        }
        .padding(14)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func snapshotsCard(_ snapshots: AlphaProgressSnapshotsResponse) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            headerRow(title: "Alpha Snapshots", subtitle: "count=\(snapshots.count)")

            if let first = snapshots.snapshots.first {
                Text("Latest: \(first.capturedAt ?? "—") • \(first.appCommit ?? "—")")
                    .font(.caption)
                    .foregroundStyle(.gray)
            }
        }
        .padding(14)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func headerRow(title: String, subtitle: String?) -> some View {
        HStack {
            Text(title)
                .font(.headline)
                .foregroundStyle(.white)
            Spacer()
            Text(subtitle ?? "—")
                .font(.caption2)
                .foregroundStyle(.gray)
        }
    }

    private func metricPill(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.gray)
            Text(value)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(white: 0.14))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func progressRow(label: String, current: Int, target: Int) -> some View {
        let pct = target > 0 ? min(1.0, Double(current) / Double(target)) : 0
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.white)
                Spacer()
                Text("\(current)/\(target)")
                    .font(.caption2)
                    .foregroundStyle(.gray)
            }
            ProgressView(value: pct)
                .tint(.blue)
        }
    }

    private func progressPctRow(label: String, pct: Double, targetPct: Double, fracText: String) -> some View {
        let p = targetPct > 0 ? min(1.0, pct / targetPct) : 0
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.white)
                Spacer()
                Text("\(pct)% (\(fracText)) / \(Int(targetPct))%")
                    .font(.caption2)
                    .foregroundStyle(.gray)
            }
            ProgressView(value: p)
                .tint(.blue)
        }
    }

    private func load() async {
        isLoading = true
        error = nil

        // Fetch independently in parallel - partial success is OK
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                do {
                    let result = try await APIClient.shared.getOpsDashboard()
                    await MainActor.run { self.ops = result }
                } catch {
                    // Silently fail - partial success is OK
                }
            }
            group.addTask {
                do {
                    let result = try await APIClient.shared.getPITDashboard()
                    await MainActor.run { self.pit = result }
                } catch {
                    // Silently fail - partial success is OK
                }
            }
            group.addTask {
                do {
                    let result = try await APIClient.shared.getAlphaProgressSnapshots(limit: 20)
                    await MainActor.run { self.snapshots = result }
                } catch {
                    // Silently fail - partial success is OK
                }
            }
        }

        // Only show error if ALL failed
        if ops == nil && pit == nil && snapshots == nil {
            self.error = "Failed to load dashboard data. Check token configuration."
        }

        isLoading = false
    }
}

#Preview {
    DashboardView()
        .preferredColorScheme(.dark)
}


