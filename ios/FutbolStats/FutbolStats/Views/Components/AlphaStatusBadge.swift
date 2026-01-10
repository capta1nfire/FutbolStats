import SwiftUI

struct AlphaStatusBadge: View {
    let progress: OpsProgress?

    var body: some View {
        let ready = progress?.readyForRetest ?? false
        let text = ready ? "✅ Listo para re-test" : "⏳ Alpha en progreso"

        return HStack(spacing: 8) {
            Text(text)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(ready ? .green : .yellow)

            if let p = progress {
                Text("\(p.pitBets30d ?? 0)/\(p.targetPitBets30d ?? 100) bets • \(p.baselineCoveragePct ?? 0)% baseline")
                    .font(.caption2)
                    .foregroundStyle(.gray)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((ready ? Color.green : Color.yellow).opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke((ready ? Color.green : Color.yellow).opacity(0.35), lineWidth: 1)
        )
    }
}

#Preview {
    ZStack {
        Color.black.ignoresSafeArea()
        AlphaStatusBadge(progress: OpsProgress(
            pitSnapshots30d: 54,
            targetPitSnapshots30d: 100,
            pitBets30d: 22,
            targetPitBets30d: 100,
            baselineCoveragePct: 61.1,
            pitWithBaseline: 33,
            pitTotalForBaseline: 54,
            targetBaselineCoveragePct: 60,
            readyForRetest: false
        ))
        .padding()
    }
    .preferredColorScheme(.dark)
}


