import SwiftUI

/// A cached version of AsyncImage that stores downloaded images in memory
struct CachedAsyncImage<Content: View, Placeholder: View>: View {
    let url: URL?
    let content: (Image) -> Content
    let placeholder: () -> Placeholder

    @State private var image: UIImage?
    @State private var isLoading = false

    init(
        url: URL?,
        @ViewBuilder content: @escaping (Image) -> Content,
        @ViewBuilder placeholder: @escaping () -> Placeholder
    ) {
        self.url = url
        self.content = content
        self.placeholder = placeholder
    }

    var body: some View {
        Group {
            if let image = image {
                content(Image(uiImage: image))
            } else {
                placeholder()
                    .onAppear {
                        loadImage()
                    }
            }
        }
    }

    private func loadImage() {
        guard let url = url, !isLoading else { return }

        // Check cache first (synchronous, fast)
        if let cached = ImageCache.shared.get(for: url) {
            self.image = cached
            return
        }

        isLoading = true

        // Use detached task to avoid inheriting MainActor context
        // This ensures URLSession.data and UIImage(data:) run off main thread
        Task.detached(priority: .utility) {
            let startTime = CFAbsoluteTimeGetCurrent()
            do {
                let (data, _) = try await URLSession.shared.data(from: url)
                // UIImage init is CPU-bound, keep it off main thread
                guard let uiImage = UIImage(data: data) else {
                    await MainActor.run {
                        self.isLoading = false
                    }
                    return
                }
                let fetchMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                ImageCache.shared.set(uiImage, for: url)
                ImageCache.shared.recordFetch(ms: fetchMs)
                // Only mutate @State on MainActor
                await MainActor.run {
                    self.image = uiImage
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.isLoading = false
                }
            }
        }
    }
}

/// Simple in-memory image cache with size limit and telemetry
final class ImageCache {
    static let shared = ImageCache()

    private var cache = NSCache<NSURL, UIImage>()

    // Telemetry counters (atomic via serial queue)
    private let statsQueue = DispatchQueue(label: "ImageCache.stats")
    private var _hits: Int = 0
    private var _misses: Int = 0
    private var _fetchTime: Double = 0  // Total fetch time in ms
    private var _fetchCount: Int = 0

    private init() {
        // Limit cache to ~50MB
        cache.totalCostLimit = 50 * 1024 * 1024
        cache.countLimit = 200
    }

    func get(for url: URL) -> UIImage? {
        let result = cache.object(forKey: url as NSURL)
        statsQueue.async {
            if result != nil {
                self._hits += 1
            } else {
                self._misses += 1
            }
        }
        return result
    }

    func set(_ image: UIImage, for url: URL) {
        let cost = image.pngData()?.count ?? 0
        cache.setObject(image, forKey: url as NSURL, cost: cost)
    }

    /// Record fetch time for telemetry
    func recordFetch(ms: Double) {
        statsQueue.async {
            self._fetchTime += ms
            self._fetchCount += 1
        }
    }

    func clear() {
        cache.removeAllObjects()
    }

    /// Get current stats for telemetry (thread-safe)
    func getStats() -> (hits: Int, misses: Int, hitRate: Double, avgFetchMs: Double) {
        statsQueue.sync {
            let total = _hits + _misses
            let hitRate = total > 0 ? Double(_hits) / Double(total) : 0
            let avgFetch = _fetchCount > 0 ? _fetchTime / Double(_fetchCount) : 0
            return (_hits, _misses, hitRate, avgFetch)
        }
    }

    /// Log stats summary (call periodically or on app background)
    func logStats() {
        let stats = getStats()
        print("[ImageCache] hits=\(stats.hits), misses=\(stats.misses), hit_rate=\(String(format: "%.1f%%", stats.hitRate * 100)), avg_fetch_ms=\(String(format: "%.1f", stats.avgFetchMs))")
    }
}

// Convenience initializer matching AsyncImage API
extension CachedAsyncImage where Content == Image, Placeholder == Color {
    init(url: URL?) {
        self.init(
            url: url,
            content: { $0 },
            placeholder: { Color.gray.opacity(0.2) }
        )
    }
}
