// Model catalog + downloader. GGUF files are far too big for an app bundle
// or a git repo, so the app downloads one on first launch into
// Application Support/Models/ (excluded from iCloud backup).

import Foundation

struct CatalogModel: Identifiable, Hashable {
    let id: String            // filename on disk
    let name: String
    let url: URL
    let sizeMB: Int
    let note: String

    /// Curated defaults sized for an iPhone 15 (6 GB RAM, ~3.3 GB app budget).
    /// If a URL breaks (Hugging Face repos occasionally reorganize), paste any
    /// direct .gguf link in the app's custom-URL field instead.
    static let catalog: [CatalogModel] = [
        CatalogModel(
            id: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            name: "Qwen2.5 0.5B Instruct",
            url: URL(string: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")!,
            sizeMB: 400,
            note: "Fastest. Good for testing the pipeline. Apache 2.0."
        ),
        CatalogModel(
            id: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            name: "Qwen2.5 1.5B Instruct",
            url: URL(string: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf")!,
            sizeMB: 990,
            note: "Recommended balance of quality and speed. Apache 2.0."
        ),
        CatalogModel(
            id: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            name: "Llama 3.2 3B Instruct",
            url: URL(string: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf")!,
            sizeMB: 2020,
            note: "Best quality that fits a base iPhone 15. Llama license."
        ),
    ]
}

enum ModelStore {
    static var modelsDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
        let dir = base.appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func localPath(for filename: String) -> URL {
        modelsDirectory.appendingPathComponent(filename)
    }

    /// Any .gguf already downloaded, largest first.
    static func downloadedModels() -> [URL] {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: modelsDirectory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return files
            .filter { $0.pathExtension == "gguf" }
            .sorted { (fileSize($0) ?? 0) > (fileSize($1) ?? 0) }
    }

    static func fileSize(_ url: URL) -> Int? {
        (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
    }

    /// Downloads `url` to `destination`, reporting progress in [0, 1] (or -1
    /// when the server doesn't announce a length). Cancellable via task
    /// cancellation.
    static func download(from url: URL, to destination: URL,
                         progress: @escaping (Double) -> Void) async throws {
        let delegate = DownloadDelegate(progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let tempURL: URL = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                delegate.continuation = continuation
                let task = session.downloadTask(with: url)
                delegate.task = task
                task.resume()
            }
        } onCancel: {
            delegate.task?.cancel()
        }

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: tempURL, to: destination)

        // model files are re-downloadable; keep them out of iCloud backups
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var dest = destination
        try? dest.setResourceValues(values)
    }
}

private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let progress: (Double) -> Void
    var continuation: CheckedContinuation<URL, Error>?
    var task: URLSessionDownloadTask?

    init(progress: @escaping (Double) -> Void) {
        self.progress = progress
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        progress(totalBytesExpectedToWrite > 0
                 ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
                 : -1)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
            continuation?.resume(throwing: URLError(.badServerResponse))
            continuation = nil
            return
        }
        // the system deletes `location` when this method returns — move it now
        let kept = location.deletingLastPathComponent()
            .appendingPathComponent("model-\(UUID().uuidString).part")
        do {
            try FileManager.default.moveItem(at: location, to: kept)
            continuation?.resume(returning: kept)
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}
