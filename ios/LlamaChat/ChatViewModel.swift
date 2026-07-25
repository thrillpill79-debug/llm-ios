import Foundation
import SwiftUI

@MainActor
final class ChatViewModel: ObservableObject {
    enum Phase: Equatable {
        case needsModel
        case downloading(name: String, progress: Double)
        case loadingModel
        case ready
        case generating
        case failed(String)
    }

    @Published var phase: Phase = .needsModel
    @Published var messages: [ChatMessage] = []
    @Published var temperature: Double = 0.7
    @Published var statsLine = ""

    let systemPrompt = "You are a helpful assistant running entirely on the user's iPhone."

    private var engine: LlamaEngine?
    private var task: Task<Void, Never>?

    init() {
        if let existing = ModelStore.downloadedModels().first {
            loadModel(at: existing)
        }
    }

    // MARK: - Model lifecycle

    func downloadAndLoad(_ model: CatalogModel) {
        download(url: model.url, filename: model.id, displayName: model.name)
    }

    func downloadCustom(urlString: String) {
        guard let url = URL(string: urlString.trimmingCharacters(in: .whitespaces)),
              url.pathExtension == "gguf" else {
            phase = .failed("Enter a direct link to a .gguf file")
            return
        }
        download(url: url, filename: url.lastPathComponent, displayName: url.lastPathComponent)
    }

    private func download(url: URL, filename: String, displayName: String) {
        let destination = ModelStore.localPath(for: filename)
        phase = .downloading(name: displayName, progress: 0)
        task = Task {
            do {
                try await ModelStore.download(from: url, to: destination) { p in
                    Task { @MainActor in
                        if case .downloading = self.phase {
                            self.phase = .downloading(name: displayName, progress: p)
                        }
                    }
                }
                loadModel(at: destination)
            } catch is CancellationError {
                phase = .needsModel
            } catch let error as URLError where error.code == .cancelled {
                phase = .needsModel
            } catch {
                phase = .failed("Download failed: \(error.localizedDescription)")
            }
        }
    }

    private func loadModel(at url: URL) {
        phase = .loadingModel
        Task.detached(priority: .userInitiated) {
            do {
                let engine = try LlamaEngine(modelPath: url.path)
                await MainActor.run {
                    self.engine = engine
                    self.startNewChat()
                    self.phase = .ready
                }
            } catch {
                await MainActor.run {
                    self.phase = .failed(error.localizedDescription)
                }
            }
        }
    }

    func deleteModelAndRestart() {
        stop()
        engine = nil
        for url in ModelStore.downloadedModels() {
            try? FileManager.default.removeItem(at: url)
        }
        messages = []
        phase = .needsModel
    }

    var modelDescription: String { engine?.modelDescription ?? "" }

    // MARK: - Chat

    func startNewChat() {
        stop()
        engine?.resetConversation()
        messages = [ChatMessage(role: .system, content: systemPrompt)]
        statsLine = ""
    }

    func send(_ text: String) {
        guard let engine, phase == .ready else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        messages.append(ChatMessage(role: .user, content: trimmed))
        messages.append(ChatMessage(role: .assistant, content: ""))
        phase = .generating
        let history = Array(messages.dropLast())
        let temp = Float(temperature)

        task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                _ = try engine.generate(messages: history,
                                        temperature: temp,
                                        maxTokens: 1024) { piece in
                    Task { @MainActor [weak self] in
                        guard let self, !self.messages.isEmpty else { return }
                        self.messages[self.messages.count - 1].content += piece
                    }
                }
                let stats = engine.lastStats
                await MainActor.run { [weak self] in
                    self?.phase = .ready
                    self?.statsLine = String(format: "%d tokens · %.1f tok/s",
                                             stats.generatedTokens, stats.tokensPerSecond)
                }
            } catch {
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    if self.messages.last?.content.isEmpty == true {
                        self.messages[self.messages.count - 1].content = "⚠️ \(error.localizedDescription)"
                    }
                    self.phase = .ready
                }
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        if phase == .generating { phase = .ready }
    }
}
