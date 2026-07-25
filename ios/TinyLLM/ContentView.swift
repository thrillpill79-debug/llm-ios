import SwiftUI

@MainActor
final class LLMViewModel: ObservableObject {
    @Published var output = ""
    @Published var isGenerating = false
    @Published var loadError: String?
    @Published var temperature: Double = 0.8
    @Published var maxNewTokens: Double = 300

    private var generator: Generator?
    private var generationTask: Task<Void, Never>?
    private(set) var paramCount = 0

    init() {
        do {
            let checkpoint = try Checkpoint.load()
            let model = TinyGPT(checkpoint: checkpoint)
            let tokenizer = CharTokenizer(chars: checkpoint.chars)
            generator = Generator(model: model, tokenizer: tokenizer)
            paramCount = checkpoint.tensors.values.reduce(0) { $0 + $1.count }
        } catch {
            loadError = error.localizedDescription
        }
    }

    func generate(prompt: String) {
        guard let generator, !isGenerating else { return }
        isGenerating = true
        output = prompt
        let sampler = Sampler(temperature: Float(temperature), topK: 40)
        let tokens = Int(maxNewTokens)
        generationTask = Task {
            for await piece in generator.generate(prompt: prompt, maxNewTokens: tokens, sampler: sampler) {
                output += piece
            }
            isGenerating = false
        }
    }

    func stop() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }
}

struct ContentView: View {
    @StateObject private var vm = LLMViewModel()
    @State private var prompt = "ROMEO:"

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                if let error = vm.loadError {
                    ContentUnavailableView("Model not loaded",
                                           systemImage: "exclamationmark.triangle",
                                           description: Text(error))
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            Text(vm.output.isEmpty ? "Output appears here…" : vm.output)
                                .font(.system(.body, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .foregroundStyle(vm.output.isEmpty ? .secondary : .primary)
                                .padding(12)
                                .id("output")
                        }
                        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
                        .onChange(of: vm.output) {
                            proxy.scrollTo("output", anchor: .bottom)
                        }
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Temperature \(vm.temperature, specifier: "%.2f")")
                                .font(.caption)
                            Slider(value: $vm.temperature, in: 0.1...1.5)
                        }
                        HStack {
                            Text("Tokens \(Int(vm.maxNewTokens))")
                                .font(.caption)
                            Slider(value: $vm.maxNewTokens, in: 50...800, step: 50)
                        }
                    }

                    HStack {
                        TextField("Prompt", text: $prompt, axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                            .lineLimit(1...3)
                        if vm.isGenerating {
                            Button("Stop", role: .destructive) { vm.stop() }
                                .buttonStyle(.bordered)
                        } else {
                            Button("Generate") { vm.generate(prompt: prompt) }
                                .buttonStyle(.borderedProminent)
                        }
                    }
                }
            }
            .padding()
            .navigationTitle("TinyLLM")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Text("\(vm.paramCount / 1000)k params · on-device")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
