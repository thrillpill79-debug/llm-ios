import SwiftUI

struct ChatView: View {
    @StateObject private var vm = ChatViewModel()
    @State private var draft = ""
    @State private var customURL = ""
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            Group {
                switch vm.phase {
                case .needsModel, .failed:
                    modelPicker
                case .downloading(let name, let progress):
                    downloadProgress(name: name, progress: progress)
                case .loadingModel:
                    ProgressView("Loading model…")
                case .ready, .generating:
                    conversation
                }
            }
            .navigationTitle("LlamaChat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if case .ready = vm.phase {
                    ToolbarItem(placement: .topBarLeading) {
                        Button("New chat", systemImage: "square.and.pencil") {
                            vm.startNewChat()
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Settings", systemImage: "gearshape") {
                            showSettings = true
                        }
                    }
                }
            }
            .sheet(isPresented: $showSettings) { settings }
        }
    }

    // MARK: - Model setup

    private var modelPicker: some View {
        List {
            if case .failed(let message) = vm.phase {
                Section {
                    Label(message, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                }
            }
            Section("Choose a model to download") {
                ForEach(CatalogModel.catalog) { model in
                    Button {
                        vm.downloadAndLoad(model)
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(model.name).font(.headline)
                                Spacer()
                                Text("\(model.sizeMB) MB")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Text(model.note)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .tint(.primary)
                }
            }
            Section("Or paste a direct .gguf URL") {
                TextField("https://…/model.gguf", text: $customURL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button("Download") { vm.downloadCustom(urlString: customURL) }
                    .disabled(customURL.isEmpty)
            }
            Section {
                Text("Downloads happen once and run on Wi-Fi sized files "
                     + "(0.4–2 GB). Everything after that is fully offline.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func downloadProgress(name: String, progress: Double) -> some View {
        VStack(spacing: 16) {
            if progress >= 0 {
                ProgressView(value: progress) {
                    Text("Downloading \(name)")
                } currentValueLabel: {
                    Text("\(Int(progress * 100))%")
                }
                .padding(.horizontal, 32)
            } else {
                ProgressView("Downloading \(name)…")
            }
            Button("Cancel", role: .destructive) { vm.stop() }
        }
    }

    // MARK: - Conversation

    private var conversation: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 10) {
                        ForEach(vm.messages.filter { $0.role != .system }) { message in
                            bubble(message)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding()
                }
                .onChange(of: vm.messages) {
                    proxy.scrollTo("bottom")
                }
            }

            if !vm.statsLine.isEmpty {
                Text(vm.statsLine)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 2)
            }

            HStack(spacing: 8) {
                TextField("Message", text: $draft, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...4)
                if vm.phase == .generating {
                    Button {
                        vm.stop()
                    } label: {
                        Image(systemName: "stop.circle.fill").font(.title2)
                    }
                } else {
                    Button {
                        vm.send(draft)
                        draft = ""
                    } label: {
                        Image(systemName: "arrow.up.circle.fill").font(.title2)
                    }
                    .disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding([.horizontal, .bottom])
            .padding(.top, 6)
        }
    }

    private func bubble(_ message: ChatMessage) -> some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }
            Text(message.content.isEmpty ? "…" : message.content)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(
                    message.role == .user ? AnyShapeStyle(.tint) : AnyShapeStyle(.quaternary),
                    in: RoundedRectangle(cornerRadius: 16)
                )
                .foregroundStyle(message.role == .user ? .white : .primary)
                .frame(maxWidth: .infinity,
                       alignment: message.role == .user ? .trailing : .leading)
            if message.role == .assistant { Spacer(minLength: 40) }
        }
    }

    // MARK: - Settings

    private var settings: some View {
        NavigationStack {
            Form {
                Section("Sampling") {
                    VStack(alignment: .leading) {
                        Text("Temperature: \(vm.temperature, specifier: "%.2f")")
                        Slider(value: $vm.temperature, in: 0...1.5)
                    }
                }
                Section("Model") {
                    Text(vm.modelDescription)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Delete model & choose another", role: .destructive) {
                        showSettings = false
                        vm.deleteModelAndRestart()
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showSettings = false }
                }
            }
        }
    }
}

#Preview {
    ChatView()
}
