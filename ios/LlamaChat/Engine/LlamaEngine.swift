// Swift wrapper around llama.cpp for on-device chat with pretrained models.
//
// Follows the same design as llama.cpp's server: each turn the full
// conversation is formatted with the model's chat template and tokenized,
// the longest common token prefix with what's already in the KV cache is
// kept, any divergent tail is dropped (llama_memory_seq_rm), and only the
// new suffix is decoded. Long chats therefore never re-prefill from scratch,
// and the cache state is self-correcting.
//
// Requires llama.xcframework (see ios/LlamaChat/README.md). Written against
// the llama.cpp API as vendored in llama-cpp-python 0.3.34.

import Foundation
import llama

enum LlamaError: Error, LocalizedError {
    case modelLoadFailed(String)
    case contextCreationFailed
    case tokenizationFailed
    case templateFailed
    case decodeFailed(Int32)
    case contextFull

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let path): return "Could not load model at \(path)"
        case .contextCreationFailed: return "Could not create llama context"
        case .tokenizationFailed: return "Tokenization failed"
        case .templateFailed: return "Chat template failed"
        case .decodeFailed(let code): return "llama_decode failed (\(code))"
        case .contextFull: return "Context window is full — start a new chat"
        }
    }
}

struct ChatMessage: Identifiable, Equatable {
    enum Role: String { case system, user, assistant }
    let id = UUID()
    let role: Role
    var content: String
}

struct GenerationStats {
    var prefillTokens = 0
    var prefillSeconds = 0.0
    var generatedTokens = 0
    var generateSeconds = 0.0
    var tokensPerSecond: Double {
        generateSeconds > 0 ? Double(generatedTokens) / generateSeconds : 0
    }
}

// @unchecked Sendable: the view model serializes all access — one generation
// runs at a time, and all engine calls happen on a single background task.
final class LlamaEngine: @unchecked Sendable {
    private let model: OpaquePointer
    private let ctx: OpaquePointer
    private let vocab: OpaquePointer
    private let nCtx: Int
    private let nBatch: Int

    /// Tokens currently represented in the KV cache, in order. Position i in
    /// the cache always holds kvTokens[i] (positions are auto-tracked).
    private var kvTokens: [llama_token] = []

    private(set) var modelDescription = ""
    private(set) var lastStats = GenerationStats()

    init(modelPath: String, contextLength: Int = 4096) throws {
        llama_backend_init()

        var mparams = llama_model_default_params()
        #if targetEnvironment(simulator)
        mparams.n_gpu_layers = 0          // no Metal in the simulator
        #else
        mparams.n_gpu_layers = 99         // offload everything to the GPU
        #endif

        guard let model = llama_model_load_from_file(modelPath, mparams) else {
            throw LlamaError.modelLoadFailed(modelPath)
        }
        self.model = model
        self.vocab = llama_model_get_vocab(model)

        var cparams = llama_context_default_params()
        cparams.n_ctx = UInt32(contextLength)
        cparams.n_batch = 512
        let threads = Int32(max(2, min(8, ProcessInfo.processInfo.processorCount - 2)))
        cparams.n_threads = threads
        cparams.n_threads_batch = threads

        guard let ctx = llama_init_from_model(model, cparams) else {
            llama_model_free(model)
            throw LlamaError.contextCreationFailed
        }
        self.ctx = ctx
        self.nCtx = Int(llama_n_ctx(ctx))
        self.nBatch = Int(llama_n_batch(ctx))

        var buf = [CChar](repeating: 0, count: 256)
        if llama_model_desc(model, &buf, buf.count) > 0 {
            modelDescription = String(cString: buf)
        }
    }

    deinit {
        llama_free(ctx)
        llama_model_free(model)
    }

    /// Wipe the KV cache to start a fresh conversation.
    func resetConversation() {
        llama_memory_clear(llama_get_memory(ctx), true)
        kvTokens.removeAll()
    }

    /// Generate the assistant reply for `messages` (the full conversation so
    /// far). Yields text pieces to `onToken` as they are generated; returns
    /// the complete reply.
    func generate(messages: [ChatMessage],
                  temperature: Float,
                  maxTokens: Int,
                  onToken: (String) -> Void) throws -> String {
        // 1. format + tokenize the full conversation, with the assistant header
        let formatted = try applyChatTemplate(messages: messages, addAssistantHeader: true)
        let target = try tokenize(formatted, addSpecial: true)
        guard target.count < nCtx - 8 else { throw LlamaError.contextFull }

        // 2. reuse the longest common prefix already in the KV cache
        var common = 0
        while common < min(kvTokens.count, target.count), kvTokens[common] == target[common] {
            common += 1
        }
        // always leave at least one suffix token to decode, so the sampler
        // reads fresh logits (matters when regenerating the same prompt)
        if common == target.count { common -= 1 }

        if common < kvTokens.count {
            let removed = llama_memory_seq_rm(llama_get_memory(ctx), 0,
                                              llama_pos(common), -1)
            if removed {
                kvTokens.removeLast(kvTokens.count - common)
            } else {
                // some cache types can't drop a tail; rebuild from scratch
                resetConversation()
                common = 0
            }
        }

        var suffix = Array(target[common...])
        var stats = GenerationStats()
        stats.prefillTokens = suffix.count

        // 3. prefill the new suffix in n_batch chunks
        let prefillStart = Date()
        var offset = 0
        while offset < suffix.count {
            let n = min(nBatch, suffix.count - offset)
            try suffix.withUnsafeMutableBufferPointer { buf in
                try decodeChecked(llama_batch_get_one(buf.baseAddress! + offset, Int32(n)))
            }
            kvTokens.append(contentsOf: suffix[offset..<(offset + n)])
            offset += n
        }
        stats.prefillSeconds = Date().timeIntervalSince(prefillStart)

        // 4. sample until end-of-generation
        let sampler = makeSampler(temperature: temperature)
        defer { llama_sampler_free(sampler) }

        var decoder = UTF8PieceDecoder()
        var response = ""
        let genStart = Date()
        for _ in 0..<maxTokens {
            if Task.isCancelled { break }
            let token = llama_sampler_sample(sampler, ctx, -1)
            if llama_vocab_is_eog(vocab, token) { break }

            if let text = decoder.feed(piece(of: token)) {
                response += text
                onToken(text)
            }
            stats.generatedTokens += 1

            var t = token
            try withUnsafeMutablePointer(to: &t) { p in
                try decodeChecked(llama_batch_get_one(p, 1))
            }
            kvTokens.append(token)
        }
        stats.generateSeconds = Date().timeIntervalSince(genStart)
        lastStats = stats
        return response
    }

    // MARK: - Internals

    private func decodeChecked(_ batch: llama_batch) throws {
        let used = Int(llama_memory_seq_pos_max(llama_get_memory(ctx), 0)) + 1
        guard used + Int(batch.n_tokens) <= nCtx else { throw LlamaError.contextFull }
        let ret = llama_decode(ctx, batch)
        guard ret == 0 else { throw LlamaError.decodeFailed(ret) }
    }

    private func makeSampler(temperature: Float) -> OpaquePointer {
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())
        if temperature <= 0.05 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05, 1))
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: .min ... .max)))
        }
        return chain!
    }

    private func tokenize(_ text: String, addSpecial: Bool) throws -> [llama_token] {
        let utf8Count = Int32(text.utf8.count)
        let needed = -llama_tokenize(vocab, text, utf8Count, nil, 0, addSpecial, true)
        guard needed > 0 else { throw LlamaError.tokenizationFailed }
        var tokens = [llama_token](repeating: 0, count: Int(needed))
        let written = llama_tokenize(vocab, text, utf8Count, &tokens, needed, addSpecial, true)
        guard written >= 0 else { throw LlamaError.tokenizationFailed }
        return Array(tokens[0..<Int(written)])
    }

    private func piece(of token: llama_token) -> [UInt8] {
        var buf = [CChar](repeating: 0, count: 256)
        let n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
        guard n > 0 else { return [] }
        return buf[0..<Int(n)].map { UInt8(bitPattern: $0) }
    }

    private func applyChatTemplate(messages: [ChatMessage], addAssistantHeader: Bool) throws -> String {
        // nil name -> the model's embedded default template; models without
        // one fall back to ChatML, the most common scheme
        let tmpl: UnsafePointer<CChar>? = llama_model_chat_template(model, nil) ?? chatmlFallback

        var cMessages: [llama_chat_message] = messages.map {
            llama_chat_message(role: strdup($0.role.rawValue), content: strdup($0.content))
        }
        defer {
            for m in cMessages {
                free(UnsafeMutablePointer(mutating: m.role))
                free(UnsafeMutablePointer(mutating: m.content))
            }
        }

        var size = max(1024, messages.reduce(0) { $0 + $1.content.utf8.count } * 2)
        while true {
            var buf = [CChar](repeating: 0, count: size)
            let n = llama_chat_apply_template(tmpl, &cMessages, cMessages.count,
                                              addAssistantHeader, &buf, Int32(size))
            guard n >= 0 else { throw LlamaError.templateFailed }
            if Int(n) <= size {
                return String(decoding: buf[0..<Int(n)].map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
            size = Int(n) + 1
        }
    }

    private let chatmlFallback: UnsafePointer<CChar> = UnsafePointer(strdup("chatml"))
}

// UTF8PieceDecoder (streaming decode of token pieces) lives in
// ios/Shared/UTF8PieceDecoder.swift so the SwiftPM tests cover it.
