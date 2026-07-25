// swift-tools-version: 5.9
// SwiftPM package wrapping the TinyLLM inference engine so it can be built
// and numerically tested on Linux/CI without Xcode. The iOS apps themselves
// are built from the project.yml specs (XcodeGen) — see .github/workflows.

import PackageDescription

let package = Package(
    name: "llm-ios",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [
        .library(name: "TinyLLMEngine", targets: ["TinyLLMEngine"]),
    ],
    targets: [
        .target(
            name: "TinyLLMEngine",
            path: "ios",
            sources: ["TinyLLM/Engine", "Shared"]  // app/UI files are built by Xcode, not SwiftPM
        ),
        .testTarget(
            name: "TinyLLMEngineTests",
            dependencies: ["TinyLLMEngine"],
            path: "swift-tests"
        ),
    ]
)
