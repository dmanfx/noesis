# Relevant Memories for WebRTC Implementation
_Status: current as of 2026-02-02._

This document summarizes key memories and constraints relevant to the WebRTC rollout plan.

## Performance & Architecture Constraints

- **GPU-only pipeline requirement**: The project's pipeline must decode RTSP streams on the GPU with DeepStream and then into inference models without any CPU usage or fallbacks. [[memory:2417767]]
- **Performance optimization preference**: User prefers performance optimization suggestions that focus on non-obvious improvements and maintaining full functionality (i.e., not disabling cameras or reducing JPEG quality). The user is not concerned about memory usage and would prefer more GPU usage to reduce CPU load. [[memory:8159538]]
- **Native Ubuntu deployment**: The project is now running in a native Ubuntu DeepStream deployment instead of the Docker container. All DeepStream components are installed natively on Ubuntu; any commands pertaining to DeepStream should be executed directly in the native environment. [[memory:2743119]]

## Development & Testing Preferences

- **RTSP testing preference**: The user prefers to test using only an RTSP stream from config.py, since testing with an .mp4 video file is not representative of the production environment. [[memory:3337297]]
- **Thorough testing requirement**: The user prefers that the assistant thoroughly tests and ensures everything works before concluding. [[memory:3333210]]
- **No virtual environments**: The user prefers not to use Python virtual environments (venv) when working on Python projects. [[memory:2977866]]

## Code Quality & Process

- **No placeholder code**: The user hates placeholder code and workarounds and prefers a clear plan before implementing any code modifications. They want step-by-step data collection and root cause analysis before the AI presents suggested changes, and require explicit confirmation before the AI executes any code fixes. Additionally, the user does not like hiding informational log messages and prefers that informational logs be visible. [[memory:2417792]]
- **Architectural change approval**: The user requests that the assistant not make architectural changes without their explicit permission and instead provide explanations, flow mapping, and performance comparisons before implementing changes. [[memory:2177169]]
- **Incremental changes**: The user prefers to make as few changes at a time and apply fixes incrementally to allow easy rollbacks. [[memory:190807]]
- **Concise responses**: The user prefers responses in as few words as possible and as honestly as possible. Specifically, after code patches or implementations, they prefer concise, minimal replies avoiding verbosity. [[memory:190796]]

## Documentation & Organization

- **Documentation preference**: The user prefers that important technical walk-throughs or documentation be saved as .MD files in the repository for future reference. [[memory:190791]]
- **Thorough debugging**: The user prefers a thorough, end-to-end review when debugging and wants all potential issues identified rather than stopping at the first problem. [[memory:190783]]

## Technology Stack

- **YOLO version preference**: The user will focus on YOLOv11/YOLOv12 for DeepStream custom parser integration and will not use YOLOv8. [[memory:3014743]]

## Communication Style

- **Objective responses**: The user prefers the assistant to be less sycophantic, more objective and technical in responses. [[memory:8159529]]

---

*These memories inform the WebRTC implementation approach, ensuring GPU-only paths, thorough testing with RTSP streams, incremental rollout, and comprehensive documentation.*
