# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow

### Environment Setup
```bash
# Install dependencies
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

### Running the Application
The system can be run in different modes:

#### Gateway Service (Main Entry Point)
```bash
python gateway.py --port 10024 --workers localhost:22400,localhost:22401
```

#### LitServe Deployment (for Lightning AI)
```bash
# Local development
python litserve_server.py --port 8000

# Deployment to Lightning Cloud
lightning deploy litserve_server.py --cloud
```

#### Direct Worker Mode
```bash
python worker.py --port 22400 --model-path openbmb/MiniCPM-o-4_5
```

### Testing
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_chat.py
pytest tests/test_duplex.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=core --cov=gateway tests/
```

### Code Quality
```bash
# Format code
black .

# Linting
flake8 .

# Import sorting
isort .
```

## Code Architecture

### High-Level Structure
```
MiniCPM-o-Demo/
├── core/                 # Core model processing logic
│   ├── processors/       # Mode-specific processors (Chat, Streaming, Duplex)
│   ├── schemas/          # Data models and request/response types
│   ├── capabilities.py   # Feature flags for different modes
│   └── factory.py        # Processor creation factory
├── gateway.py            # Request routing and session management
├── litserve_server.py    # Lightning AI deployment adapter
├── worker.py             # Individual worker process
├── docs-app/             # Documentation website (Next.js)
└── assets/               # Configuration presets and reference files
```

### Key Components

#### 1. Core Processing (`core/`)
- **UnifiedProcessor**: Single model instance supporting all modes with <1ms switching
- **Processors**: 
  - `ChatView`: Standard request/response mode
  - `HalfDuplexView`: Voice activity detection based audio chat
  - `DuplexView`: Real-time full-duplex audio/video interaction
- **Schemas**: Type-safe data models for all API endpoints
- **Capabilities**: Feature flags indicating supported features per mode

#### 2. Service Layer
- **gateway.py**: 
  - Routes requests to appropriate workers
  - Manages session-to-worker mapping and KV cache LRU
  - Implements FIFO queueing with capacity 1000 and ETA estimation
  - Performs worker health checks
- **worker.py**: 
  - Loads MiniCPM-o 4.5 model
  - Handles actual inference processing
  - Communicates with gateway via TCP
- **litserve_server.py**: 
  - Adapter for Lightning AI platform
  - Provides HTTP/WebSocket endpoints
  - Serves static frontend assets

#### 3. Configuration
- `config.example.json`: Template configuration with defaults
- `config.json`: Active configuration (gitignored)
- Modes configurable via system prompts and reference audio

### Operation Modes
The system supports four interchangeable modes with millisecond-level switching:

1. **Turn-based Chat**: Standard Q&A with button-triggered responses
2. **Half-Duplex Audio**: VAD-activated hands-free voice conversation
3. **Omnimodal Full-Duplex**: Concurrent vision/audio input with text/voice output
4. **Audio Full-Duplex**: Simultaneous voice input and output

### Important Implementation Notes
- Model loads once in `UnifiedProcessor` and is shared across all modes
- TTS mode requires special handling: `mode="default"` ignores `ref_audio`; must use `AUDIO_ASSISTANT`
- Duplex mode requires specific system prompt token format to avoid garbled output
- Audio format: 16kHz mono input, 24kHz output
- Force listen count in duplex mode provides startup protection (typically fixed at 3)

### Common Debug Commands
```bash
# Check worker status
curl http://localhost:10024/workers

# View session stats
curl http://localhost:10024/sessions

# Test API endpoint
curl -X POST http://localhost:10024/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'
```