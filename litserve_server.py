"""
LitServe adapter for MiniCPM-o 4.5 Full-Duplex Demo
Enables deployment to Lightning AI Studio/Lightning Cloud with one command:
    lightning deploy litserve_server.py --cloud
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List, Iterator
from datetime import datetime

import numpy as np
import torch
import litserve as ls
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Import MiniCPM-o 4.5 components
import sys
sys.path.append('.')

from core.processors.unified import UnifiedProcessor
from core.schemas import (
    ChatRequest, ChatResponse,
    StreamingRequest, StreamingChunk,
    DuplexConfig, DuplexGenerateResult,
    Message, Role
)
from session_recorder import (
    DuplexSessionRecorder,
    TurnBasedSessionRecorder,
    generate_session_id
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("litserve_minicpmo")

# Global state for session management (similar to Gateway's worker pooling)
class SessionManager:
    """Manages KV cache sessions for streaming modes."""
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def create_session(self, session_id: str, mode: str, worker_id: int = 0) -> Dict:
        """Create a new session entry."""
        self.sessions[session_id] = {
            "session_id": session_id,
            "mode": mode,
            "worker_id": worker_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "kv_cache_length": 0
        }
        return self.sessions[session_id]

    def update_session(self, session_id: str, **kwargs):
        """Update session attributes."""
        if session_id in self.sessions:
            self.sessions[session_id].update(kwargs)
            self.sessions[session_id]["last_activity"] = datetime.now()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        """Remove session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            sid for sid, sess in self.sessions.items()
            if sess["last_activity"] < cutoff
        ]
        for sid in to_remove:
            self.remove_session(sid)

session_manager = SessionManager()


class MiniCPMOChatAPI(ls.LitAPI):
    """LitAPI for Chat mode (turn-based, stateless)."""

    def setup(self, device: str):
        """Load the model once per worker."""
        from config import get_config
        cfg = get_config()

        self.processor = UnifiedProcessor(
            model_path=cfg.model.model_path,
            pt_path=cfg.model.pt_path,
            ref_audio_path=cfg.audio.ref_audio_path,
            device=device,
            compile=cfg.service.compile,
            chat_vocoder=cfg.audio.chat_vocoder,
            attn_implementation=cfg.model.attn_implementation,
        )
        logger.info("MiniCPM-o Chat API initialized")

    def decode_request(self, request: Dict[str, Any]) -> ChatRequest:
        """Convert HTTP request to ChatRequest."""
        # Handle both direct schema and raw format
        if "messages" in request:
            # Convert raw messages to schema Messages
            messages = []
            for msg in request["messages"]:
                role = Role(msg["role"])
                content = msg["content"]
                if isinstance(content, list):
                    content_items = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and item.get("text"):
                                content_items.append(TextContent(text=item["text"]))
                            elif item.get("type") == "audio" and item.get("data"):
                                content_items.append(AudioContent(data=item["data"]))
                            elif item.get("type") == "image" and item.get("data"):
                                content_items.append(ImageContent(data=item["data"]))
                            elif item.get("type") == "video" and item.get("data"):
                                content_items.append(VideoContent(
                                    data=item["data"],
                                    stack_frames=item.get("stack_frames", 1),
                                ))
                    if content_items:
                        messages.append(Message(role=role, content=content_items))
                else:
                    messages.append(Message(role=role, content=content))

            # Build ChatRequest
            chat_request = ChatRequest(
                messages=messages,
                generation=request.get("generation", {}),
                tts=request.get("tts", {"enabled": False}),
                use_tts_template=request.get("use_tts_template", False),
                omni_mode=request.get("omni_mode", False),
                enable_thinking=request.get("enable_thinking", False),
                return_prompt=request.get("return_prompt", False)
            )

            # Handle image info if present
            if "image" in request:
                chat_request.image = request["image"]

            return chat_request
        else:
            # Assume it's already a proper ChatRequest dict
            return ChatRequest(**request)

    def predict(self, request: ChatRequest) -> ChatResponse:
        """Run chat inference (non-streaming)."""
        # Switch to chat mode
        chat_view = self.processor.set_chat_mode()

        # Run inference
        response = chat_view.chat(
            request,
            max_new_tokens=request.generation.max_new_tokens,
            do_sample=request.generation.do_sample,
            generate_audio=request.tts.enabled if request.tts else False,
        )

        return response

    def encode_response(self, response: ChatResponse) -> Dict[str, Any]:
        """Convert ChatResponse to HTTP response."""
        return {
            "text": response.text,
            "audio_data": response.audio_data,
            "audio_path": response.audio_path,
            "audio_sample_rate": response.audio_sample_rate,
            "duration_ms": response.duration_ms,
            "prompt": response.prompt,
            "tokens_generated": response.tokens_generated,
            "token_stats": response.token_stats,
            "success": response.success,
            "error": response.error,
            "recording_session_id": getattr(response, 'recording_session_id', None)
        }


class MiniCPMOHalfDuplexAPI(ls.LitAPI):
    """LitAPI for Half-Duplex Audio mode (stateful, streaming)."""

    def setup(self, device: str):
        """Load the model once per worker."""
        from config import get_config
        cfg = get_config()

        self.processor = UnifiedProcessor(
            model_path=cfg.model.model_path,
            pt_path=cfg.model.pt_path,
            ref_audio_path=cfg.audio.ref_audio_path,
            device=device,
            compile=cfg.service.compile,
            chat_vocoder=cfg.audio.chat_vocoder,
            attn_implementation=cfg.model.attn_implementation,
        )
        logger.info("MiniCPM-o Half-Duplex API initialized")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Store request for predict() - we need session_id for statefulness."""
        return request

    def predict(self, request: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response for half-duplex mode."""
        session_id = request.get("session_id")
        if not session_id:
            session_id = f"hdx_{uuid.uuid4().hex[:8]}"

        # Ensure session exists
        session = session_manager.get_session(session_id)
        if not session:
            session_manager.create_session(session_id, "half_duplex")
            session_manager.update_session(session_id, status="initialized")

        # Switch to half-duplex mode
        half_duplex_view = self.processor.set_half_duplex_mode()

        # Handle TTS initialization if needed
        generate_audio = request.get("generate_audio", True)
        if generate_audio:
            # Initialize TTS with reference audio if provided
            ref_audio_b64 = request.get("ref_audio_base64")
            if ref_audio_b64:
                try:
                    audio_bytes = base64.b64decode(ref_audio_b64)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    half_duplex_view.init_ref_audio_from_data(audio_np)
                except Exception as e:
                    logger.warning(f"Failed to init TTS from base64: {e}")
                    half_duplex_view.init_ref_audio(self.processor.ref_audio_path)
            else:
                half_duplex_view.init_ref_audio(self.processor.ref_audio_path)

        # Handle prefill if messages provided
        if "messages" in request:
            # Convert messages to schema format (similar to ChatAPI)
            messages = []
            for msg in request["messages"]:
                role = Role(msg["role"])
                content = msg["content"]
                if isinstance(content, list):
                    content_items = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text" and item.get("text"):
                                content_items.append(TextContent(text=item["text"]))
                            elif item.get("type") == "audio" and item.get("data"):
                                content_items.append(AudioContent(data=item["data"]))
                            elif item.get("type") == "image" and item.get("data"):
                                content_items.append(ImageContent(data=item["data"]))
                            elif item.get("type") == "video" and item.get("data"):
                                content_items.append(VideoContent(
                                    data=item["data"],
                                    stack_frames=item.get("stack_frames", 1),
                                ))
                    if content_items:
                        messages.append(Message(role=role, content=content_items))
                else:
                    messages.append(Message(role=role, content=content))

            # Create streaming request
            streaming_req = StreamingRequest(
                session_id=session_id,
                messages=messages,
                is_last_chunk=request.get("is_last_chunk", True),
                use_tts_template=request.get("use_tts_template", True),
                omni_mode=request.get("omni_mode", False),
                enable_thinking=request.get("enable_thinking", False)
            )

            # Add image info if present
            if "image" in request:
                streaming_req.image = request["image"]

            # Prefill
            try:
                prompt = half_duplex_view.prefill(streaming_req)
                session_manager.update_session(session_id,
                                             prefill_prompt=prompt,
                                             status="prefilled")
                yield {"type": "prefill_done", "input_tokens": len(prompt.split())}  # Approximate
            except Exception as e:
                logger.error(f"Prefill failed: {e}")
                yield {"type": "error", "error": str(e)}
                return

        # Stream generation
        try:
            chunk_index = 0
            full_text = ""

            for chunk in half_duplex_view.generate(
                session_id=session_id,
                generate_audio=generate_audio,
                max_new_tokens=request.get("max_new_tokens", 256),
                do_sample=request.get("do_sample", True),
                length_penalty=request.get("length_penalty", 1.1),
            ):
                # Update session KV cache length
                kv_len = self.processor.kv_cache_length
                session_manager.update_session(session_id, kv_cache_length=kv_len)

                # Prepare chunk response
                chunk_data = {
                    "type": "chunk",
                    "chunk_index": chunk_index,
                    "text_delta": chunk.text_delta,
                    "audio_data": chunk.audio_data,
                    "audio_sample_rate": chunk.audio_sample_rate,
                    "is_final": chunk.is_final,
                    "duration_ms": getattr(chunk, 'duration_ms', 0),
                    "kv_cache_length": kv_len
                }

                if chunk.text_delta:
                    full_text += chunk.text_delta

                yield chunk_data
                chunk_index += 1

                if chunk.is_final:
                    break

            # Final response
            yield {
                "type": "done",
                "session_id": session_id,
                "full_text": full_text,
                "total_chunks": chunk_index,
                "final_kv_cache_length": self.processor.kv_cache_length
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            yield {"type": "error", "error": str(e)}
        finally:
            # Clean up session if needed (optional - keep for continuity)
            pass

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through for streaming responses."""
        return output


class MiniCPMODuplexAPI(ls.LitAPI):
    """LitAPI for Duplex mode (full-duplex, bidirectional streaming)."""

    def setup(self, device: str):
        """Load the model once per worker."""
        from config import get_config
        cfg = get_config()

        self.processor = UnifiedProcessor(
            model_path=cfg.model.model_path,
            pt_path=cfg.model.pt_path,
            ref_audio_path=cfg.audio.ref_audio_path,
            device=device,
            compile=cfg.service.compile,
            chat_vocoder=cfg.audio.chat_vocoder,
            attn_implementation=cfg.model.attn_implementation,
        )
        logger.info("MiniCPM-o Duplex API initialized")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Store request for predict()."""
        return request

    def predict(self, request: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle duplex mode - this is complex due to bidirectional nature.

        For LitServe, we'll implement a unidirectional audio-to-(text+audio)
        streaming endpoint. For true bidirectional, WebSockets are better,
        but this provides HTTP streaming compatibility.
        """
        session_id = request.get("session_id")
        if not session_id:
            session_id = f"dux_{uuid.uuid4().hex[:8]}"

        # Extract parameters
        system_prompt = request.get("system_prompt", "You are a helpful assistant.")
        ref_audio_b64 = request.get("ref_audio_base64")
        audio_chunks = request.get("audio_chunks", [])  # List of base64 audio chunks
        generate_audio = request.get("generate_audio", True)
        max_slice_nums = request.get("max_slice_nums", 1)
        use_deferred_finalize = request.get("deferred_finalize", True)

        # Initialize or get session
        session = session_manager.get_session(session_id)
        if not session:
            session_manager.create_session(session_id, "duplex")
            session_manager.update_session(session_id, status="initialized")

        try:
            # Prepare duplex session
            duplex_view = self.processor.set_duplex_mode()

            # Handle reference audio
            import tempfile
            import soundfile as sf
            import numpy as np

            actual_ref_audio_path = None
            actual_tts_audio_path = None
            temp_files = []

            # LLM ref audio
            if ref_audio_b64:
                try:
                    ref_bytes = base64.b64decode(ref_audio_b64)
                    ref_ndarray = np.frombuffer(ref_bytes, dtype=np.float32)
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="dux_llm_ref_")
                    sf.write(tmp.name, ref_ndarray, 16000)
                    actual_ref_audio_path = tmp.name
                    temp_files.append(tmp.name)
                except Exception as e:
                    logger.warning(f"Failed to process ref audio: {e}")
                    if self.processor.ref_audio_path:
                        actual_ref_audio_path = self.processor.ref_audio_path

            # TTS ref audio (can be different)
            tts_ref_audio_b64 = request.get("tts_ref_audio_base64", ref_audio_b64)
            if tts_ref_audio_b64 and tts_ref_audio_b64 != ref_audio_b64:
                try:
                    tts_bytes = base64.b64decode(tts_ref_audio_b64)
                    tts_ndarray = np.frombuffer(tts_bytes, dtype=np.float32)
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="dux_tts_ref_")
                    sf.write(tmp.name, tts_ndarray, 16000)
                    actual_tts_audio_path = tmp.name
                    temp_files.append(tmp.name)
                except Exception as e:
                    logger.warning(f"Failed to process TTS ref audio: {e}")
                    if actual_ref_audio_path:
                        actual_tts_audio_path = actual_ref_audio_path
                    elif self.processor.ref_audio_path:
                        actual_tts_audio_path = self.processor.ref_audio_path
            else:
                actual_tts_audio_path = actual_ref_audio_path or self.processor.ref_audio_path

            # Prepare the session
            prompt = duplex_view.prepare(
                system_prompt_text=system_prompt,
                ref_audio_path=actual_ref_audio_path,
                prompt_wav_path=actual_tts_audio_path
            )

            session_manager.update_session(session_id,
                                         prompt=prompt,
                                         status="prepared")

            yield {"type": "prepared", "prompt_length": len(prompt)}

            # Process audio chunks if provided
            if audio_chunks:
                chunk_index = 0

                for audio_b64 in audio_chunks:
                    try:
                        # Decode audio
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_waveform = np.frombuffer(audio_bytes, dtype=np.float32)

                        # Prefill
                        prefill_result = duplex_view.prefill(
                            audio_waveform=audio_waveform,
                            max_slice_nums=max_slice_nums
                        )

                        # Generate
                        result = duplex_view.generate()

                        # Convert audio to base64 if present
                        audio_data = None
                        if result.get("audio_waveform") is not None:
                            waveform = result["audio_waveform"]
                            if isinstance(waveform, torch.Tensor):
                                waveform = waveform.cpu().numpy()
                            audio_bytes_out = waveform.astype(np.float32).tobytes()
                            audio_data = base64.b64encode(audio_bytes_out).decode('utf-8')

                        # Prepare response
                        response_data = {
                            "type": "result",
                            "chunk_index": chunk_index,
                            "is_listen": result.get("is_listen", True),
                            "text": result.get("text", ""),
                            "audio_data": audio_data,
                            "end_of_turn": result.get("end_of_turn", False),
                            "current_time": result.get("current_time", 0),
                            "kv_cache_length": self.processor.kv_cache_length
                        }

                        # Add timing info if available
                        if result.get("cost_all_ms"):
                            response_data["wall_clock_ms"] = result["cost_all_ms"]

                        yield response_data

                        # Finalize if using deferred mode
                        if use_deferred_finalize:
                            duplex_view.finalize()

                        chunk_index += 1

                        # Break if end of turn
                        if result.get("end_of_turn", False):
                            break

                    except Exception as e:
                        logger.error(f"Error processing audio chunk {chunk_index}: {e}")
                        yield {"type": "error", "error": str(e), "chunk_index": chunk_index}
                        break

            # Cleanup temp files
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"Duplex predict failed: {e}")
            yield {"type": "error", "error": str(e)}
        finally:
            # Clean up session (optional)
            pass

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through for streaming responses."""
        return output


def create_litserve_app():
    """Create and configure the LitServe server with all APIs."""
    # Create APIs
    chat_api = MiniCPMOChatAPI(stream=False)
    half_duplex_api = MiniCPMOHalfDuplexAPI(stream=True)
    duplex_api = MiniCPMODuplexAPI(stream=True)

    # Create LitServe server
    server = ls.LitServer(
        api=chat_api,  # Default API
        accelerator="auto",
        device_ids=[0],  # Will be overridden by Lightning Cloud
        batch_size=1,
        max_batch_size=1,
        timeout=30,
        stream=False,
        api_path="/predict",
    )

    # Note: For true multi-api support, we'd need to mount different APIs
    # on different routes, but LitServe's current API is designed for
    # single-model serving. We'll use the ChatAPI as default and
    # extend with custom routes if needed.

    return server


# Custom FastAPI app for additional endpoints (static files, health, etc.)
def create_custom_app():
    """Create a custom FastAPI app that includes LitServe and additional endpoints."""
    from fastapi import FastAPI
    import uvicorn

    # Create LitServe server
    lit_server = create_litserve_app()

    # Create FastAPI app
    app = FastAPI(title="MiniCPM-o 4.5 LitServe Server")

    # Mount LitServe app
    app.mount("/lit", lit_server.app)

    # Health endpoint
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "minicpmo-litserve"}

    # Serve static files (frontend)
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Root endpoint
    @app.get("/")
    async def root():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "MiniCPM-o 4.5 LitServe Server", "docs": "/docs"}

    # API info endpoint
    @app.get("/api/info")
    async def api_info():
        return {
            "model": "MiniCPM-o 4.5",
            "modes": ["chat", "half_duplex", "duplex"],
            "endpoints": {
                "chat": "/lit/predict",
                "half_duplex": "/lit/predict",  # Use same endpoint with different params
                "duplex": "/lit/predict"
            }
        }

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 LitServe Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--mode", type=str, default="chat",
                       choices=["chat", "half_duplex", "duplex"],
                       help="Default mode to operate in")
    args = parser.parse_args()

    # Create and run server
    app = create_custom_app()

    logger.info(f"Starting MiniCPM-o 4.5 LitServe server on {args.host}:{args.port}")
    logger.info(f"Default mode: {args.mode}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")