import torch
import transformers
import faster_whisper
from TTS.api import TTS
import numpy as np
import os
import asyncio
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import webrtcvad
from collections import deque
from pydub import AudioSegment

# --- FastAPI App Initialization ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Agent Configuration & Initialization ---
class RealTimeS2SAgent:
    def __init__(self):
        print("--- Initializing S2S Agent ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Using device: {self.device.upper()}")

        # VAD Initialization
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.chunk_size = int(self.sample_rate * self.frame_duration / 1000)

        # STT Model
        print("Loading STT model...")
        self.stt_model = faster_whisper.WhisperModel("distil-large-v3", device=self.device, compute_type="float16")
        print("STT model loaded.")

        # LLM
        print("Loading LLM...")
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": self.torch_dtype},
            device_map=self.device,
        )
        print("LLM loaded.")

        # TTS Model
        print("Loading TTS model...")
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print("TTS model loaded.")
        print("\n--- Agent is Ready ---")

    async def _send_json(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_json(data)
        except Exception:
            pass # Client might have disconnected

    async def _process_s2s_pipeline(self, websocket: WebSocket, audio_data: bytes):
        await self._send_json(websocket, {"type": "status", "data": "Transcribing..."})
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = await asyncio.to_thread(self.stt_model.transcribe, audio_np, beam_size=5)
        user_text = " ".join([s.text for s in segments])
        print(f"User: {user_text}")
        if not user_text.strip():
            await self._send_json(websocket, {"type": "status", "data": "Listening..."})
            return
        await self._send_json(websocket, {"type": "transcription_user", "data": user_text})

        await self._send_json(websocket, {"type": "status", "data": "Thinking..."})
        messages = [
            {"role": "system", "content": "You are a friendly conversational AI named Deva. Keep responses concise."},
            {"role": "user", "content": user_text},
        ]
        terminators = [
            self.llm_pipeline.tokenizer.eos_token_id,
            self.llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = await asyncio.to_thread(
            self.llm_pipeline, messages, max_new_tokens=256, eos_token_id=terminators, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
        )
        agent_response = outputs[0]["generated_text"][-1]['content']
        print(f"Agent: {agent_response}")
        await self._send_json(websocket, {"type": "transcription_agent", "data": agent_response})

        await self._send_json(websocket, {"type": "status", "data": "Speaking..."})
        chunks = self.tts_model.tts(text=agent_response, speaker="Claribel Dervla", language="en", stream=True)
        for chunk in chunks:
            encoded_chunk = base64.b64encode(chunk.cpu().numpy().tobytes()).decode('utf-8')
            await self._send_json(websocket, {"type": "audio_chunk", "data": encoded_chunk})
        await self._send_json(websocket, {"type": "status", "data": "Listening..."})

    async def handle_audio_stream(self, websocket: WebSocket):
        # VAD state variables
        silence_threshold_ms = 700
        padding_ms = 200
        num_padding_chunks = padding_ms // self.frame_duration
        num_silence_chunks = silence_threshold_ms // self.frame_duration
        voice_buffer = deque(maxlen=num_silence_chunks)
        padding_buffer = deque(maxlen=num_padding_chunks)
        voiced_frames = []
        is_speaking = False

        # --- THE FFMPEG PIPE ---
        # This command reads WebM/Opus from stdin and outputs raw 16kHz 16-bit mono PCM to stdout
        ffmpeg_command = [
            "ffmpeg",
            "-i", "pipe:0",      # Input from stdin
            "-f", "s16le",       # Format: signed 16-bit little-endian (raw PCM)
            "-ar", "16000",      # Audio sample rate: 16kHz
            "-ac", "1",          # Audio channels: 1 (mono)
            "pipe:1"             # Output to stdout
        ]
        
        # Create the subprocess
        ffmpeg_process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE  # Capture stderr to see ffmpeg logs
        )

        await self._send_json(websocket, {"type": "status", "data": "Listening..."})

        try:
            while True:
                # 1. Get audio from browser
                webm_chunk = await websocket.receive_bytes()

                # 2. Write it to ffmpeg's stdin
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.write(webm_chunk)
                    await ffmpeg_process.stdin.drain()

                # 3. Read the converted PCM data from ffmpeg's stdout
                # We expect a certain amount of data based on the input, but we'll read what's available
                while True:
                    pcm_chunk = await ffmpeg_process.stdout.read(self.chunk_size * 2) # Read one frame's worth
                    if not pcm_chunk:
                        break # No more data from ffmpeg for this input chunk

                    # 4. Process the clean PCM chunk with the VAD
                    if len(pcm_chunk) != self.chunk_size * 2:
                        continue # Skip incomplete frames

                    is_speech = self.vad.is_speech(pcm_chunk, self.sample_rate)
                    
                    if is_speech:
                        if not is_speaking:
                            is_speaking = True
                            voiced_frames.extend([f for f, _ in padding_buffer])
                        voiced_frames.append(pcm_chunk)
                        voice_buffer.clear()
                    else:
                        padding_buffer.append((pcm_chunk, is_speech))
                        if is_speaking:
                            voice_buffer.append(pcm_chunk)
                            if len(voice_buffer) >= num_silence_chunks:
                                is_speaking = False
                                audio_data = b''.join(voiced_frames)
                                asyncio.create_task(self._process_s2s_pipeline(websocket, audio_data))
                                voiced_frames.clear()
                                voice_buffer.clear()
                                padding_buffer.clear()
        
        except WebSocketDisconnect:
            print("Client disconnected.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print("Cleaning up ffmpeg process...")
            if ffmpeg_process.returncode is None:
                ffmpeg_process.kill()
            await ffmpeg_process.wait()
            print("ffmpeg process cleaned up.")

# Instantiate the agent once
agent = RealTimeS2SAgent()

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await agent.handle_audio_stream(websocket)
