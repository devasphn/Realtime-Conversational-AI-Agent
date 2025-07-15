import torch
import transformers
import faster_whisper
from TTS.api import TTS
import numpy as np
import os
import asyncio
import base64

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import webrtcvad
from collections import deque

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
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
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
        await websocket.send_json(data)

    async def _process_s2s_pipeline(self, websocket: WebSocket, audio_data: bytes):
        # 1. Transcribe Audio
        await self._send_json(websocket, {"type": "status", "data": "Transcribing..."})
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = await asyncio.to_thread(self.stt_model.transcribe, audio_np, beam_size=5)
        user_text = " ".join([s.text for s in segments])
        print(f"User: {user_text}")
        if not user_text.strip():
            await self._send_json(websocket, {"type": "status", "data": "Listening..."})
            return
        await self._send_json(websocket, {"type": "transcription_user", "data": user_text})

        # 2. Generate LLM Response
        await self._send_json(websocket, {"type": "status", "data": "Thinking..."})
        messages = [
            {"role": "system", "content": "You are a friendly and helpful conversational AI. Your name is Deva. Keep your responses concise and to the point."},
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

        # 3. Stream TTS Audio
        await self._send_json(websocket, {"type": "status", "data": "Speaking..."})
        # Use the streaming feature of the TTS model
        chunks = self.tts_model.tts(text=agent_response, speaker="Claribel Dervla", language="en", stream=True)
        for chunk in chunks:
            encoded_chunk = base64.b64encode(chunk.cpu().numpy().tobytes()).decode('utf-8')
            await self._send_json(websocket, {"type": "audio_chunk", "data": encoded_chunk})
        await self._send_json(websocket, {"type": "status", "data": "Listening..."})

    async def handle_audio_stream(self, websocket: WebSocket):
        # VAD state
        silence_threshold_ms = 700  # 0.7 seconds of silence
        padding_ms = 200 # Add a bit of audio padding
        num_padding_chunks = padding_ms // self.frame_duration
        num_silence_chunks = silence_threshold_ms // self.frame_duration

        voice_buffer = deque(maxlen=num_silence_chunks)
        padding_buffer = deque(maxlen=num_padding_chunks)
        
        voiced_frames = []
        is_speaking = False

        await self._send_json(websocket, {"type": "status", "data": "Listening..."})

        while True:
            try:
                data = await websocket.receive_bytes()
                is_speech = self.vad.is_speech(data, self.sample_rate)

                if is_speech:
                    if not is_speaking:
                        is_speaking = True
                        # Add padding frames to the start of speech
                        voiced_frames.extend([f for f, _ in padding_buffer])
                    voiced_frames.append(data)
                    voice_buffer.clear()
                else:
                    padding_buffer.append((data, is_speech))
                    if is_speaking:
                        voice_buffer.append(data)
                        if len(voice_buffer) >= num_silence_chunks:
                            # End of speech detected
                            is_speaking = False
                            audio_data = b''.join(voiced_frames)
                            asyncio.create_task(self._process_s2s_pipeline(websocket, audio_data))
                            # Reset buffers
                            voiced_frames.clear()
                            voice_buffer.clear()
                            padding_buffer.clear()
            except Exception as e:
                print(f"Connection closed or error: {e}")
                break

# Instantiate agent
agent = RealTimeS2SAgent()

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await agent.handle_audio_stream(websocket)
