import torch
import transformers
import faster_whisper
from TTS.api import TTS
import numpy as np
import os
import asyncio
import base64
import io

from fastapi import FastAPI, WebSocket
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
        self.frame_duration = 30 # ms
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

    def _convert_audio(self, webm_data: bytes) -> bytes:
        """Converts WebM/Opus audio data from browser to raw PCM for VAD."""
        try:
            audio = AudioSegment.from_file(io.BytesIO(webm_data), format="webm")
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
            return audio.raw_data
        except Exception as e:
            print(f"Pydub conversion error: {e}")
            return b''

    async def handle_audio_stream(self, websocket: WebSocket):
        silence_threshold_ms = 700
        padding_ms = 200
        num_padding_chunks = padding_ms // self.frame_duration
        num_silence_chunks = silence_threshold_ms // self.frame_duration

        voice_buffer = deque(maxlen=num_silence_chunks)
        padding_buffer = deque(maxlen=num_padding_chunks)
        
        voiced_frames = []
        is_speaking = False

        await self._send_json(websocket, {"type": "status", "data": "Listening..."})

        while True:
            try:
                webm_data = await websocket.receive_bytes()
                
                # **THE CRITICAL FIX IS HERE**
                # Convert the incoming webm audio to raw PCM before processing
                pcm_data = await asyncio.to_thread(self._convert_audio, webm_data)

                if not pcm_data:
                    continue

                # Now, process the raw PCM data in chunks for the VAD
                for i in range(0, len(pcm_data), self.chunk_size * 2): # 2 bytes per sample
                    chunk = pcm_data[i:i + self.chunk_size * 2]
                    if len(chunk) != self.chunk_size * 2:
                        continue # Skip incomplete chunks

                    is_speech = self.vad.is_speech(chunk, self.sample_rate)
                    
                    if is_speech:
                        if not is_speaking:
                            is_speaking = True
                            voiced_frames.extend([f for f, _ in padding_buffer])
                        voiced_frames.append(chunk)
                        voice_buffer.clear()
                    else:
                        padding_buffer.append((chunk, is_speech))
                        if is_speaking:
                            voice_buffer.append(chunk)
                            if len(voice_buffer) >= num_silence_chunks:
                                is_speaking = False
                                audio_data = b''.join(voiced_frames)
                                asyncio.create_task(self._process_s2s_pipeline(websocket, audio_data))
                                voiced_frames.clear()
                                voice_buffer.clear()
                                padding_buffer.clear()
            except Exception as e:
                print(f"Connection closed or error: {e}")
                break

agent = RealTimeS2SAgent()

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await agent.handle_audio_stream(websocket)
