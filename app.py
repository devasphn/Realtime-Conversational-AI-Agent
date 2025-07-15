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
        self.frame_duration = 30
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
            pass

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
        
        # 1. Generate the entire audio clip as a list of floats
        wav_chunks = self.tts_model.tts(text=agent_response, speaker="Claribel Dervla", language="en")
        
        # 2. Convert float audio to 16-bit PCM bytes, the standard WAV format
        audio_array = np.array(wav_chunks)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # 3. Stream the audio to the client, framed by start/end signals for WAV assembly
        await self._send_json(websocket, {"type": "audio_start"})
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            encoded_chunk = base64.b64encode(chunk).decode('utf-8')
            await self._send_json(websocket, {"type": "audio_chunk", "data": encoded_chunk})
        await self._send_json(websocket, {"type": "audio_end"})

        await self._send_json(websocket, {"type": "status", "data": "Listening..."})

    async def _log_ffmpeg_stderr(self, stderr):
        while True:
            line = await stderr.readline()
            if not line: break
            print(f"[ffmpeg stderr] {line.decode().strip()}")

    async def handle_audio_stream(self, websocket: WebSocket):
        silence_threshold_ms = 700
        padding_ms = 200
        num_padding_chunks = padding_ms // self.frame_duration
        num_silence_chunks = silence_threshold_ms // self.frame_duration
        voice_buffer = deque(maxlen=num_silence_chunks)
        padding_buffer = deque(maxlen=num_padding_chunks)
        voiced_frames = []
        is_speaking = False

        ffmpeg_command = [
            "ffmpeg", '-hide_banner', '-loglevel', 'error', '-f', 'webm', '-i', 'pipe:0',
            '-f', 's16le', '-ar', '16000', '-ac', '1', '-fflags', 'nobuffer',
            '-probesize', '32', 'pipe:1'
        ]
        
        ffmpeg_process = await asyncio.create_subprocess_exec(
            *ffmpeg_command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stderr_logger_task = asyncio.create_task(self._log_ffmpeg_stderr(ffmpeg_process.stderr))

        await self._send_json(websocket, {"type": "status", "data": "Listening..."})

        try:
            while True:
                webm_chunk = await websocket.receive_bytes()
                if ffmpeg_process.stdin and not ffmpeg_process.stdin.is_closing():
                    ffmpeg_process.stdin.write(webm_chunk)
                    await ffmpeg_process.stdin.drain()

                while True:
                    try:
                        pcm_chunk = await asyncio.wait_for(ffmpeg_process.stdout.read(self.chunk_size * 2), timeout=0.01)
                        if not pcm_chunk: break
                        if len(pcm_chunk) != self.chunk_size * 2: continue

                        is_speech = self.vad.is_speech(pcm_chunk, self.sample_rate)
                        
                        if is_speech:
                            if not is_speaking:
                                is_speaking = True
                                voiced_frames.extend(list(padding_buffer))
                            voiced_frames.append(pcm_chunk)
                            voice_buffer.clear()
                        else:
                            padding_buffer.append(pcm_chunk)
                            if is_speaking:
                                voice_buffer.append(pcm_chunk)
                                if len(voice_buffer) >= num_silence_chunks:
                                    is_speaking = False
                                    audio_data = b''.join(voiced_frames)
                                    asyncio.create_task(self._process_s2s_pipeline(websocket, audio_data))
                                    voiced_frames.clear()
                                    voice_buffer.clear()
                                    padding_buffer.clear()
                    except asyncio.TimeoutError:
                        break
        
        except WebSocketDisconnect:
            print("Client disconnected.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print("Cleaning up ffmpeg process...")
            if ffmpeg_process.stdin and not ffmpeg_process.stdin.is_closing():
                try:
                    ffmpeg_process.stdin.close()
                    await ffmpeg_process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            if ffmpeg_process.returncode is None:
                ffmpeg_process.kill()
            await ffmpeg_process.wait()
            stderr_logger_task.cancel()
            print("ffmpeg process cleaned up.")

agent = RealTimeS2SAgent()

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await agent.handle_audio_stream(websocket)
