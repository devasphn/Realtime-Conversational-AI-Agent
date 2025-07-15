document.addEventListener('DOMContentLoaded', () => {
    const startCallBtn = document.getElementById('start-call-btn');
    const endCallBtn = document.getElementById('end-call-btn');
    const muteBtn = document.getElementById('mute-btn');
    const transcriptContainer = document.getElementById('transcript-container');
    const statusText = document.getElementById('status-text');
    const statusIndicator = document.getElementById('status-indicator');

    let socket;
    let mediaRecorder;
    let audioContext;
    let audioQueue = [];
    let isPlaying = false;
    let isMuted = false;
    let stream;
    let pcmBuffer = [];

    const updateStatus = (text, indicatorClass) => {
        statusText.textContent = text;
        statusIndicator.className = 'status-indicator';
        if (indicatorClass) {
            statusIndicator.classList.add(indicatorClass);
        }
    };

    const addTranscript = (text, sender) => {
        const entry = document.createElement('div');
        entry.classList.add('transcript-entry', sender);
        entry.textContent = text;
        transcriptContainer.appendChild(entry);
        transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
    };

    const createWavFile = (pcmData) => {
        const sampleRate = 24000; // XTTS-v2 sample rate
        const numChannels = 1;
        const bitsPerSample = 16;
        const dataSize = pcmData.byteLength;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true);
        view.setUint16(32, numChannels * (bitsPerSample / 8), true);
        view.setUint16(34, bitsPerSample, true);
        writeString(view, 36, 'data');
        view.setUint32(40, dataSize, true);

        const pcmView = new Uint8Array(pcmData);
        const dataView = new Uint8Array(buffer, 44);
        dataView.set(pcmView);

        return buffer;
    };

    const processAudioQueue = async () => {
        if (audioQueue.length === 0 || isPlaying) return;
        
        isPlaying = true;
        const wavData = audioQueue.shift();
        try {
            const buffer = await audioContext.decodeAudioData(wavData);
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.start();
            source.onended = () => {
                isPlaying = false;
                processAudioQueue();
            };
        } catch (error) {
            console.error("Error decoding audio data:", error);
            isPlaying = false;
        }
    };

    const startCall = async () => {
        startCallBtn.disabled = true;
        endCallBtn.disabled = false;
        muteBtn.disabled = false;
        updateStatus("Connecting...", "thinking");

        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000 } });
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            socket = new WebSocket(wsUrl);

            socket.onopen = () => {
                console.log("WebSocket connected.");
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                
                mediaRecorder.addEventListener('dataavailable', event => {
                    if (event.data.size > 0 && socket.readyState === WebSocket.OPEN && !isMuted) {
                        socket.send(event.data);
                    }
                });
                mediaRecorder.start(200);
            };

            socket.onmessage = async (event) => {
                const message = JSON.parse(event.data);
                switch (message.type) {
                    case 'status':
                        const indicatorClass = { "Listening...": "listening", "Thinking...": "thinking", "Speaking...": "speaking" }[message.data] || "";
                        updateStatus(message.data, indicatorClass);
                        break;
                    case 'transcription_user':
                        addTranscript(message.data, 'user');
                        break;
                    case 'transcription_agent':
                        addTranscript(message.data, 'agent');
                        break;
                    case 'audio_start':
                        pcmBuffer = [];
                        break;
                    case 'audio_chunk':
                        const audioData = atob(message.data);
                        const audioBytes = new Uint8Array(audioData.length);
                        for (let i = 0; i < audioData.length; i++) {
                            audioBytes[i] = audioData.charCodeAt(i);
                        }
                        pcmBuffer.push(audioBytes);
                        break;
                    case 'audio_end':
                        if (pcmBuffer.length === 0) break;
                        const completePcmBlob = new Blob(pcmBuffer, { type: 'application/octet-stream' });
                        const reader = new FileReader();
                        reader.onload = (e) => {
                            const wavData = createWavFile(e.target.result);
                            audioQueue.push(wavData);
                            processAudioQueue();
                        };
                        reader.readAsArrayBuffer(completePcmBlob);
                        pcmBuffer = [];
                        break;
                }
            };

            socket.onclose = () => { console.log("WebSocket disconnected."); endCall(); };
            socket.onerror = (error) => { console.error("WebSocket error:", error); endCall(); };

        } catch (error) {
            console.error("Error starting call:", error);
            alert("Could not start call. Check microphone permissions.");
            endCall();
        }
    };

    const endCall = () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
        if (socket) socket.close();
        if (stream) stream.getTracks().forEach(track => track.stop());
        if (audioContext && audioContext.state !== 'closed') audioContext.close();

        audioQueue = []; pcmBuffer = []; isPlaying = false;
        startCallBtn.disabled = false; endCallBtn.disabled = true; muteBtn.disabled = true;
        muteBtn.textContent = 'Mute'; muteBtn.classList.remove('muted'); isMuted = false;
        updateStatus("Disconnected", "");
    };

    const toggleMute = () => {
        isMuted = !isMuted;
        muteBtn.textContent = isMuted ? 'Unmute' : 'Mute';
        muteBtn.classList.toggle('muted', isMuted);
    };

    startCallBtn.addEventListener('click', startCall);
    endCallBtn.addEventListener('click', endCall);
    muteBtn.addEventListener('click', toggleMute);
});
