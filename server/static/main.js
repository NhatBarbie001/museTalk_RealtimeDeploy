let pc = null;
let currentSessionId = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// DOM Elements
const avatarSelect = document.getElementById("avatarSelect");
const btnConnect = document.getElementById("btnConnect");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnInterrupt = document.getElementById("btnInterrupt");
const avatarVideo = document.getElementById("avatarVideo");
const connPill = document.getElementById("connPill");
const connText = document.getElementById("connText");
const vramText = document.getElementById("vramText");
const streamState = document.getElementById("streamState");
const botStateVal = document.getElementById("botStateVal");
const latencyVal = document.getElementById("latencyVal");
const btnSendText = document.getElementById("btnSendText");
const ttsTextInput = document.getElementById("ttsTextInput");
const btnUploadAudio = document.getElementById("btnUploadAudio");
const audioFileInput = document.getElementById("audioFileInput");
const btnRecordMic = document.getElementById("btnRecordMic");

// 1. Fetch available avatars on load
async function loadAvatars() {
  try {
    const res = await fetch("/api/v1/avatars");
    const data = await res.json();
    avatarSelect.innerHTML = "";

    if (!data.avatars || data.avatars.length === 0) {
      avatarSelect.innerHTML = '<option value="default">default (Auto)</option>';
      return;
    }

    data.avatars.forEach(av => {
      const opt = document.createElement("option");
      opt.value = av.avatar_id;
      opt.textContent = `${av.avatar_id} ${av.is_loaded ? "(Đã nạp)" : ""}`;
      avatarSelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load avatars:", err);
    avatarSelect.innerHTML = '<option value="avatar1">avatar1</option>';
  }
}

// 2. Poll server telemetry & health
async function pollHealth() {
  try {
    const res = await fetch("/api/v1/healthz");
    const data = await res.json();
    if (data.gpu && data.gpu.allocated_vram_mb !== undefined) {
      vramText.textContent = `${data.gpu.allocated_vram_mb} / ${data.gpu.total_vram_mb} MB`;
    }
  } catch (err) {
    // server unreachable
  }
}
setInterval(pollHealth, 3000);

// 3. WebRTC Start & Stop
async function startWebRTC() {
  const avatarId = avatarSelect.value || "default";
  setConnectionState("connecting", "Đang kết nối WebRTC...");

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
  });

  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });

  pc.ontrack = (event) => {
    if (event.track.kind === "video") {
      avatarVideo.srcObject = event.streams[0];
      setConnectionState("connected", "Đã kết nối Live Avatar");
      streamState.textContent = "Trạng thái: Đang phát trực tiếp";
      btnConnect.style.display = "none";
      btnDisconnect.style.display = "inline-flex";
    }
  };

  pc.oniceconnectionstatechange = () => {
    console.log("ICE Connection State:", pc.iceConnectionState);
    if (pc.iceConnectionState === "disconnected" || pc.iceConnectionState === "failed") {
      stopWebRTC();
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const startTime = Date.now();
  try {
    const res = await fetch("/api/v1/webrtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
        avatar_id: avatarId,
      }),
    });

    const answer = await res.json();
    currentSessionId = answer.session_id;
    await pc.setRemoteDescription(new RTCSessionDescription(answer));

    const rtt = Date.now() - startTime;
    latencyVal.textContent = `${rtt} ms`;
  } catch (err) {
    console.error("WebRTC Negotiation failed:", err);
    setConnectionState("disconnected", "Kết nối thất bại");
    stopWebRTC();
  }
}

function stopWebRTC() {
  if (pc) {
    pc.close();
    pc = null;
  }
  currentSessionId = null;
  avatarVideo.srcObject = null;
  setConnectionState("disconnected", "Chưa kết nối");
  streamState.textContent = "Trạng thái: Đã dừng";
  btnConnect.style.display = "inline-flex";
  btnDisconnect.style.display = "none";
  botStateVal.textContent = "IDLE";
}

function setConnectionState(state, text) {
  connPill.className = `pill ${state}`;
  connText.textContent = text;
}

// 4. Send Audio to Avatar (Talk)
async function sendAudioFile(fileBlob) {
  if (!currentSessionId) {
    alert("Vui lòng bắt đầu kết nối WebRTC trước khi gửi audio!");
    return;
  }

  const formData = new FormData();
  formData.append("session_id", currentSessionId);
  formData.append("audio_file", fileBlob, "speech.wav");

  botStateVal.textContent = "TALKING";
  botStateVal.style.color = "var(--accent-cyan)";

  try {
    const t0 = Date.now();
    const res = await fetch("/api/v1/talk", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    const cost = Date.now() - t0;
    latencyVal.textContent = `${cost} ms`;
  } catch (err) {
    console.error("Failed to send audio:", err);
  } finally {
    setTimeout(() => {
      botStateVal.textContent = "IDLE";
      botStateVal.style.color = "var(--accent-emerald)";
    }, 2000);
  }
}

// 5. Barge-in / Interrupt
async function triggerInterrupt() {
  if (!currentSessionId) return;

  try {
    await fetch("/api/v1/interrupt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: currentSessionId }),
    });
    botStateVal.textContent = "INTERRUPTED";
    botStateVal.style.color = "var(--accent-rose)";
    setTimeout(() => {
      botStateVal.textContent = "IDLE";
      botStateVal.style.color = "var(--accent-emerald)";
    }, 800);
  } catch (err) {
    console.error("Interrupt failed:", err);
  }
}

// Event Listeners
btnConnect.addEventListener("click", startWebRTC);
btnDisconnect.addEventListener("click", stopWebRTC);
btnInterrupt.addEventListener("click", triggerInterrupt);

btnUploadAudio.addEventListener("click", () => audioFileInput.click());
audioFileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    sendAudioFile(e.target.files[0]);
  }
});

// Microphone Recorder
btnRecordMic.addEventListener("click", async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: "audio/wav" });
        sendAudioFile(blob);
      };
      mediaRecorder.start();
      isRecording = true;
      btnRecordMic.classList.add("active");
      btnRecordMic.textContent = "⏹ Dừng & Gửi Âm Thanh";
    } catch (err) {
      alert("Không thể truy cập Microphone: " + err.message);
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    btnRecordMic.classList.remove("active");
    btnRecordMic.textContent = "🎙️ Thu Âm Từ Micro";
  }
});

// Text-to-Speech simulation trigger
btnSendText.addEventListener("click", () => {
  const text = ttsTextInput.value.trim();
  if (!text) {
    alert("Vui lòng nhập nội dung câu nói!");
    return;
  }
  // If your TTS API endpoint is ready, you can fetch the WAV blob directly from your TTS backend
  // For demo: uses web audio synthesis or prompt
  alert("Gợi ý: Hãy kết nối đầu ra audio stream từ backend TTS của bạn vào endpoint /api/v1/talk!");
});

// Init
window.addEventListener("DOMContentLoaded", () => {
  loadAvatars();
  pollHealth();
});
