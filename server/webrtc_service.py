import asyncio
import fractions
import time
import uuid
import av
import cv2
import numpy as np
import torch
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from typing import Dict, Optional
from musetalk.live.avatar import LiveAvatar
from musetalk.live.avatar_manager import AvatarManager


class AvatarVideoStreamTrack(MediaStreamTrack):
    """
    WebRTC Video Stream Track generating 25 FPS video frames.
    Yields talking frames when available; otherwise falls back to seamless idle loops.
    """

    kind = "video"

    def __init__(self, avatar: LiveAvatar, fps: int = 25):
        super().__init__()
        self.avatar = avatar
        self.fps = fps
        self.time_base = fractions.Fraction(1, self.fps)
        self.frame_queue = asyncio.Queue()
        self.pts = 0
        self._start_time = None

    async def recv(self):
        pts, time_base = self.pts, self.time_base
        self.pts += 1

        # Check if talking frames are queued
        if not self.frame_queue.empty():
            frame_bgr = await self.frame_queue.get()
        else:
            # Yield natural idle frame
            frame_bgr = self.avatar.get_next_idle_frame()

        # Convert BGR (OpenCV) to RGB for WebRTC
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        # Regulate playback speed to 25 FPS
        if self._start_time is None:
            self._start_time = time.time()
        else:
            expected_time = self._start_time + (self.pts / self.fps)
            delay = expected_time - time.time()
            if delay > 0:
                await asyncio.sleep(delay)

        return video_frame

    def flush_queue(self):
        """Flushes all queued talking frames on interruption."""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class AvatarAudioStreamTrack(MediaStreamTrack):
    """
    WebRTC Audio Stream Track generating 16kHz/48kHz audio.
    Streams TTS speech when speaking, or silence/room tone when idle.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000, frame_duration_ms: int = 20):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_frame = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.audio_queue = asyncio.Queue()
        self.pts = 0
        self._start_time = None

    async def recv(self):
        pts, time_base = self.pts, self.time_base
        self.pts += self.samples_per_frame

        if not self.audio_queue.empty():
            audio_data = await self.audio_queue.get()
        else:
            # Silence / ambient tone (zeros in float32 format)
            audio_data = np.zeros(self.samples_per_frame, dtype=np.int16)

        # Ensure correct shape and type: (1, samples) mono int16
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)

        audio_frame = av.AudioFrame.from_ndarray(audio_data, format="s16", layout="mono")
        audio_frame.sample_rate = self.sample_rate
        audio_frame.pts = pts
        audio_frame.time_base = time_base

        if self._start_time is None:
            self._start_time = time.time()
        else:
            expected_time = self._start_time + (self.pts / self.sample_rate)
            delay = expected_time - time.time()
            if delay > 0:
                await asyncio.sleep(delay)

        return audio_frame

    def flush_queue(self):
        """Flushes queued audio frames on interruption."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class WebRTCSession:
    """
    Represents an active WebRTC session for a connected client.
    """

    def __init__(self, session_id: str, avatar: LiveAvatar, pc: RTCPeerConnection):
        self.session_id = session_id
        self.avatar = avatar
        self.pc = pc
        self.video_track = AvatarVideoStreamTrack(avatar=avatar)
        self.audio_track = AvatarAudioStreamTrack()
        self.created_at = time.time()

    async def interrupt(self):
        """Cancels any ongoing speech and returns avatar to idle state."""
        self.avatar.state_manager.trigger_interrupt()
        self.video_track.flush_queue()
        self.audio_track.flush_queue()


class WebRTCManager:
    """
    Manages WebRTC PeerConnections and active client sessions.
    """

    def __init__(self, avatar_manager: AvatarManager):
        self.avatar_manager = avatar_manager
        self.sessions: Dict[str, WebRTCSession] = {}

    async def create_session(
        self,
        sdp_offer: str,
        sdp_type: str,
        avatar_id: str,
        stun_server: Optional[str] = "stun:stun.l.google.com:19302",
    ) -> dict:
        """
        Creates a new WebRTC session, negotiates SDP, and adds media tracks.
        """
        avatar = self.avatar_manager.get_or_load_avatar(avatar_id)

        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=stun_server)] if stun_server else []
        )
        pc = RTCPeerConnection(configuration=config)
        session_id = str(uuid.uuid4())

        session = WebRTCSession(session_id=session_id, avatar=avatar, pc=pc)
        self.sessions[session_id] = session

        # Add Avatar Video and Audio Tracks to PeerConnection
        pc.addTrack(session.video_track)
        pc.addTrack(session.audio_track)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"[WebRTC] Session {session_id} state changed to: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await self.close_session(session_id)

        # Set remote offer and create answer
        offer = RTCSessionDescription(sdp=sdp_offer, type=sdp_type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "session_id": session_id,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def handle_talk(self, session_id: str, audio_bytes: bytes):
        """
        Feeds audio into the session's avatar pipeline and enqueues rendered frames.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")

        # 1. Load and resample audio to 16kHz
        audio_data = self.avatar_manager.audio_processor.load_audio_bytes(audio_bytes)
        if len(audio_data) == 0:
            return

        # 2. Extract Whisper features
        prompts, num_frames = self.avatar_manager.audio_processor.extract_whisper_chunks(
            audio_data=audio_data,
            whisper_model=self.avatar_manager.whisper,
            device=self.avatar_manager.device,
            weight_dtype=self.avatar_manager.weight_dtype,
        )

        if num_frames == 0:
            return

        # 3. Generate speech frames via UNet + VAE + Fast Blending
        frame_generator = session.avatar.generate_talking_frames(
            audio_prompts=prompts,
            unet_model=self.avatar_manager.unet.model,
            vae_model=self.avatar_manager.vae,
            pe_layer=self.avatar_manager.pe,
            timesteps=self.avatar_manager.timesteps,
            batch_size=4,
        )

        # 4. Enqueue rendered video frames
        for frame in frame_generator:
            await session.video_track.frame_queue.put(frame)

    async def handle_interrupt(self, session_id: str):
        session = self.sessions.get(session_id)
        if session:
            await session.interrupt()

    async def close_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            await session.pc.close()
            print(f"[WebRTC] Closed session {session_id}")
