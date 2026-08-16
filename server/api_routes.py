import os
import shutil
import subprocess
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/api/v1")


class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str
    avatar_id: str
    stun_server: Optional[str] = "stun:stun.l.google.com:19302"


class InterruptRequest(BaseModel):
    session_id: str


@router.post("/webrtc/offer")
async def webrtc_offer(request: WebRTCOfferRequest):
    """
    Handles WebRTC SDP offer exchange and starts a live avatar streaming session.
    """
    from .main import webrtc_mgr
    try:
        res = await webrtc_mgr.create_session(
            sdp_offer=request.sdp,
            sdp_type=request.type,
            avatar_id=request.avatar_id,
            stun_server=request.stun_server,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WebRTC negotiation failed: {str(e)}")


@router.post("/talk")
async def talk(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    Injects an audio chunk (from TTS) into an active WebRTC avatar session.
    """
    from .main import webrtc_mgr
    if session_id not in webrtc_mgr.sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    audio_bytes = await audio_file.read()
    try:
        await webrtc_mgr.handle_talk(session_id=session_id, audio_bytes=audio_bytes)
        return {"status": "ok", "session_id": session_id, "audio_size": len(audio_bytes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process talk request: {str(e)}")


@router.post("/interrupt")
async def interrupt(request: InterruptRequest):
    """
    Triggers immediate speech interruption (Barge-in).
    """
    from .main import webrtc_mgr
    await webrtc_mgr.handle_interrupt(request.session_id)
    return {"status": "ok", "message": f"Session {request.session_id} interrupted."}


@router.get("/avatars")
async def list_avatars():
    """
    Lists all pre-cached avatars available on the server.
    """
    from .main import avatar_mgr
    return {"avatars": avatar_mgr.list_avatars()}


@router.post("/avatars/prepare")
async def prepare_avatar(
    avatar_id: str = Form(...),
    video_file: UploadFile = File(...),
    bbox_shift: int = Form(0),
    extra_margin: int = Form(10),
    parsing_mode: str = Form("jaw"),
):
    """
    Uploads a source video and runs offline pre-computation (landmarks, latents, masks).
    """
    upload_dir = f"./data/avatars_upload/{avatar_id}"
    os.makedirs(upload_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, video_file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)

    cmd = [
        "python", "-m", "scripts.realtime_inference",
        "--avatar_id", avatar_id,
        "--video_path", video_path,
        "--bbox_shift", str(bbox_shift),
        "--extra_margin", str(extra_margin),
        "--parsing_mode", parsing_mode,
        "--preparation", "True",
        "--version", "v15",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            "status": "success",
            "avatar_id": avatar_id,
            "message": "Avatar preparation completed successfully.",
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Avatar preparation failed: {e.stderr}")


@router.get("/healthz")
async def health_check():
    """
    Returns server health, active WebRTC sessions, and GPU VRAM statistics.
    """
    from .main import webrtc_mgr
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "allocated_vram_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 2),
            "reserved_vram_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 2),
            "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2),
        }

    return {
        "status": "healthy",
        "active_sessions": len(webrtc_mgr.sessions),
        "gpu": gpu_info,
    }
