import os
import argparse
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from musetalk.live.avatar_manager import AvatarManager
from server.webrtc_service import WebRTCManager

app = FastAPI(
    title="MuseTalk Realtime Live Avatar Server",
    description="Low-latency WebRTC & REST API Server for AI Digital Human Chatbots",
    version="1.5.0",
)

# Enable CORS for all frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Manager Singletons
avatar_mgr: AvatarManager = None
webrtc_mgr: WebRTCManager = None


@app.on_event("startup")
async def startup_event():
    global avatar_mgr, webrtc_mgr
    print("[Server] Starting MuseTalk Realtime Server...")
    avatar_mgr = AvatarManager(
        unet_model_path=os.getenv("UNET_MODEL_PATH", "models/musetalkV15/unet.pth"),
        unet_config=os.getenv("UNET_CONFIG", "models/musetalkV15/musetalk.json"),
        vae_type=os.getenv("VAE_TYPE", "sd-vae"),
        whisper_dir=os.getenv("WHISPER_DIR", "models/whisper/"),
        avatars_base_dir=os.getenv("AVATARS_DIR", "./results/v15/avatars"),
        use_float16=os.getenv("USE_FLOAT16", "true").lower() == "true",
    )
    webrtc_mgr = WebRTCManager(avatar_manager=avatar_mgr)
    print("[Server] MuseTalk Engine & WebRTC Manager Ready.")


# Register API Routes
from server.api_routes import router as api_router
app.include_router(api_router)

# Mount Static UI Files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Auto reload")
    args = parser.parse_args()

    uvicorn.run("server.main:app", host=args.host, port=args.port, reload=args.reload)
