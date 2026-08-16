"""
MuseTalk Live Streaming & Real-time Avatar Engine
"""

from .avatar import LiveAvatar
from .avatar_manager import AvatarManager
from .audio_streamer import StreamingAudioProcessor
from .cuda_blender import fast_image_blending

__all__ = ["LiveAvatar", "AvatarManager", "StreamingAudioProcessor", "fast_image_blending"]
