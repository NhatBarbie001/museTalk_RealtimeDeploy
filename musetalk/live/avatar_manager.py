import os
import glob
import torch
from typing import Dict, Optional, List
from transformers import WhisperModel

from musetalk.utils.utils import load_all_model
from .avatar import LiveAvatar
from .audio_streamer import StreamingAudioProcessor


class AvatarManager:
    """
    Global Avatar Manager that maintains loaded AI models (UNet, VAE, Whisper)
    and manages active avatar instances for concurrent streaming sessions.
    """

    def __init__(
        self,
        unet_model_path: str = "models/musetalkV15/unet.pth",
        unet_config: str = "models/musetalkV15/musetalk.json",
        vae_type: str = "sd-vae",
        whisper_dir: str = "models/whisper/",
        avatars_base_dir: str = "./results/v15/avatars",
        use_float16: bool = True,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_float16 = use_float16
        self.avatars_base_dir = avatars_base_dir
        self.weight_dtype = torch.float16 if (self.use_float16 and torch.cuda.is_available()) else torch.float32

        print(f"[AvatarManager] Initializing on device: {self.device}, dtype: {self.weight_dtype}")

        # 1. Load VAE, UNet, PositionalEncoding
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type=vae_type,
            unet_config=unet_config,
            device=self.device,
        )

        if self.use_float16 and torch.cuda.is_available():
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)
        self.timesteps = torch.tensor([0], device=self.device)

        # 2. Load Whisper Audio Model & Processor
        print(f"[AvatarManager] Loading Whisper model from {whisper_dir}...")
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        self.audio_processor = StreamingAudioProcessor(
            feature_extractor_path=whisper_dir,
            sr=16000,
            fps=25,
        )

        # 3. Avatar registry
        self.loaded_avatars: Dict[str, LiveAvatar] = {}
        self.auto_discover_avatars()

    def auto_discover_avatars(self):
        """
        Discovers and registers all prepared avatars in the avatars base directory.
        """
        if not os.path.exists(self.avatars_base_dir):
            os.makedirs(self.avatars_base_dir, exist_ok=True)
            return

        for entry in os.listdir(self.avatars_base_dir):
            avatar_dir = os.path.join(self.avatars_base_dir, entry)
            if os.path.isdir(avatar_dir) and os.path.exists(os.path.join(avatar_dir, "latents.pt")):
                try:
                    self.get_or_load_avatar(entry)
                except Exception as e:
                    print(f"[AvatarManager] Failed to pre-load avatar '{entry}': {e}")

    def list_avatars(self) -> List[dict]:
        """
        Lists available avatars and their metadata.
        """
        results = []
        if not os.path.exists(self.avatars_base_dir):
            return results

        for entry in os.listdir(self.avatars_base_dir):
            avatar_dir = os.path.join(self.avatars_base_dir, entry)
            latents_file = os.path.join(avatar_dir, "latents.pt")
            if os.path.isdir(avatar_dir) and os.path.exists(latents_file):
                info_file = os.path.join(avatar_dir, "avator_info.json")
                info = {}
                if os.path.exists(info_file):
                    import json
                    try:
                        with open(info_file, "r") as f:
                            info = json.load(f)
                    except:
                        pass
                results.append({
                    "avatar_id": entry,
                    "is_loaded": entry in self.loaded_avatars,
                    "info": info,
                })
        return results

    def get_or_load_avatar(self, avatar_id: str) -> LiveAvatar:
        """
        Gets a loaded avatar or instantiates and caches it into the registry.
        """
        if avatar_id in self.loaded_avatars:
            return self.loaded_avatars[avatar_id]

        avatar = LiveAvatar(
            avatar_id=avatar_id,
            base_dir=self.avatars_base_dir,
            device=self.device,
            weight_dtype=self.weight_dtype,
        )
        self.loaded_avatars[avatar_id] = avatar
        return avatar
