import os
import glob
import pickle
import copy
import json
import torch
import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
from tqdm import tqdm

from .cuda_blender import fast_image_blending
from .state_manager import StateManager, AvatarState


class LiveAvatar:
    """
    Live Avatar class representing a pre-cached digital human.
    Maintains avatar assets in memory for zero-latency frame serving.
    """

    def __init__(
        self,
        avatar_id: str,
        base_dir: str = "./results/v15/avatars",
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        weight_dtype: torch.dtype = torch.float16,
    ):
        self.avatar_id = avatar_id
        self.avatar_path = os.path.join(base_dir, avatar_id)
        self.device = device
        self.weight_dtype = weight_dtype

        self.full_imgs_path = os.path.join(self.avatar_path, "full_imgs")
        self.coords_path = os.path.join(self.avatar_path, "coords.pkl")
        self.latents_path = os.path.join(self.avatar_path, "latents.pt")
        self.mask_path = os.path.join(self.avatar_path, "mask")
        self.mask_coords_path = os.path.join(self.avatar_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_path, "avator_info.json")

        self.state_manager = StateManager()
        self.idle_frame_idx = 0
        self.talking_frame_idx = 0

        self.is_loaded = False
        self.load_cache()

    def load_cache(self):
        """
        Loads pre-processed avatar assets from disk into RAM/VRAM.
        """
        if not os.path.exists(self.avatar_path):
            raise FileNotFoundError(f"Avatar directory does not exist: {self.avatar_path}")

        print(f"[LiveAvatar] Loading avatar cache for '{self.avatar_id}' from {self.avatar_path}...")

        # 1. Load coordinates
        with open(self.coords_path, "rb") as f:
            self.coord_list = pickle.load(f)

        # 2. Load mask coordinates
        with open(self.mask_coords_path, "rb") as f:
            self.mask_coords_list = pickle.load(f)

        # 3. Load full frames
        img_files = sorted(
            glob.glob(os.path.join(self.full_imgs_path, "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        self.frame_list = []
        for img_p in img_files:
            frame = cv2.imread(img_p)
            self.frame_list.append(frame)

        # 4. Load masks
        mask_files = sorted(
            glob.glob(os.path.join(self.mask_path, "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )
        self.mask_list = []
        for mask_p in mask_files:
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            self.mask_list.append(mask)

        # 5. Load latents
        self.input_latents = torch.load(self.latents_path, map_location=self.device)
        if isinstance(self.input_latents, list):
            self.input_latents = [lat.to(device=self.device, dtype=self.weight_dtype) for lat in self.input_latents]

        self.num_frames = len(self.frame_list)
        self.is_loaded = True
        print(f"[LiveAvatar] Avatar '{self.avatar_id}' successfully loaded with {self.num_frames} cyclical frames.")

    def get_next_idle_frame(self) -> np.ndarray:
        """
        Retrieves the next frame in the natural idle loop sequence.
        """
        frame = self.frame_list[self.idle_frame_idx % self.num_frames]
        self.idle_frame_idx = (self.idle_frame_idx + 1) % self.num_frames
        return frame

    @torch.no_grad()
    def generate_talking_frames(
        self,
        audio_prompts: torch.Tensor,
        unet_model: torch.nn.Module,
        vae_model,
        pe_layer: torch.nn.Module,
        timesteps: torch.Tensor,
        batch_size: int = 4,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generates lip-synced video frames for the provided audio prompts.
        Yields frames as they are rendered.
        """
        num_prompts = len(audio_prompts)
        if num_prompts == 0:
            return

        self.state_manager.set_talking()

        for start_idx in range(0, num_prompts, batch_size):
            if self.state_manager.is_interrupted():
                print(f"[LiveAvatar] Interruption detected. Flushed remaining {num_prompts - start_idx} frames.")
                self.state_manager.reset_interrupt()
                break

            end_idx = min(start_idx + batch_size, num_prompts)
            batch_prompts = audio_prompts[start_idx:end_idx].to(self.device)
            current_batch_size = end_idx - start_idx

            # Prepare latents batch corresponding to cycle position
            batch_latents = []
            frame_indices = []
            for b in range(current_batch_size):
                cycle_idx = (self.talking_frame_idx + b) % self.num_frames
                batch_latents.append(self.input_latents[cycle_idx])
                frame_indices.append(cycle_idx)

            batch_latents = torch.cat(batch_latents, dim=0).to(device=self.device, dtype=self.weight_dtype)

            # Positional encoding for audio features
            audio_feats = pe_layer(batch_prompts)

            # UNet single-step forward
            pred_latents = unet_model(
                batch_latents,
                timesteps,
                encoder_hidden_states=audio_feats,
            ).sample

            # VAE decode
            pred_latents = pred_latents.to(dtype=vae_model.vae.dtype)
            recon_mouth_patches = vae_model.decode_latents(pred_latents)

            # Fast Blending
            for b_i, recon_mouth in enumerate(recon_mouth_patches):
                cycle_idx = frame_indices[b_i]
                bbox = self.coord_list[cycle_idx]
                ori_frame = self.frame_list[cycle_idx]
                mask = self.mask_list[cycle_idx]
                crop_box = self.mask_coords_list[cycle_idx]

                blended_frame = fast_image_blending(
                    ori_frame=ori_frame,
                    res_frame=recon_mouth,
                    face_box=bbox,
                    mask_array=mask,
                    crop_box=crop_box,
                )

                yield blended_frame

            self.talking_frame_idx = (self.talking_frame_idx + current_batch_size) % self.num_frames

        self.state_manager.set_idle()
