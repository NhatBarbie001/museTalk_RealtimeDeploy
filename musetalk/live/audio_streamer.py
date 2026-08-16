import math
import io
import torch
import numpy as np
import soundfile as sf
import librosa
from einops import rearrange
from transformers import AutoFeatureExtractor
from typing import List, Tuple, Optional


class StreamingAudioProcessor:
    """
    Streaming Audio Processor for real-time lip-sync generation.
    Supports in-memory audio chunking, resampling to 16kHz mono,
    and Whisper feature extraction.
    """

    def __init__(
        self,
        feature_extractor_path: str = "models/whisper/",
        sr: int = 16000,
        fps: int = 25,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)
        self.sr = sr
        self.fps = fps
        self.audio_fps = 50
        self.audio_padding_left = audio_padding_length_left
        self.audio_padding_right = audio_padding_length_right
        self.whisper_multiplier = self.audio_fps / self.fps
        self.feature_length_per_frame = 2 * (self.audio_padding_left + self.audio_padding_right + 1)

    def load_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """
        Loads arbitrary audio bytes (wav, mp3, ogg, pcm) into a 16kHz mono numpy array.
        """
        try:
            audio_io = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(audio_io)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if samplerate != self.sr:
                data = librosa.resample(data.astype(np.float32), orig_sr=samplerate, target_sr=self.sr)
            return data.astype(np.float32)
        except Exception as e:
            # Fallback to librosa directly
            audio_io = io.BytesIO(audio_bytes)
            data, _ = librosa.load(audio_io, sr=self.sr, mono=True)
            return data.astype(np.float32)

    def extract_whisper_chunks(
        self,
        audio_data: np.ndarray,
        whisper_model: torch.nn.Module,
        device: torch.device,
        weight_dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, int]:
        """
        Converts 16kHz audio data to whisper latent features for the UNet model.

        Returns:
            audio_prompts: Tensor of shape (num_frames, 50, 384)
            num_frames: Total number of video frames corresponding to this audio
        """
        if len(audio_data) == 0:
            return torch.empty(0), 0

        total_samples = len(audio_data)
        num_frames = math.floor((total_samples / self.sr) * self.fps)
        if num_frames == 0:
            return torch.empty(0), 0

        # Segment into 30s segments
        segment_len = 30 * self.sr
        segments = [audio_data[i:i + segment_len] for i in range(0, total_samples, segment_len)]

        whisper_feature_list = []
        for seg in segments:
            feats = self.feature_extractor(
                seg,
                return_tensors="pt",
                sampling_rate=self.sr
            ).input_features
            feats = feats.to(device=device, dtype=weight_dtype)

            with torch.no_grad():
                encoder_out = whisper_model.encoder(feats, output_hidden_states=True).hidden_states
                seg_feats = torch.stack(encoder_out, dim=2)
                whisper_feature_list.append(seg_feats)

        whisper_feats = torch.cat(whisper_feature_list, dim=1)

        actual_len = math.floor((total_samples / self.sr) * self.audio_fps)
        whisper_feats = whisper_feats[:, :actual_len, ...]

        # Padding
        padding_nums = math.ceil(self.whisper_multiplier)
        pad_left = torch.zeros_like(whisper_feats[:, :padding_nums * self.audio_padding_left])
        pad_right = torch.zeros_like(whisper_feats[:, :padding_nums * 3 * self.audio_padding_right])
        whisper_feats = torch.cat([pad_left, whisper_feats, pad_right], dim=1)

        audio_prompts = []
        for idx in range(num_frames):
            audio_idx = math.floor(idx * self.whisper_multiplier)
            clip = whisper_feats[:, audio_idx: audio_idx + self.feature_length_per_frame]
            if clip.shape[1] == self.feature_length_per_frame:
                audio_prompts.append(clip)
            else:
                # Handle boundary case
                pad_needed = self.feature_length_per_frame - clip.shape[1]
                clip = torch.cat([clip, torch.zeros_like(clip[:, :pad_needed])], dim=1)
                audio_prompts.append(clip)

        if len(audio_prompts) == 0:
            return torch.empty(0), 0

        audio_prompts = torch.cat(audio_prompts, dim=0)  # (N, 10, 5, 384)
        audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')  # (N, 50, 384)
        return audio_prompts, num_frames
