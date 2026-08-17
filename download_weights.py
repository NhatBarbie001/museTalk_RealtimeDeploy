import os
import sys
import urllib.request
import gdown
from huggingface_hub import hf_hub_download

MODELS_DIR = "models"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_hf_file(repo_id: str, filename: str, subfolder: str = None, target_path: str = None):
    """
    Downloads a single file from Hugging Face Hub using the Python API.
    """
    print(f"📥 Downloading {repo_id}/{filename} -> {target_path}...")
    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
        )
        ensure_dir(os.path.dirname(target_path))
        if os.path.abspath(downloaded_file) != os.path.abspath(target_path):
            import shutil
            shutil.copyfile(downloaded_file, target_path)
        print(f"✅ Saved to {target_path}")
    except Exception as e:
        print(f"❌ Error downloading {filename} from {repo_id}: {e}")
        raise e


def download_all():
    print("==================================================")
    print("🚀 Starting Automatic Weight Download for MuseTalk")
    print("==================================================")

    # 1. MuseTalk V1.0 & V1.5
    ensure_dir(os.path.join(MODELS_DIR, "musetalk"))
    ensure_dir(os.path.join(MODELS_DIR, "musetalkV15"))

    download_hf_file(
        repo_id="TMElyralab/MuseTalk",
        filename="musetalk.json",
        subfolder="musetalk",
        target_path=os.path.join(MODELS_DIR, "musetalk", "musetalk.json"),
    )
    download_hf_file(
        repo_id="TMElyralab/MuseTalk",
        filename="pytorch_model.bin",
        subfolder="musetalk",
        target_path=os.path.join(MODELS_DIR, "musetalk", "pytorch_model.bin"),
    )
    download_hf_file(
        repo_id="TMElyralab/MuseTalk",
        filename="musetalk.json",
        subfolder="musetalkV15",
        target_path=os.path.join(MODELS_DIR, "musetalkV15", "musetalk.json"),
    )
    download_hf_file(
        repo_id="TMElyralab/MuseTalk",
        filename="unet.pth",
        subfolder="musetalkV15",
        target_path=os.path.join(MODELS_DIR, "musetalkV15", "unet.pth"),
    )

    # 2. SD VAE (ft-mse)
    ensure_dir(os.path.join(MODELS_DIR, "sd-vae"))
    download_hf_file(
        repo_id="stabilityai/sd-vae-ft-mse",
        filename="config.json",
        target_path=os.path.join(MODELS_DIR, "sd-vae", "config.json"),
    )
    download_hf_file(
        repo_id="stabilityai/sd-vae-ft-mse",
        filename="diffusion_pytorch_model.bin",
        target_path=os.path.join(MODELS_DIR, "sd-vae", "diffusion_pytorch_model.bin"),
    )

    # 3. Whisper Tiny
    ensure_dir(os.path.join(MODELS_DIR, "whisper"))
    download_hf_file(
        repo_id="openai/whisper-tiny",
        filename="config.json",
        target_path=os.path.join(MODELS_DIR, "whisper", "config.json"),
    )
    download_hf_file(
        repo_id="openai/whisper-tiny",
        filename="pytorch_model.bin",
        target_path=os.path.join(MODELS_DIR, "whisper", "pytorch_model.bin"),
    )
    download_hf_file(
        repo_id="openai/whisper-tiny",
        filename="preprocessor_config.json",
        target_path=os.path.join(MODELS_DIR, "whisper", "preprocessor_config.json"),
    )

    # 4. DWPose
    ensure_dir(os.path.join(MODELS_DIR, "dwpose"))
    download_hf_file(
        repo_id="yzd-v/DWPose",
        filename="dw-ll_ucoco_384.pth",
        target_path=os.path.join(MODELS_DIR, "dwpose", "dw-ll_ucoco_384.pth"),
    )

    # 5. SyncNet
    ensure_dir(os.path.join(MODELS_DIR, "syncnet"))
    download_hf_file(
        repo_id="ByteDance/LatentSync",
        filename="latentsync_syncnet.pt",
        target_path=os.path.join(MODELS_DIR, "syncnet", "latentsync_syncnet.pt"),
    )

    # 6. Face Parse BiSeNet
    ensure_dir(os.path.join(MODELS_DIR, "face-parse-bisent"))
    bisent_model_path = os.path.join(MODELS_DIR, "face-parse-bisent", "79999_iter.pth")
    if not os.path.exists(bisent_model_path):
        print(f"📥 Downloading Face Parsing 79999_iter.pth from Google Drive...")
        try:
            gdown.download(
                id="154JgKpzCPW82qINcVieuPH3fZ2e0P812",
                output=bisent_model_path,
                quiet=False,
            )
            print(f"✅ Saved to {bisent_model_path}")
        except Exception as e:
            print(f"⚠️ gdown failed: {e}. Trying direct URL...")
            try:
                gdown.download(
                    "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812",
                    output=bisent_model_path,
                    quiet=False,
                )
            except Exception as e2:
                print(f"❌ Could not download 79999_iter.pth: {e2}")

    # 7. ResNet18 backbone for Face Parsing
    resnet_path = os.path.join(MODELS_DIR, "face-parse-bisent", "resnet18-5c106cde.pth")
    if not os.path.exists(resnet_path):
        print(f"📥 Downloading ResNet-18 backbone -> {resnet_path}...")
        url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        urllib.request.urlretrieve(url, resnet_path)
        print(f"✅ Saved to {resnet_path}")

    print("==================================================")
    print("🎉 All MuseTalk weights successfully downloaded!")
    print("==================================================")


if __name__ == "__main__":
    download_all()
