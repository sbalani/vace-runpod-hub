import os
import sys
import tempfile
import base64
from typing import List, Optional
import runpod
import torch
from huggingface_hub import hf_hub_download, snapshot_download

print("Starting VACE RunPod handler...", flush=True)

# Add project root to python path for vace imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Add current directory
sys.path.insert(0, os.path.join(current_dir, "vace"))  # Add vace directory

print("Python path set up", flush=True)

from vace.vace_wan_inference import main as vace_wan_inference
from vace.models.wan import WanVace
from vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES

print("VACE imports successful", flush=True)

def download_models():
    """Download required model files if they don't exist"""
    print("Starting model download...", flush=True)
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Download the full Wan2.1-VACE-14B repository
    #print("Downloading Wan2.1-VACE-14B repository...", flush=True)
    #snapshot_download(
    #    repo_id="Wan-AI/Wan2.1-VACE-14B",
    #    local_dir="models",
    #    local_dir_use_symlinks=False
    #)

    # Download model files if they don't exist
    model_files = {
        "special_tokens_map.json": {
            "repo_id": "google/umt5-xxl",
            "filename": "special_tokens_map.json"
        },
        "Wan2.1_T2V_14B_FusionX_VACE-FP16.safetensors": {
            "repo_id": "QuantStack/Wan2.1_T2V_14B_FusionX_VACE",
            "filename": "Wan2.1_T2V_14B_FusionX_VACE-FP16.safetensors"
        },

        "tokenizer_config.json": {
            "repo_id": "google/umt5-xxl",
            "filename": "tokenizer_config.json"
        },
        "spiece.model": {
            "repo_id": "google/umt5-xxl",
            "filename": "spiece.model"
        }
    }

    for filename, info in model_files.items():
        filepath = os.path.join("models", filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from {info['repo_id']}...", flush=True)
            hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                local_dir="models",
                local_dir_use_symlinks=False
            )
    print("Model download complete", flush=True)

def save_base64_to_file(base64_string: str, filepath: str) -> None:
    """Save a base64 encoded file to disk"""
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_string))

def generate_video(job):
    """
    Generate a video using VACE model
    """
    print(f"Received job: {job}", flush=True)
    try:
        # Get input parameters
        input_data = job["input"]
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Handle source video if provided
            src_video = None
            if "src_video" in input_data:
                src_video_path = os.path.join(temp_dir, "src_video.mp4")
                save_base64_to_file(input_data["src_video"], src_video_path)
                src_video = src_video_path

            # Handle source mask if provided
            src_mask = None
            if "src_mask" in input_data:
                src_mask_path = os.path.join(temp_dir, "src_mask.mp4")
                save_base64_to_file(input_data["src_mask"], src_mask_path)
                src_mask = src_mask_path

            # Handle source reference images if provided
            src_ref_images = None
            if "src_ref_images" in input_data:
                src_ref_images = []
                for i, img_base64 in enumerate(input_data["src_ref_images"]):
                    img_path = os.path.join(temp_dir, f"src_ref_image_{i}.png")
                    save_base64_to_file(img_base64, img_path)
                    src_ref_images.append(img_path)
                src_ref_images = ",".join(src_ref_images)

            # Prepare arguments for vace_wan_inference
            args = {
                "model_name": input_data.get("model_name", "vace-14B"),
                "prompt": input_data.get("prompt", ""),
                "src_video": src_video,
                "src_mask": src_mask,
                "src_ref_images": src_ref_images,
                "size": input_data.get("size", "480p"),
                "frame_num": input_data.get("frame_num", 81),
                "base_seed": input_data.get("base_seed", -1),
                "sample_solver": input_data.get("sample_solver", "unipc"),
                "sample_steps": input_data.get("sample_steps", 25),
                "sample_shift": input_data.get("sample_shift", 5.0),
                "sample_guide_scale": input_data.get("sample_guide_scale", 5.0),
                "ckpt_dir": "models",
                "offload_model": True,
                "ulysses_size": 1,
                "ring_size": 1,
                "t5_fsdp": False,
                "t5_cpu": False,
                "dit_fsdp": False,
                "use_prompt_extend": "plain",
                "save_file": os.path.join(temp_dir, "output.mp4")
            }

            print("Starting inference...", flush=True)
            # Run inference
            result = vace_wan_inference(args)
            print("Inference complete", flush=True)

            # Read output video and convert to base64
            with open(result["out_video"], "rb") as f:
                output_video_base64 = base64.b64encode(f.read()).decode()

            # Prepare response
            response = {
                "output_video": output_video_base64
            }

            # Add source video if available
            if "src_video" in result:
                with open(result["src_video"], "rb") as f:
                    response["src_video"] = base64.b64encode(f.read()).decode()

            # Add source mask if available
            if "src_mask" in result:
                with open(result["src_mask"], "rb") as f:
                    response["src_mask"] = base64.b64encode(f.read()).decode()

            # Add source reference images if available
            for key in result:
                if key.startswith("src_ref_image_"):
                    with open(result[key], "rb") as f:
                        response[key] = base64.b64encode(f.read()).decode()

            return response

    except Exception as e:
        print(f"Error in generate_video: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

print("Starting model download...", flush=True)
# Download models on startup
download_models()
print("Model download complete, starting RunPod serverless handler...", flush=True)

# Start the RunPod serverless handler
runpod.serverless.start({"handler": generate_video}) 