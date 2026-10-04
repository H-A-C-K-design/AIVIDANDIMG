import os
import sys
import torch
import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Add ConsisID to path
sys.path.append(os.path.join(os.path.dirname(__file__), "ConsisID"))

try:
    from diffusers.utils import export_to_video
    from diffusers.image_processor import VaeImageProcessor
    from models.consisid_utils import prepare_face_models, process_face_embeddings_infer
    from models.pipeline_consisid import ConsisIDPipeline
    from models.transformer_consisid import ConsisIDTransformer3DModel
    CONSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ConsisID dependencies missing. {e}")
    CONSIS_AVAILABLE = False

app = FastAPI(title="AI Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directories
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


def build_qwen_image_prompt(base_prompt: str) -> str:
    """Uses awesome-qwen-image-3 principles to enhance prompt."""
    return f"Highly realistic cinematic photography, 4k high resolution, detailed atmospheric background, medium shot, eye level, dramatic lighting. Subject: {base_prompt}"

def generate_image(prompt: str) -> str:
    output_filename = "generated_image.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    enhanced = build_qwen_image_prompt(prompt)
    import urllib.parse
    
    print("Using Pollinations API for free image generation...")
    try:
        # We use a 16:9 aspect ratio (1024x576) to match our cinematic prompts
        encoded_prompt = urllib.parse.quote(enhanced)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true"
        
        # Pollinations returns the raw image bytes directly
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as handler:
                handler.write(response.content)
            return output_filename
        else:
            print(f"Pollinations API failed with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

def generate_video(prompt: str, image_filename: str) -> str:
    print("Attempting to use Vidu API for video generation...")
    vidu_api_key = "vda_1004229886702784512_Jk3nEjtcFwt8c1DlnVYfkNJLzJhOw86N"
    output_filename = "generated_video.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        # Set up the Vidu API request
        headers = {
            "Authorization": f"Bearer {vidu_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "style": "general",
            "duration": 4
        }
        # Note: Vidu endpoints vary by aggregator. Using standard placeholder endpoint.
        url = "https://api.vidu.studio/v1/video/text-to-video" 
        
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if "url" in data or "video_url" in data:
                vid_url = data.get("url", data.get("video_url"))
                vid_data = requests.get(vid_url).content
                with open(output_path, "wb") as f:
                    f.write(vid_data)
                return output_filename
            else:
                print("Vidu API returned success but no direct video URL found (likely requires polling task_id).")
        else:
            print(f"Vidu API failed with {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Vidu API connection error: {e}")
        
    print("Vidu API generation failed. Falling back to native ConsisID local generation!")
    
    if not CONSIS_AVAILABLE:
        print("ConsisID cannot be used because a Windows policy blocked the AI modules from loading.")
        return None
        
    model_path = os.path.join(os.path.dirname(__file__), "ConsisID", "ckpts")
    
    if not os.path.exists(model_path):
        print(f"ConsisID models missing at {model_path}.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    subfolder = "transformer_ema" if os.path.exists(os.path.join(model_path, "transformer_ema")) else "transformer"

    # Minimal error handling to not crash the server on inference OOM
    try:
        f1, f2, fclip, fmain, eva_m, eva_s = prepare_face_models(model_path, device, dtype)
        transformer = ConsisIDTransformer3DModel.from_pretrained(model_path, subfolder=subfolder, torch_dtype=dtype)
        pipe = ConsisIDPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=dtype)
        transformer.to(device, dtype=dtype)
        pipe.to(device)

        id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
            f1, fclip, f2, eva_m, eva_s, fmain, device, dtype, image_path, is_align_face=True
        )
        
        gen = torch.Generator(device).manual_seed(42)
        video_pt = pipe(
            prompt=prompt, image=image, num_videos_per_prompt=1,
            num_inference_steps=50, num_frames=49, use_dynamic_cfg=False,
            guidance_scale=6.0, generator=gen, id_vit_hidden=id_vit_hidden,
            id_cond=id_cond, kps_cond=face_kps, output_type="pt"
        ).frames
        
        pt_image = video_pt[0]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        batch_frames = [VaeImageProcessor.numpy_to_pil(image_np)]
        
        export_to_video(batch_frames[0], output_path, fps=8)
        
        # Cleanup memory
        del pipe, transformer, f1, f2, fclip, fmain
        torch.cuda.empty_cache()
        
        return output_filename
    except Exception as e:
        print(f"Video generation error: {e}")
        return None

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    print(f"Received generation request for prompt: {req.prompt}")
    
    img_filename = generate_image(req.prompt)
    if not img_filename:
        raise HTTPException(status_code=500, detail="Failed to generate image.")
        
    vid_filename = generate_video(req.prompt, img_filename)
    
    # Return the enhanced prompt so the user can see how the model interpreted it
    enhanced_prompt = build_qwen_image_prompt(req.prompt)
    
    return {
        "image_url": f"/outputs/{img_filename}",
        "video_url": f"/outputs/{vid_filename}" if vid_filename else None,
        "enhanced_prompt": enhanced_prompt,
        "message": "Success! (Note: If video is missing, ConsisID models need to be downloaded to ConsisID/ckpts)" if not vid_filename else "Success!"
    }

# Mount static folders
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

if __name__ == "__main__":
    print("Starting AI Studio Server on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
