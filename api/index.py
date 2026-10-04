from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import urllib.parse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI(title="AI Studio API (Vercel Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_qwen_image_prompt(base_prompt: str) -> str:
    """Uses awesome-qwen-image-3 principles to enhance prompt."""
    return f"Highly realistic cinematic photography, 4k high resolution, detailed atmospheric background, medium shot, eye level, dramatic lighting. Subject: {base_prompt}"

def generate_image(prompt: str) -> str:
    enhanced = build_qwen_image_prompt(prompt)
    print("Using Pollinations API for serverless image generation...")
    # Because Vercel is read-only, we don't save the image locally. We just return the direct URL.
    encoded_prompt = urllib.parse.quote(enhanced)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true"

def generate_video(prompt: str, image_url: str) -> str:
    print("Attempting to use Vidu API for serverless video generation...")
    vidu_api_key = os.getenv("VIDU_API_KEY", "vda_1004229886702784512_Jk3nEjtcFwt8c1DlnVYfkNJLzJhOw86N")
    
    # If the user has added their exact endpoint URL as an environment variable, use it.
    vidu_endpoint = os.getenv("VIDU_ENDPOINT_URL", "https://api.vidu.studio/v1/video/text-to-video")
    
    try:
        headers = {
            "Authorization": f"Bearer {vidu_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "image_url": image_url, # Pass the public image URL
            "style": "general",
            "duration": 4
        }
        
        response = requests.post(vidu_endpoint, headers=headers, json=payload, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            if "url" in data or "video_url" in data:
                return data.get("url", data.get("video_url"))
    except Exception as e:
        print(f"Vidu API Error: {e}")
        
    return None

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    print(f"Received Serverless generation request for prompt: {req.prompt}")
    
    img_url = generate_image(req.prompt)
    if not img_url:
        raise HTTPException(status_code=500, detail="Failed to generate image URL.")
        
    vid_url = generate_video(req.prompt, img_url)
    enhanced_prompt = build_qwen_image_prompt(req.prompt)
    
    return {
        "image_url": img_url,
        "video_url": vid_url,
        "enhanced_prompt": enhanced_prompt,
        "message": "Generated on Vercel!" if vid_url else "Image generated! (Video failed: Vidu API endpoint missing or incorrect in Vercel Env Vars)"
    }
