import gradio as gr
import os
import sys
import torch
import requests
from PIL import Image

# Add ConsisID to path so its modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "ConsisID"))

# Import ConsisID dynamically
try:
    from diffusers.utils import export_to_video # type: ignore
    from diffusers.image_processor import VaeImageProcessor # type: ignore
    from models.consisid_utils import prepare_face_models, process_face_embeddings_infer # type: ignore
    from models.pipeline_consisid import ConsisIDPipeline # type: ignore
    from models.transformer_consisid import ConsisIDTransformer3DModel # type: ignore
    CONSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ConsisID dependencies are not fully installed. {e}")
    CONSIS_AVAILABLE = False


def build_qwen_image_prompt(base_prompt):
    """
    Uses the awesome-qwen-image-3 base framework to enhance the user's prompt.
    """
    enhanced = f"""Use case: cinematic photography / character
Asset type: 16:9 4k high resolution
Goal: create a highly realistic and consistent character image
Scene/backdrop: detailed background, atmospheric
Subject: {base_prompt}
Style/medium: photo, hyper-realistic
Composition/framing: medium shot, eye level
Lighting/mood: cinematic lighting, dramatic
"""
    return enhanced


def generate_image_qwen(prompt):
    """
    Uses DashScope API to generate an image from text.
    (Requires DASHSCOPE_API_KEY environment variable, otherwise generates a dummy image for testing).
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    output_path = "temp_generated_image.png"
    
    if not api_key:
        print("No DASHSCOPE_API_KEY found. Generating dummy image for demonstration.")
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        img.save(output_path)
        return output_path
        
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
    headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Use the enhanced prompt format from awesome-qwen-image-3
    enhanced_prompt = build_qwen_image_prompt(prompt)
    
    payload = {
        "model": "wanx-v1", # Wanx is Aliyun's image model (often powering Qwen image gen)
        "input": {"prompt": enhanced_prompt},
        "parameters": {"size": "1024*1024", "n": 1}
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        task_id = response.json().get("output", {}).get("task_id")
        import time
        while True:
            res_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            res = requests.get(res_url, headers={"Authorization": f"Bearer {api_key}"})
            status = res.json().get("output", {}).get("task_status")
            if status == "SUCCEEDED":
                img_url = res.json()["output"]["results"][0]["url"]
                img_data = requests.get(img_url).content
                with open(output_path, 'wb') as handler:
                    handler.write(img_data)
                return output_path
            elif status == "FAILED":
                print("Image generation failed.")
                return None
            time.sleep(2)
    return None

def generate_video_consisid(prompt, image_path):
    """
    Uses ConsisID to generate a video based on the text prompt and generated image.
    """
    if not CONSIS_AVAILABLE:
        print("ConsisID is not available. Please install its dependencies.")
        return None
        
    output_video_path = "temp_generated_video.mp4"
    model_path = "ConsisID/ckpts" 
    
    if not os.path.exists(model_path):
        print(f"ConsisID models not found in {model_path}. Please download them first.")
        return None
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    subfolder = "transformer_ema" if os.path.exists(os.path.join(model_path, "transformer_ema")) else "transformer"

    # 1. Prepare models
    face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models(model_path, device, dtype)
    transformer = ConsisIDTransformer3DModel.from_pretrained(model_path, subfolder=subfolder, torch_dtype=dtype)
    pipe = ConsisIDPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=dtype)
    
    transformer.to(device, dtype=dtype)
    pipe.to(device)

    # 2. Process face from the generated image
    id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
        face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std,
        face_main_model, device, dtype, image_path, is_align_face=True
    )
    
    # 3. Generate video
    generator = torch.Generator(device).manual_seed(42)
    video_pt = pipe(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=False,
        guidance_scale=6.0,
        generator=generator,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=face_kps,
        output_type="pt",
    ).frames
    
    # 4. Save to MP4
    pt_image = video_pt[0]
    pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
    image_np = VaeImageProcessor.pt_to_numpy(pt_image)
    batch_video_frames = [VaeImageProcessor.numpy_to_pil(image_np)]
    
    export_to_video(batch_video_frames[0], output_video_path, fps=8)
    return output_video_path

def process_pipeline(prompt):
    print(f"Step 1: Generating Image using Qwen architecture for prompt: {prompt}")
    image_path = generate_image_qwen(prompt)
    
    video_path = None
    if image_path:
        print("Step 2: Generating Video using ConsisID...")
        video_path = generate_video_consisid(prompt, image_path)
    else:
        print("Image generation failed. Skipping video generation.")
        
    return image_path, video_path

def create_ui():
    with gr.Blocks(title="AI Image & Video Generator") as app:
        gr.Markdown("# 🚀 Text-to-Image & Video LLM Pipeline")
        gr.Markdown("This LLM application uses **awesome-qwen-image-3** prompt structures to generate an image and uses **ConsisID** to animate the image into a consistent video.")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Enter your text prompt here:", 
                    lines=4, 
                    placeholder="A boy walking along a city street, 35mm camera, classic film grain..."
                )
                generate_btn = gr.Button("Generate Image & Video", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(label="Generated Image (Qwen / Wanx)", type="filepath")
                video_output = gr.Video(label="Generated Video (ConsisID)")
                
        generate_btn.click(fn=process_pipeline, inputs=prompt_input, outputs=[image_output, video_output])
        
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()
