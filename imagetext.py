from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import base64
from io import BytesIO
from pydantic import BaseModel
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model Initialization ----
image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch_device)

# ---- Define Request Model ----
class TextRequest(BaseModel):
    prompt: str

# ---- Image Captioning Route ----
@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        inputs = image_processor(images=image, return_tensors="pt")
        output = image_model.generate(**inputs)
        caption = image_processor.decode(output[0], skip_special_tokens=True)
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# ---- Image Generation Route ----
@app.post("/generate-image/")
def generate_image(request: TextRequest):
    try:
        image = stable_diffusion_pipe(request.prompt, height=512, width=512).images[0]
        if image.mode == "RGBA":
            image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
