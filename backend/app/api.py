from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io, os
from .vision_model import VisionToPrompt
import dotenv
dotenv.load_dotenv()
router = APIRouter()
VISION_MODEL_PATH=os.getenv("VISION_MODEL_PATH","vision-model")
vision=VisionToPrompt(VISION_MODEL_PATH)
import genai
client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
@router.post("/prompt/from-image")
async def prompt_from_image(file:UploadFile=File(...)):
    img=Image.open(io.BytesIO(await file.read())).convert("RGB")
    return {"prompt": vision.generate_prompt(img)}
