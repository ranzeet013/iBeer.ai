from PIL import Image
from fastapi import HTTPException, UploadFile
import io

async def read_image(file: UploadFile) -> Image.Image:
    """Read and return the uploaded image file."""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
