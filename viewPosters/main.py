from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Path to the base directory where beer folders are stored
BASE_DIR = Path("/Users/Raneet/Desktop/IBeer.ai/poster/realImage")

# CORS settings to allow API access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def welcome():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the IBeer.ai Image Retrieval API!"}

@app.get("/retrieve-images/{folder_name}")
def retrieve_images(folder_name: str):
    """
    Retrieve the URLs for 'poster.jpeg' and 'logo.jpeg' from a specific folder.
    """
    # Construct the path to the folder
    folder_path = BASE_DIR / folder_name

    # Check if the folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found")

    # Construct paths to the images
    poster_path = folder_path / "poster.jpeg"
    logo_path = folder_path / "logo.jpeg"

    # Check if both images exist in the folder
    missing_files = []
    if not poster_path.exists():
        missing_files.append("poster.jpeg")
    if not logo_path.exists():
        missing_files.append("logo.jpeg")

    if missing_files:
        raise HTTPException(
            status_code=404,
            detail=f"Missing files: {', '.join(missing_files)} in folder '{folder_name}'",
        )

    # Return URLs to access the images
    return {
        "poster_image_url": f"http://127.0.0.1:8000/image/{folder_name}/poster.jpeg",
        "logo_image_url": f"http://127.0.0.1:8000/image/{folder_name}/logo.jpeg",
    }

@app.get("/image/{folder_name}/{image_name}")
def get_image(folder_name: str, image_name: str):
    """
    Serve an image file given its folder name and file name.
    """
    # Construct the full path to the requested image
    image_path = BASE_DIR / folder_name / image_name

    # Validate the image path
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Image '{image_name}' not found in folder '{folder_name}'"
        )

    # Serve the image file as a response
    return FileResponse(str(image_path))
