from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
import numpy as np
import torch
import cv2
import io
import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# local
from utils.posedetector import PoseDetector
from utils.gmm import load_gmm, tps_transform

# Initiate app
app = FastAPI(title="Virtual Try-On API")

# Handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Angular's dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
pose_detector = PoseDetector(model_path="models/graph_opt.pb")
gmm_model = load_gmm("models/gmm_final.pth")

# Move GMM model to GPU (H100)
device = "cuda" if torch.cuda.is_available() else "cpu"
gmm_model = gmm_model.to(device)

# Create output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """
    Reads uploaded user file, converts contents to
    numpy array, which is then used to decode the
    array, and returns RGB image.
    """
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Virtual Try-On API",
        "version": "1.0.0",
        "description": "API for trying on clothing items on person images",
        "redirects": {},
        "endpoints": {
            "/try-on": "POST - Try on a clothing item",
            "/": "GET - This information",
        },
    }


@app.post("/try-on")
async def try_on(
    user_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
):
    """
    Try on a clothing item on a person image.

    Parameters:
    - user_image: Image of the person
    - reference_image: Image of the clothing item

    Returns:
    - Image file with the clothing item warped onto the person
    """
    try:
        # Process uploaded images
        user_img = process_uploaded_image(user_image)
        ref_img = process_uploaded_image(reference_image)
        print(user_img)
        print(ref_img)
        # Get pose points
        pose_points = pose_detector.detect(user_img)

        # Check key points
        keypoints = [2, 5, 8, 11]  # Example indices for shoulders and hips
        if not all(k in pose_points for k in keypoints):
            raise HTTPException(
                status_code=400,
                detail="Missing critical pose keypoints in the person image",
            )

        # Resize reference and user image to make GMM input
        ref_img = cv2.resize(ref_img, (192, 256))
        user_img = cv2.resize(user_img, (192, 256))

        # Convert images to tensors
        ref_tensor = (
            torch.FloatTensor(ref_img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        )
        user_tensor = (
            torch.FloatTensor(user_img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        )

        # Validate user_tensor
        if user_tensor.shape[1:] != (3, 192, 256) or ref_tensor.shape[1:] != (
            3,
            192,
            256,
        ):
            raise HTTPException(
                status_code=400, detail="Input tensors have unexpected dimensions"
            )

        # Run GMM model
        with torch.no_grad():
            theta = gmm_model(user_tensor, ref_tensor)
            warped_ref = tps_transform(theta, ref_tensor)
            warped_ref_np = warped_ref.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
            warped_ref_np = (warped_ref_np).astype(np.uint8)

        # Resize warped shirt to original size
        warped_ref_resized = cv2.resize(
            warped_ref_np, (user_img.shape[1], user_img.shape[0])
        )

        # Blend and weight reference and user images
        blended_img = cv2.addWeighted(user_img, 0.7, warped_ref_resized, 0.3, 0)

        # Save blended image
        output_path = output_dir / "blended_result.jpg"
        cv2.imwrite(str(output_path), blended_img)

        # Encode the image and stream it
        _, buffer = cv2.imencode(".png", blended_img)
        stream = io.BytesIO(buffer)
        return StreamingResponse(
            stream,
            media_type="image/jpg",
            headers={"Content-Disposition": "attachment; filename=blended_result.jpg"},
        )
    finally:
        user_image.file.close()
        reference_image.file.close()
