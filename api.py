from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from fastapi import HTTPException
from pathlib import Path
import matplotlib as plt
import numpy as np
import requests
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

app.mount("/static", StaticFiles(directory="output"), name="static")

# Handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins="http://localhost:4200",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
try:
    pose_detector = PoseDetector(model_path="models/graph_opt.pb")
    gmm_model = load_gmm("models/gmm_final.pth")

    # Move GMM model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gmm_model = gmm_model.to(device)
    print(f"Models loaded successfully. Using device: {device}")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

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
        "endpoints": {
            "/try-on": "POST - Try on a clothing item",
            "/": "GET - This information",
        },
    }


@app.post("/try-on")
async def try_on(user_image: UploadFile = File(...), reference_image: str = Form(...)):
    print(f"\n=== Starting try-on request ===")
    print(f"Reference image: {reference_image}")
    print(f"User image filename: {user_image.filename}")

    try:
        # Handle reference_image
        if reference_image.startswith(("http://", "https://")):
            try:
                print("Fetching reference image from URL...")
                response = requests.get(reference_image)
                response.raise_for_status()
                ref_img_data = response.content
                print("Successfully fetched reference image")
            except requests.RequestException as e:
                print(f"Error fetching reference image: {str(e)}")
                raise HTTPException(
                    status_code=400, detail=f"Failed to fetch reference image: {str(e)}"
                )
        else:
            print("Invalid reference image URL")
            raise HTTPException(
                status_code=400, detail="Reference image must be a valid HTTP(S) URL"
            )

        # Process user_image
        try:
            print("Reading user image data...")
            user_img_data = await user_image.read()
            print(f"Successfully read user image, size: {len(user_img_data)} bytes")
        except Exception as e:
            print(f"Error reading user image: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Failed to read uploaded image: {str(e)}"
            )

        # Decode images
        try:
            print("Decoding images...")
            ref_img = cv2.imdecode(
                np.frombuffer(ref_img_data, np.uint8), cv2.IMREAD_COLOR
            )
            user_img = cv2.imdecode(
                np.frombuffer(user_img_data, np.uint8), cv2.IMREAD_COLOR
            )

            if ref_img is None:
                raise ValueError("Failed to decode reference image")
            if user_img is None:
                raise ValueError("Failed to decode user image")

            print(f"Successfully decoded images:")
            print(f"- Reference image shape: {ref_img.shape}")
            print(f"- User image shape: {user_img.shape}")
        except Exception as e:
            print(f"Error decoding images: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Failed to decode images: {str(e)}"
            )

        # Store original dimensions
        orig_h, orig_w = user_img.shape[:2]
        print(f"Original dimensions: {orig_w}x{orig_h}")

        # Resize for pose detection
        print("Resizing image for pose detection...")
        pose_detect_img = cv2.resize(user_img, (368, 368))
        print(f"Resized for pose detection: {pose_detect_img.shape}")

        # Detect pose
        try:
            print("Running pose detection...")
            pose_points = pose_detector.detect(pose_detect_img)
            if not pose_points:
                raise ValueError("No pose points detected")

            # Scale pose points
            pose_points = [
                (
                    (int(p[0] * orig_w / 368), int(p[1] * orig_h / 368))
                    if p is not None
                    else None
                )
                for p in pose_points
            ]
            print(
                f"Detected {sum(1 for p in pose_points if p is not None)} pose points"
            )
        except Exception as e:
            print(f"Error in pose detection: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Pose detection failed: {str(e)}"
            )

        # Process keypoints
        print("Processing keypoints...")
        keypoints = [2, 5, 8, 11]  # Shoulders and hips
        for k in keypoints:
            if k not in range(len(pose_points)):
                print(f"Estimating missing keypoint {k}...")
                if k == 2:  # Left shoulder
                    pose_points[k] = estimate_keypoint(pose_points, 8, 5)
                elif k == 5:  # Right shoulder
                    pose_points[k] = estimate_keypoint(pose_points, 11, 2)
                elif k == 8:  # Left hip
                    pose_points[k] = estimate_keypoint(pose_points, 2, 11)
                elif k == 11:  # Right hip
                    pose_points[k] = estimate_keypoint(pose_points, 5, 8)
                else:
                    pose_points[k] = (0, 0)
                print(f"Keypoint {k}: {pose_points[k]}")

        print("Preparing tensors...")
        # Resize for GMM
        ref_img = cv2.resize(ref_img, (192, 256))
        user_img = cv2.resize(user_img, (192, 256))

        # Create pose heatmap and masks
        pose_map = pose_points_to_heatmap(pose_points, h=256, w=192)
        torso_mask = np.ones((256, 192), dtype=np.float32)
        face_hair_mask = np.zeros((256, 192), dtype=np.float32)

        # Prepare tensors
        try:
            print("Creating person representation...")
            person_repr = prepare_person_representation(
                pose_points, torso_mask, face_hair_mask
            )
            person_repr = torch.FloatTensor(person_repr).unsqueeze(0).to(device)
            print(f"Person representation shape: {person_repr.shape}")

            print("Creating reference tensor...")
            # Transpose the image to get the correct dimensions
            ref_img_transposed = cv2.transpose(ref_img)
            ref_tensor = (
                torch.FloatTensor(ref_img_transposed)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
                / 255.0
            )
            print(f"Reference tensor shape: {ref_tensor.shape}")
        except Exception as e:
            print(f"Error preparing tensors: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to prepare tensors: {str(e)}"
            )

        # Validate tensor shapes
        if ref_tensor.shape[1:] != (3, 192, 256):
            msg = f"Reference tensor has unexpected dimensions: {ref_tensor.shape[1:]}, expected (3, 192, 256)"
            print(msg)
            raise HTTPException(status_code=400, detail=msg)
        if person_repr.shape[1:] != (7, 256, 192):
            msg = f"Person representation tensor has unexpected dimensions: {person_repr.shape[1:]}, expected (7, 256, 192)"
            print(msg)
            raise HTTPException(status_code=400, detail=msg)

        # Run GMM model
        try:
            print("Running GMM model...")
            with torch.no_grad():
                theta = gmm_model(person_repr, ref_tensor)
                print("Generated theta parameters")
                warped_ref = tps_transform(theta, ref_tensor)
                print("Applied TPS transform")
                warped_ref_np = (
                    warped_ref.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
                )
                warped_ref_np = warped_ref_np.astype(np.uint8)
                print("Converted warped reference to numpy array")
                print(f"Warped reference shape: {warped_ref_np.shape}")
        except Exception as e:
            print(f"Error in GMM model execution: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"GMM model execution failed: {str(e)}"
            )

        # Final processing
        print("Performing final image processing...")
        try:
            # Ensure both images are in the same size for blending
            print(f"Original user image shape: {user_img.shape}")
            print(f"Warped reference shape before resize: {warped_ref_np.shape}")

            # Resize warped clothing to match user image size
            warped_ref_resized = cv2.resize(
                warped_ref_np, (user_img.shape[1], user_img.shape[0])
            )
            print(f"Warped reference shape after resize: {warped_ref_resized.shape}")

            # Ensure both images have the same number of channels
            if len(user_img.shape) != len(warped_ref_resized.shape):
                if len(user_img.shape) == 3:
                    warped_ref_resized = cv2.cvtColor(
                        warped_ref_resized, cv2.COLOR_GRAY2BGR
                    )
                else:
                    user_img = cv2.cvtColor(user_img, cv2.COLOR_GRAY2BGR)

            print(
                f"Final shapes - User: {user_img.shape}, Warped: {warped_ref_resized.shape}"
            )

            # Blend images
            blended_img = cv2.addWeighted(user_img, 0.7, warped_ref_resized, 0.3, 0)
            _, buffer = cv2.imencode(".jpg", blended_img)

            # Save the blended image
            blended_image_path = output_dir / "blended_result.jpg"
            cv2.imwrite(str(blended_image_path), blended_img)

            print("Successfully blended images")

            print("=== Completed try-on request successfully ===\n")
            # Encode result
            return Response(buffer.tobytes(), media_type="image/jpeg")
        except Exception as e:
            print(f"Error in final image processing: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process final image: {str(e)}"
            )
    except HTTPException as e:
        print(f"=== Request failed with HTTP error: {e.detail} ===\n")
        raise e
    except Exception as e:
        print(f"=== Request failed with unexpected error: {str(e)} ===\n")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


def estimate_keypoint(pose_points, k1, k2):
    if k1 < len(pose_points) and k2 < len(pose_points):
        x = (pose_points[k1][0] + pose_points[k2][0]) // 2
        y = (pose_points[k1][1] + pose_points[k2][1]) // 2
        return (x, y)
    return None


def pose_points_to_heatmap(pose_points, h, w):
    heatmap = np.zeros((h, w), dtype=np.float32)
    for i, point in enumerate(pose_points):
        if point is not None:
            x, y = point
            if 0 <= y < h and 0 <= x < w:  # Ensure point is within bounds
                heatmap[y, x] = 1.0
    return heatmap


def prepare_person_representation(pose_points, torso_mask, face_hair_mask):
    person_repr = np.zeros((7, 256, 192), dtype=np.float32)

    # Add mask channels
    person_repr[0, :, :] = cv2.resize(torso_mask, (192, 256))
    person_repr[1, :, :] = cv2.resize(face_hair_mask, (192, 256))

    # Add pose point channels (we only use 5 key points)
    key_indices = [2, 5, 8, 11, 1]  # shoulders, hips, and neck
    for i, idx in enumerate(key_indices):
        if idx < len(pose_points) and pose_points[idx] is not None:
            x, y = pose_points[idx]
            if 0 <= y < 256 and 0 <= x < 192:  # Ensure point is within bounds
                person_repr[i + 2, y, x] = 1.0

    return person_repr
