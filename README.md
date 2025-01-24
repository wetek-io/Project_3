# Project_3

Image ML for virtual Gown try on.

**From A High Level**
This app is ment to allow brides to try on gown virtually while visiting our website. This project
uses a custom UNet model which will be trained on the Meta AI segmentation dataset SA-1B.

The model needs to be pre-trained for the app.

When a user uploads an image it will will be segmented by the model in parallel with OpenPose.

## Environment

[Miniconda MacOSX arm64](https://pytorch.org/get-started/locally/#mac-anaconda)

## model selection

This is a multi(3) model application.

- Segmentation ([Convolutional Neural Network]('utils.py')):

  - [Image-to-Image G]
  - A CNN is especially adapt at edge detection such as the edge of a dress of body part.
  - This project uses a model based on the pytorch tutorial model. This is a good starting
    point for a custom CNN build. This model is currently only taking one input and applying
    softmax with a dimension of one

- Pose Evaluation (OpenPose):

  - [OpenPoses (A collection of poses for OpenPose)](https://openposes.com/)
  - For detecting the shoulders, hips, and bust anchor points

- Image Generation (CycleGAN):

  - Generate image of bride wearing gown

- **!!Important!! Further reading**

  - [CycleGAN (Medium)](https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca)

  - [Convolutional Pose Machines (PDF)](https://arxiv.org/pdf/1602.00134)
  - [CycleGAN.ipynb (Google colab)](https://colab.research.google.com/drive/1BuI-9P1-ku00Nc1tPbBhoeL006-3tNUS?usp=sharing)

## training data

- [SA-1B](https://ai.meta.com/datasets/segment-anything-downloads/)
- [tensorflow](https://tensorflow.google.cn/datasets/catalog/segment_anything)

  - A Meta AI dataset with 11M images and 1.1B mask annotations.

## Purpose

To bring brides just a little something extra. Let them see themselves in that perfect dress on that
perfect day.

## Workflow

- Data Wrangling
  - Collect images and corresponding masks (grayscale masks for segmentation).
  - Resize images and masks to a fixed size (e.g., 250x250).
  - Convert images and masks to PyTorch tensors for use in the model.
- Modeling
  - Build a segmentation model with two output channels (user, gown)
  - Maintain consistent spacial dimensions with appropriate convolution layer padding
- Training
  - Load data with PyTorch DataLoader
  - Use CrossEntropyLoss on images and masks
  - monitor for epoch loss changes
- Evaluation
  - Switch to evaluation mode (model.eval())
  - Generate prediction on model's validation/test data
  - Visualize the prediction masks against the truth masks
- Inference
  - Save training model
  - Load the model for inference on new image
  - Generate and visualize segmentation masks of the user's image and the product's images
- Visualization and debugging
  - Visualization is handled by matplotlib or saved as images

## User Experience

Example Flow Visualization 1.

- Welcome Screen:
  - Title:
    - “Welcome to the Image Segmentation Tool!”
  - Button:
    - “Get Started”
  - Upload:
    - Drag-and-drop or click to upload.
  - Preview:
    - Uploaded image is shown.
  - Results:
    - Spinner with “Processing…” message.
    - Display image of bride in dress.
    - Option to download or share.
  - Next Steps:
    - “Upload Another” or “Try Advanced Tools.”

## Deployment

## CI/CD

## Env Reset

```bash
conda deactivate
conda deactivate
conda env remove -n project_3 -y
conda env create -f env.yml -y
conda activate project_3
conda clean --all --yes
```

### Developer Notes:

Project development data: [Zander](https://www.maggiesottero.com/sottero-and-midgley/zander/11869)

### Research:

- [virtual-gown-tryon (A previous personal project)](https://github.com/steven-midgley/virtual-gown-tryon)

- [LIP(Look Into Person)](https://www.sysu-hcp.net/lip/index.php) - A large scale dataset for the sematic understanding of person

- [DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)

- [U-Net](https://github.com/milesial/Pytorch-UNet)

  - [U-Net White paper](https://arxiv.org/pdf/1505.04597v1)
  - [The U-Net: A complete guide (Medium)](https://medium.com/@alejandro.itoaramendia/decoding-the-u-net-a-complete-guide-810b1c6d56d8#https://medium.com/@alejandro.itoaramendia/convolutional-neural-networks-cnns-a-complete-guide-a803534a1930)
  - [tensorflow segment anything](https://github.com/tensorflow/datasets/blob/master/docs/catalog/segment_anything.md)

- [Convolutional Image Segmentation](https://arxiv.org/pdf/1706.05587v3)

## Focus Areas for the 2.5-Week Timeline:

1. Pose Detection & Body Segmentation: Use pre-trained models to save time.

2. Gown Overlay & Warping: Focus on 2D overlays with minimal physics for now.

3. Frontend & Backend Integration: Ensure seamless communication between user inputs and the models.

4. Testing & Iteration: Prioritize basic functionality over perfection.

## Team Breakdown and Task Assignments

## Pose Detection & Keypoints Extraction (JD, Roger):

Goal: Quickly implement pose estimation to detect the user’s posture.
Tools: Use a pre-trained model like TensorFlow MoveNet or PoseNet.

Tasks:

- Integrate the pose detection model for single images.
- Output keypoints (e.g., shoulders, hips) for gown alignment.
- Deliverable: API/module that takes an image and returns pose keypoints.
- Time Allocation: 4 days

## Body Segmentation Specialist (Carson, Steven):

Goal: Identify and segment the user’s body from the background.
Tools: Use a pre-trained segmentation model like MediaPipe Selfie Segmentation or DeepLabV3+.

Tasks:

- Segment the body to create a mask for overlaying the gown.
- Handle simple backgrounds for faster processing.
- Deliverable: API/module to return body masks from uploaded images.
- Time Allocation: 4 days.

## Gown Overlay and Warping (Keri, Dane):

Goal: Develop a module to overlay and align the gown with the detected pose.
Tools: Use simple transformations (e.g., Thin Plate Splines or Affine Transformations).

Tasks:

- Align gowns to shoulders, hips, and legs based on pose keypoints.
- Add minimal scaling to adjust for body size variations.
- Deliverable: A functional overlay module for at least 2–3 sample gowns.
- Time Allocation: 6 days.

## Frontend Developer (Steven):

Goal: Build a simple, user-friendly interface for the try-on experience.
Tools: React Native or Flutter.

Tasks:

- Allow users to upload photos and select gowns.
- Display results from the backend (pose, segmentation, and gown overlay).
- Deliverable: A working UI that integrates with the backend.
- Time Allocation: 7 days.

## Backend Developer (Reis, Ian):

Goal: Set up the server for processing pose detection, segmentation, and gown fitting.
Tools: Use FastAPI or Flask.

Tasks:

- Host models and manage API calls from the frontend.
- Handle image uploads and return processed results.
- Deliverable: A functional backend with API endpoints for the frontend.
- Time Allocation: 7 days.

## Project Manager & QA Specialist (Ian, Keri):

Goal: Ensure coordination, testing, and iterative improvements.

Tasks:

- Manage timelines and ensure deliverables align with milestones.
- Test each module for performance and accuracy.
- Provide feedback for quick fixes and optimizations.
- Deliverable: A project roadmap, test cases, and a consolidated MVP demo.
- Time Allocation: Continuous, with focused testing in the last 4 days.
