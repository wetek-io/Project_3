# Project 3 Overview - Transforming Bridal and Apparel Shopping with AI

### Business Problem  

Bridal shopping is often **stressful, time-consuming, and logistically challenging**. Brides may not have access to a **wide range of dresses** in physical stores, and trying on multiple gowns can be exhausting. This limits their ability to make confident purchasing decisions, especially in an era where online shopping is increasingly preferred.  

### Solution  

The **Virtual Try-On App** addresses these challenges by providing a seamless, interactive shopping experience. The app is:  

1. **Site Agnostic** – Works with images from any website or store, allowing broad compatibility.  
2. **Product Agnostic** – Enables users to try on any outfit (gowns, casual wear, etc.) by uploading a digital image of the product and themselves.  
3. **Convenient** – Delivers a **personalized** shopping experience with instant visualizations, eliminating the need for physical store visits.  

### Impact  

- **Empowers customers** to visualize how they’ll look in various outfits, increasing confidence in their purchase decisions.  
- **Enhances online shopping conversion rates** by reducing uncertainty and improving engagement.  
- **Saves time and effort**, making bridal shopping more enjoyable and efficient.  

The **Virtual Try-On App** is designed to **transform the shopping experience** by bridging the gap between digital browsing and real-world confidence. 

# Model Pipeline

Image ML for virtual gown try on.

**From A High Level**

This app is meant to allow brides to try on gowns virtually while visiting our website. This project uses a custom UNet model which will be trained on the Meta AI segmentation dataset SA-1B.

The model needs to be pre-trained for the app.

When a user uploads an image it will will be segmented by the model in parallel with OpenPose.

## Environment

[Anaconda](https://www.anaconda.com/)

## Model Selection

This is a multi-model application.

- Segmentation ([Convolutional Neural Network]('utils.py')):

  - This model utilizes two inputs and applies Sigmoid.

- Pose Evaluation (OpenPose):

  - [OpenPoses (A collection of poses for OpenPose)](https://openposes.com/)
  - For detecting the shoulders, hips, and bust anchor points

- Segmentation ([Convolutional Neural Network]('utils.py')):
  
  - This model utilizes two inputs and applies Softmax.
 
- Image Generation (CycleGAN):

  - Generate image of bride wearing the selected gown.
 
### Why Angular 18?
The front end of the **Virtual Try-On App** is built using **Angular 18**, chosen specifically for its efficiency, scalability, and ease of use for the client. This framework ensures a **loosely coupled** architecture, keeping the UI lightweight while relying on the API for processing.

### Key Benefits of Angular 18
- **Client Simplicity** – The client, **Maggie Sottero**, only needs to supply existing **image links** without additional processing.
- **Efficient Data Flow** – The **reference image URL** (product image) and **user-uploaded image** are sent **unchanged** directly to the API, maintaining a clean and structured data pipeline.
- **Scalability & Maintainability** – Angular’s component-based architecture allows for easy scaling and future enhancements.
- **Optimized Performance** – Angular 18 leverages Ivy rendering and improved reactivity, ensuring a **smooth** and **fast user experience**.
- **Strict TypeScript Support** – Ensures robust type safety and fewer runtime errors.

  **Future Additions**

  - [CycleGAN (Medium)](https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca)
  - [Convolutional Pose Machines (PDF)](https://arxiv.org/pdf/1602.00134)
  - [CycleGAN.ipynb (Google colab)](https://colab.research.google.com/drive/1BuI-9P1-ku00Nc1tPbBhoeL006-3tNUS?usp=sharing)
 
# Data Collection, Cleanup & Exploration

## Data Collection
- **[SA-1B](https://ai.meta.com/datasets/segment-anything-downloads/) Dataset (Meta AI)**
  - A Meta AI dataset with 11M images and 1.1B mask annotations
- Additional fashion datasets for model fine-tuning
- Collected images and corresponding masks (grayscale masks for segmentation)

## Data Cleanup & Preprocessing
- Resized images and masks to a fixed size (e.g., 250x250)
- Converted images and masks to **PyTorch** tensors for deep learning models
- Removed noise and ensured data consistency across samples

## Exploration & Workflow
- **Segmentation Model Development** – Built a segmentation model with two output channels
- **Pose Detection** – Integrated **OpenPose** to capture key points for garment alignment
- **Training Process** – Loaded data using **Torch DataLoader**, applied **CrossEntropyLoss**, and monitored epoch loss changes
- **Model Evaluation** – Used **IoU** (Intersection over Union) to assess accuracy

## Training & Evaluation
- Loaded data with **Torch DataLoader**, applied **CrossEntropyLoss**, and monitored loss
- Evaluated using **IoU** (Intersection over Union), which measures segmentation accuracy
- **IoU** is a metric that evaluates segmentation accuracy by comparing the overlap between the predicted mask and the actual mask
- **Higher IoU** values indicate better segmentation quality; a value close to 1 suggests near-perfect segmentation
- **Results**: Achieved **98-99% IoU**, ensuring highly accurate segmentation & garment overlay

---

## Purpose

To bring brides just a little something extra. Let them see themselves in that perfect dress on that perfect day.

## Why this Approach

- Agnostic Segmentation: The app is not tied to specific brands or clothing styles, making it universally adaptable for any product line.

- Minimal Client-Side Requirements: Clients only need to provide image URLs of their products, drastically reducing onboarding complexity and setup time.

- Scalable and Modular Architecture: Loosely coupled front-end, API, and back-end components allow for seamless scaling and efficient maintenance.

- Pre-Trained Model Integration: Leveraging pre-trained models ensures a faster time-to-market while retaining room for customization and improvement.

- User-Centric Design: The interface is intuitive, allowing brides to upload a single photo and instantly see results without needing complex inputs.

- Future-Ready Framework: Built with Angular18 and modular design principles, the app is prepared for rapid iteration and feature expansion.

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
    - “Go ahead. Try it on.”
  - Image Slider:
    - These are client product image i.e. Maggie Sottero wedding dress
    - reference_image
    - User interacts with slider until to select a reference_image
  - Upload:
    - This section is hidden until the user has selected a reference_image
    - They can either drag-and-drop or click to upload.
    - Users user_image is displayed under the upload CTA
  - Image Generation
    - both reference_image and user_image are sent to the backend models via the fastapi endpoint
      - AI machine stuff magic
  - Display Results:
    - Spinner with “Processing…” message.
    - Display image of bride in dress.
  - Next Steps:
    - Option to download or share.
    - “Upload Another” or “Try Advanced Tools. (Future)”

## Env Reset

```bash
conda deactivate
conda deactivate
conda env remove -n project_3 -y
conda env create -f env.yml -y
conda activate project_3
conda clean --all --yes
```

## Local Deployment

To run locally you will need to manually start both the fastapi server and the angular server.

fastapi:

```bash
conda activate project_3 # if applicable
fastapi dev api.py
```

angular(ui):

```bash
cd ui
ng serve
```

## Remote Deployment

This app will be entirely deployed from a DigitalOcean GPU droplet.

## Continuous Integration/Continuous Deployment

None at the moment. Future plans include automation for CI/CD.

### 📈 Future Development  

The **Virtual Try-On App** has several planned enhancements aimed at improving user experience, enhancing realism, and expanding functionality.  

#### 360-Degree Video Uploads  
- Users can upload **360-degree videos** to get a full view of the outfit.  

#### Adjust Sizes / Try on Real Sizes  
- Allow users to enter their **measurements** for precise gown fitting and realistic size adjustments.  

#### Sustainability Insights  
- Highlight **eco-friendly gown options** or outfits made from sustainable materials.  

#### Hair & Makeup Customization  
- Provide options for **hairstyles, hair colors, and makeup try-ons** to complete the look.  

#### Body Shape Customization  
- Add a feature to adjust the **user’s digital body representation** to match specific body shapes or sizes.  

#### Gown Filtering & Recommendations  
- Introduce **AI-driven recommendations** based on uploaded user photos and preferences (e.g., body shape, color preferences).  

#### Fabric & Motion Simulation  
- Enhance gown visualization with **fabric flow and motion effects** for a more realistic experience.  

#### Retail Integration  
- Collaborate with **bridal and fashion retailers** to include their inventory directly within the app.  

#### Accessories & Shoes  
- Enable users to **mix and match** shoes, jewelry, and other accessories to complete their virtual try-on experience.  

#### Virtual Backgrounds  
- Allow users to **select virtual backgrounds** to see the gown in different settings.  

#### Virtual Mirror Functionality  
- Mimic an **in-store fitting room** with a real-time, interactive experience.  

#### Inclusivity Features  
- Support users with **diverse physical needs** (e.g., wheelchair poses) to create a fully inclusive experience.  

#### Real-Time Feedback from Friends & Family  
- Include a **"share" feature** so users can get feedback on try-ons via social media or private links.  

These future developments will make the **Virtual Try-On App** more accurate, interactive, and accessible, ensuring a seamless digital shopping experience for all users.  


### Research:

- [virtual-gown-tryon (A previous personal project)](https://github.com/steven-midgley/virtual-gown-tryon)

- [DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)

- [U-Net](https://github.com/milesial/Pytorch-UNet)

  - [U-Net White paper](https://arxiv.org/pdf/1505.04597v1)
  - [The U-Net: A complete guide (Medium)](https://medium.com/@alejandro.itoaramendia/decoding-the-u-net-a-complete-guide-810b1c6d56d8#https://medium.com/@alejandro.itoaramendia/convolutional-neural-networks-cnns-a-complete-guide-a803534a1930)
    
- [Convolutional Image Segmentation](https://arxiv.org/pdf/1706.05587v3)

## Focus Areas for the 2 Week Timeline:

1. Pose Detection & Body Segmentation: Use pre-trained models to save time.

2. Gown Overlay & Warping: Focus on 2D overlays with minimal physics for now.

3. Frontend & Backend Integration: Ensure seamless communication between user inputs and the models.

4. Testing & Iteration: Prioritize basic functionality over perfection.

# Team Breakdown and Task Assignments

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

## Gown Overlay and Warping (Keri):

Goal: Develop a module to overlay and align the gown with the detected pose.
Tools: Use simple transformations (e.g., Thin Plate Splines or Affine Transformations).

Tasks:

- Align gowns to shoulders, hips, and legs based on pose keypoints.
- Add minimal scaling to adjust for body size variations.
- Deliverable: A functional overlay module for at least 2–3 sample gowns.
- Time Allocation: 12 days.

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
