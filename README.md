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
  - [The U-Net: A complete guide (Medium)]: (https://medium.com/@alejandro.itoaramendia/decoding-the-u-net-a-complete-guide-810b1c6d56d8#https://medium.com/@alejandro.itoaramendia/convolutional-neural-networks-cnns-a-complete-guide-a803534a1930)

- [Convolutional Image Segmentation](https://arxiv.org/pdf/1706.05587v3)
