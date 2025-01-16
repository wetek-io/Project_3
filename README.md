# Project_3

Image ML for virtual Gown try on.

**From A High Level**
This app is ment to allow brides to try on gown virtually while visiting our website. The model training data
will be the publicly available product images from https://www.maggiesottero.com. These images are of the
model(human) and the and the gown. The challenge with this is nuanced wedding gowns are highly detailed
so the images must be as high quality as achievable. Luckily each product only has reality few images typically 10-20. For this project I plan on selecting just a single gown. I chose this one because it
presents the fewest issues. The [Zander](https://www.maggiesottero.com/sottero-and-midgley/zander/11869) gown
has good contract, a large number of images, including different angles, as well as close up images of the
gown on a maniquen, there is also a black and a white version of the dress which will help a great deal for
maintain gown detail in the final generation.

## Environment

[Miniconda MacOSX arm64](https://pytorch.org/get-started/locally/#mac-anaconda)

## model selection

This is a multi(3) model application.

- Segmentation ([Convolutional Neural Network]('utils.py')):

  - A CNN is especially adapt at edge detection such as the edge of a dress of body part.

  - This project uses a model based on the pytorch tutorial model. This is a good starting
    point for a custom CNN build. This model is currently only taking one input and applying
    softmax with a dimension of one

- Pose Evaluation (OpenPose or MediaPipe Pose):

  - For detecting the shoulders, hips, and bust anchor points

- Image Generation (Pix2Pix, CycleGAN or VITON/CP-VTON):

  - Generate image of bride wearing gown

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
conda env remove --name=project_3 -y
conda env create -f env.yml -y
conda activate project_3
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
