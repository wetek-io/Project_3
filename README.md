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

## model selection

This is a multi(3) model application.

- Segmentation (DeepLabv3+ or U-Net):

  - identify which segments of the image are the model(human) and the gown it self

- Pose Evaluation (OpenPose or MediaPipe Pose):

  - For detecting the shoulders, hips, and bust anchor points

- Image Generation (Pix2Pix, CycleGAN or VITON/CP-VTON):

  - Generate image of bride wearing gown

## training data

The training data will be the product images
[Zander](https://www.maggiesottero.com/sottero-and-midgley/zander/11869)

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

## Deployment

## CI/CD

## Env Reset

```bash
conda deactivate
conda deactivate
conda env remove --name=project_3 -y
conda env create -f environment.yml -y
conda activate project_3
```

### Developer Notes:

Project development data: [Zander](https://www.maggiesottero.com/sottero-and-midgley/zander/11869)
