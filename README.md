# Project_3

Image ML for virtual Gown try on.

# model selection

# training data

The training data will be the product images

# Purpose

To bring brides just a little something extra. Let them see themselves in that perfect dress on that
perfect day.

# Workflow

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

# Deployment

# CI/CD

# Env Reset

```bash
conda deactivate
conda deactivate
conda env remove --name=project_3 -y
conda env create -f environment.yml -y
conda activate project_3
```
