import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3):
        super(FeatureExtraction, self).__init__()
        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 2),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 4),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 8),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 8),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        try:
            for i, layer in enumerate(self.model):
                x = layer(x)
            return x
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        feature_A = F.normalize(feature_A, p=2, dim=1)
        feature_B = F.normalize(feature_B, p=2, dim=1)
        feature_A = feature_A.view(b, c, -1)
        feature_B = feature_B.view(b, c, -1).transpose(1, 2)
        correlation = torch.bmm(feature_B, feature_A)
        return correlation.view(b, h * w, h, w)


class FeatureRegression(nn.Module):
    def __init__(self):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(192, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.linear = nn.Linear(768, 50)

    def forward(self, x):
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (12, 1))
        x = x.reshape(x.size(0), -1)
        return self.linear(x)


class GMM(nn.Module):
    def __init__(self, input_nc, output_nc=3):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(input_nc, ngf=64, n_layers=3)
        self.extractionB = FeatureExtraction(3, ngf=64, n_layers=3)
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression()

    def forward(self, person, clothing):
        try:
            with open("debug_gmm.log", "w") as f:
                f.write("\nGMM Forward Pass:\n")
                f.write(f"Input person shape: {person.shape}\n")
                f.write(f"Input clothing shape: {clothing.shape}\n")
                f.write(f"person.dtype: {person.dtype}\n")
                f.write(f"clothing.dtype: {clothing.dtype}\n")
                f.write(f"person min/max: {person.min():.2f}/{person.max():.2f}\n")
                f.write(
                    f"clothing min/max: {clothing.min():.2f}/{clothing.max():.2f}\n"
                )

            person_features = self.extractionA(person)
            clothing_features = self.extractionB(clothing)

            with open("debug_gmm.log", "a") as f:
                f.write(f"\nExtracted person features shape: {person_features.shape}\n")
                f.write(
                    f"Extracted clothing features shape: {clothing_features.shape}\n"
                )
                f.write(
                    f"person_features min/max: {person_features.min():.2f}/{person_features.max():.2f}\n"
                )
                f.write(
                    f"clothing_features min/max: {clothing_features.min():.2f}/{clothing_features.max():.2f}\n"
                )

            correlation = self.correlation(person_features, clothing_features)

            with open("debug_gmm.log", "a") as f:
                f.write(f"\nCorrelation output shape: {correlation.shape}\n")
                f.write(
                    f"correlation min/max: {correlation.min():.2f}/{correlation.max():.2f}\n"
                )

            theta = self.regression(correlation)

            with open("debug_gmm.log", "a") as f:
                f.write(f"\nFinal theta shape: {theta.shape}\n")
                f.write(f"theta min/max: {theta.min():.2f}/{theta.max():.2f}\n")

            return theta
        except Exception as e:
            with open("debug_gmm.log", "a") as f:
                f.write("\nError in GMM forward pass:\n")
                import traceback

                traceback.print_exc(file=f)
            raise


def load_gmm(model_path):
    """Load the GMM model from a checkpoint file."""
    model = GMM(7)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pose_points_to_heatmap(pose_points, w=192, h=256, sigma=6):
    """Convert pose points to heatmap representation."""
    pose_map = np.zeros((w, h, 3), dtype=np.float32)
    shoulder_neck_indices = [2, 5, 1]
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    for i, idx in enumerate(shoulder_neck_indices):
        if pose_points[idx] is not None:
            x, y = pose_points[idx]
            if x is not None and y is not None:
                heatmap = np.exp(
                    -((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma**2)
                )
                pose_map[..., i] = heatmap

    return pose_map


def prepare_person_representation(
    rgb_image, pose_points, torso_mask, face_hair_mask, width=192, height=256
):
    """Prepare the person representation for the GMM model."""
    rgb_image = rgb_image.astype(np.float32) / 255.0
    torso_mask = cv2.resize(torso_mask, (rgb_image.shape[1], rgb_image.shape[0]))
    face_hair_mask = cv2.resize(
        face_hair_mask, (rgb_image.shape[1], rgb_image.shape[0])
    )

    person_repr = np.concatenate(
        [rgb_image, pose_points, torso_mask[..., np.newaxis]], axis=2
    )
    person_repr = torch.FloatTensor(person_repr).permute(2, 0, 1).unsqueeze(0)
    person_repr = F.interpolate(
        person_repr, size=(height, width), mode="bilinear", align_corners=True
    )

    return person_repr


def warp_clothing(gmm_model, person_repr, clothing_img):
    """Warp clothing image using GMM prediction."""
    with torch.no_grad():
        theta = gmm_model(person_repr, clothing_img)
        return tps_transform(theta, clothing_img)


def tps_transform(theta, clothing, target_height=256, target_width=192):
    """Apply Thin Plate Spline transformation to warp the clothing image."""
    batch_size = theta.size(0)
    num_control_points = theta.size(1) // 2

    grid_size = int(np.sqrt(num_control_points))
    target_control_points = []
    padding = 0.15
    for i in range(grid_size):
        for j in range(grid_size):
            x = 2 * ((j / (grid_size - 1)) * (1 - 2 * padding) + padding - 0.5)
            y = 2 * ((i / (grid_size - 1)) * (1 - 2 * padding) + padding - 0.5)
            target_control_points.append([x, y])

    target_control_points = (
        torch.FloatTensor(target_control_points).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    source_control_points = (
        torch.tanh(theta.view(batch_size, num_control_points, 2)) * 0.8
    )

    x = torch.linspace(-1, 1, target_width)
    y = torch.linspace(-1, 1, target_height)
    grid_y, grid_x = torch.meshgrid(y, x)
    grid_points = (
        torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )

    D = torch.cdist(target_control_points, target_control_points)
    K = D * torch.log(D + 1e-6)
    P = torch.cat(
        [torch.ones(batch_size, num_control_points, 1), target_control_points], dim=2
    )

    reg_lambda = 0.001
    L = torch.zeros(batch_size, num_control_points + 3, num_control_points + 3)
    L[:, :num_control_points, :num_control_points] = K + reg_lambda * torch.eye(
        num_control_points
    )
    L[:, :num_control_points, num_control_points:] = P
    L[:, num_control_points:, :num_control_points] = P.transpose(1, 2)

    Y = torch.cat([source_control_points, torch.zeros(batch_size, 3, 2)], dim=1)
    weights = torch.linalg.solve(L, Y)

    D = torch.cdist(grid_points, target_control_points)
    K = D * torch.log(D + 1e-6)
    P = torch.cat(
        [torch.ones(batch_size, target_height * target_width, 1), grid_points], dim=2
    )

    transformed_points = torch.bmm(K, weights[:, :num_control_points, :]) + torch.bmm(
        P, weights[:, num_control_points:, :]
    )
    transformed_points = transformed_points.view(
        batch_size, target_height, target_width, 2
    )
    transformed_points = torch.tanh(transformed_points) * 0.9

    return F.grid_sample(
        clothing,
        transformed_points,
        mode="bicubic",
        padding_mode="border",
        align_corners=True,
    )
