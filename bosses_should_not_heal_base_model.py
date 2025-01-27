import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class PoseEstimator:
    def __init__(self, model_type="lightning"):
        if model_type not in ["lightning", "thunder"]:
            raise ValueError("Model type must be either 'lightning' or 'thunder'")
            
        model_name = f"movenet_singlepose_{model_type}"
        model_handle = f"https://tfhub.dev/google/movenet/{model_name}/4"
        self.model = hub.load(model_handle)
        self.movenet = self.model.signatures['serving_default']
        
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def _process_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 192, 192)
        return tf.cast(input_image, dtype=tf.int32)

    def detect_pose(self, image_path):
        input_image = self._process_image(image_path)
        outputs = self.movenet(input_image)
        keypoints = outputs['output_0'].numpy().squeeze()
        
        return {
            name: {
                'x': int(x * input_image.shape[2]),
                'y': int(y * input_image.shape[1]),
                'confidence': float(confidence)
            }
            for idx, (y, x, confidence) in enumerate(keypoints)
            for name in [self.keypoint_names[idx]]
        }

    def get_specific_keypoints(self, keypoints, required_points=None):
        if required_points is None:
            required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        return {k: v for k, v in keypoints.items() if k in required_points}

    def validate_pose(self, keypoints, confidence_threshold=0.3):
        required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        return all(
            keypoints.get(point, {}).get('confidence', 0) >= confidence_threshold
            for point in required_points
        )