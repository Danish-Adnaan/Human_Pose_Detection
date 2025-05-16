import cv2
import numpy as np
import pickle
import logging
from ultralytics import YOLO
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pose_classifier')

class PoseClassifier:
    def __init__(self, model_path='models/pose_classifier.pkl'):
        """Initialize the pose classifier"""
        # Load the trained model
        logger.info("Loading pose classifier model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.classes = model_data['classes']
        self.n_features = 67  # Expected number of features
        
        # Initialize YOLO pose detector
        logger.info("Loading YOLO pose model...")
        self.pose_detector = YOLO('yolov8n-pose.pt')
        
        # Add class name mapping
        self.class_display_names = {
            'sleeping': 'Watching_screen'
        }
        
        logger.info("Pose classifier initialized successfully")
    
    def _extract_features(self, keypoints):
        """Extract features from keypoints for classification"""
        try:
            # Basic features (normalized coordinates and confidences)
            basic_features = keypoints.flatten()
            
            # Relative distances between key joints
            distances = []
            important_pairs = [
                (0, 1),   # nose to left eye
                (0, 2),   # nose to right eye
                (5, 6),   # shoulder width
                (11, 12), # hip width
                (5, 11),  # left body height
                (6, 12),  # right body height
                (13, 15), # left leg length
                (14, 16), # right leg length
                (7, 9),   # left arm length
                (8, 10)   # right arm length
            ]
            
            for i, j in important_pairs:
                dist = np.linalg.norm(keypoints[i, :2] - keypoints[j, :2])
                distances.append(dist)
            
            # Angles between joints
            angles = []
            angle_triplets = [
                (9, 7, 5),    # left arm angle
                (10, 8, 6),   # right arm angle
                (7, 5, 11),   # left shoulder angle
                (8, 6, 12),   # right shoulder angle
                (15, 13, 11), # left leg angle
                (16, 14, 12)  # right leg angle
            ]
            
            for i, j, k in angle_triplets:
                v1 = keypoints[i, :2] - keypoints[j, :2]
                v2 = keypoints[k, :2] - keypoints[j, :2]
                angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                angles.append(angle)
            
            # Combine all features
            features = np.concatenate([
                keypoints[:, :2].flatten(),  # Only x,y coordinates (34 features)
                keypoints[:, 2],             # Confidence scores (17 features)
                distances,                   # 10 distance features
                angles                       # 6 angle features
            ])
            
            # Ensure we have the correct number of features
            if len(features) != self.n_features:
                logger.error(f"Feature mismatch: got {len(features)}, expected {self.n_features}")
                return None
                
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def classify_pose(self, frame):
        """Detect and classify pose in a frame"""
        # Run YOLO pose detection
        results = self.pose_detector(frame, verbose=False)
        
        if not results or len(results) == 0:
            return frame, None, None
        
        result = results[0]
        if not result.keypoints or len(result.keypoints) == 0:
            return frame, None, None
        
        # Get keypoints
        keypoints = result.keypoints.data[0].cpu().numpy()  # Shape: (17, 3)
        
        # Normalize keypoint coordinates
        img_height, img_width = frame.shape[:2]
        keypoints[:, 0] = keypoints[:, 0] / img_width
        keypoints[:, 1] = keypoints[:, 1] / img_height
        
        # Extract features
        features = self._extract_features(keypoints)
        
        if features is None:
            return result.plot(), None, None
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and probability
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = probabilities.max()

            # Add this line to map the prediction to display name
            display_prediction = self.class_display_names.get(prediction, prediction)
            
            # Map the prediction to display name
            prediction = self.class_display_names.get(prediction, prediction)
            
            # Draw skeleton on frame
            annotated_frame = result.plot()
            
            return annotated_frame, prediction, confidence
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return result.plot(), None, None
    
    def run_webcam(self):
        """Run real-time pose classification on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        logger.info("Starting real-time pose classification...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Classify pose
                annotated_frame, prediction, confidence = self.classify_pose(frame)
                
                if prediction is not None:
                    logger.info(f"Detected pose: {prediction} ({confidence:.2f})")
                
                # Display result
                cv2.imshow('Pose Classification', annotated_frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Stopped pose classification")

if __name__ == "__main__":
    # Create and run classifier
    classifier = PoseClassifier()
    classifier.run_webcam() 