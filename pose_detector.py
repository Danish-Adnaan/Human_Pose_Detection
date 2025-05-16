import cv2
import numpy as np
import torch
import logging
import os
import time
import math
import traceback
import pickle
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from PIL import Image
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)

# Try to import TensorFlow-related libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    tensorflow_available = True
except ImportError:
    tensorflow_available = False
    logging.warning("TensorFlow not available, some features will be disabled")

class PoseDetector:
    """
    A class to detect and classify human poses using YOLOv8 and a machine learning classifier.
    """
    def __init__(self, model_path="yolov8m-pose.pt", confidence=0.25, use_mediapipe=False):
        """
        Initialize the PoseDetector with YOLOv8 model and pose classifier.
        
        Args:
            model_path (str): Path to YOLOv8 model
            confidence (float): Confidence threshold for detections
            use_mediapipe (bool): Whether to use MediaPipe as fallback
        """
        self.logger = logging.getLogger('pose_detector')
        self.confidence_threshold = confidence
        self.use_mediapipe = use_mediapipe
        
        # Load YOLO model
        self.logger.info("Loading YOLOv8 model...")
        start_time = time.time()
        try:
            self.model = YOLO(model_path)
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model = None
        
        # Initialize classifier
        self.classifier = None
        self.pose_classes = []
        self.feature_scaler = None
        self.expected_feature_length = 67  # Updated to match training implementation
        
        # Default location for classifier
        default_classifier_path = 'models/pose_classifier.pkl'
        
        if os.path.exists(default_classifier_path):
            self._load_classifier(default_classifier_path)
        else:
            self.logger.warning(f"No pose classifier found at {default_classifier_path}. Only keypoint detection will be available.")
        
        # Create fallback classifier if needed
        if self.classifier is None:
            self._create_fallback_classifier()
        
        # Initialize performance metrics
        self.fps = 0
        self.processing_time = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        # Initialize detection results
        self.current_pose = "Unknown"
        self.current_confidence = 0.0
        self.keypoints = None
        
        self.lock = threading.Lock()
        self.result_queue = queue.Queue()
    
    def _load_classifier(self, classifier_path):
        """Load the pose classifier model from disk"""
        try:
            self.logger.info(f"Loading pose classifier from {classifier_path}")
            with open(classifier_path, 'rb') as f:
                classifier_data = pickle.load(f)
            
            # Verify the contents of the classifier data    
            if 'model' not in classifier_data:
                self.logger.error("Missing 'model' in classifier data")
                self.classifier = None
                return
                
            if 'classes' not in classifier_data:
                self.logger.error("Missing 'classes' in classifier data")
                self.classifier = None
                return
            
            self.classifier = classifier_data['model']
            self.pose_classes = classifier_data['classes']
            self.feature_scaler = classifier_data.get('scaler', None)
            
            # Verify classifier is a valid model
            if not hasattr(self.classifier, 'predict') or not hasattr(self.classifier, 'predict_proba'):
                self.logger.error("Loaded classifier doesn't have required methods")
                self.classifier = None
                return
                
            self.logger.info(f"Loaded pose classifier with {len(self.pose_classes)} classes: {self.pose_classes}")
            
            # Additional validation of the model
            if isinstance(self.classifier, RandomForestClassifier):
                self.logger.info(f"RandomForest classifier with {self.classifier.n_estimators} trees")
                self.logger.info(f"Feature importances: {self.classifier.feature_importances_[:5]}...")
            
            # Indicate if we have a scaler
            if self.feature_scaler is not None:
                self.logger.info("Feature scaler is available for normalization")
            else:
                self.logger.info("No feature scaler found - using raw features")
                
        except FileNotFoundError:
            self.logger.error(f"Classifier file not found at {classifier_path}")
            self.classifier = None
        except Exception as e:
            self.logger.error(f"Failed to load classifier from {classifier_path}: {str(e)}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.classifier = None
    
    def _create_fallback_classifier(self):
        """Create a simple fallback classifier with basic poses"""
        self.logger.info("Creating fallback classifier with basic poses")
        try:
            # Define basic pose classes
            self.fallback_classes = ["Standing", "Sitting", "Walking/Running"]
            
            # Create a simple RandomForest classifier
            self.fallback_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
            
            # Create some synthetic training data for basic poses
            X_train = []
            y_train = []
            
            # Standing features (10 samples with variations)
            for _ in range(10):
                # Create feature vector with some randomness to simulate variation
                features = []
                
                # X,Y coordinates (34 features)
                for i in range(17):
                    x = 0.5 + np.random.normal(0, 0.05)  # Centered with noise
                    y = 0.3 + i * 0.05 + np.random.normal(0, 0.02)  # Vertically arranged
                    features.extend([x, y])
                
                # Confidence scores (17 features)
                features.extend([0.8 + np.random.normal(0, 0.1) for _ in range(17)])
                
                # Distance features (10 features)
                features.extend([0.2 + np.random.normal(0, 0.05) for _ in range(10)])
                
                # Angle features (6 features)
                features.extend([np.pi/2 + np.random.normal(0, 0.2) for _ in range(6)])
                
                X_train.append(features)
                y_train.append(0)  # Standing class
            
            # Sitting features (10 samples)
            for _ in range(10):
                features = []
                
                # X,Y coordinates (34 features)
                for i in range(17):
                    x = 0.5 + np.random.normal(0, 0.05)
                    if i < 11:  # Upper body
                        y = 0.3 + i * 0.05 + np.random.normal(0, 0.02)
                    else:  # Lower body more compressed
                        y = 0.6 + (i-11) * 0.02 + np.random.normal(0, 0.02)
                    features.extend([x, y])
                
                # Confidence scores (17 features)
                features.extend([0.8 + np.random.normal(0, 0.1) for _ in range(17)])
                
                # Distance features (10 features)
                features.extend([0.15 + np.random.normal(0, 0.05) for _ in range(10)])
                
                # Angle features (6 features)
                features.extend([np.pi/2 + np.random.normal(0, 0.3) for _ in range(6)])
                
                X_train.append(features)
                y_train.append(1)  # Sitting class
            
            # Walking/Running features (10 samples)
            for _ in range(10):
                features = []
                
                # X,Y coordinates (34 features)
                for i in range(17):
                    x = 0.5 + np.random.normal(0, 0.1)  # More horizontal variation
                    y = 0.3 + i * 0.05 + np.random.normal(0, 0.05)  # More vertical variation
                    features.extend([x, y])
                
                # Confidence scores (17 features)
                features.extend([0.8 + np.random.normal(0, 0.1) for _ in range(17)])
                
                # Distance features (10 features)
                features.extend([0.25 + np.random.normal(0, 0.1) for _ in range(10)])
                
                # Angle features (6 features)
                features.extend([np.pi/3 + np.random.normal(0, 0.4) for _ in range(6)])
                
                X_train.append(features)
                y_train.append(2)  # Walking/Running class
            
            # Train the classifier
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Verify feature dimensions
            if X_train.shape[1] != 67:
                self.logger.error(f"Fallback classifier feature mismatch: got {X_train.shape[1]}, expected 67")
                raise ValueError("Feature dimension mismatch")
            
            # Fit the model
            self.fallback_classifier.fit(X_train, y_train)
            self.logger.info("Fallback classifier created and trained successfully")
            
            # For compatibility with the main classifier interface
            if self.classifier is None:
                self.pose_classes = self.fallback_classes
                self.classifier = self.fallback_classifier
                self.logger.info("Using fallback classifier as primary classifier")
            
        except Exception as e:
            self.logger.error(f"Failed to create fallback classifier: {str(e)}")
            self.fallback_classifier = None
            self.fallback_classes = []
    
    def process_frame(self, frame):
        """Process a frame to detect poses and keypoints"""
        if frame is None:
            logging.error("Received None frame in process_frame")
            self.current_pose = "No detection"
            self.current_confidence = 0.0
            return None, [], [], None
            
        try:
            # Make a copy to avoid modifying the original
            processed_frame = frame.copy()
            
            # Process with YOLOv8
            with self.lock:  # Prevent concurrent model inference
                try:
                    results = self.model.predict(
                        source=frame,
                        conf=self.confidence_threshold,
                        verbose=False,
                        stream=False
                    )
                except Exception as e:
                    logging.error(f"Error in YOLOv8 prediction: {str(e)}")
                    self.current_pose = "Error"
                    self.current_confidence = 0.0
                    return processed_frame, [], [], None
            
            # Extract keypoints if available
            poses = []
            confidences = []
            keypoints_list = []
            
            if len(results) > 0:
                result = results[0]  # Get first result from batch
                
                # Draw only the skeleton and detection lines
                processed_frame = result.plot(labels=False)  # Disable label text
                
                if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints.data) > 0:
                    logging.info(f"Detected {len(result.keypoints.data)} people with keypoints")
                    
                    # Process each detected person (we'll use the first person's pose for stats)
                    for i, keypoints in enumerate(result.keypoints.data):
                        try:
                            # Get keypoints as numpy array
                            person_kpts = keypoints.cpu().numpy()
                            keypoints_list.append(person_kpts)
                            
                            # Extract features and classify pose
                            features = self._extract_features(person_kpts)
                            
                            if features is not None:
                                pose_class, confidence = self._classify_pose(features)
                                poses.append(pose_class)
                                confidences.append(confidence)
                                
                                # Update stats panel with first person's pose
                                if i == 0:
                                    self.current_pose = pose_class
                                    self.current_confidence = confidence * 100  # Convert to percentage
                                    logging.info(f"Current pose: {pose_class} with {confidence:.2f} confidence")
                            else:
                                poses.append("Standing")
                                confidences.append(0.6)
                                if i == 0:
                                    self.current_pose = "Standing"
                                    self.current_confidence = 60.0
                            
                        except Exception as e:
                            logging.error(f"Error processing keypoints for person {i}: {str(e)}")
                            poses.append("Standing")
                            confidences.append(0.6)
                            if i == 0:
                                self.current_pose = "Standing"
                                self.current_confidence = 60.0
                else:
                    self.current_pose = "No pose detected"
                    self.current_confidence = 0.0
                    logging.warning("No keypoints detected in the frame")
            else:
                self.current_pose = "No detection"
                self.current_confidence = 0.0
                logging.info("No person detected in frame")
            
            # Ensure we always return valid data
            if not poses:
                poses = [self.current_pose]
                confidences = [self.current_confidence / 100.0]  # Convert back to 0-1 range
            
            return processed_frame, poses, confidences, keypoints_list
            
        except Exception as e:
            logging.error(f"Error in process_frame: {str(e)}")
            self.current_pose = "Error"
            self.current_confidence = 0.0
            return frame, ["Error"], [0.0], []
    
    def _extract_features(self, keypoints):
        """Extract features from keypoints for classification"""
        try:
            if keypoints is None or not isinstance(keypoints, np.ndarray) or len(keypoints) < 17:
                logging.warning(f"Not enough keypoints for feature extraction: {len(keypoints) if keypoints is not None else 0}")
                return None
            
            # Convert to numpy format if needed
            keypoints_np = []
            for kp in keypoints:
                if isinstance(kp, tuple) and len(kp) >= 3:
                    keypoints_np.append([kp[0], kp[1], kp[2]])
                elif isinstance(kp, np.ndarray) and kp.size >= 3:
                    keypoints_np.append([kp[0], kp[1], kp[2]])
                else:
                    continue
            
            keypoints_np = np.array(keypoints_np)
            
            # Basic features (normalized coordinates and confidences)
            # Only x,y coordinates (34 features)
            coord_features = keypoints_np[:, :2].flatten()
            
            # Confidence scores (17 features)
            conf_features = keypoints_np[:, 2]
            
            # Relative distances between key joints (10 features)
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
                dist = np.linalg.norm(keypoints_np[i, :2] - keypoints_np[j, :2])
                distances.append(dist)
            
            # Angles between joints (6 features)
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
                v1 = keypoints_np[i, :2] - keypoints_np[j, :2]
                v2 = keypoints_np[k, :2] - keypoints_np[j, :2]
                angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                angles.append(angle)
            
            # Combine all features
            features = np.concatenate([
                coord_features,     # 34 features (x,y coordinates)
                conf_features,      # 17 features (confidence scores)
                distances,         # 10 features (distances)
                angles            # 6 features (angles)
            ])
            
            # Verify feature count
            if len(features) != 67:
                logging.error(f"Generated {len(features)} features, expected 67")
                return None
                
            logging.info(f"Extracted {len(features)} features successfully")
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def _calculate_angle(self, a, b, c):
        """Calculate the angle between three points"""
        ba = a - b
        bc = c - b
        
        # Ensure vectors have valid magnitude
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-5 or norm_bc < 1e-5:
            return 0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure within valid range
        
        return np.arccos(cosine_angle) * 180.0 / np.pi
    
    def _classify_pose(self, features):
        """Classify pose based on extracted features"""
        try:
            if self.classifier is None:
                logging.warning("No classifier loaded")
                return self._rule_based_classification(features)
            
            # Reshape features for the classifier
            features_array = np.array([features])
            
            try:
                # Scale features if available
                if hasattr(self, 'feature_scaler') and self.feature_scaler is not None:
                    try:
                        features_scaled = self.feature_scaler.transform(features_array)
                    except Exception as e:
                        logging.error(f"Feature scaling error: {str(e)}")
                        features_scaled = features_array
                else:
                    features_scaled = features_array
                
                # Get prediction and probability
                prediction = self.classifier.predict(features_scaled)[0]
                probabilities = self.classifier.predict_proba(features_scaled)[0]
                
                # Get the highest confidence prediction
                max_idx = np.argmax(probabilities)
                confidence = probabilities[max_idx]
                
                # Convert numerical prediction to class name if needed
                if isinstance(prediction, (int, np.integer)):
                    if 0 <= prediction < len(self.pose_classes):
                        prediction = self.pose_classes[prediction]
                
                # Use rule-based classification if confidence is too low
                if confidence < 0.4:
                    logging.info(f"Low ML confidence ({confidence:.2f}) - using rule-based classification")
                    return self._rule_based_classification(features)
                
                logging.info(f"ML classifier: '{prediction}' with {confidence:.2f} confidence")
                return prediction, float(confidence)
                
            except Exception as e:
                logging.error(f"Classifier prediction error: {str(e)}")
                return self._rule_based_classification(features)
            
        except Exception as e:
            logging.error(f"Error in pose classification: {str(e)}")
            return "Standing", 0.6  # Safe default
    
    def _rule_based_classification(self, features):
        """Rule-based classification when ML model fails or has low confidence"""
        try:
            # Extract relevant feature indices
            # Coordinates are first 34 features (17 keypoints * 2)
            # Confidences are next 17 features
            # Distances are next 10 features
            # Angles are last 6 features
            
            coord_end = 34
            conf_end = coord_end + 17
            dist_end = conf_end + 10
            
            # Get confidence scores
            confidences = features[coord_end:conf_end]
            
            # Get distances
            distances = features[conf_end:dist_end]
            shoulder_width = distances[2]  # shoulder width is 3rd distance
            hip_width = distances[3]      # hip width is 4th distance
            left_leg = distances[6]       # left leg length is 7th distance
            right_leg = distances[7]      # right leg length is 8th distance
            
            # Get angles
            angles = features[dist_end:]
            left_knee = angles[4]   # left leg angle
            right_knee = angles[5]  # right leg angle
            
            # Average confidence of detection
            avg_conf = np.mean(confidences)
            
            # 1. Check for running pose
            leg_asymmetry = abs(left_leg - right_leg) / max(left_leg, right_leg)
            knee_bend = min(abs(left_knee), abs(right_knee))
            if leg_asymmetry > 0.2 and knee_bend < 2.0:
                if knee_bend < 1.0:  # More bent knees indicate running
                    return "Running", 0.85
                else:
                    return "Walking", 0.75
            
            # 2. Check for sitting pose
            hip_shoulder_ratio = hip_width / shoulder_width
            knee_angle_avg = (abs(left_knee) + abs(right_knee)) / 2
            if hip_shoulder_ratio > 0.8 and knee_angle_avg < 1.5:
                return "Sitting", 0.85
            
            # 3. Check for lying down/Watching_screen
            # We'll keep this but with adjusted confidence
            vertical_extent = np.max(features[:coord_end:2]) - np.min(features[:coord_end:2])
            horizontal_extent = np.max(features[1:coord_end:2]) - np.min(features[1:coord_end:2])
            if horizontal_extent > 1.5 * vertical_extent:
                return "Watching_screen", 0.75  # Reduced confidence for Watching_screen
            
            # 4. Check for other poses
            if avg_conf > 0.7:  # Only classify with good confidence
                if knee_angle_avg > 2.8:  # Straight legs
                    return "Standing", 0.8
                elif hip_shoulder_ratio < 0.7:  # Narrow stance
                    return "Walking", 0.7
            
            # Default case
            return "Standing", 0.6
            
        except Exception as e:
            logging.error(f"Rule-based classification error: {str(e)}")
            return "Standing", 0.6  # Final fallback
    
    def _estimate_3d_pose(self, keypoints):
        """
        Estimate 3D pose from 2D keypoints
        This is a simple estimation for visualization purposes
        
        Args:
            keypoints: 2D keypoints with confidence values
            
        Returns:
            keypoints_3d: Estimated 3D keypoints
        """
        if keypoints is None or len(keypoints) == 0:
            return None

        # Convert to numpy array if it's a list
        if isinstance(keypoints, list):
            keypoints = np.array(keypoints)
        
        # Extract 2D points and confidences
        points_2d = keypoints[:, :2]
        confidences = keypoints[:, 2]
        
        # Filter low-confidence keypoints
        valid_indices = confidences > 0.5
        if np.sum(valid_indices) < 5:  # Need at least 5 valid keypoints
            return None
            
        # Basic 3D estimation: use 2D x,y and estimate z based on relative positions
        keypoints_3d = []
        
        # Normalize coordinates for better visualization
        x_min, y_min = np.min(points_2d[valid_indices], axis=0)
        x_max, y_max = np.max(points_2d[valid_indices], axis=0)
        
        width = max(x_max - x_min, 1e-5)
        height = max(y_max - y_min, 1e-5)
        
        # Create a scaling factor to normalize points
        scale_x = 1.0 / width
        scale_y = 1.0 / height
        
        # Create 3D keypoints with estimated depth
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:
                # Normalize to [-0.5, 0.5] range for better visualization
                norm_x = (x - x_min) * scale_x - 0.5
                norm_y = -((y - y_min) * scale_y - 0.5)  # Flip Y axis for 3D visualization
                
                # Estimate Z based on keypoint position
                # This is a simplified model that makes certain assumptions about human pose
                z = 0.0
                
                # Add some depth based on body part
                if i in [0, 1, 2, 3, 4]:  # Face keypoints slightly forward
                    z = 0.1
                elif i in [5, 6]:  # Shoulders
                    z = 0.05
                elif i in [7, 8]:  # Elbows slightly forward
                    z = 0.1
                elif i in [9, 10]:  # Wrists can be more forward
                    z = 0.15
                elif i in [11, 12]:  # Hips
                    z = 0.0
                elif i in [13, 14]:  # Knees slightly forward
                    z = 0.05
                elif i in [15, 16]:  # Ankles
                    z = 0.1
                
                keypoints_3d.append([norm_x, norm_y, z, conf])
            else:
                # For low confidence keypoints, still include but mark with low confidence
                keypoints_3d.append([0, 0, 0, 0])
        
        return keypoints_3d

    def classify_pose(self, keypoints):
        """Classify the pose from keypoints"""
        try:
            # Extract features from keypoints
            features = self._extract_features(keypoints)
            
            if features is None or len(features) == 0:
                logging.warning("No valid features extracted for classification")
                return "Standing", 0.7
                
            # Classify using either ML classifier or rule-based approach
            pose_class, confidence = self._classify_pose(features)
            
            # Always return a valid pose with reasonable confidence
            if pose_class in ["Unknown", "Classification error"] or confidence < 0.4:
                return self._rule_based_classification(features)
                
            # Log the classification result
            logging.info(f"Pose classified as '{pose_class}' with {confidence:.2f} confidence")
            
            return pose_class, confidence
            
        except Exception as e:
            logging.error(f"Error in pose classification: {str(e)}")
            logging.error(traceback.format_exc())
            return "Standing", 0.7  # Default fallback 