import os
import numpy as np
import cv2
import logging
import time
import pickle
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
import urllib.parse
import random
import traceback
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pose_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pose_data_collector')

class PoseDataCollector:
    def __init__(self, pose_classes=None):
        """Initialize the pose data collector"""
        if pose_classes is None:
            self.pose_classes = [
                'standing', 'sitting', 'walking', 'handwave', 'thinking', 
                'running', 'dancing', 'jumping', 't-pose', 'squats', 
                'leaning', 'Watching_screen'
            ]
        else:
            self.pose_classes = pose_classes
            
        # Create necessary directories
        os.makedirs('data/pose_images', exist_ok=True)
        os.makedirs('data/keypoints', exist_ok=True)
        
        # Initialize YOLO model
        try:
            self.yolo_model = YOLO('yolov8x-pose.pt')
            logger.info("Successfully loaded YOLOv8x-pose model")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise

        # Define search queries for each pose class
        self.search_queries = {
            'standing': [
                'person standing straight professional photo',
                'person standing pose full body',
                'standing posture human clear'
            ],
            'sitting': [
                'person sitting chair pose professional',
                'sitting posture office clear photo',
                'person sitting position full body'
            ],
            'walking': [
                'person walking side view clear',
                'walking pose full body photo',
                'human walking position professional'
            ],
            'handwave': [
                'person waving hand clear photo',
                'hand wave gesture full body',
                'human waving hello pose'
            ],
            'thinking': [
                'person thinking pose professional',
                'thinking gesture full body photo',
                'human contemplating pose clear'
            ],
            'running': [
                'person running side view clear',
                'running pose athletic photo',
                'human jogging position professional'
            ],
            'dancing': [
                'person dancing pose clear photo',
                'dance position full body professional',
                'human dancing movement clear'
            ],
            'jumping': [
                'person jumping clear photo',
                'jump pose athletic professional',
                'human jumping position full body'
            ],
            't-pose': [
                'person t pose clear photo',
                't position arms spread professional',
                'human t stance full body'
            ],
            'squats': [
                'person doing squats clear',
                'squat position athletic photo',
                'proper squat form full body'
            ],
            'leaning': [
                'person leaning pose clear',
                'leaning position full body photo',
                'human leaning stance professional'
            ],
            'Watching_screen': [
                'person lying down pose clear',
                'Watching_screen position full body photo',
                'human resting pose professional'
            ]
        }

    def _download_image(self, url, timeout=5):
        """Download an image from URL with validation"""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code != 200:
                return None

            # Validate image using PIL
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for OpenCV processing
            image_np = np.array(image)
            
            # Convert from RGB to BGR for OpenCV
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            return image_cv2

        except Exception as e:
            logger.debug(f"Error downloading image from {url}: {str(e)}")
            return None

    def _search_images(self, query, max_images=20):
        """Search for images using a query"""
        try:
            # Encode query for URL with more specific terms
            search_query = f"{query} person full body clear visible"
            search_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(search_query)}&first=1&qft=+filterui:photo-photo"
            
            # Send request with headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Search failed for query: {query}")
                return []

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            image_elements = soup.find_all('img')
            
            # Extract image URLs with better filtering
            image_urls = []
            for img in image_elements:
                if 'src' in img.attrs:
                    url = img['src']
                    if (url.startswith('http') and 
                        (url.endswith('.jpg') or url.endswith('.jpeg') or 
                         url.endswith('.png') or 'blob:' not in url)):
                        image_urls.append(url)
                        if len(image_urls) >= max_images:
                            break

            return image_urls

        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            return []

    def _extract_keypoints(self, image):
        """Extract keypoints from image using YOLO"""
        try:
            # Run YOLO pose detection
            results = self.yolo_model(image)
            
            # Get keypoints from first person detected
            if len(results) > 0 and len(results[0].keypoints.data) > 0:
                keypoints = results[0].keypoints.data[0].cpu().numpy()
                
                # Normalize keypoints
                height, width = image.shape[:2]
                keypoints[:, 0] = keypoints[:, 0] / width
                keypoints[:, 1] = keypoints[:, 1] / height
                
                # Validate keypoints
                if self._validate_keypoints(keypoints):
                    return keypoints
            return None
        except Exception as e:
            logger.error(f"Error extracting keypoints: {str(e)}")
            return None

    def _validate_keypoints(self, keypoints):
        """Validate keypoints quality"""
        if keypoints is None or len(keypoints) != 17:
            return False
            
        # Check confidence scores - lower threshold
        confidences = keypoints[:, 2]
        if np.mean(confidences) < 0.2:  # Reduced from 0.3 to 0.2
            return False
            
        # Check if key joints are detected (head, shoulders, hips)
        key_joints = [0, 5, 6, 11, 12]  # Indices for nose, shoulders, and hips
        if np.any(confidences[key_joints] < 0.2):  # Reduced from 0.3 to 0.2
            return False
            
        return True

    def collect_pose_data(self, min_images_per_class=20):
        """Collect pose data for all classes"""
        collected_data = {
            'keypoints': [],
            'labels': [],
            'images': []
        }
        
        for pose_class in self.pose_classes:
            logger.info(f"Collecting data for pose class: {pose_class}")
            
            images_collected = 0
            for query in self.search_queries[pose_class]:
                if images_collected >= min_images_per_class:
                    break
                    
                logger.info(f"Using search query: {query}")
                image_urls = self._search_images(query, max_images=40)
                
                for url in tqdm(image_urls, desc=f"Processing {pose_class} images"):
                    if images_collected >= min_images_per_class:
                        break
                        
                    # Download and process image
                    image = self._download_image(url)
                    if image is None:
                        continue
                    
                    # Extract keypoints
                    keypoints = self._extract_keypoints(image)
                    if keypoints is None:
                        continue
                    
                    # Save data
                    try:
                        # Save image
                        image_filename = f"{pose_class}_{images_collected}.jpg"
                        image_path = os.path.join('data/pose_images', image_filename)
                        cv2.imwrite(image_path, image)
                        
                        # Save keypoints
                        keypoints_filename = f"{pose_class}_{images_collected}.npy"
                        keypoints_path = os.path.join('data/keypoints', keypoints_filename)
                        np.save(keypoints_path, keypoints)
                        
                        # Add to collected data
                        collected_data['keypoints'].append(keypoints)
                        collected_data['labels'].append(pose_class)
                        collected_data['images'].append(image_path)
                        
                        images_collected += 1
                        logger.info(f"Successfully collected {images_collected} images for {pose_class}")
                        
                    except Exception as e:
                        logger.error(f"Error saving data: {str(e)}")
                        continue
            
            logger.info(f"Completed collection for {pose_class}: {images_collected} images")
        
        # Save collected data
        try:
            with open('data/collected_data.pkl', 'wb') as f:
                pickle.dump(collected_data, f)
            logger.info("Successfully saved collected data")
        except Exception as e:
            logger.error(f"Error saving collected data: {str(e)}")
        
        return collected_data

if __name__ == "__main__":
    collector = PoseDataCollector()
    collector.collect_pose_data(min_images_per_class=20) 