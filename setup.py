from setuptools import setup, find_packages
import os
import torch
from ultralytics import YOLO
import shutil
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_yolo_model():
    """Download YOLOv8 pose model if not exists"""
    try:
        model_path = os.path.join('models', 'yolov8n-pose.pt')
        if not os.path.exists(model_path):
            logger.info("Downloading YOLOv8 pose model...")
            model = YOLO('yolov8n-pose.pt')
            # Save the model
            os.makedirs('models', exist_ok=True)
            shutil.copy(model.ckpt_path, model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.info("YOLO model already exists")
    except Exception as e:
        logger.error(f"Error downloading YOLO model: {str(e)}")
        sys.exit(1)

def create_project_structure():
    """Create the project directory structure"""
    directories = [
        'models',
        'data/pose_images',
        'data/keypoints',
        'data/collected_poses',
        'static/uploads',
        'static/processed',
        'templates'
    ]
    
    try:
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        sys.exit(1)

# Create project structure before setup
create_project_structure()
download_yolo_model()

setup(
    name="human-pose-detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "ultralytics>=8.0.0",
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.45.0",
        "requests>=2.25.0",
        "werkzeug>=2.0.0"
    ],
    package_data={
        '': [
            'models/*.pt',
            'models/*.pkl',
            'static/*',
            'templates/*',
        ],
    },
    include_package_data=True,
)

def post_install():
    """Post-installation steps"""
    try:
        # Create a basic index.html template if it doesn't exist
        template_path = os.path.join('templates', 'index.html')
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Human Pose Detection</title>
</head>
<body>
    <h1>Human Pose Detection System</h1>
    <div id="camera-feed"></div>
</body>
</html>
''')
            logger.info("Created basic index.html template")
            
        logger.info("Setup completed successfully!")
        logger.info("\nTo run the application:")
        logger.info("1. Activate your virtual environment")
        logger.info("2. Run: python app.py")
        logger.info("3. Open http://localhost:8501 in your browser")
        
    except Exception as e:
        logger.error(f"Error in post-installation: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    post_install()
