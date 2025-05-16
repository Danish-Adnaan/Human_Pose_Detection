# Human Pose Detection & Classification System

A machine learning application for real-time human pose detection, 3D visualization, and pose classification using YOLOv8 and custom ML models.

## Features

- **Real-time pose detection** using YOLOv8 keypoint model
- **Pose classification** for 12 different human poses: standing, sitting, walking, handwave, thinking, running, dancing, jumping, t-pose, squats, leaning, and sleeping
- **3D visualization** of detected poses
- **Image upload** for processing static images
- **Webcam support** for real-time detection
- **Modern web interface** with responsive design

## Requirements

- Python 3.7.5 or higher
- Compatible with Windows, macOS, and Linux
- Webcam (for real-time detection)

## Installation

### 1. Clone this Repository on your PC 

```bash
git clone <repository-url>
cd human_pose_detection
```

### 2. Run the setup script
```bash
python setup.py
```

This script will:
- Create a virtual environment
- Install all dependencies
- Download the YOLOv8 model
- Create necessary directories
- Train the pose classifier model

### 3. Activate the virtual environment

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

### 4. Run the application

```bash
python app.py
```

Visit `http://localhost:8501` in your web browser to access the application.

## Usage

### Image Mode
1. Click on the "Image" tab
2. Upload an image by dragging and dropping or clicking "Browse files"
3. View the pose detection results and 3D visualization

### Webcam Mode
1. Click on the "Webcam" tab
2. Click "Start Camera" to begin real-time detection
3. Click "Stop Camera" to end the session

## Pose Classes

The system can recognize the following poses:
- Standing
- Sitting
- Walking
- Hand Wave
- Thinking
- Running
- Dancing
- Jumping
- T-Pose
- Squats
- Leaning
- Sleeping

## Technical Details

The system uses:
- YOLOv8m-pose for keypoint detection
- RandomForest classifier for pose classification
- Flask web framework for the backend
- Three.js for 3D visualization
- Custom synthetic data generation for training

## Troubleshooting

### Camera not working
- Make sure your webcam is connected and not being used by another application
- Check browser permissions to allow camera access

### Model performance issues
- For slower devices, try closing other applications to free up resources
- The application automatically adjusts processing parameters based on available resources

## Acknowledgments

- YOLOv8 by Ultralytics
- Three.js for 3D visualization
- Flask web framework 
