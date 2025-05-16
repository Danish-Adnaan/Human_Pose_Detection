from flask import Flask, render_template, Response, request, jsonify, abort
import cv2
import numpy as np
import os
import time
import logging
import threading
import base64
import gc
from werkzeug.utils import secure_filename
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Import PoseDetector
try:
    from pose_detector import PoseDetector
    from pose_classifier import PoseClassifier
    logging.info("Successfully imported PoseDetector and PoseClassifier")
except ImportError as e:
    logging.error(f"Failed to import: {str(e)}")
    raise
except Exception as e:
    logging.error(f"Unknown error importing modules: {str(e)}")
    logging.error(traceback.format_exc())
    raise

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize pose detector and classifier
detector = None
pose_classifier = None

# Global variables for performance monitoring
performance_stats = {
    'fps': 0,
    'processing_time': 0,
    'memory_usage': 0
}

# Global variable to track webcam state
webcam_active = False
webcam_cap = None
webcam_lock = threading.Lock()  # Add thread lock for webcam access

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', demo_mode=False)

@app.route('/webcam_feed')
def webcam_feed():
    """Video streaming route for webcam"""
    global webcam_active, webcam_cap, webcam_lock
    
    def generate_frames():
        global webcam_active, webcam_cap, webcam_lock
        
        with webcam_lock:
            # Open webcam with optimized settings
            try:
                webcam_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow API for faster startup on Windows
                webcam_active = True
                
                # Set buffer size to minimum to reduce latency
                webcam_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Set lower resolution for better performance
                webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Set FPS to balance performance
                webcam_cap.set(cv2.CAP_PROP_FPS, 15)
                
                # Check if webcam opened successfully
                if not webcam_cap.isOpened():
                    logging.error("Error: Could not open webcam")
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          open('static/error.jpg', 'rb').read() + b'\r\n')
                    webcam_active = False
                    return
            except Exception as e:
                logging.error(f"Error initializing webcam: {str(e)}")
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + 
                      open('static/error.jpg', 'rb').read() + b'\r\n')
                webcam_active = False
                return
        
        # Discard first few frames to let camera adjust
        for _ in range(5):
            with webcam_lock:
                if webcam_cap is None or not webcam_active:
                    break
                webcam_cap.read()
        
        # Send initial frame quickly before processing begins
        with webcam_lock:
            if webcam_cap is None or not webcam_active:
                return
            success, first_frame = webcam_cap.read()
            
        if success:
            # Just add text to show it's starting
            cv2.putText(first_frame, "Starting camera...", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', first_frame)
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # FPS calculation variables
        prev_frame_time = 0
        new_frame_time = 0
        fps_values = []
        skip_frames = 0  # For frame skipping when CPU is under load
        
        # Add class name mapping
        class_display_names = {
            'sleeping': 'Watching_screen'
        }
        
        try:
            while webcam_active:
                # Safely read frame with lock
                with webcam_lock:
                    if webcam_cap is None or not webcam_active:
                        break
                    success, frame = webcam_cap.read()
                
                if not success:
                    logging.error("Error: Failed to capture image")
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          open('static/error.jpg', 'rb').read() + b'\r\n')
                    break
                
                # Frame skipping for better performance under load
                skip_frames += 1
                if skip_frames < 2:  # Process every other frame initially
                    # Just encode and send the raw frame for faster display
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in byte format
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    continue
                else:
                    skip_frames = 0
                
                # Calculate FPS
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
                prev_frame_time = new_frame_time
                
                # Add to rolling average
                fps_values.append(fps)
                if len(fps_values) > 5:  # Reduce from 10 to 5 for faster response
                    fps_values.pop(0)
                
                # Calculate average FPS
                avg_fps = sum(fps_values) / len(fps_values)
                performance_stats['fps'] = avg_fps
                
                try:
                    # Process frame
                    start_time = time.time()
                    processed_frame, poses, confidences, extracted_keypoints = detector.process_frame(frame)
                    
                    # Add pose classification if keypoints are detected
                    detected_pose = "-"  # Default value
                    pose_confidence = 0.0  # Default value
                    
                    if extracted_keypoints is not None and len(extracted_keypoints) > 0:
                        for person_keypoints in extracted_keypoints:
                            # Normalize keypoints for classification
                            img_height, img_width = frame.shape[:2]
                            person_keypoints[:, 0] = person_keypoints[:, 0] / img_width
                            person_keypoints[:, 1] = person_keypoints[:, 1] / img_height
                            
                            # Get pose classification
                            features = pose_classifier._extract_features(person_keypoints)
                            features_scaled = pose_classifier.scaler.transform(features)
                            pose_class = pose_classifier.model.predict(features_scaled)[0]
                            class_confidence = pose_classifier.model.predict_proba(features_scaled)[0].max()
                            
                            # Update the detected pose for display (using first person detected)
                            detected_pose = class_display_names.get(pose_class, pose_class)
                            pose_confidence = class_confidence * 100  # Convert to percentage
                            break  # Only use first person for now
                    
                    # Calculate FPS
                    new_frame_time = time.time()
                    fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
                    prev_frame_time = new_frame_time
                    
                    # Add to rolling average
                    fps_values.append(fps)
                    if len(fps_values) > 5:
                        fps_values.pop(0)
                    
                    # Calculate average FPS
                    avg_fps = sum(fps_values) / len(fps_values)
                    
                    # Update performance stats
                    processing_time = time.time() - start_time
                    performance_stats.update({
                        'fps': avg_fps,
                        'processing_time': processing_time * 1000,  # Convert to milliseconds
                        'detected_pose': class_display_names.get(detected_pose, detected_pose),
                        'confidence': pose_confidence
                    })
                    
                    # Add FPS and processing time to frame
                    cv2.putText(processed_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Process: {processing_time:.3f}s", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Encode frame
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in byte format
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                except Exception as e:
                    logging.error(f"Error processing webcam frame: {str(e)}")
                    logging.error(traceback.format_exc())
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          open('static/error.jpg', 'rb').read() + b'\r\n')
                
                # Force garbage collection to free memory less frequently
                if len(fps_values) % 60 == 0:
                    gc.collect()
                
        finally:
            # Release resources
            with webcam_lock:
                if webcam_cap is not None:
                    webcam_cap.release()
                    webcam_cap = None
                webcam_active = False
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop the webcam feed"""
    global webcam_active, webcam_cap, webcam_lock
    
    with webcam_lock:
        webcam_active = False
        
        # Release webcam if it's open
        if webcam_cap is not None:
            webcam_cap.release()
            webcam_cap = None
    
    logging.info("Webcam stopped by client request")
    return jsonify({'success': True})

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process uploaded image for pose detection"""
    try:
        # Check if request contains file
        if 'file' not in request.files:
            logging.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logging.error("No selected file (empty filename)")
            return jsonify({'error': 'No selected file'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            logging.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed. Please upload only PNG, JPG, or JPEG files.'}), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        logging.info(f"Processing uploaded file: {filename}, content type: {file.content_type}")
        
        # Check file size
        if request.content_length > app.config['MAX_CONTENT_LENGTH']:
            logging.error(f"File too large: {request.content_length} bytes")
            return jsonify({'error': 'File too large'}), 413
        
        # Read image
        try:
            file_bytes = file.read()
            if len(file_bytes) == 0:
                logging.error("Empty file uploaded")
                return jsonify({'error': 'Empty file'}), 400
                
            logging.info(f"Read {len(file_bytes)} bytes from uploaded file")
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"Error reading file: {str(e)}")
            logging.error(traceback.format_exc())
            return jsonify({'error': f'Could not read file: {str(e)}'}), 400
        
        # Check if image is valid
        if img is None:
            logging.error("Invalid image file - could not decode image")
            return jsonify({'error': 'Invalid image file - could not decode image'}), 400
        
        logging.info(f"Successfully decoded image with shape: {img.shape}")
        
        # Resize large images for better performance
        max_dimension = 1280
        h, w = img.shape[:2]
        if h > max_dimension or w > max_dimension:
            scale = max_dimension / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            logging.info(f"Resized image from {w}x{h} to {int(w * scale)}x{int(h * scale)}")
        
        # Initialize detector if needed
        global detector
        if detector is None:
            logging.info("Initializing pose detector...")
            detector = PoseDetector()
            logging.info("Pose detector initialized")
        
        # Save original image for the original image panel
        _, original_buffer = cv2.imencode('.jpg', img)
        original_img_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        # Add class name mapping
        class_display_names = {
            'sleeping': 'Watching_screen'
        }
        
        # Process image
        start_time = time.time()
        logging.info("Processing image with pose detector...")
        try:
            processed_img, poses, confidences, extracted_keypoints_list = detector.process_frame(img)
            
            # Add pose classification
            detected_pose = "-"
            pose_confidence = 0.0
            
            if extracted_keypoints_list is not None and len(extracted_keypoints_list) > 0:
                for person_keypoints in extracted_keypoints_list:
                    if person_keypoints is not None and len(person_keypoints) > 0:
                        # Normalize keypoints for classification
                        person_keypoints[:, 0] = person_keypoints[:, 0] / img.shape[1]
                        person_keypoints[:, 1] = person_keypoints[:, 1] / img.shape[0]
                        
                        # Get pose classification
                        features = pose_classifier._extract_features(person_keypoints)
                        features_scaled = pose_classifier.scaler.transform(features)
                        pose_class = pose_classifier.model.predict(features_scaled)[0]
                        class_confidence = pose_classifier.model.predict_proba(features_scaled)[0].max()
                        
                        # Update the detected pose (using first person detected)
                        detected_pose = class_display_names.get(pose_class, pose_class)
                        pose_confidence = class_confidence * 100  # Convert to percentage
                        break  # Only use first person for now
            
            # Generate 3D keypoints for visualization with enhanced clarity and better normalization
            all_keypoints_3d = []
            if extracted_keypoints_list is not None:
                try:
                    # Now we might have multiple people's keypoints
                    num_people = len(extracted_keypoints_list)
                    logging.info(f"Processing 3D keypoints for {num_people} detected people")
                    
                    for person_idx, keypoints in enumerate(extracted_keypoints_list):
                        # Check if we have valid keypoints for this person
                        if keypoints is not None and len(keypoints) > 0:
                            logging.info(f"Generating 3D keypoints for person {person_idx} with {len(keypoints)} keypoints")
                            
                            # Use the detector's built-in 3D estimation for this person
                            keypoints_3d = detector._estimate_3d_pose(keypoints)
                            
                            if keypoints_3d and len(keypoints_3d) > 0:
                                logging.info(f"Person {person_idx}: Generated {len(keypoints_3d)} 3D keypoints")
                                all_keypoints_3d.append({
                                    'person_idx': person_idx,
                                    'keypoints': keypoints_3d
                                })
                            else:
                                logging.warning(f"Person {person_idx}: No valid 3D keypoints generated")
                
                    logging.info(f"Total people with 3D poses: {len(all_keypoints_3d)}")
                except Exception as e:
                    logging.error(f"Error generating 3D keypoints: {str(e)}")
                    logging.error(traceback.format_exc())
            else:
                logging.warning("No extracted keypoints available for 3D visualization")
            
            processing_time = time.time() - start_time
            performance_stats['processing_time'] = processing_time
            
            # Log the poses and confidences
            logging.info(f"Detected poses: {poses}")
            logging.info(f"Confidence levels: {confidences}")
            logging.info(f"Generated 3D keypoints for {len(all_keypoints_3d)} people")
            
            # Encode processed image
            try:
                _, buffer = cv2.imencode('.jpg', processed_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                logging.error(f"Error encoding processed image: {str(e)}")
                logging.error(traceback.format_exc())
                return jsonify({'error': f'Error encoding processed image: {str(e)}'}), 500
            
            # Log performance
            logging.info(f"Image processed in {processing_time:.3f}s, size: {img.shape}")
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_native_types(obj):
                if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: convert_to_native_types(v) for k, v in obj.items()}
                else:
                    return obj
            
            # Prepare the response
            response_data = {
                'success': True,
                'image': img_base64,
                'original_image': original_img_base64,
                'poses': poses,
                'confidences': [float(c) if isinstance(c, (np.float32, np.float64)) else c for c in confidences],
                'detected_pose': detected_pose,
                'confidence': pose_confidence,
                'processing_time': processing_time * 1000,  # Convert to milliseconds
                'num_people': len(poses)
            }
            
            # Add 3D keypoints if they exist and are valid
            if all_keypoints_3d and len(all_keypoints_3d) > 0:
                response_data['all_keypoints_3d'] = convert_to_native_types(all_keypoints_3d)
                logging.info(f"Added 3D keypoints for {len(all_keypoints_3d)} people to response")
                
                # For backward compatibility, include the first person's keypoints as keypoints_3d
                if len(all_keypoints_3d) > 0:
                    response_data['keypoints_3d'] = convert_to_native_types(all_keypoints_3d[0]['keypoints'])
            else:
                logging.warning("No 3D keypoints available for response")
                response_data['all_keypoints_3d'] = []
                response_data['keypoints_3d'] = []
            
            logging.info("Successfully prepared response")
            return jsonify(response_data)
            
        except Exception as e:
            logging.error(f"Error in pose detection: {str(e)}")
            logging.error(traceback.format_exc())
            return jsonify({'error': f'Error in pose detection: {str(e)}'}), 500
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/original_image', methods=['POST'])
def original_image():
    """Return original image without keypoints"""
    try:
        # Get base64 image from request
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Return the image as-is
        return jsonify({
            'success': True,
            'image': data['image_base64']
        })
    except Exception as e:
        logging.error(f"Error in original_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/performance_stats')
def get_performance_stats():
    """Get current performance statistics"""
    global performance_stats
    
    # Create a copy of the stats to avoid modifying the original
    stats = dict(performance_stats)
    
    return jsonify(stats)

# Add a basic health check endpoint
@app.route('/health')
def health_check():
    """Basic health check endpoint"""
    return jsonify({'status': 'ok'})

# Add error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def initialize_detector():
    """Initialize the pose detector and classifier"""
    global detector, pose_classifier
    try:
        logging.info("Initializing pose detector...")
        
        # Set lower confidence threshold for faster startup
        startup_confidence = 0.2
        
        # Initialize the detector with optimized settings
        detector = PoseDetector(confidence=startup_confidence)
        
        # Initialize the pose classifier
        logging.info("Initializing pose classifier...")
        pose_classifier = PoseClassifier(model_path='models/pose_classifier.pkl')
        
        # Pre-warm the models with a dummy inference
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.process_frame(dummy_img)
        
        logging.info("Pose detector and classifier initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    # Initialize detector
    try:
        initialize_detector()
    except IndentationError as e:
        logging.error(f"Indentation error in pose_detector.py at line {e.lineno}, position {e.offset}")
        logging.error("Please fix the indentation errors before running the application")
        exit(2)
    except SyntaxError as e:
        logging.error(f"Syntax error in pose_detector.py at line {e.lineno}, position {e.offset}")
        logging.error("Please fix the syntax errors before running the application")
        exit(3)
    except ImportError as e:
        logging.error(f"Import error: {e}")
        logging.error("Make sure all required modules are installed")
        exit(4)
    except Exception as e:
        logging.error(f"Failed to initialize detector: {e}")
        logging.error(traceback.format_exc())
        exit(1)
    
    # Create error image if it doesn't exist
    error_img_path = 'static/error.jpg'
    if not os.path.exists(error_img_path):
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(error_img_path, error_img)
    
    # Print startup message
    print("\n" + "="*50)
    print("Human Pose Detection Application")
    print("="*50)
    print(f"Server running at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=8501, threaded=True) 