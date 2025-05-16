document.addEventListener('DOMContentLoaded', function() {
    // Tab switching
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Initialize 3D visualizer for pose 
    let pose3DVisualizer = null;
    
    // Initialize Pose3D Visualizer
    if (typeof initPose3DVisualizer === 'function') {
        pose3DVisualizer = initPose3DVisualizer('pose-3d');
        
        // Setup 3D controls
        setupPose3DControls();
        
        // Test visualization is only for initial loading
        // Actual visualization will use the processed image keypoints
        setTimeout(() => {
            testPose3DVisualization();
        }, 1000);
    } else {
        console.warn('Pose3D Visualizer not found');
    }
    
    // Setup 3D visualization controls
    function setupPose3DControls() {
        const toggleRotateBtn = document.getElementById('toggle-rotate');
        const resetViewBtn = document.getElementById('reset-view');
        
        if (toggleRotateBtn && pose3DVisualizer) {
            toggleRotateBtn.addEventListener('click', () => {
                const isRotating = pose3DVisualizer.toggleAutoRotate();
                toggleRotateBtn.classList.toggle('active', isRotating);
            });
        }
        
        if (resetViewBtn && pose3DVisualizer) {
            resetViewBtn.addEventListener('click', () => {
                pose3DVisualizer.resetView();
            });
        }
    }
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked tab
            tab.classList.add('active');
            
            // Hide all tab contents
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Show corresponding tab content
            const tabId = tab.getAttribute('data-tab');
            const tabContent = document.getElementById(tabId + '-tab');
            if (tabContent) {
                tabContent.classList.add('active');
                
                // For webcam tab, ensure webcam is not auto-started
                if (tabId === 'webcam' && webcamFeed) {
                    webcamFeed.src = '';
                    webcamFeed.style.display = 'none';
                }
            }
        });
    });
    
    // Image upload handling
    const imageDropzone = document.getElementById('image-dropzone');
    const imageFileInput = document.getElementById('image-upload');
    const imageBrowseBtn = document.getElementById('image-browse');
    const originalImage = document.getElementById('original-image');
    const detectionImage = document.getElementById('detection-image');
    const imageDisplay = document.getElementById('image-display');
    const imagePose = document.getElementById('image-pose');
    const imageConfidence = document.getElementById('image-confidence');
    
    // Webcam handling
    const startWebcamBtn = document.getElementById('start-webcam');
    const stopWebcamBtn = document.getElementById('stop-webcam');
    const webcamFeed = document.getElementById('webcam-feed');
    const webcamFps = document.getElementById('webcam-fps');
    const webcamPose = document.getElementById('webcam-pose');
    const webcamConfidence = document.getElementById('webcam-confidence');
    
    // Setup image upload
    if (imageDropzone && imageFileInput) {
        setupDropzone(imageDropzone, imageFileInput, handleImageUpload);
        
        imageFileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                handleImageUpload(this.files[0]);
            }
        });
        
        if (imageBrowseBtn) {
            imageBrowseBtn.addEventListener('click', () => {
                imageFileInput.click();
            });
        }
    }
    
    // Setup webcam
    if (startWebcamBtn && stopWebcamBtn) {
        startWebcamBtn.addEventListener('click', () => {
            if (webcamFeed) {
                // Show loading state immediately
                webcamFeed.style.display = 'block';
                const webcamBox = document.querySelector('.webcam-box');
                if (webcamBox) {
                    const loadingDiv = document.createElement('div');
                    loadingDiv.className = 'loading-indicator';
                    loadingDiv.innerHTML = '<div class="spinner"></div><p>Starting camera...</p>';
                    loadingDiv.style.position = 'absolute';
                    loadingDiv.style.top = '50%';
                    loadingDiv.style.left = '50%';
                    loadingDiv.style.transform = 'translate(-50%, -50%)';
                    loadingDiv.style.zIndex = '10';
                    webcamBox.style.position = 'relative';
                    webcamBox.appendChild(loadingDiv);
                }
                
                // Hide start button and show stop button
                startWebcamBtn.style.display = 'none';
                stopWebcamBtn.style.display = 'inline-block';
                
                // Add a timestamp to prevent caching
                const timestamp = new Date().getTime();
                webcamFeed.src = `/webcam_feed?t=${timestamp}`;
                
                // Add an onload event to remove loading indicator
                webcamFeed.onload = function() {
                    const loadingIndicator = document.querySelector('.loading-indicator');
                    if (loadingIndicator) {
                        loadingIndicator.remove();
                    }
                };
                
                // Add a timeout to remove loading indicator even if onload doesn't fire
                setTimeout(() => {
                    const loadingIndicator = document.querySelector('.loading-indicator');
                    if (loadingIndicator) {
                        loadingIndicator.remove();
                    }
                }, 5000);
                
                // Start FPS counter
                startFpsCounter();
            }
        });
        
        stopWebcamBtn.addEventListener('click', () => {
            stopWebcam();
        });
    }
    
    function stopWebcam() {
        if (webcamFeed) {
            webcamFeed.src = '';
            webcamFeed.style.display = 'none';
            
            // Send stop request to server
            fetch('/stop_webcam', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log('Webcam stopped:', data);
            })
            .catch(error => {
                console.error('Error stopping webcam:', error);
            });
            
            if (startWebcamBtn && stopWebcamBtn) {
                startWebcamBtn.style.display = 'inline-block';
                stopWebcamBtn.style.display = 'none';
            }
        }
    }
    
    function setupDropzone(dropzone, fileInput, handleFunction) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, unhighlight, false);
        });
        
        dropzone.addEventListener('drop', handleDrop, false);
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        function highlight() {
            dropzone.classList.add('highlight');
        }
        
        function unhighlight() {
            dropzone.classList.remove('highlight');
        }
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                handleFunction(files[0]);
            }
        }
    }
    
    function handleImageUpload(file) {
        console.log('Handling file:', file);
        
        // Check if file is an image
        if (!file.type.match('image.*')) {
            alert('Please upload an image file (JPEG, PNG, etc.)');
            return;
        }
        
        // Check file size (limit to 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size too large. Please upload an image smaller than 10MB.');
            return;
        }
        
        // Show loading state
        imageDisplay.style.display = 'block'; // Make sure display area is visible
        imageDisplay.classList.add('loading');
        originalImage.style.display = 'none';
        detectionImage.style.display = 'none';
        
        // Clear any previous results
        imagePose.textContent = 'Processing...';
        imageConfidence.textContent = '';
        
        // Show 3D visualizer loading message
        if (pose3DVisualizer) {
            pose3DVisualizer.showMessage("Processing image...");
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/process_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Image processing result:', data);
            
            // Remove loading state
            imageDisplay.classList.remove('loading');
            
            if (data.success) {
                // Display original image
                if (originalImage) {
                    // Create a new URL for the base64 image for the original (without keypoints)
                    fetch('/original_image', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ image_base64: data.original_image || data.image })
                    })
                    .then(response => response.json())
                    .then(imgData => {
                        if (imgData.success) {
                            originalImage.src = 'data:image/jpeg;base64,' + imgData.image;
                        } else {
                            originalImage.src = 'data:image/jpeg;base64,' + data.image;
                        }
                        originalImage.style.display = 'block';
                    })
                    .catch(err => {
                        console.error("Error retrieving original image:", err);
                    originalImage.src = 'data:image/jpeg;base64,' + data.image;
                    originalImage.style.display = 'block';
                    });
                }
                
                // Display detection image
                if (detectionImage) {
                    detectionImage.src = 'data:image/jpeg;base64,' + data.image;
                    detectionImage.style.display = 'block';
                }
                
                // Display pose information
                if (data.poses && data.poses.length > 0) {
                    // Show total number of people detected
                    imagePose.textContent = `Detected ${data.poses.length} ${data.poses.length > 1 ? 'people' : 'person'}`;
                    
                    // Show confidence of first person if available
                    if (data.confidences && data.confidences.length > 0) {
                        imageConfidence.textContent = `Person 0: ${data.poses[0]} (Confidence: ${(data.confidences[0] * 100).toFixed(1)}%)`;
                        
                        // Add more confidence info for additional people
                        if (data.confidences.length > 1) {
                            const confidenceElement = document.getElementById('image-confidence');
                            for (let i = 1; i < Math.min(data.confidences.length, 3); i++) {
                                const additionalConf = document.createElement('div');
                                additionalConf.className = 'additional-confidence';
                                additionalConf.textContent = `Person ${i}: ${data.poses[i]} (Confidence: ${(data.confidences[i] * 100).toFixed(1)}%)`;
                                confidenceElement.parentNode.appendChild(additionalConf);
                            }
                            
                            // If more than 3 people, add a note
                            if (data.confidences.length > 3) {
                                const moreNote = document.createElement('div');
                                moreNote.className = 'additional-confidence';
                                moreNote.textContent = `(+ ${data.confidences.length - 3} more people detected)`;
                                confidenceElement.parentNode.appendChild(moreNote);
                            }
                        }
                    }
                    
                    // Update 3D visualization if available
                    if (pose3DVisualizer) {
                        // Check if we have multi-person data
                        if (data.all_keypoints_3d && data.all_keypoints_3d.length > 0) {
                            console.log(`Received 3D keypoints for ${data.all_keypoints_3d.length} people`);
                        
                        // Make sure we have the 3D container visible
                        const pose3DContainer = document.getElementById('pose-3d');
                        if (pose3DContainer) {
                                pose3DContainer.style.display = 'block';
                            pose3DContainer.style.opacity = '1';
                            }
                            
                            // Remove any test labels
                            const testLabel = pose3DContainer.querySelector('.pose3d-test-label');
                            if (testLabel) {
                                testLabel.remove();
                            }
                            
                            // Clear any previous message
                            pose3DVisualizer.clearMessage();
                            
                            // Set image mode for better visualization
                            pose3DVisualizer.setImageMode(true);
                            
                            // Select the person with the most confident pose for visualization
                            let bestPersonIdx = 0;
                            let maxConfidence = data.confidences && data.confidences.length > 0 ? data.confidences[0] : 0;
                            
                            for (let i = 1; i < data.confidences.length; i++) {
                                if (data.confidences[i] > maxConfidence) {
                                    maxConfidence = data.confidences[i];
                                    bestPersonIdx = i;
                                }
                            }
                            
                            console.log(`Selecting person ${bestPersonIdx} with highest confidence ${maxConfidence} for visualization`);
                            
                            // Find the keypoints for the best person
                            const bestPersonData = data.all_keypoints_3d.find(p => p.person_idx === bestPersonIdx) || data.all_keypoints_3d[0];
                            
                            // Short delay to ensure the container is rendered before updating
                            setTimeout(() => {
                                try {
                                    // Clean and validate 3D keypoints before passing to visualizer
                                    const validKeypoints = bestPersonData.keypoints.filter(kp => {
                                        // Check if keypoint is an array with at least 3 elements (x,y,z)
                                        // and all values are valid numbers
                                        return Array.isArray(kp) && kp.length >= 3 && 
                                            !isNaN(parseFloat(kp[0])) && 
                                            !isNaN(parseFloat(kp[1])) && 
                                            !isNaN(parseFloat(kp[2]));
                                    });
                                    
                                    if (validKeypoints.length > 0) {
                                        console.log(`Visualizing person ${bestPersonData.person_idx} with ${validKeypoints.length} valid keypoints`);
                                        console.log("Keypoint sample:", JSON.stringify(validKeypoints[0]));
                                        
                                        // Stop auto-rotation to match the reference
                                        if (pose3DVisualizer.autoRotate) {
                                            pose3DVisualizer.toggleAutoRotate();
                                        }
                                        
                                        // Force reset view and then update pose
                                        pose3DVisualizer.resetView();
                                        
                                        // Add the current pose label to the 3D viz
                                        const poseLabel = data.poses[bestPersonIdx] || '';
                                        if (poseLabel) {
                                            const labelElement = document.createElement('div');
                                            labelElement.className = 'pose3d-test-label';
                                            labelElement.textContent = poseLabel;
                                            labelElement.style.cssText = `
                                                position: absolute;
                                                top: 10px;
                                                left: 50%;
                                                transform: translateX(-50%);
                                                background-color: #00ff00;
                                                color: black;
                                                padding: 3px 8px;
                                                border-radius: 4px;
                                                font-size: 14px;
                                                font-weight: bold;
                                                z-index: 100;
                                            `;
                                            pose3DContainer.appendChild(labelElement);
                                        }
                                        
                                        // Add message if showing a different person than first detected
                                        if (bestPersonData.person_idx > 0) {
                                            const message = document.createElement('div');
                                            message.className = 'pose3d-person-indicator';
                                            message.textContent = `Showing Person ${bestPersonData.person_idx}`;
                                            message.style.cssText = `
                                                position: absolute;
                                                bottom: 10px;
                                                left: 50%;
                                                transform: translateX(-50%);
                                                background-color: rgba(0,0,0,0.7);
                                                color: white;
                                                padding: 5px 10px;
                                                border-radius: 4px;
                                                font-size: 12px;
                                                z-index: 100;
                                            `;
                                            pose3DContainer.appendChild(message);
                                        }
                                        
                                        // Small delay to ensure reset is applied
                                        setTimeout(() => {
                                            pose3DVisualizer.updatePose(validKeypoints);
                                        }, 100);
                                    } else {
                                        console.warn(`No valid 3D keypoints found for person ${bestPersonData.person_idx}`);
                                        pose3DVisualizer.showMessage("Invalid 3D keypoints data");
                                    }
                                } catch (err) {
                                    console.error("Error updating 3D visualization:", err);
                                    pose3DVisualizer.showMessage("Error in 3D visualization: " + err.message);
                                }
                            }, 300);  // Increased delay for rendering
                        } else if (data.keypoints_3d && data.keypoints_3d.length > 0) {
                            // Legacy support for single person data
                            console.log('Using legacy single-person 3D keypoints data');
                            
                            // Make sure we have the 3D container visible
                            const pose3DContainer = document.getElementById('pose-3d');
                            if (pose3DContainer) {
                                pose3DContainer.style.display = 'block';
                                pose3DContainer.style.opacity = '1';
                                
                                // Remove any test labels
                                const testLabel = pose3DContainer.querySelector('.pose3d-test-label');
                                if (testLabel) {
                                    testLabel.remove();
                                }
                            }
                            
                            // Clear any previous message
                            pose3DVisualizer.clearMessage();
                            
                            // Set image mode for better visualization
                            pose3DVisualizer.setImageMode(true);
                            
                            // Short delay to ensure the container is rendered before updating
                            setTimeout(() => {
                                try {
                                    // Clean and validate 3D keypoints before passing to visualizer
                                    const validKeypoints = data.keypoints_3d.filter(kp => {
                                        // Check if keypoint is an array with at least 3 elements (x,y,z)
                                        // and all values are valid numbers
                                        return Array.isArray(kp) && kp.length >= 3 && 
                                            !isNaN(parseFloat(kp[0])) && 
                                            !isNaN(parseFloat(kp[1])) && 
                                            !isNaN(parseFloat(kp[2]));
                                    });
                                    
                                    if (validKeypoints.length > 0) {
                                        console.log("Using valid keypoints:", validKeypoints.length);
                                        console.log("Sample keypoint:", JSON.stringify(validKeypoints[0]));
                                        
                                        // Stop auto-rotation to match the reference
                                        if (pose3DVisualizer.autoRotate) {
                                            pose3DVisualizer.toggleAutoRotate();
                                        }
                                        
                                        // Force reset view and then update pose
                                        pose3DVisualizer.resetView();
                                        
                                        // Add the current pose label to the 3D viz if available
                                        if (data.poses && data.poses.length > 0) {
                                            const poseLabel = data.poses[0] || '';
                                            if (poseLabel) {
                                                const labelElement = document.createElement('div');
                                                labelElement.className = 'pose3d-test-label';
                                                labelElement.textContent = poseLabel;
                                                labelElement.style.cssText = `
                                                    position: absolute;
                                                    top: 10px;
                                                    left: 50%;
                                                    transform: translateX(-50%);
                                                    background-color: #00ff00;
                                                    color: black;
                                                    padding: 3px 8px;
                                                    border-radius: 4px;
                                                    font-size: 14px;
                                                    font-weight: bold;
                                                    z-index: 100;
                                                `;
                                                pose3DContainer.appendChild(labelElement);
                                            }
                                        }
                                        
                                        // Small delay to ensure reset is applied
                                        setTimeout(() => {
                                            pose3DVisualizer.updatePose(validKeypoints);
                        }, 100);
                                    } else {
                                        console.warn("No valid 3D keypoints found after filtering");
                                        pose3DVisualizer.showMessage("Invalid 3D keypoints data");
                                    }
                                } catch (err) {
                                    console.error("Error updating 3D visualization:", err);
                                    pose3DVisualizer.showMessage("Error in 3D visualization: " + err.message);
                                }
                            }, 300);  // Increased delay for rendering
                        } else {
                            console.warn("No 3D keypoints available in response data");
                        pose3DVisualizer.showMessage("No 3D keypoints available for visualization");
                        }
                    } else {
                        console.error("Pose3D visualizer not initialized");
                    }
                } else {
                    imagePose.textContent = 'No pose detected';
                    imageConfidence.textContent = '';
                    
                    // Show message in 3D visualizer
                    if (pose3DVisualizer) {
                        pose3DVisualizer.showMessage("No pose detected for 3D visualization");
                    }
                }
                
                // Show processing time
                if (data.processing_time) {
                    const processingInfo = document.getElementById('processing-time');
                    if (processingInfo) {
                        processingInfo.textContent = 'Processing time: ' + 
                            data.processing_time.toFixed(2) + ' seconds';
                    }
                }
            } else {
                // Display error
                if (data.error) {
                    alert('Error processing image: ' + data.error);
                    
                    // Show error in 3D visualizer
                    if (pose3DVisualizer) {
                        pose3DVisualizer.showMessage("Error: " + data.error);
                    }
                } else {
                    alert('Unknown error processing image');
                    
                    // Show generic error in 3D visualizer
                    if (pose3DVisualizer) {
                        pose3DVisualizer.showMessage("Unknown error in processing");
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error processing image:', error);
            imageDisplay.classList.remove('loading');
            alert('Error processing image: ' + error.message);
            
            // Show error in 3D visualizer
            if (pose3DVisualizer) {
                pose3DVisualizer.showMessage("Error: " + error.message);
            }
        });
    }
    
    function startFpsCounter() {
        // Update FPS and pose info every second for webcam
        const fpsInterval = setInterval(() => {
            if (!webcamFeed || webcamFeed.style.display === 'none') {
                clearInterval(fpsInterval);
                return;
            }
            
            fetch('/performance_stats')
                .then(response => response.json())
                .then(data => {
                    // Update FPS display
                    if (webcamFps && typeof data.fps === 'number') {
                        webcamFps.textContent = 'FPS: ' + data.fps.toFixed(1);
                    }
                    
                    // Update processing time display
                    const processingTime = document.getElementById('webcam-processing-time');
                    if (processingTime && typeof data.processing_time === 'number') {
                        processingTime.textContent = 'Processing: ' + 
                            (data.processing_time * 1000).toFixed(1) + 'ms';
                    }
                })
                .catch(error => {
                    console.error('Error fetching performance stats:', error);
                });
        }, 1000);
    }
    
    // Handle window resize for 3D visualizer
    window.addEventListener('resize', function() {
        if (pose3DVisualizer) {
            pose3DVisualizer.onWindowResize();
        }
    });
    
    // Function to test 3D visualization with a sample pose
    function testPose3DVisualization() {
        console.log('Testing 3D visualization with sample pose');
        if (!pose3DVisualizer) return;
        
        // Create a test pose for a person with arms crossed (matching the reference image)
        const testPose = [
            [0, -0.2, 0.1, 1.0],      // 0: nose
            [-0.05, -0.22, 0.1, 0.9],  // 1: left eye
            [0.05, -0.22, 0.1, 0.9],   // 2: right eye
            [-0.1, -0.22, 0.05, 0.8],  // 3: left ear
            [0.1, -0.22, 0.05, 0.8],   // 4: right ear
            [-0.15, 0, 0, 1.0],        // 5: left shoulder
            [0.15, 0, 0, 1.0],         // 6: right shoulder
            [0.05, 0.05, 0.1, 0.9],    // 7: left elbow (crossed)
            [-0.05, 0.05, 0.1, 0.9],   // 8: right elbow (crossed)
            [0.25, 0.05, 0.15, 0.9],   // 9: left wrist (crossed)
            [-0.25, 0.05, 0.15, 0.9],  // 10: right wrist (crossed)
            [-0.15, 0.4, 0, 1.0],      // 11: left hip
            [0.15, 0.4, 0, 1.0],       // 12: right hip
            [-0.15, 0.7, 0.05, 0.9],   // 13: left knee
            [0.15, 0.7, 0.05, 0.9],    // 14: right knee
            [-0.15, 1.0, 0.1, 0.9],    // 15: left ankle
            [0.15, 1.0, 0.1, 0.9]      // 16: right ankle
        ];
        
        // Update 3D visualization with test pose
        pose3DVisualizer.updatePose(testPose);
        
        // Start auto-rotation for better view
        pose3DVisualizer.toggleAutoRotate();
        
        // Show message indicating this is the arms crossed pose
        const pose3DContainer = document.getElementById('pose-3d');
        if (pose3DContainer) {
            const testLabel = document.createElement('div');
            testLabel.className = 'pose3d-test-label';
            testLabel.textContent = 'Arms_Cross';
            testLabel.style.cssText = `
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                background-color: rgba(0,200,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                z-index: 100;
            `;
            pose3DContainer.appendChild(testLabel);
        }
    }
}); 