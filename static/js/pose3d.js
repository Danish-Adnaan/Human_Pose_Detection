/**
 * Enhanced Pose3D Visualizer using Three.js
 * Creates a clean, modern 3D visualization of human pose keypoints
 */
class Pose3DVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.isInitialized = false;
        this.autoRotate = false;
        this.joints = [];
        this.bones = [];
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.clock = new THREE.Clock();
        
        // Define joint connections for skeleton (based on COCO keypoint indices)
        this.connections = [
            [0, 1], [0, 2],           // Nose to eyes
            [1, 3], [2, 4],           // Eyes to ears
            [0, 5], [0, 6],           // Nose to shoulders
            [5, 7], [7, 9],           // Left arm
            [6, 8], [8, 10],          // Right arm
            [5, 11], [6, 12],         // Shoulders to hips
            [11, 13], [13, 15],       // Left leg
            [12, 14], [14, 16],       // Right leg
            [5, 6],                   // Connect shoulders
            [11, 12]                  // Connect hips - emphasized
        ];
        
        // Initialize if container exists
        if (this.container) {
            this.init();
        } else {
            console.error(`Container with ID "${containerId}" not found.`);
        }
    }
    
    init() {
        console.log(`Initializing Enhanced Pose3D Visualizer in container: ${this.containerId}`);
        
        // Create scene with clean background
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        
        // Improved camera setup
        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 1.5, 3);
        
        // Enhanced renderer settings
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Improved orbit controls
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.screenSpacePanning = true;
            this.controls.minDistance = 1;
            this.controls.maxDistance = 7;
            this.controls.maxPolarAngle = Math.PI / 1.5;
        }
        
        // Modern lighting setup
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);
        
        const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
        mainLight.position.set(5, 5, 5);
        mainLight.castShadow = true;
        this.scene.add(mainLight);
        
        // Add transparent coordinate grid
        this.addCoordinateGrid();
        
        // Create enhanced joints and bones
        this.createSkeleton();
        
        // Start animation
        this.animate();
        window.addEventListener('resize', () => this.onWindowResize());
        this.isInitialized = true;
    }
    
    addCoordinateGrid() {
        // Create transparent coordinate planes
        const gridSize = 4;
        const divisions = 10;
        
        // XY plane (blue)
        const gridXY = new THREE.GridHelper(gridSize, divisions);
        gridXY.material.opacity = 0.15;
        gridXY.material.transparent = true;
        gridXY.rotateX(Math.PI/2);
        this.scene.add(gridXY);
        
        // XZ plane (red)
        const gridXZ = new THREE.GridHelper(gridSize, divisions);
        gridXZ.material.opacity = 0.15;
        gridXZ.material.transparent = true;
        this.scene.add(gridXZ);
        
        // YZ plane (green)
        const gridYZ = new THREE.GridHelper(gridSize, divisions);
        gridYZ.material.opacity = 0.15;
        gridYZ.material.transparent = true;
        gridYZ.rotateZ(Math.PI/2);
        this.scene.add(gridYZ);
        
        // Add axes helper
        const axesHelper = new THREE.AxesHelper(2);
        axesHelper.material.opacity = 0.5;
        axesHelper.material.transparent = true;
        this.scene.add(axesHelper);
    }
    
    createSkeleton() {
        // Create joints
        const jointGeometry = new THREE.SphereGeometry(0.03, 16, 16);
        const jointMaterial = new THREE.MeshPhongMaterial({
            color: 0xff0000,          // Red color for joints
            shininess: 30,
            emissive: 0x330000,       // Slight red glow
            transparent: true,
            opacity: 1.0
        });

        for (let i = 0; i < 17; i++) {
            const joint = new THREE.Mesh(jointGeometry, jointMaterial.clone());
            joint.visible = false;
            this.joints.push(joint);
            this.scene.add(joint);
        }

        // Create bones (connections) using LineBasicMaterial for clean lines
        const boneMaterial = new THREE.LineBasicMaterial({
            color: 0x00ff00,          // Bright green color
            linewidth: 2,             // Note: linewidth > 1 might not work in WebGL
            transparent: true,
            opacity: 1.0
        });

        for (const [i, j] of this.connections) {
            const geometry = new THREE.BufferGeometry();
            const bone = new THREE.Line(geometry, boneMaterial.clone());
            bone.visible = false;
            this.bones.push({
                line: bone,
                from: i,
                to: j
            });
            this.scene.add(bone);
        }
    }

    updatePose(keypoints) {
        if (!keypoints || keypoints.length === 0 || !this.isInitialized) return;
    
        // Reset visibility
        this.joints.forEach(joint => joint.visible = false);
        this.bones.forEach(bone => bone.line.visible = false);
    
        // Update joints
        for (let i = 0; i < Math.min(keypoints.length, this.joints.length); i++) {
            const kp = keypoints[i];
            if (!kp || kp.length < 3) continue;
    
            const [x, y, z] = kp.map(coord => parseFloat(coord));
            if (isNaN(x) || isNaN(y) || isNaN(z)) continue;
    
            const joint = this.joints[i];
            joint.position.set(x * 2, y * 2, z * 2);  // Scale all coordinates by 2
            joint.visible = true;
        }
    
        // Update bones with clean lines
        this.bones.forEach(bone => {
            const fromJoint = this.joints[bone.from];
            const toJoint = this.joints[bone.to];
    
            if (fromJoint.visible && toJoint.visible) {
                const positions = new Float32Array([
                    fromJoint.position.x, fromJoint.position.y, fromJoint.position.z,
                    toJoint.position.x, toJoint.position.y, toJoint.position.z
                ]);
    
                bone.line.geometry.setAttribute('position', 
                    new THREE.BufferAttribute(positions, 3));
                bone.line.geometry.computeBoundingSphere();
                bone.line.visible = true;
            }
        });
    
        // Ensure continuous rendering
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    updateJoints(keypoints) {
        if (!this.isInitialized || !keypoints || keypoints.length === 0) return;

        // Clear existing joints and bones
        this.clearSkeleton();
        
        // Create joints with improved materials
        const jointGeometry = new THREE.SphereGeometry(0.03, 16, 16);
        const jointMaterial = new THREE.MeshPhongMaterial({
            color: 0xff0000,
            emissive: 0xff0000,
            emissiveIntensity: 0.3,
            shininess: 100
        });

        // Create bones with improved materials
        const boneMaterial = new THREE.LineBasicMaterial({
            color: 0x00ff00,
            linewidth: 2,
            transparent: true,
            opacity: 0.8
        });

        // Process keypoints with confidence threshold
        const validKeypoints = keypoints.filter(kp => kp[2] > 0.3);
        
        // Create joints for valid keypoints
        validKeypoints.forEach((kp, i) => {
            const joint = new THREE.Mesh(jointGeometry, jointMaterial);
            // Scale coordinates appropriately
            joint.position.set(
                (kp[0] - 300) / 100,  // Center and scale X
                -(kp[1] - 300) / 100, // Center and scale Y (inverted)
                0                     // Z coordinate
            );
            this.scene.add(joint);
            this.joints.push(joint);
        });

        // Create bones between valid connections
        this.connections.forEach(([startIdx, endIdx]) => {
            if (startIdx < validKeypoints.length && endIdx < validKeypoints.length) {
                const start = validKeypoints[startIdx];
                const end = validKeypoints[endIdx];
                
                if (start[2] > 0.3 && end[2] > 0.3) {  // Check confidence
                    const points = [
                        new THREE.Vector3(
                            (start[0] - 300) / 100,
                            -(start[1] - 300) / 100,
                            0
                        ),
                        new THREE.Vector3(
                            (end[0] - 300) / 100,
                            -(end[1] - 300) / 100,
                            0
                        )
                    ];
                    
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const line = new THREE.Line(geometry, boneMaterial);
                    this.scene.add(line);
                    this.bones.push(line);
                }
            }
        });

        // Request render
        this.render();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls if available
        if (this.controls) {
            this.controls.update();
        }
        
        // Auto-rotate if enabled
        if (this.autoRotate && this.isInitialized) {
            const rotationSpeed = 0.005;
            this.scene.rotation.y += rotationSpeed;
        }
        
        // Render scene
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    onWindowResize() {
        if (!this.isInitialized) return;
        
        // Update camera aspect ratio
            this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
            this.camera.updateProjectionMatrix();
        
        // Update renderer size
            this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    toggleAutoRotate() {
        this.autoRotate = !this.autoRotate;
        return this.autoRotate;
    }
    
    resetView() {
        if (!this.isInitialized) return;
        
        // Reset camera position to match reference image view
        this.camera.position.set(0, 0, 3.5);
        this.camera.lookAt(0, 0, 0);
            
        // Reset scene rotation
        this.scene.rotation.set(0, 0, 0);
        
        // Reset controls if available
        if (this.controls) {
            this.controls.reset();
        }
        
        // Ensure controls update immediately
        this.renderer.render(this.scene, this.camera);
    }
    
    setImageMode(isImageMode) {
        // Add specific adjustments for image vs webcam mode
        if (isImageMode) {
            // For image mode - slower rotation, closer camera
            this.camera.position.set(0.5, 0, 2.2);
            if (this.controls) {
                this.controls.dampingFactor = 0.1;
            }
        } else {
            // For webcam mode - faster rotation, further camera
            this.camera.position.set(0.5, 0, 2.8);
        if (this.controls) {
                this.controls.dampingFactor = 0.05;
            }
        }
        this.camera.lookAt(0, 0, 0);
    }
    
    showMessage(message) {
        // Remove any existing message
        this.clearMessage();
        
        // Create message display in the container
        const messageDiv = document.createElement('div');
        messageDiv.className = 'pose3d-message';
        messageDiv.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 14px;
            text-align: center;
            pointer-events: none;
            z-index: 100;
            max-width: 80%;
        `;
        messageDiv.innerText = message;
            this.container.appendChild(messageDiv);
    }
    
    clearMessage() {
        // Remove any existing message
        const existingMessage = this.container.querySelector('.pose3d-message');
        if (existingMessage) {
            existingMessage.remove();
        }
    }
}

// Function to initialize the visualizer
function initPose3DVisualizer(containerId) {
    console.log(`Initializing Pose3D Visualizer with container: ${containerId}`);
    return new Pose3DVisualizer(containerId);
}