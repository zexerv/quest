#!/usr/bin/env python3
"""
Enhanced Multi-Sensor Quaternion Estimator for Arm-Hand Tracking
"""

import numpy as np
import serial
import time
import threading
import json
import os
from collections import deque
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from PyQt5.QtCore import pyqtSignal, QObject, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QComboBox, 
                           QProgressBar, QTabWidget, QGroupBox, QSlider)
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Configuration
DEFAULT_PORTS = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2']  # Default ports for three sensors
DEFAULT_BAUD_RATE = 115200
COMPLEMENTARY_FILTER_ALPHA = 0.98  # Higher = more gyro, less accel
CALIBRATION_FILE = "sensor_calibration.json"

class SerialReader(QObject):
    """Serial reader for multiple IMU sensors"""
    data_ready = pyqtSignal(object)
    
    def __init__(self, ports=DEFAULT_PORTS, baud_rate=DEFAULT_BAUD_RATE):
        super().__init__()
        self.ports = ports
        self.baud_rate = baud_rate
        self.running = False
        self.serial_connections = [None] * len(ports)
        self.threads = [None] * len(ports)
    
    def start(self):
        """Start the serial reading threads for all sensors"""
        self.running = True
        for i, port in enumerate(self.ports):
            self.threads[i] = threading.Thread(target=self._read_serial, args=(i, port), daemon=True)
            self.threads[i].start()
    
    def _read_serial(self, sensor_id, port):
        """Thread function to continuously read from serial for a specific sensor"""
        try:
            # Open serial port
            self.serial_connections[sensor_id] = serial.Serial(port, self.baud_rate, timeout=0.1)
            print(f"Connected to sensor {sensor_id} on {port} at {self.baud_rate} baud")
            
            while self.running:
                try:
                    # Read a line and decode
                    line = self.serial_connections[sensor_id].readline().decode('utf-8').rstrip()
                    
                    # Split the line by commas
                    data = line.split(',')
                    
                    # Check for 6 values (ax, ay, az, gx, gy, gz)
                    if len(data) == 6:
                        try:
                            # Convert to floats
                            ax, ay, az, gx, gy, gz = map(float, data)
                            
                            # Convert gyro from degrees/s to rad/s
                            gx = np.radians(gx)
                            gy = np.radians(gy)
                            gz = np.radians(gz)
                            
                            # Emit the data
                            self.data_ready.emit({
                                'sensor_id': sensor_id,
                                'accel': np.array([ax, ay, az]),
                                'gyro': np.array([gx, gy, gz]),
                                'timestamp': time.time()
                            })
                        except ValueError:
                            print(f"Sensor {sensor_id}: Invalid data: {line}")
                    elif line:
                        print(f"Sensor {sensor_id}: Incomplete data: {line}")
                        
                except serial.SerialException as e:
                    print(f"Sensor {sensor_id} serial error: {e}")
                    time.sleep(1)  # Wait before trying again
                    break
                except Exception as e:
                    print(f"Error in serial thread for sensor {sensor_id}: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Failed to open {port} for sensor {sensor_id}: {e}")
        
        finally:
            # Try to close the serial port
            if self.serial_connections[sensor_id] and self.serial_connections[sensor_id].is_open:
                self.serial_connections[sensor_id].close()
                print(f"Serial port for sensor {sensor_id} closed")
    
    def stop(self):
        """Stop all serial reading threads"""
        self.running = False
        for i in range(len(self.ports)):
            if self.threads[i]:
                self.threads[i].join(timeout=1.0)
            if self.serial_connections[i] and self.serial_connections[i].is_open:
                self.serial_connections[i].close()

class MultiSensorEstimator:
    """Quaternion estimator for multiple IMU sensors with calibration"""
    
    def __init__(self, num_sensors=3, alpha=COMPLEMENTARY_FILTER_ALPHA):
        self.num_sensors = num_sensors
        self.alpha = alpha
        
        # Initialize quaternions for each sensor
        self.sensor_quaternions = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(num_sensors)]  # w, x, y, z
        self.anatomical_quaternions = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(num_sensors)]
        self.last_times = [time.time() for _ in range(num_sensors)]
        
        # Calibration quaternions (sensor to anatomical frame)
        self.calibration_quaternions = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(num_sensors)]
        self.is_calibrated = [False] * num_sensors
        
        # Store calibration data
        self.calibration_data = [[] for _ in range(num_sensors)]
        
        # Current calibration state
        self.calibration_mode = False
        self.current_calibration_sensor = -1
        self.current_calibration_axis = -1
        self.current_calibration_step = 0
        
        # Joint angles
        self.joint_angles = np.zeros(7)  # 7 DOF arm-hand model
        
        # Data storage
        self.max_samples = 500
        self.timestamps = [deque(maxlen=self.max_samples) for _ in range(num_sensors)]
        self.accel_data = [deque(maxlen=self.max_samples) for _ in range(num_sensors)]
        self.gyro_data = [deque(maxlen=self.max_samples) for _ in range(num_sensors)]
        self.quaternions = [deque(maxlen=self.max_samples) for _ in range(num_sensors)]
        
        # Try to load calibration data
        self.load_calibration()
    
    def update_sensor(self, sensor_id, accel, gyro, timestamp=None):
        """Update quaternion estimation for a specific sensor"""
        if sensor_id < 0 or sensor_id >= self.num_sensors:
            print(f"Invalid sensor ID: {sensor_id}")
            return self.sensor_quaternions[0]  # Return something safe
        
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate time delta
        dt = timestamp - self.last_times[sensor_id] if self.last_times[sensor_id] else 0.01
        self.last_times[sensor_id] = timestamp
        
        # Limit dt to reasonable values
        dt = min(max(dt, 0.001), 0.05)
        
        # Store raw data
        self.timestamps[sensor_id].append(timestamp)
        self.accel_data[sensor_id].append(accel)
        self.gyro_data[sensor_id].append(gyro)
        
        # In calibration mode, just store calibration data
        if self.calibration_mode and sensor_id == self.current_calibration_sensor:
            self.calibration_data[sensor_id].append({
                'accel': accel.copy(),
                'gyro': gyro.copy(),
                'timestamp': timestamp,
                'step': self.current_calibration_step,
                'axis': self.current_calibration_axis
            })
            
        # Normalize accelerometer data to extract gravity direction
        accel_magnitude = np.linalg.norm(accel)
        if accel_magnitude > 0.1:  # Check for non-zero acceleration
            accel_normalized = accel / accel_magnitude
            
            # Convert current quaternion to gravity vector
            qw, qx, qy, qz = self.sensor_quaternions[sensor_id]
            
            # Expected gravity direction based on current orientation
            # This is a fast way to rotate [0,0,1] by the quaternion
            gx = 2.0 * (qx*qz - qw*qy)
            gy = 2.0 * (qy*qz + qw*qx)
            gz = qw*qw - qx*qx - qy*qy + qz*qz
            expected_gravity = np.array([gx, gy, gz])
            
            # Calculate rotation error using cross product
            error = np.cross(accel_normalized, expected_gravity)
            
            # Apply error correction to gyro (optional, helps with drift)
            error_gain = 0.1
            gyro = gyro + error * error_gain
        
        # Quaternion derivative from angular velocity
        qw, qx, qy, qz = self.sensor_quaternions[sensor_id]
        half_dt = 0.5 * dt
        
        dqw = -half_dt * (qx*gyro[0] + qy*gyro[1] + qz*gyro[2])
        dqx =  half_dt * (qw*gyro[0] + qy*gyro[2] - qz*gyro[1])
        dqy =  half_dt * (qw*gyro[1] + qz*gyro[0] - qx*gyro[2])
        dqz =  half_dt * (qw*gyro[2] + qx*gyro[1] - qy*gyro[0])
        
        # Update quaternion
        qw += dqw
        qx += dqx
        qy += dqy
        qz += dqz
        
        # Normalize quaternion
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if norm > 0:
            self.sensor_quaternions[sensor_id] = np.array([qw, qx, qy, qz]) / norm
        
        # Store quaternion
        self.quaternions[sensor_id].append(self.sensor_quaternions[sensor_id].copy())
        
        # Apply calibration transform if calibrated
        if self.is_calibrated[sensor_id]:
            self.anatomical_quaternions[sensor_id] = self.apply_calibration(
                sensor_id, self.sensor_quaternions[sensor_id]
            )
        else:
            self.anatomical_quaternions[sensor_id] = self.sensor_quaternions[sensor_id].copy()
        
        # If all sensors are calibrated, update joint angles
        if all(self.is_calibrated):
            self.update_joint_angles()
        
        return self.sensor_quaternions[sensor_id]
    
    def apply_calibration(self, sensor_id, q_sensor):
        """Apply calibration to transform sensor quaternion to anatomical frame"""
        q_cal = self.calibration_quaternions[sensor_id]
        
        # Transform quaternion: q_anatomical = q_cal ⊗ q_sensor ⊗ q_cal*
        q_anatomical = quaternion_multiply(
            q_cal,
            quaternion_multiply(q_sensor, quaternion_conjugate(q_cal))
        )
        
        return q_anatomical
    
    def update_joint_angles(self):
        """Update joint angles based on calibrated sensor quaternions"""
        if self.num_sensors < 2:
            return
        
        # For a complete implementation, this would use the relative quaternions
        # between segments to calculate the 7 DOF joint angles
        # This is a placeholder implementation
        
        # Example for extracting relative orientation between segments
        if self.num_sensors >= 2:
            # Relative rotation from upper arm to lower arm
            q_rel_elbow = quaternion_multiply(
                quaternion_conjugate(self.anatomical_quaternions[0]),
                self.anatomical_quaternions[1]
            )
            
            # Convert to Euler angles (simple approximation)
            euler_elbow = quaternion_to_euler(q_rel_elbow)
            
            # Update joint angles (just an example)
            self.joint_angles[3] = euler_elbow[0]  # elbow flexion
            self.joint_angles[4] = euler_elbow[2]  # forearm rotation
        
        if self.num_sensors >= 3:
            # Relative rotation from lower arm to hand
            q_rel_wrist = quaternion_multiply(
                quaternion_conjugate(self.anatomical_quaternions[1]),
                self.anatomical_quaternions[2]
            )
            
            # Convert to Euler angles
            euler_wrist = quaternion_to_euler(q_rel_wrist)
            
            # Update joint angles
            self.joint_angles[5] = euler_wrist[0]  # wrist flexion
            self.joint_angles[6] = euler_wrist[1]  # wrist deviation
    
    def start_calibration(self, sensor_id=0):
        """Start calibration process for a specific sensor"""
        if sensor_id < 0 or sensor_id >= self.num_sensors:
            print(f"Invalid sensor ID for calibration: {sensor_id}")
            return False
        
        self.calibration_mode = True
        self.current_calibration_sensor = sensor_id
        self.current_calibration_axis = 0
        self.current_calibration_step = 0
        self.calibration_data[sensor_id] = []
        
        print(f"Starting calibration for sensor {sensor_id}")
        return True
    
    def next_calibration_step(self):
        """Move to the next calibration step"""
        if not self.calibration_mode:
            return False
        
        self.current_calibration_step += 1
        
        # After collecting samples for the current axis, move to the next
        if self.current_calibration_step >= 2:
            self.current_calibration_axis += 1
            self.current_calibration_step = 0
            
            # After all axes, process calibration
            if self.current_calibration_axis >= 3:
                self.process_calibration()
                return False
        
        return True
    
    def process_calibration(self):
        """Process collected calibration data to compute calibration quaternion"""
        sensor_id = self.current_calibration_sensor
        
        if not self.calibration_data[sensor_id]:
            print(f"No calibration data for sensor {sensor_id}")
            self.calibration_mode = False
            return False
        
        print(f"Processing calibration data for sensor {sensor_id}")
        
        # Group data by axis and step
        reference_data = []
        axis_data = []
        
        for data in self.calibration_data[sensor_id]:
            if data['step'] == 0 and data['axis'] == 0:
                # Reference position data
                reference_data.append(data)
            else:
                # Axis rotation data
                axis_data.append(data)
        
        if not reference_data or not axis_data:
            print("Insufficient calibration data")
            self.calibration_mode = False
            return False
        
        # Calculate average reference quaternion
        ref_quats = [quaternion_from_acc_gyro(d['accel']) for d in reference_data]
        q_ref = average_quaternions(ref_quats)
        
        # Initialize optimization
        initial_q_cal = np.array([1.0, 0.0, 0.0, 0.0])  # Initial guess: identity quaternion
        
        # Define objective function for optimization
        def objective(q_cal_vec):
            # Normalize quaternion
            q_cal = q_cal_vec / np.linalg.norm(q_cal_vec)
            
            total_error = 0
            
            for data in axis_data:
                # Get measured sensor quaternion
                q_sensor = quaternion_from_acc_gyro(data['accel'])
                
                # Get expected anatomical quaternion based on axis and step
                axis = data['axis']
                angle = np.radians(45.0 if data['step'] == 1 else 0.0)  # Assuming 45° rotation
                
                if axis == 0:
                    # X-axis rotation
                    q_expected = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
                elif axis == 1:
                    # Y-axis rotation
                    q_expected = np.array([np.cos(angle/2), 0, np.sin(angle/2), 0])
                else:
                    # Z-axis rotation
                    q_expected = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
                
                # Apply calibration
                q_anatomical = quaternion_multiply(
                    q_cal,
                    quaternion_multiply(q_sensor, quaternion_conjugate(q_cal))
                )
                
                # Calculate error (quaternion distance)
                error = 1.0 - np.abs(np.dot(q_anatomical, q_expected))**2
                total_error += error
            
            return total_error
        
        # Run optimization
        result = minimize(objective, initial_q_cal, method='BFGS')
        
        if result.success:
            # Get optimized calibration quaternion
            q_cal = result.x / np.linalg.norm(result.x)
            self.calibration_quaternions[sensor_id] = q_cal
            self.is_calibrated[sensor_id] = True
            
            print(f"Calibration successful for sensor {sensor_id}: {q_cal}")
            self.save_calibration()
        else:
            print(f"Calibration optimization failed: {result.message}")
        
        self.calibration_mode = False
        return result.success
    
    def get_calibration_instruction(self):
        """Get instruction text for current calibration step"""
        if not self.calibration_mode:
            return "Not in calibration mode"
        
        sensor_names = ["Upper Arm", "Lower Arm", "Hand"]
        axis_names = ["X (elevation/flexion)", "Y (abduction/deviation)", "Z (rotation)"]
        
        sensor_name = sensor_names[self.current_calibration_sensor] if self.current_calibration_sensor < len(sensor_names) else f"Sensor {self.current_calibration_sensor}"
        
        if self.current_calibration_axis == 0 and self.current_calibration_step == 0:
            return f"Hold {sensor_name} in reference position (arm hanging naturally)"
        
        if self.current_calibration_step == 0:
            return f"Hold {sensor_name} in rest position"
        else:
            action = "Raise/Lower" if self.current_calibration_axis == 0 else "Move Side-to-Side" if self.current_calibration_axis == 1 else "Rotate"
            return f"{action} the {sensor_name} around {axis_names[self.current_calibration_axis]} axis"
    
    def save_calibration(self):
        """Save calibration data to file"""
        calibration_data = {
            "quaternions": [q.tolist() for q in self.calibration_quaternions],
            "is_calibrated": self.is_calibrated
        }
        
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(calibration_data, f)
            print(f"Calibration saved to {CALIBRATION_FILE}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self):
        """Load calibration data from file"""
        if not os.path.exists(CALIBRATION_FILE):
            print(f"Calibration file {CALIBRATION_FILE} not found")
            return False
        
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                data = json.load(f)
            
            self.calibration_quaternions = [np.array(q) for q in data["quaternions"]]
            self.is_calibrated = data["is_calibrated"]
            
            print(f"Calibration loaded from {CALIBRATION_FILE}")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

class ArmKinematicsVisualizer(QMainWindow):
    """Visualization GUI for arm kinematics with multiple sensors"""
    
    def __init__(self, num_sensors=3):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.sensor_reader = SerialReader(DEFAULT_PORTS[:num_sensors])
        self.estimator = MultiSensorEstimator(num_sensors)
        
        # Initialize UI
        self.init_ui()
        
        # Connect signals
        self.sensor_reader.data_ready.connect(self.process_sensor_data)
        
        # Setup update timer (for UI updates)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.start(33)  # ~30 FPS
        
        # Start sensor reading
        self.sensor_reader.start()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Arm Kinematics Tracker")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create tab widget for different views
        tab_widget = QTabWidget()
        
        # Tab 1: 3D Visualization
        self.viz_tab = QWidget()
        viz_layout = QVBoxLayout()
        
        # 3D Visualization widget
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=1.0, elevation=30, azimuth=45)
        
        # Add coordinate axes
        self.axis = gl.GLAxisItem()
        self.axis.setSize(0.5, 0.5, 0.5)
        self.gl_view.addItem(self.axis)
        
        # Add grid
        self.grid = gl.GLGridItem()
        self.grid.setSize(2, 2, 1)
        self.grid.setSpacing(0.1, 0.1, 0.1)
        self.gl_view.addItem(self.grid)
        
        # Add arm segments
        self.arm_segments = []
        segment_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        
        for i in range(num_sensors):
            segment = gl.GLLinePlotItem(width=5, color=segment_colors[i % len(segment_colors)])
            self.gl_view.addItem(segment)
            self.arm_segments.append(segment)
        
        # Add joints
        self.joints = []
        joint_color = (1, 1, 0, 1)
        
        for i in range(num_sensors + 1):  # One more joint than segments
            joint = gl.GLScatterPlotItem(size=15, color=joint_color)
            self.gl_view.addItem(joint)
            self.joints.append(joint)
        
        viz_layout.addWidget(self.gl_view)
        self.viz_tab.setLayout(viz_layout)
        tab_widget.addTab(self.viz_tab, "3D Visualization")
        
        # Tab 2: Sensor Data
        self.data_tab = QWidget()
        data_layout = QVBoxLayout()
        
        # Sensor data displays
        sensor_group = QGroupBox("Sensor Data")
        sensor_layout = QVBoxLayout()
        
        self.sensor_labels = []
        self.quaternion_labels = []
        
        for i in range(num_sensors):
            sensor_name = ["Upper Arm", "Lower Arm", "Hand"][i] if i < 3 else f"Sensor {i}"
            sensor_label = QLabel(f"{sensor_name} Sensor:")
            sensor_label.setFont(QFont("Arial", 12, QFont.Bold))
            
            quat_label = QLabel("Quaternion: [1, 0, 0, 0]")
            
            sensor_layout.addWidget(sensor_label)
            sensor_layout.addWidget(quat_label)
            sensor_layout.addSpacing(10)
            
            self.sensor_labels.append(sensor_label)
            self.quaternion_labels.append(quat_label)
        
        sensor_group.setLayout(sensor_layout)
        data_layout.addWidget(sensor_group)
        
        # Joint angles display
        angles_group = QGroupBox("Joint Angles")
        angles_layout = QVBoxLayout()
        
        self.joint_labels = []
        joint_names = [
            "Shoulder Elevation",
            "Shoulder Abduction",
            "Shoulder Rotation",
            "Elbow Flexion",
            "Forearm Rotation",
            "Wrist Flexion",
            "Wrist Deviation"
        ]
        
        for i, name in enumerate(joint_names):
            angle_label = QLabel(f"{name}: 0.0°")
            angles_layout.addWidget(angle_label)
            self.joint_labels.append(angle_label)
        
        angles_group.setLayout(angles_layout)
        data_layout.addWidget(angles_group)
        
        self.data_tab.setLayout(data_layout)
        tab_widget.addTab(self.data_tab, "Sensor Data")
        
        # Tab 3: Calibration
        self.cal_tab = QWidget()
        cal_layout = QVBoxLayout()
        
        # Calibration status
        cal_status_group = QGroupBox("Calibration Status")
        cal_status_layout = QVBoxLayout()
        
        self.cal_status_labels = []
        
        for i in range(num_sensors):
            sensor_name = ["Upper Arm", "Lower Arm", "Hand"][i] if i < 3 else f"Sensor {i}"
            status_label = QLabel(f"{sensor_name}: Not Calibrated")
            cal_status_layout.addWidget(status_label)
            self.cal_status_labels.append(status_label)
        
        cal_status_group.setLayout(cal_status_layout)
        cal_layout.addWidget(cal_status_group)
        
        # Calibration controls
        cal_control_group = QGroupBox("Calibration Controls")
        cal_control_layout = QVBoxLayout()
        
        self.sensor_combo = QComboBox()
        for i in range(num_sensors):
            sensor_name = ["Upper Arm", "Lower Arm", "Hand"][i] if i < 3 else f"Sensor {i}"
            self.sensor_combo.addItem(sensor_name)
        
        self.cal_button = QPushButton("Start Calibration")
        self.cal_button.clicked.connect(self.start_calibration)
        
        self.cal_next_button = QPushButton("Next Step")
        self.cal_next_button.clicked.connect(self.next_calibration_step)
        self.cal_next_button.setEnabled(False)
        
        self.cal_instruction = QLabel("Select a sensor and press 'Start Calibration'")
        self.cal_instruction.setWordWrap(True)
        self.cal_instruction.setMinimumHeight(60)
        self.cal_instruction.setFont(QFont("Arial", 12))
        
        self.cal_progress = QProgressBar()
        self.cal_progress.setRange(0, 6)  # 2 steps x 3 axes
        self.cal_progress.setValue(0)
        
        cal_control_layout.addWidget(QLabel("Select Sensor:"))
        cal_control_layout.addWidget(self.sensor_combo)
        cal_control_layout.addWidget(self.cal_button)
        cal_control_layout.addWidget(self.cal_next_button)
        cal_control_layout.addWidget(QLabel("Instructions:"))
        cal_control_layout.addWidget(self.cal_instruction)
        cal_control_layout.addWidget(self.cal_progress)
        
        cal_control_group.setLayout(cal_control_layout)
        cal_layout.addWidget(cal_control_group)
        
        self.cal_tab.setLayout(cal_layout)
        tab_widget.addTab(self.cal_tab, "Calibration")
        
        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)
        
        # Set main layout
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Update the calibration status display
        self.update_calibration_status()
    
    def process_sensor_data(self, data):
        """Process incoming sensor data"""
        sensor_id = data['sensor_id']
        accel = data['accel']
        gyro = data['gyro']
        timestamp = data['timestamp']
        
        # Update quaternion estimation
        self.estimator.update_sensor(sensor_id, accel, gyro, timestamp)
    
    def update_displays(self):
        """Update UI displays with latest data"""
        # Update quaternion labels
        for i in range(self.num_sensors):
            if i < len(self.estimator.quaternions) and self.estimator.quaternions[i]:
                q = self.estimator.quaternions[i][-1]
                self.quaternion_labels[i].setText(
                    f"Quaternion: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
                )
                
                if self.estimator.is_calibrated[i]:
                    self.sensor_labels[i].setText(f"{self.sensor_combo.itemText(i)} (Calibrated):")
                else:
                    self.sensor_labels[i].setText(f"{self.sensor_combo.itemText(i)} (Not Calibrated):")
        
        # Update joint angle labels
        for i, label in enumerate(self.joint_labels):
            angle_deg = np.degrees(self.estimator.joint_angles[i])
            label.setText(f"{label.text().split(':')[0]}: {angle_deg:.1f}°")
        
        # Update 3D visualization
        self.update_3d_visualization()
        
        # Update calibration instruction if in calibration mode
        if self.estimator.calibration_mode:
            self.cal_instruction.setText(self.estimator.get_calibration_instruction())
            progress_value = self.estimator.current_calibration_axis * 2 + self.estimator.current_calibration_step
            self.cal_progress.setValue(progress_value)
    
    def update_3d_visualization(self):
        """Update the 3D visualization of the arm"""
        if not all(self.estimator.is_calibrated):
            # If not all sensors are calibrated, just show them individually
            for i in range(self.num_sensors):
                if self.estimator.quaternions[i]:
                    # Show sensor orientation using a short line in the direction of local Z-axis
                    q = self.estimator.quaternions[i][-1]
                    r = R.from_quat([q[1], q[2], q[3], q[0]])  # xyzw to wxyz
                    
                    # Transform a unit vector along Z
                    v_local = np.array([0, 0, 0.2])  # Local Z axis
                    v_world = r.apply(v_local)
                    
                    # Position this at origin plus a small offset
                    pos = np.array([0.1 * i, 0, 0])
                    
                    # Update line segment
                    self.arm_segments[i].setData(
                        pos=np.array([pos, pos + v_world])
                    )
                    
                    # Update joint position
                    self.joints[i].setData(pos=np.array([pos]))
            
            # Hide any extra joints
            if self.num_sensors < len(self.joints):
                self.joints[self.num_sensors].setData(pos=np.empty((0, 3)))
            
            return
        
        # If all sensors are calibrated, show the full arm kinematic chain
        # This is a simple implementation - a full one would use the kinematic model
        
        # Define joint positions (simplified)
        joint_positions = [np.array([0, 0, 0])]  # Base position
        
        # Segment lengths (in meters)
        lengths = [0.3, 0.25, 0.08]  # Upper arm, lower arm, hand
        
        for i in range(self.num_sensors):
            if i < len(self.estimator.anatomical_quaternions):
                q = self.estimator.anatomical_quaternions[i]
                r = R.from_quat([q[1], q[2], q[3], q[0]])  # xyzw to wxyz
                
                # Direction vector
                if i == 0:
                    # Upper arm: typically points down in anatomical pose
                    v_local = np.array([0, 0, lengths[i]])
                elif i == 1:
                    # Lower arm: typically points forward
                    v_local = np.array([0, lengths[i], 0])
                else:
                    # Hand: typically points forward
                    v_local = np.array([0, lengths[i], 0])
                
                # Apply rotation
                v_world = r.apply(v_local)
                
                # Calculate next joint position
                next_pos = joint_positions[-1] + v_world
                joint_positions.append(next_pos)
                
                # Update segment
                self.arm_segments[i].setData(
                    pos=np.array([joint_positions[-2], joint_positions[-1]])
                )
        
        # Update joint positions
        for i, pos in enumerate(joint_positions):
            if i < len(self.joints):
                self.joints[i].setData(pos=np.array([pos]))
    
    def start_calibration(self):
        """Start the calibration process"""
        sensor_id = self.sensor_combo.currentIndex()
        
        if self.estimator.start_calibration(sensor_id):
            self.cal_button.setEnabled(False)
            self.cal_next_button.setEnabled(True)
            self.sensor_combo.setEnabled(False)
            self.cal_progress.setValue(0)
            self.cal_instruction.setText(self.estimator.get_calibration_instruction())
    
    def next_calibration_step(self):
        """Move to the next calibration step"""
        if not self.estimator.next_calibration_step():
            # Calibration completed
            self.cal_button.setEnabled(True)
            self.cal_next_button.setEnabled(False)
            self.sensor_combo.setEnabled(True)
            self.cal_instruction.setText("Calibration completed. Select a sensor for new calibration.")
            self.update_calibration_status()
    
    def update_calibration_status(self):
        """Update the calibration status display"""
        for i in range(self.num_sensors):
            status_text = "Calibrated" if self.estimator.is_calibrated[i] else "Not Calibrated"
            sensor_name = self.sensor_combo.itemText(i)
            self.cal_status_labels[i].setText(f"{sensor_name}: {status_text}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.sensor_reader.stop()
        self.update_timer.stop()
        event.accept()

# Helper functions for quaternion operations

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Return the conjugate of a quaternion"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_from_acc_gyro(acc):
    """Create a simple quaternion from accelerometer data (gravity direction)"""
    # Normalize acceleration vector
    acc = acc / np.linalg.norm(acc)
    
    # Find rotation from [0,0,1] to acc
    # This is a simple approach - more robust methods exist
    v = np.cross([0, 0, 1], acc)
    s = np.linalg.norm(v)
    
    if s < 1e-10:
        # Vectors are parallel, return identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    c = np.dot([0, 0, 1], acc)
    v = v / s
    
    # Convert axis-angle to quaternion
    angle = np.arctan2(s, c)
    sin_half = np.sin(angle/2)
    
    w = np.cos(angle/2)
    x = v[0] * sin_half
    y = v[1] * sin_half
    z = v[2] * sin_half
    
    return np.array([w, x, y, z])

def average_quaternions(quaternions):
    """Average a list of quaternions"""
    if not quaternions:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    # Ensure consistent sign (avoid averaging q with -q)
    ref = quaternions[0]
    for i in range(1, len(quaternions)):
        if np.dot(ref, quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]
    
    # Simple averaging (not perfect but works for small rotations)
    avg = np.mean(quaternions, axis=0)
    return avg / np.linalg.norm(avg)

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

# Main application
if __name__ == "__main__":
    import sys
    
    # Number of sensors (default 3 for full arm-hand system)
    num_sensors = 3
    
    # Create application
    app = QApplication(sys.argv)
    window = ArmKinematicsVisualizer(num_sensors)
    window.show()
    
    sys.exit(app.exec_())