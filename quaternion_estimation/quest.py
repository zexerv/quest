#!/usr/bin/env python3
"""
Simple, Low-Latency Quaternion Estimation for MPU6050
"""

import numpy as np
import serial
import time
import threading
from collections import deque
from PyQt5.QtCore import pyqtSignal, QObject

# Configuration
DEFAULT_PORT = '/dev/ttyACM0'
DEFAULT_BAUD_RATE = 115200
COMPLEMENTARY_FILTER_ALPHA = 0.98  # Higher = more gyro, less accel

class SerialReader(QObject):
    """Simple serial reader that doesn't block the main thread"""
    data_ready = pyqtSignal(object)
    
    def __init__(self, port=DEFAULT_PORT, baud_rate=DEFAULT_BAUD_RATE):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = False
        self.ser = None
        self.thread = None
    
    def start(self):
        """Start the serial reading thread"""
        self.running = True
        self.thread = threading.Thread(target=self._read_serial, daemon=True)
        self.thread.start()
    
    def _read_serial(self):
        """Thread function to continuously read from serial"""
        try:
            # Open serial port
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            
            while self.running:
                try:
                    # Read a line and decode
                    line = self.ser.readline().decode('utf-8').rstrip()
                    
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
                                'accel': np.array([ax, ay, az]),
                                'gyro': np.array([gx, gy, gz]),
                                'timestamp': time.time()
                            })
                        except ValueError:
                            print(f"Invalid data: {line}")
                    elif line:
                        print(f"Incomplete data: {line}")
                        
                except serial.SerialException as e:
                    print(f"Serial error: {e}")
                    time.sleep(1)  # Wait before trying again
                    break
                except Exception as e:
                    print(f"Error in serial thread: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Failed to open {self.port}: {e}")
        
        finally:
            # Try to close the serial port
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("Serial port closed")
    
    def stop(self):
        """Stop the serial reading thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()

class QuaternionEstimator:
    """Fast quaternion estimation using sensor fusion"""
    
    def __init__(self, alpha=COMPLEMENTARY_FILTER_ALPHA):
        self.alpha = alpha
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.last_time = time.time()
        
        # Data storage
        self.max_samples = 500
        self.timestamps = deque(maxlen=self.max_samples)
        self.accel_data = deque(maxlen=self.max_samples)
        self.gyro_data = deque(maxlen=self.max_samples)
        self.quaternions = deque(maxlen=self.max_samples)
        self.processing_times = deque(maxlen=self.max_samples)
    
    def update(self, accel, gyro, timestamp=None):
        """Update quaternion estimation with new sensor data"""
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate time delta
        dt = timestamp - self.last_time if self.last_time else 0.01
        self.last_time = timestamp
        
        # Limit dt to reasonable values
        dt = min(max(dt, 0.001), 0.05)
        
        # Store raw data
        self.timestamps.append(timestamp)
        self.accel_data.append(accel)
        self.gyro_data.append(gyro)
        
        # Normalize accelerometer data to extract gravity direction
        accel_magnitude = np.linalg.norm(accel)
        if accel_magnitude > 0.1:  # Check for non-zero acceleration
            accel_normalized = accel / accel_magnitude
            print(accel, accel_magnitude)
            
            # Convert current quaternion to gravity vector
            qw, qx, qy, qz = self.quaternion
            
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
        qw, qx, qy, qz = self.quaternion
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
            self.quaternion = np.array([qw, qx, qy, qz]) / norm
        
        # Store quaternion
        self.quaternions.append(self.quaternion.copy())
        
        # Record processing time
        self.processing_times.append((time.time() - start_time) * 1000)  # in ms
        
        return self.quaternion

# Simple usage example
if __name__ == "__main__":
    import time
    
    # Create and start the serial reader
    reader = SerialReader()
    reader.start()
    
    # Create quaternion estimator
    estimator = QuaternionEstimator()
    
    # Process data when it arrives
    def process_data(data):
        accel = data['accel']
        gyro = data['gyro']
        timestamp = data['timestamp']
        
        # Update quaternion
        quaternion = estimator.update(accel, gyro, timestamp)
        print(f"Quaternion: [w={quaternion[0]:.4f}, x={quaternion[1]:.4f}, y={quaternion[2]:.4f}, z={quaternion[3]:.4f}]")
    
    # Connect the signal
    reader.data_ready.connect(process_data)
    
    # Run for a while
    try:
        print("Reading quaternion data. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()