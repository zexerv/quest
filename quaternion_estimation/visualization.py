# visualization.py - 3D Visualization for Quaternion Orientation
# Updated to use direct serial reading approach

import sys
import numpy as np
import serial
import time
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

class SerialReader(QObject):
    """Simple serial reader for MPU6050 data"""
    data_ready = pyqtSignal(object)
    
    def __init__(self, port='/dev/ttyACM0', baud_rate=115200):
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
    
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.last_time = time.time()
    
    def update(self, accel, gyro, timestamp=None):
        """Update quaternion estimation with new sensor data"""
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate time delta
        dt = timestamp - self.last_time if self.last_time else 0.01
        self.last_time = timestamp
        
        # Limit dt to reasonable values
        dt = min(max(dt, 0.001), 0.05)
        
        # Normalize accelerometer data to extract gravity direction
        accel_magnitude = np.linalg.norm(accel)
        if accel_magnitude > 0.1:  # Check for non-zero acceleration
            accel_normalized = accel / accel_magnitude
            
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
        
        return self.quaternion

class SphereVisualization(QOpenGLWidget):
    def __init__(self, parent=None):
        super(SphereVisualization, self).__init__(parent)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # initial quaternion [w, x, y, z]
        self.sphere_radius = 0.8  # Sphere radius
        self.dot_radius = 0.05    # Indicator dot radius
        
        # Initialize 8 colors for the 8 regions of the sphere
        self.colors = [
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0, 1.0],  # Orange
            [0.5, 0.0, 1.0, 1.0],  # Purple
        ]
        
        # Start serial reader and quaternion estimator
        self.serial_reader = SerialReader()
        self.serial_reader.data_ready.connect(self.process_sensor_data)
        self.serial_reader.start()
        
        self.estimator = QuaternionEstimator()
        
        # Set up update timer for visualization
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(16)  # ~60 FPS
    
    def process_sensor_data(self, data):
        """Process incoming sensor data"""
        accel = data['accel']
        gyro = data['gyro']
        timestamp = data['timestamp']
        
        # Update quaternion estimation
        self.quaternion = self.estimator.update(accel, gyro, timestamp)
    
    def initializeGL(self):
        # Set up OpenGL
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Set up light
        light_pos = [10.0, 10.0, 10.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
    
    def resizeGL(self, width, height):
        # Set up viewport
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Position camera
        glTranslatef(0.0, 0.0, -3.0)
        
        # Apply rotation from quaternion
        qw, qx, qy, qz = self.quaternion
        
        # Convert quaternion to rotation matrix (column-major for OpenGL)
        m = np.zeros(16, dtype=np.float32)
        
        m[0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        m[1] = 2.0 * (qx * qy + qw * qz)
        m[2] = 2.0 * (qx * qz - qw * qy)
        m[3] = 0.0
        
        m[4] = 2.0 * (qx * qy - qw * qz)
        m[5] = 1.0 - 2.0 * (qx * qx + qz * qz)
        m[6] = 2.0 * (qy * qz + qw * qx)
        m[7] = 0.0
        
        m[8] = 2.0 * (qx * qz + qw * qy)
        m[9] = 2.0 * (qy * qz - qw * qx)
        m[10] = 1.0 - 2.0 * (qx * qx + qy * qy)
        m[11] = 0.0
        
        m[12] = 0.0
        m[13] = 0.0
        m[14] = 0.0
        m[15] = 1.0
        
        glMultMatrixf(m)
        
        # Draw colored sphere segments
        self.draw_colored_sphere()
        
        # Draw indicator dot at the top of the sphere
        self.draw_indicator_dot()
        
        # Draw coordinate axes
        self.draw_axes()
    
    def draw_colored_sphere(self):
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        
        # Draw sphere with alternating colored octants
        stacks = 32
        slices = 32
        
        for i in range(stacks):
            lat0 = np.pi * (-0.5 + (i / stacks))
            z0 = np.sin(lat0)
            zr0 = np.cos(lat0)
            
            lat1 = np.pi * (-0.5 + ((i + 1) / stacks))
            z1 = np.sin(lat1)
            zr1 = np.cos(lat1)
            
            for j in range(slices):
                lng0 = 2 * np.pi * (j / slices)
                x0 = np.cos(lng0) * zr0
                y0 = np.sin(lng0) * zr0
                
                lng1 = 2 * np.pi * ((j + 1) / slices)
                x1 = np.cos(lng1) * zr0
                y1 = np.sin(lng1) * zr0
                
                x2 = np.cos(lng0) * zr1
                y2 = np.sin(lng0) * zr1
                
                x3 = np.cos(lng1) * zr1
                y3 = np.sin(lng1) * zr1
                
                # Determine which octant this quad is in
                octant = 0
                if x0 > 0:
                    octant |= 1
                if y0 > 0:
                    octant |= 2
                if z0 > 0:
                    octant |= 4
                
                # Set color for this octant
                glColor4fv(self.colors[octant])
                
                # Draw quad
                glBegin(GL_QUADS)
                
                # Vertex normals for proper lighting
                glNormal3f(x0, y0, z0)
                glVertex3f(x0 * self.sphere_radius, y0 * self.sphere_radius, z0 * self.sphere_radius)
                
                glNormal3f(x1, y1, z0)
                glVertex3f(x1 * self.sphere_radius, y1 * self.sphere_radius, z0 * self.sphere_radius)
                
                glNormal3f(x3, y3, z1)
                glVertex3f(x3 * self.sphere_radius, y3 * self.sphere_radius, z1 * self.sphere_radius)
                
                glNormal3f(x2, y2, z1)
                glVertex3f(x2 * self.sphere_radius, y2 * self.sphere_radius, z1 * self.sphere_radius)
                
                glEnd()
    
    def draw_indicator_dot(self):
        # Draw a white dot at the top of the sphere
        glColor4f(1.0, 1.0, 1.0, 1.0)
        
        sphere = gluNewQuadric()
        gluQuadricNormals(sphere, GLU_SMOOTH)
        
        glPushMatrix()
        glTranslatef(0, 0, self.sphere_radius)
        gluSphere(sphere, self.dot_radius, 16, 16)
        glPopMatrix()
        
        gluDeleteQuadric(sphere)
    
    def draw_axes(self):
        # Draw coordinate axes
        glLineWidth(2.0)
        
        # X axis (red)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glEnd()
        
        # Y axis (green)
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glEnd()
        
        # Z axis (blue)
        glBegin(GL_LINES)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Quaternion Visualization")
        self.setGeometry(100, 100, 800, 600)
        
        self.visualization = SphereVisualization(self)
        self.setCentralWidget(self.visualization)
    
    def closeEvent(self, event):
        """Clean up when window is closed"""
        if hasattr(self.visualization, 'serial_reader'):
            self.visualization.serial_reader.stop()
        super().closeEvent(event)

# Main execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())