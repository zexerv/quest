#!/usr/bin/env python3
"""
3D Visualization for Quaternion Orientation using PyQtGraph
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl
import pyqtgraph as pg

# Import from our quest module
from quest import SerialReader, QuaternionEstimator

# Configuration
PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

class CubeVisualizer(gl.GLViewWidget):
    """3D visualization of quaternion orientation using a cube"""
    
    def __init__(self):
        super().__init__()
        
        # Set up the 3D view
        self.setCameraPosition(distance=5)
        self.setBackgroundColor('#222222')
        
        # Add coordinate axes
        self.addItem(gl.GLAxisItem(size=2))
        
        # Create the cube
        vertices, faces, colors = self._create_cube_data()
        self.cube = gl.GLMeshItem(
            vertexes=vertices, 
            faces=faces, 
            faceColors=colors,
            smooth=False, 
            drawEdges=True,
            edgeColor=(0.5, 0.5, 0.5, 1.0)
        )
        self.addItem(self.cube)
        
        # Current orientation
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    
    def _create_cube_data(self):
        """Create vertices, faces and colors for a cube"""
        # Vertices for a cube centered at origin
        vertices = np.array([
            [ 0.5,  0.5,  0.5],  # 0
            [-0.5,  0.5,  0.5],  # 1
            [ 0.5, -0.5,  0.5],  # 2
            [-0.5, -0.5,  0.5],  # 3
            [ 0.5,  0.5, -0.5],  # 4
            [-0.5,  0.5, -0.5],  # 5
            [ 0.5, -0.5, -0.5],  # 6
            [-0.5, -0.5, -0.5]   # 7
        ])
        
        # Faces defined by vertex indices
        faces = np.array([
            [0, 1, 3, 2],  # Front
            [4, 6, 7, 5],  # Back
            [0, 4, 5, 1],  # Top
            [2, 3, 7, 6],  # Bottom
            [0, 2, 6, 4],  # Right
            [1, 5, 7, 3]   # Left
        ])
        
        # Face colors
        colors = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0, 1.0]   # Cyan
        ])
        
        return vertices, faces, colors
    
    def update_orientation(self, quaternion):
        """Update cube orientation from quaternion"""
        self.quaternion = quaternion
        
        # Convert quaternion to rotation matrix
        w, x, y, z = quaternion
        
        # Row-major rotation matrix from quaternion
        m11 = 1.0 - 2.0 * (y*y + z*z)
        m12 = 2.0 * (x*y - z*w)
        m13 = 2.0 * (x*z + y*w)
        
        m21 = 2.0 * (x*y + z*w)
        m22 = 1.0 - 2.0 * (x*x + z*z)
        m23 = 2.0 * (y*z - x*w)
        
        m31 = 2.0 * (x*z - y*w)
        m32 = 2.0 * (y*z + x*w)
        m33 = 1.0 - 2.0 * (x*x + y*y)
        
        # Convert to 4x4 matrix for OpenGL
        matrix = np.array([
            [m11, m12, m13, 0],
            [m21, m22, m23, 0],
            [m31, m32, m33, 0],
            [0, 0, 0, 1]
        ])
        
        # Apply transform to cube
        self.cube.setTransform(matrix)

class MainWindow(QMainWindow):
    """Main application window with plots and visualization"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast Quaternion Estimation")
        self.resize(1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)
        
        # Configure plots
        pg.setConfigOptions(antialias=True)
        
        # Accelerometer plot
        self.accel_plot = pg.PlotWidget(title="Accelerometer")
        self.accel_plot.setLabel('left', 'Acceleration', 'g')
        self.accel_plot.addLegend()
        self.accel_plot.showGrid(x=True, y=True)
        self.accel_x_curve = self.accel_plot.plot(pen='r', name='X')
        self.accel_y_curve = self.accel_plot.plot(pen='g', name='Y')
        self.accel_z_curve = self.accel_plot.plot(pen='b', name='Z')
        
        # Gyroscope plot
        self.gyro_plot = pg.PlotWidget(title="Gyroscope")
        self.gyro_plot.setLabel('left', 'Angular Velocity', 'rad/s')
        self.gyro_plot.addLegend()
        self.gyro_plot.showGrid(x=True, y=True)
        self.gyro_x_curve = self.gyro_plot.plot(pen='r', name='X')
        self.gyro_y_curve = self.gyro_plot.plot(pen='g', name='Y')
        self.gyro_z_curve = self.gyro_plot.plot(pen='b', name='Z')
        
        # Quaternion plot
        self.quat_plot = pg.PlotWidget(title="Quaternion")
        self.quat_plot.setLabel('left', 'Value')
        self.quat_plot.addLegend()
        self.quat_plot.showGrid(x=True, y=True)
        self.quat_w_curve = self.quat_plot.plot(pen='#FFF', name='W')
        self.quat_x_curve = self.quat_plot.plot(pen='r', name='X')
        self.quat_y_curve = self.quat_plot.plot(pen='g', name='Y')
        self.quat_z_curve = self.quat_plot.plot(pen='b', name='Z')
        
        # Timing plot
        self.timing_plot = pg.PlotWidget(title="Processing Time")
        self.timing_plot.setLabel('left', 'Time', 'ms')
        self.timing_plot.setLabel('bottom', 'Sample')
        self.timing_plot.showGrid(x=True, y=True)
        self.timing_curve = self.timing_plot.plot(pen='y')
        
        # 3D visualization
        try:
            self.visualizer = CubeVisualizer()
        except Exception as e:
            print(f"3D visualization not available: {e}")
            self.visualizer = None
        
        # Add widgets to layout
        layout.addWidget(self.accel_plot, 0, 0)
        layout.addWidget(self.gyro_plot, 0, 1)
        layout.addWidget(self.quat_plot, 1, 0)
        layout.addWidget(self.timing_plot, 1, 1)
        
        if self.visualizer:
            layout.addWidget(self.visualizer, 2, 0, 1, 2)
        
        # Set up serial reader from quest.py
        self.serial_reader = SerialReader(PORT, BAUD_RATE)
        self.serial_reader.data_ready.connect(self.process_sensor_data)
        self.serial_reader.start()
        
        # Set up quaternion estimator from quest.py
        self.estimator = QuaternionEstimator()
        
        # Update timer for plots
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(33)  # ~30 fps
    
    def process_sensor_data(self, data):
        """Process incoming sensor data"""
        accel = data['accel']
        gyro = data['gyro']
        timestamp = data['timestamp']
        
        # Update quaternion estimation
        quaternion = self.estimator.update(accel, gyro, timestamp)
        
        # Update 3D visualization immediately for responsiveness
        if self.visualizer:
            self.visualizer.update_orientation(quaternion)
    
    def update_plots(self):
        """Update all plots with latest data"""
        if not hasattr(self.estimator, 'timestamps') or len(self.estimator.timestamps) == 0:
            return
        
        # Get relative timestamps for x-axis
        start_time = self.estimator.timestamps[0]
        times = np.array([t - start_time for t in self.estimator.timestamps])
        
        # Update accelerometer plot
        accel_x = [a[0] for a in self.estimator.accel_data]
        accel_y = [a[1] for a in self.estimator.accel_data]
        accel_z = [a[2] for a in self.estimator.accel_data]
        self.accel_x_curve.setData(times, accel_x)
        self.accel_y_curve.setData(times, accel_y)
        self.accel_z_curve.setData(times, accel_z)
        
        # Update gyroscope plot
        gyro_x = [g[0] for g in self.estimator.gyro_data]
        gyro_y = [g[1] for g in self.estimator.gyro_data]
        gyro_z = [g[2] for g in self.estimator.gyro_data]
        self.gyro_x_curve.setData(times, gyro_x)
        self.gyro_y_curve.setData(times, gyro_y)
        self.gyro_z_curve.setData(times, gyro_z)
        
        # Update quaternion plot
        quat_w = [q[0] for q in self.estimator.quaternions]
        quat_x = [q[1] for q in self.estimator.quaternions]
        quat_y = [q[2] for q in self.estimator.quaternions]
        quat_z = [q[3] for q in self.estimator.quaternions]
        self.quat_w_curve.setData(times, quat_w)
        self.quat_x_curve.setData(times, quat_x)
        self.quat_y_curve.setData(times, quat_y)
        self.quat_z_curve.setData(times, quat_z)
        
        # Update timing plot
        self.timing_curve.setData(self.estimator.processing_times)
        
        # Auto-range plots occasionally
        if len(times) % 100 == 0:
            self.accel_plot.enableAutoRange()
            self.gyro_plot.enableAutoRange()
            self.quat_plot.enableAutoRange()
            self.timing_plot.enableAutoRange()
    
    def closeEvent(self, event):
        """Clean up when window is closed"""
        self.serial_reader.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())