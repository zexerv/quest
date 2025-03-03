import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QGroupBox, QPushButton)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from hand import ArmHandKinematics

class ArmHandVisualization(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize the arm kinematics
        self.arm = ArmHandKinematics()
        
        # Set initial joint angles (in degrees)
        self.joint_angles = [
            0,   # shoulder elevation
            0,   # shoulder abduction
            0,   # shoulder rotation
            45,  # elbow flexion
            0,   # forearm rotation
            0,   # wrist flexion
            0    # wrist deviation
        ]
        
        # Convert to radians
        self.joint_angles_rad = np.radians(self.joint_angles)
        
        # Define link colors
        self.link_colors = [
            (1, 0, 0, 1),     # Red for upper arm
            (0, 1, 0, 1),     # Green for lower arm
            (0, 0, 1, 1)      # Blue for hand
        ]
        
        # Initialize the UI
        self.initUI()
        
        # Update the 3D visualization
        self.update_visualization()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle('Arm-Hand Kinematics Visualization')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # Add sliders for each joint
        joint_names = [
            "Shoulder Elevation", 
            "Shoulder Abduction", 
            "Shoulder Rotation", 
            "Elbow Flexion", 
            "Forearm Rotation", 
            "Wrist Flexion", 
            "Wrist Deviation"
        ]
        
        # Define slider ranges and initial values
        slider_ranges = [
            (-90, 90),     # Shoulder elevation
            (-90, 90),     # Shoulder abduction
            (-180, 180),   # Shoulder rotation
            (0, 120),      # Elbow flexion
            (-180, 180),   # Forearm rotation
            (-60, 60),     # Wrist flexion
            (-45, 45)      # Wrist deviation
        ]
        
        self.sliders = []
        self.value_labels = []
        
        for i, (name, angle_range) in enumerate(zip(joint_names, slider_ranges)):
            group = QGroupBox(name)
            group_layout = QVBoxLayout()
            
            # Create slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(angle_range[0])
            slider.setMaximum(angle_range[1])
            slider.setValue(self.joint_angles[i])
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(30)
            
            # Create value label
            value_label = QLabel(f"Value: {self.joint_angles[i]}°")
            
            # Connect slider signal to update function
            slider.valueChanged.connect(lambda value, idx=i: self.update_joint_angle(idx, value))
            
            # Add slider and label to group layout
            group_layout.addWidget(slider)
            group_layout.addWidget(value_label)
            
            # Set the group layout
            group.setLayout(group_layout)
            
            # Add group to control layout
            control_layout.addWidget(group)
            
            # Store slider and label references
            self.sliders.append(slider)
            self.value_labels.append(value_label)
        
        # Add preset buttons
        preset_group = QGroupBox("Preset Poses")
        preset_layout = QVBoxLayout()
        
        # Default pose button
        default_btn = QPushButton("Default Pose")
        default_btn.clicked.connect(self.set_default_pose)
        preset_layout.addWidget(default_btn)
        
        # Reaching pose button
        reaching_btn = QPushButton("Reaching Pose")
        reaching_btn.clicked.connect(self.set_reaching_pose)
        preset_layout.addWidget(reaching_btn)
        
        # Waving pose button
        waving_btn = QPushButton("Waving Pose")
        waving_btn.clicked.connect(self.set_waving_pose)
        preset_layout.addWidget(waving_btn)
        
        # Set preset group layout
        preset_group.setLayout(preset_layout)
        control_layout.addWidget(preset_group)
        
        # Add information panel
        info_group = QGroupBox("Joint Position Information")
        info_layout = QVBoxLayout()
        
        # Create labels for joint positions
        self.position_labels = []
        joint_pos_names = ["Base", "Shoulder", "Elbow", "Wrist", "End Effector"]
        for name in joint_pos_names:
            label = QLabel(f"{name}: (0.000, 0.000, 0.000)")
            info_layout.addWidget(label)
            self.position_labels.append(label)
        
        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)
        
        # Add stretching space at bottom
        control_layout.addStretch()
        
        # Set control panel layout
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(300)
        
        # Create 3D visualization widget
        self.view_widget = gl.GLViewWidget()
        self.view_widget.setCameraPosition(distance=1.0, elevation=30, azimuth=45)
        
        # Add coordinate axes for reference
        self.axis = gl.GLAxisItem()
        self.axis.setSize(0.5, 0.5, 0.5)
        self.view_widget.addItem(self.axis)
        
        # Add grid for reference
        self.grid = gl.GLGridItem()
        self.grid.setSize(2, 2, 1)
        self.grid.setSpacing(0.1, 0.1, 0.1)
        self.view_widget.addItem(self.grid)
        
        # Add links and joints
        self.links = []
        for i in range(3):  # 3 links: upper arm, lower arm, hand
            link = gl.GLLinePlotItem(width=5)
            self.view_widget.addItem(link)
            self.links.append(link)
        
        self.joints = []
        for i in range(5):  # 5 joints: base, shoulder, elbow, wrist, end effector
            joint = gl.GLScatterPlotItem(size=15)
            self.view_widget.addItem(joint)
            self.joints.append(joint)
        
        # Add widgets to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.view_widget, 1)
        
        # Set main widget layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def update_joint_angle(self, index, value):
        # Update joint angle value
        self.joint_angles[index] = value
        self.joint_angles_rad[index] = np.radians(value)
        
        # Update value label
        self.value_labels[index].setText(f"Value: {value}°")
        
        # Update visualization
        self.update_visualization()
    
    def set_default_pose(self):
        # Set default pose values
        default_pose = [0, 0, 0, 45, 0, 0, 0]
        self.set_pose(default_pose)
    
    def set_reaching_pose(self):
        # Set reaching pose values
        reaching_pose = [30, 45, 10, 60, 20, 10, 5]
        self.set_pose(reaching_pose)
    
    def set_waving_pose(self):
        # Set waving pose values
        waving_pose = [0, 90, -30, 90, 0, 0, 0]
        self.set_pose(waving_pose)
    
    def set_pose(self, pose_angles):
        # Update sliders and internal values
        for i, angle in enumerate(pose_angles):
            self.sliders[i].setValue(angle)
            # The sliders will trigger the update_joint_angle method
    
    def update_visualization(self):
        # Compute forward kinematics
        positions = self.arm.get_joint_positions(self.joint_angles_rad)
        
        # Convert positions to numpy arrays for plotting
        pos_array = np.array(positions)
        
        # Update joint positions (spheres)
        joint_colors = [(1, 1, 0, 1), (1, 0.7, 0, 1), (0.7, 0.7, 0, 1), 
                        (0, 0.7, 0.7, 1), (0.7, 0, 0.7, 1)]
        
        for i, joint in enumerate(self.joints):
            joint.setData(pos=pos_array[i:i+1], color=joint_colors[i])
        
        # Update links (lines)
        self.links[0].setData(pos=pos_array[1:3], color=self.link_colors[0])  # Upper arm
        self.links[1].setData(pos=pos_array[2:4], color=self.link_colors[1])  # Lower arm
        self.links[2].setData(pos=pos_array[3:5], color=self.link_colors[2])  # Hand
        
        # Update position labels
        joint_names = ["Base", "Shoulder", "Elbow", "Wrist", "End Effector"]
        for i, (name, pos, label) in enumerate(zip(joint_names, positions, self.position_labels)):
            label.setText(f"{name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

# Main application
def main():
    app = QApplication(sys.argv)
    window = ArmHandVisualization()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()