import numpy as np
from scipy.spatial.transform import Rotation as R

class ArmHandKinematics:
    """
    A class to model the kinematics of a human arm and hand system with 7 DOF:
    - Upper arm: 3 DOF (shoulder)
    - Lower arm: 2 DOF (elbow and forearm rotation)
    - Hand: 2 DOF (wrist)
    """
    
    def __init__(self):
        # Initialize link lengths (in meters)
        self.upper_arm_length = 0.3  # ~30 cm
        self.lower_arm_length = 0.25  # ~25 cm
        self.hand_length = 0.08  # ~8 cm
        
        # Joint constraints (in radians)
        # Upper arm (shoulder) constraints
        self.shoulder_elevation_limits = (-np.pi/2, np.pi/2)  # Up/Down
        self.shoulder_abduction_limits = (-np.pi/2, np.pi/2)  # Side to side
        self.shoulder_rotation_limits = (-np.pi, np.pi)  # Rotation
        
        # Lower arm constraints
        self.elbow_flexion_limits = (0, 2*np.pi/3)  # Elbow bend
        self.forearm_rotation_limits = (-np.pi, np.pi)  # Forearm rotation
        
        # Hand (wrist) constraints
        self.wrist_flexion_limits = (-np.pi/3, np.pi/3)  # Wrist up/down
        self.wrist_deviation_limits = (-np.pi/4, np.pi/4)  # Wrist side to side
    
    def apply_joint_constraints(self, joint_angles):
        """
        Apply joint constraints to ensure physiologically plausible poses.
        
        Args:
            joint_angles: List of 7 joint angles [shoulder_elev, shoulder_abd, shoulder_rot,
                                                  elbow_flex, forearm_rot,
                                                  wrist_flex, wrist_dev]
        
        Returns:
            Constrained joint angles
        """
        constrained_angles = joint_angles.copy()
        
        # Apply constraints to each joint
        constrained_angles[0] = np.clip(constrained_angles[0], *self.shoulder_elevation_limits)
        constrained_angles[1] = np.clip(constrained_angles[1], *self.shoulder_abduction_limits)
        constrained_angles[2] = np.clip(constrained_angles[2], *self.shoulder_rotation_limits)
        constrained_angles[3] = np.clip(constrained_angles[3], *self.elbow_flexion_limits)
        constrained_angles[4] = np.clip(constrained_angles[4], *self.forearm_rotation_limits)
        constrained_angles[5] = np.clip(constrained_angles[5], *self.wrist_flexion_limits)
        constrained_angles[6] = np.clip(constrained_angles[6], *self.wrist_deviation_limits)
        
        return constrained_angles
    
    def forward_kinematics(self, joint_angles):
        """
        Compute forward kinematics to get the positions of each joint and the end effector.
        
        Args:
            joint_angles: List of 7 joint angles [shoulder_elev, shoulder_abd, shoulder_rot,
                                                  elbow_flex, forearm_rot,
                                                  wrist_flex, wrist_dev]
        
        Returns:
            List of 4x4 transformation matrices for each joint and the end effector
        """
        # Apply constraints to joint angles
        joint_angles = self.apply_joint_constraints(joint_angles)
        
        # Extract individual joint angles
        shoulder_elev, shoulder_abd, shoulder_rot, elbow_flex, forearm_rot, wrist_flex, wrist_dev = joint_angles
        
        # Initialize the transformation matrices
        T_base = np.eye(4)  # Base transformation (identity)
        transformations = [T_base]
        
        # 1. Shoulder elevation (rotation around x-axis)
        T_shoulder_elev = np.array([
            [1, 0, 0, 0],
            [0, np.cos(shoulder_elev), -np.sin(shoulder_elev), 0],
            [0, np.sin(shoulder_elev), np.cos(shoulder_elev), 0],
            [0, 0, 0, 1]
        ])
        
        # 2. Shoulder abduction (rotation around y-axis)
        T_shoulder_abd = np.array([
            [np.cos(shoulder_abd), 0, np.sin(shoulder_abd), 0],
            [0, 1, 0, 0],
            [-np.sin(shoulder_abd), 0, np.cos(shoulder_abd), 0],
            [0, 0, 0, 1]
        ])
        
        # 3. Shoulder rotation (rotation around z-axis)
        T_shoulder_rot = np.array([
            [np.cos(shoulder_rot), -np.sin(shoulder_rot), 0, 0],
            [np.sin(shoulder_rot), np.cos(shoulder_rot), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Combine shoulder transformations
        T_shoulder = T_base @ T_shoulder_elev @ T_shoulder_abd @ T_shoulder_rot
        transformations.append(T_shoulder)
        
        # 4. Upper arm to elbow (translation along z-axis)
        T_upper_arm = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.upper_arm_length],
            [0, 0, 0, 1]
        ])
        
        # 5. Elbow flexion (rotation around x-axis)
        T_elbow_flex = np.array([
            [1, 0, 0, 0],
            [0, np.cos(elbow_flex), -np.sin(elbow_flex), 0],
            [0, np.sin(elbow_flex), np.cos(elbow_flex), 0],
            [0, 0, 0, 1]
        ])
        
        # 6. Forearm rotation (rotation around z-axis)
        T_forearm_rot = np.array([
            [np.cos(forearm_rot), -np.sin(forearm_rot), 0, 0],
            [np.sin(forearm_rot), np.cos(forearm_rot), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Combine elbow transformations
        T_elbow = T_shoulder @ T_upper_arm @ T_elbow_flex @ T_forearm_rot
        transformations.append(T_elbow)
        
        # 7. Lower arm to wrist (translation along z-axis)
        T_lower_arm = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.lower_arm_length],
            [0, 0, 0, 1]
        ])
        
        # 8. Wrist flexion (rotation around x-axis)
        T_wrist_flex = np.array([
            [1, 0, 0, 0],
            [0, np.cos(wrist_flex), -np.sin(wrist_flex), 0],
            [0, np.sin(wrist_flex), np.cos(wrist_flex), 0],
            [0, 0, 0, 1]
        ])
        
        # 9. Wrist deviation (rotation around y-axis)
        T_wrist_dev = np.array([
            [np.cos(wrist_dev), 0, np.sin(wrist_dev), 0],
            [0, 1, 0, 0],
            [-np.sin(wrist_dev), 0, np.cos(wrist_dev), 0],
            [0, 0, 0, 1]
        ])
        
        # Combine wrist transformations
        T_wrist = T_elbow @ T_lower_arm @ T_wrist_flex @ T_wrist_dev
        transformations.append(T_wrist)
        
        # 10. Hand to end effector (translation along z-axis)
        T_hand = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.hand_length],
            [0, 0, 0, 1]
        ])
        
        # Final end effector transformation
        T_end_effector = T_wrist @ T_hand
        transformations.append(T_end_effector)
        
        return transformations
    
    def get_joint_positions(self, joint_angles):
        """
        Compute the positions of each joint and the end effector.
        
        Args:
            joint_angles: List of 7 joint angles [shoulder_elev, shoulder_abd, shoulder_rot,
                                                  elbow_flex, forearm_rot,
                                                  wrist_flex, wrist_dev]
        
        Returns:
            List of 3D positions for the base, shoulder, elbow, wrist, and end effector
        """
        transformations = self.forward_kinematics(joint_angles)
        
        # Extract positions from the transformation matrices
        positions = []
        for transform in transformations:
            positions.append(transform[:3, 3])
        
        return positions
    
    def print_joint_positions(self, joint_angles):
        """
        Print the positions of each joint and the end effector.
        
        Args:
            joint_angles: List of 7 joint angles [shoulder_elev, shoulder_abd, shoulder_rot,
                                                  elbow_flex, forearm_rot,
                                                  wrist_flex, wrist_dev]
        """
        positions = self.get_joint_positions(joint_angles)
        joint_names = ["Base", "Shoulder", "Elbow", "Wrist", "End Effector"]
        
        print("Joint Positions:")
        for i, (name, pos) in enumerate(zip(joint_names, positions)):
            print(f"{name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

# Example usage
if __name__ == "__main__":
    # Create an instance of the kinematics model
    arm = ArmHandKinematics()
    
    # Define some example joint angles (in radians)
    example_angles = [
        np.radians(30),  # shoulder elevation
        np.radians(20),  # shoulder abduction
        np.radians(10),  # shoulder rotation
        np.radians(45),  # elbow flexion
        np.radians(20),  # forearm rotation
        np.radians(10),  # wrist flexion
        np.radians(5)    # wrist deviation
    ]
    
    # Apply forward kinematics and print the results
    print("Example joint angles (degrees):")
    for i, angle in enumerate(example_angles):
        print(f"Joint {i+1}: {np.degrees(angle):.1f}°")
    
    print("\nForward Kinematics:")
    arm.print_joint_positions(example_angles)