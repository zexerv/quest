# Hand Pose Estimation with Tri-Gyro System

This repository explores quaternion estimation and optimization using a tri-gyro system for human hand pose estimation.

**Key Features:**

* **Hardware:** 3 MPU6050 gyroscopes, Arduino, multiplexer.
* **Kinematics:** Models the human hand with 7 degrees of freedom (DOF).
* **Quaternion Estimation:** Estimates the attitude of each gyro using gravitational force.
* **Calibration:** Utilizes domain knowledge of hand kinematics for accurate calibration.
* **Pose Estimation:** Derives the hand tip pose using calibrated data and optimization.

**Steps:**

1. **Quaternion Estimation**
2. **Calibration**
3. **Pose Estimation**

**This project aims to achieve accurate and robust hand pose estimation using a low-cost and accessible tri-gyro setup.**

To create this file, simply:

1. **Create a new file named README.md in your project directory.**
2. **Copy and paste the Markdown content above into the file.**
3. **Save the file.**

This will provide a clear and concise overview of your project for anyone visiting your repository.



# Quaternion Estimation from MPU6050 Sensor

This project implements quaternion estimation from an MPU6050 sensor (accelerometer and gyroscope) to track orientation in 3D space. The system uses sensor fusion to combine accelerometer and gyroscope data for accurate and stable orientation tracking.

## Overview

The system consists of three main components:

1. **Arduino Code (MPU6050_Quaternion.ino)**: Reads raw data from the MPU6050 sensor and sends it to the computer via serial connection.
2. **Quaternion Estimator (quest.py)**: Processes the sensor data and estimates the orientation as a quaternion.
3. **Visualization (visualization.py)**: Displays the orientation as a colored sphere with a reference point.

## Mathematical Framework

### Quaternion Basics

Quaternions are a mathematical notation used to represent 3D rotations. A quaternion has the form:

```
q = w + xi + yj + zk
```

Where:
- `w` is the scalar part
- `x`, `y`, `z` form the vector part

Quaternions avoid the gimbal lock problem that can occur with Euler angles and are more computationally efficient than rotation matrices for many operations.

### Gravity-Based Orientation

The key insight in our quaternion estimation is using gravity as a reference. The accelerometer measures the combined effect of linear acceleration and gravity. When the sensor is relatively stationary, the accelerometer reading approximates the gravity vector direction.

By knowing the gravity direction in the sensor frame, we can determine the orientation of the sensor relative to the Earth's reference frame. The gravity vector always points "down" in the Earth frame (along the Z-axis in our convention). 

The transformation from the gravity vector to a quaternion involves:

1. Finding the rotation axis (cross product between reference vector and measured gravity)
2. Finding the rotation angle (dot product between vectors)
3. Converting to quaternion representation

### Gyroscope Integration

While accelerometer data provides an absolute reference to gravity, it's susceptible to noise and non-gravitational accelerations. The gyroscope measures angular velocity, which can be integrated over time to track orientation changes:

```
q(t+dt) = q(t) ⊗ (1 + 0.5*dt*[0, ω_x, ω_y, ω_z])
```

Where:
- `q(t)` is the quaternion at time t
- `⊗` represents quaternion multiplication
- `[0, ω_x, ω_y, ω_z]` is angular velocity as a quaternion
- `dt` is the time step

Gyroscope integration provides smooth, responsive tracking but suffers from drift over time due to numerical integration errors.

### Sensor Fusion with Complementary Filter

To get the best of both sensors, we use a complementary filter:

```
q_fused = α * q_gyro + (1-α) * q_accel
```

Where:
- `α` is the filter coefficient (typically 0.98)
- `q_gyro` is the orientation estimated from gyroscope
- `q_accel` is the orientation estimated from accelerometer

For better results, we use spherical linear interpolation (SLERP) instead of simple linear interpolation:

```
q_fused = SLERP(q_gyro, q_accel, 1-α)
```

This approach combines the stability of the accelerometer (for long-term reference) with the responsiveness of the gyroscope (for short-term changes).

## Implementation Details

### quest.py

The quaternion estimation algorithm performs:

1. **Sensor Reading**: Reads accelerometer and gyroscope data from Arduino
2. **Gravity Direction Extraction**: Normalizes accelerometer data to obtain gravity direction
3. **Quaternion from Gravity**: Converts gravity vector to orientation quaternion
4. **Gyroscope Integration**: Converts angular velocity to quaternion rate and integrates
5. **Sensor Fusion**: Blends gyroscope and accelerometer quaternions using SLERP
6. **Output**: Provides the estimated quaternion for visualization or other uses

The core mathematical operations include:
- Quaternion multiplication
- Quaternion normalization
- Axis-angle to quaternion conversion
- Quaternion derivative calculation
- Spherical linear interpolation (SLERP)

### visualization.py

The visualization creates an 8-colored sphere where:
- Each octant has a different color (based on the signs of x, y, z coordinates)
- A white dot at the top of the sphere acts as a reference point
- The sphere rotates according to the estimated quaternion
- Coordinate axes provide reference for orientation

This visualization makes it easy to see the orientation of the sensor in real-time, with the colored octants providing clear visual reference for rotation.

### MPU6050_Quaternion.ino

The Arduino code:
1. Initializes the MPU6050 via I2C communication
2. Reads raw accelerometer and gyroscope values
3. Converts raw values to physical units (g for accelerometer, degrees/second for gyroscope)
4. Transmits data via serial connection at 100Hz
5. Uses a simple comma-separated format for easy parsing

## Usage

1. Wire the MPU6050 to the Arduino:
   - VCC → 3.3V
   - GND → GND
   - SCL → A5 (Arduino Uno)
   - SDA → A4 (Arduino Uno)
   
2. Upload the MPU6050_Quaternion.ino sketch to your Arduino board

3. Install required Python dependencies:
   ```
   pip install numpy scipy pyserial pyqt5 pyopengl
   ```

4. Run the visualization:
   ```
   python visualization.py
   ```

5. Observe the colored sphere rotating in accordance with the sensor orientation

## Future Improvements

- Implement Mahony or Madgwick filters for better sensor fusion
- Add magnetometer support for absolute yaw reference
- Add calibration routine for each sensor
- Implement Extended Kalman Filter for more accurate state estimation
- Support different sensor configurations and IMUs


