// MPU6050_Quaternion.ino - Read MPU6050 sensor data and send to Python
// for quaternion estimation

#include <Wire.h>

// MPU6050 registers
#define MPU6050_ADDR         0x68
#define MPU6050_ACCEL_XOUT_H 0x3B
#define MPU6050_GYRO_XOUT_H  0x43
#define MPU6050_PWR_MGMT_1   0x6B

// MPU6050 scale factors
#define ACCEL_SCALE_FACTOR 16384.0  // LSB/g for +/- 2g range
#define GYRO_SCALE_FACTOR  131.0    // LSB/(deg/s) for +/- 250 deg/s range

// Variables for sensor data
float accel_x, accel_y, accel_z;
float gyro_x, gyro_y, gyro_z;

// Timing
unsigned long last_time = 0;
unsigned long sample_period = 10;  // 100 Hz sampling rate

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize MPU6050
  initMPU6050();
  
  // Wait for serial to be ready
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("MPU6050 Quaternion Sensor Data");
  Serial.println("Format: accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z");
}

void loop() {
  unsigned long current_time = millis();
  
  // Sample at specified rate
  if (current_time - last_time >= sample_period) {
    last_time = current_time;
    
    // Read sensor data
    readSensorData();
    
    // Send data to Python
    Serial.print(accel_x); Serial.print(",");
    Serial.print(accel_y); Serial.print(",");
    Serial.print(accel_z); Serial.print(",");
    Serial.print(gyro_x); Serial.print(",");
    Serial.print(gyro_y); Serial.print(",");
    Serial.println(gyro_z);
  }
}

void initMPU6050() {
  // Wake up MPU6050
  writeMPU6050(MPU6050_PWR_MGMT_1, 0);
  
}

void readSensorData() {
  // Start reading from the accelerometer registers
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 14, true);  // Request 14 bytes
  
  // Read accelerometer data (6 bytes)
  int16_t accel_x_raw = Wire.read() << 8 | Wire.read();
  int16_t accel_y_raw = Wire.read() << 8 | Wire.read();
  int16_t accel_z_raw = Wire.read() << 8 | Wire.read();
  
  // Skip temperature (2 bytes)
  Wire.read();
  Wire.read();
  
  // Read gyroscope data (6 bytes)
  int16_t gyro_x_raw = Wire.read() << 8 | Wire.read();
  int16_t gyro_y_raw = Wire.read() << 8 | Wire.read();
  int16_t gyro_z_raw = Wire.read() << 8 | Wire.read();
  
  // Convert to physical units
  // Accelerometer: Convert to g (9.81 m/s^2)
  accel_x = accel_x_raw / ACCEL_SCALE_FACTOR;
  accel_y = accel_y_raw / ACCEL_SCALE_FACTOR;
  accel_z = accel_z_raw / ACCEL_SCALE_FACTOR;
  
  // Gyroscope: Convert to degrees per second
  gyro_x = gyro_x_raw / GYRO_SCALE_FACTOR;
  gyro_y = gyro_y_raw / GYRO_SCALE_FACTOR;
  gyro_z = gyro_z_raw / GYRO_SCALE_FACTOR;
}

void writeMPU6050(byte reg, byte data) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission(true);
}