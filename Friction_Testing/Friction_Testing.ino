/*
   Arduino and MPU6050 Accelerometer and Gyroscope Sensor Tutorial
   by Dejan, https://howtomechatronics.com
*/
#include <Wire.h>
const int MPU = 0x68; // MPU6050 I2C address
//Includes the Arduino Stepper Library
#include <Stepper.h>

// Defines the number of steps per rotation
const int stepsPerRevolution = 2038;

// Creates an instance of stepper class
// Pins entered in sequence IN1-IN3-IN2-IN4 for proper step sequence
Stepper myStepper = Stepper(stepsPerRevolution, 8, 10, 9, 11);

volatile byte state = LOW;
const int cal_but = 3;
const int stop_but = 2;
float AccErrorX_sam, GyroErrorX_sam, zeroing;
float d_gyro = 0.98; // 0.89
float d_acc = 0.02;  // 0.11

float AccX, AccY, AccZ;
float GyroX, GyroY, GyroZ;
float accAngleX, accAngleY, gyroAngleX, gyroAngleY, gyroAngleZ;
float roll, pitch, yaw;
float AccErrorX, AccErrorY, GyroErrorX, GyroErrorY, GyroErrorZ;
float elapsedTime, currentTime, previousTime;
int c = 0;
float angle;

int samples = 5;


void setup() {
  Serial.begin(19200);
  Wire.begin();                      // Initialize comunication
  Wire.beginTransmission(MPU);       // Start communication with MPU6050 // MPU=0x68
  Wire.write(0x6B);                  // Talk to the register 6B
  Wire.write(0x00);                  // Make reset - place a 0 into the 6B register
  Wire.endTransmission(true);        //end the transmission

  pinMode(cal_but, INPUT);
  pinMode(stop_but, INPUT);

  AccErrorX_sam = -1.69;
  GyroErrorX_sam = -1.70;
  zeroing = 40;
}
void loop() { 
  
  //angle = readGyro(AccErrorX_sam, GyroErrorX_sam);
  angle = readGyro(AccErrorX_sam, GyroErrorX_sam) - zeroing;
  Serial.print("Incline: ");
  Serial.println(angle);
  delay(10);

  if (digitalRead(cal_but) == HIGH){
    Serial.print("CALIBRATED");
    zeroing = angle + zeroing;
    AccErrorX_sam = calculate_IMUacc_error();
    GyroErrorX_sam = calculate_IMUgyro_error();
  }
  if (digitalRead(stop_but) == HIGH){
    
    myStepper.setSpeed(1);
    Serial.print("TESTING");
    int count = 0;
    while(digitalRead(cal_but) == LOW){
      myStepper.step(1);
      count = count + 1;
      delay(100);
    }
    delay(10000);

    float angle_f = 0;
    int c_trials = 800;
    for (int i = 0; i < c_trials; i++){
      float val_t = readGyro(AccErrorX_sam, GyroErrorX_sam) - zeroing;
      angle_f = angle_f + val_t;
      delay(20);
    }
    Serial.print("Slide Incline: ");
    Serial.println(angle_f/c_trials);
    Serial.print("Stepper count: ");
    Serial.println(count);
    delay(10000);
    myStepper.step(-count);
  }
  
}


float readGyro(float AccErrorX_sam, float GyroErrorX_sam){

    // === Read acceleromter data === //
  Wire.beginTransmission(MPU);
  Wire.write(0x3B); // Start with register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 6, true); // Read 6 registers total, each axis value is stored in 2 registers
  //For a range of +-2g, we need to divide the raw values by 16384, according to the datasheet
  AccX = (Wire.read() << 8 | Wire.read()) / 16384.0; // X-axis value
  AccY = (Wire.read() << 8 | Wire.read()) / 16384.0; // Y-axis value
  AccZ = (Wire.read() << 8 | Wire.read()) / 16384.0; // Z-axis value

  // Calculating Roll and Pitch from the accelerometer data
  accAngleX = (atan(AccY / sqrt(pow(AccX, 2) + pow(AccZ, 2))) * 180 / PI) + AccErrorX_sam; // AccErrorX ~(0.58) See the calculate_IMU_error()custom function for more details
  accAngleY = (atan(-1 * AccX / sqrt(pow(AccY, 2) + pow(AccZ, 2))) * 180 / PI) - 0.70; // AccErrorY ~(-1.58)
  
  // === Read gyroscope data === //
  previousTime = currentTime;        // Previous time is stored before the actual time read
  currentTime = millis();            // Current time actual time read
  elapsedTime = (currentTime - previousTime) / 1000; // Divide by 1000 to get seconds
  Wire.beginTransmission(MPU);
  Wire.write(0x43); // Gyro data first register address 0x43
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 6, true); // Read 4 registers total, each axis value is stored in 2 registers
  GyroX = (Wire.read() << 8 | Wire.read()) / 131.0; // For a 250deg/s range we have to divide first the raw value by 131.0, according to the datasheet
  //GyroY = (Wire.read() << 8 | Wire.read()) / 131.0;
  //GyroZ = (Wire.read() << 8 | Wire.read()) / 131.0;

  // Correct the outputs with the calculated error values
  GyroX = GyroX + GyroErrorX_sam; // GyroErrorX ~(-0.56)
  //GyroY = GyroY - 2.59; // GyroErrorY ~(2)
  //GyroZ = GyroZ - 2.19; // GyroErrorZ ~ (-0.8)

  // Currently the raw values are in degrees per seconds, deg/s, so we need to multiply by sendonds (s) to get the angle in degrees
  gyroAngleX = gyroAngleX + GyroX * elapsedTime; // deg/s * s = deg
  //gyroAngleY = gyroAngleY + GyroY * elapsedTime;
  //yaw =  yaw + GyroZ * elapsedTime;
  // Complementary filter - combine acceleromter and gyro angle values
  gyroAngleX = d_gyro * gyroAngleX + d_acc * accAngleX;
  //gyroAngleY = 0.92 * gyroAngleY + 0.08 * accAngleY;

  roll = gyroAngleX;

  return roll;
  
}

float calculate_IMUacc_error() {
  c = 0;
  int trials = 500;
  while (c < trials) {
    Wire.beginTransmission(MPU);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU, 6, true);
    AccX = (Wire.read() << 8 | Wire.read()) / 16384.0 ;
    AccY = (Wire.read() << 8 | Wire.read()) / 16384.0 ;
    AccZ = (Wire.read() << 8 | Wire.read()) / 16384.0 ;
    // Sum all readings
    AccErrorX = AccErrorX + ((atan((AccY) / sqrt(pow((AccX), 2) + pow((AccZ), 2))) * 180 / PI));
    AccErrorY = AccErrorY + ((atan(-1 * (AccX) / sqrt(pow((AccY), 2) + pow((AccZ), 2))) * 180 / PI));
    c++;
  }
  //Divide the sum by 200 to get the error value
  AccErrorX = AccErrorX / trials;
  AccErrorY = AccErrorY / trials;

  //AccErrorX_sam = AccErrorX;
  return AccErrorX; 
}

float calculate_IMUgyro_error() {
  c = 0;
  int trials = 500;
  // Read gyro values 200 times
  while (c < trials) {
    Wire.beginTransmission(MPU);
    Wire.write(0x43);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU, 6, true);
    GyroX = Wire.read() << 8 | Wire.read();
    GyroY = Wire.read() << 8 | Wire.read();
    GyroZ = Wire.read() << 8 | Wire.read();
    // Sum all readings
    GyroErrorX = GyroErrorX + (GyroX / 131.0);
    GyroErrorY = GyroErrorY + (GyroY / 131.0);
    GyroErrorZ = GyroErrorZ + (GyroZ / 131.0);
    c++;
  }
  //Divide the sum by 200 to get the error value
  GyroErrorX = GyroErrorX / trials;
  GyroErrorY = GyroErrorY / trials;
  GyroErrorZ = GyroErrorZ / trials;
  // Print the error values on the Serial Monitor
  
  return GyroErrorX;
}
