/* FSR Testing
adapted from www.ladyada.net/learn/sensors/fsr.html */
int fsrPin = A0;               // analog pin A0
int fsrReading;               // analog input from FSR
int fsrVoltage;               // analog input converted to voltage
unsigned long fsrResistance;  // voltage converted to resistance
unsigned long fsrConductance;
void setup(void) {
  Serial.begin(9600);
}
void loop(void) {
  fsrReading = analogRead(fsrPin);
  // print analog input
  //Serial.print("Analog reading = ");
  //Serial.println(fsrReading);
  // map analog input to voltage [mV]
  fsrVoltage = map(fsrReading, 0, 1023, 0, 5000);
  // print voltage
  //Serial.print("Voltage reading in mV = ");
  //Serial.println(fsrVoltage);
  // if no input from FSR, print &quot;no pressure&quot;
  if (fsrVoltage == 0) {
    //Serial.println("No pressure & ");
    // from circuit diagram, calculations for analog input voltage (V0):
    // V0 = Vcc * R1 / (R1 + FSR) where R1 = 10K and Vcc = 5V
    // therefore FSR = ((Vcc - V0) * R1) / V0
  } else {
    fsrResistance = 5000 - fsrVoltage;  // (Vcc - V0)
    fsrResistance *= 10000;             // *R1
    fsrResistance /= fsrVoltage;        // / V0
    // print resistance
    //Serial.print("FSR resistance in ohms = ");
    //Serial.println(fsrResistance);


    // conductance in microMhos
    fsrConductance = 1000000;
    fsrConductance /= fsrResistance;
    // print conductance
    //Serial.print("Conductance in microMhos: ");
    //Serial.println(fsrConductance);

    // Force
    float fsr_force;
    fsr_force = 10.2*exp(-3.294*0.00001*fsrVoltage) + 0.2102*exp(1.591*0.001*fsrVoltage);
    // print conductance
    Serial.print("Force in gf ");
    Serial.println(fsr_force);
  }
  Serial.println("-- -- -- -- -- -- -- -- -- --");
  delay(200);
}
