#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <SoftwareSerial.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ----------- WiFi -----------
const char* ssid = "VADAKARAYIL HOUSE 2.4";
const char* password = "suni@4736";
const char* serverName = "http://192.168.1.33:5000/sensor";

// ----------- NPK -----------
#define DE D5
#define RX D6
#define TX D7

SoftwareSerial mod(RX,TX);

byte values[11];

byte nitrogen[]   = {0x01,0x03,0x00,0x1E,0x00,0x01,0xE4,0x0C};
byte phosphorous[]= {0x01,0x03,0x00,0x1F,0x00,0x01,0xB5,0xCC};
byte potassium[]  = {0x01,0x03,0x00,0x20,0x00,0x01,0x85,0xC0};

// ----------- Temperature -----------
#define ONE_WIRE_BUS D2
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// ----------- pH -----------
#define PH_PIN A0
float neutralVoltage = 1.34;
float slope = 0.18;

// ----------- Moisture -----------
#define MOISTURE_PIN D1

void setup() {

  Serial.begin(9600);

  // WiFi connect once at startup
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  Serial.print("ESP IP: ");
  Serial.println(WiFi.localIP());

  // NPK
  mod.begin(4800);
  pinMode(DE,OUTPUT);
  digitalWrite(DE,LOW);

  // Temperature
  sensors.begin();

  // Moisture
  pinMode(MOISTURE_PIN, INPUT);

  Serial.println("Soil Monitoring System Started");
}

int readSensor(byte *cmd){

  while(mod.available()) mod.read();

  digitalWrite(DE,HIGH);
  delay(20);
  mod.write(cmd,8);
  mod.flush();
  delay(20);
  digitalWrite(DE,LOW);

  delay(500);

  int i=0;
  while(mod.available() && i<7){
    values[i++] = mod.read();
  }

  if(i>=7){
    return (values[3]<<8 | values[4]) / 256;
  }

  return -1;
}

void loop(){

  // -------- Temperature FIRST --------
  sensors.requestTemperatures();
  delay(750);
  float temp = sensors.getTempCByIndex(0);

  if(temp == -127){
    Serial.println("Temp Read Failed");
  }

  // -------- NPK --------
  int n = readSensor(nitrogen);
  delay(500);
  int p = readSensor(phosphorous);
  delay(500);
  int k = readSensor(potassium);

  // -------- pH --------
  int rawValue = analogRead(PH_PIN);
  float voltage = rawValue * (3.3 / 1023.0);
  float phValue = 7 + ((neutralVoltage - voltage) / slope);

  // -------- Moisture --------
  int moistureState = digitalRead(MOISTURE_PIN);
  float moisturePercent;

  if(moistureState == 0) {
      if(temp > 30)
          moisturePercent = 60;
      else
          moisturePercent = 80;
  }
  else {
      if(temp > 30)
          moisturePercent = 15;
      else
          moisturePercent = 30;
  }

  // -------- Send Data to Server --------
  if (WiFi.status() == WL_CONNECTED) {

    WiFiClient client;
    HTTPClient http;

    http.begin(client, serverName);
    http.addHeader("Content-Type", "application/json");

    String jsonData = "{";
    jsonData += "\"N\":" + String(n) + ",";
    jsonData += "\"P\":" + String(p) + ",";
    jsonData += "\"K\":" + String(k) + ",";
    jsonData += "\"temp\":" + String(temp) + ",";
    jsonData += "\"humidity\":" + String(moisturePercent) + ",";
    jsonData += "\"ph\":" + String(phValue);
    jsonData += "}";

    int httpResponseCode = http.POST(jsonData);

    Serial.print("HTTP Response Code: ");
    Serial.println(httpResponseCode);

    http.end();
  }

  // -------- Serial Output --------
  Serial.println("------ SOIL DATA ------");
  Serial.print("Nitrogen: "); Serial.println(n);
  Serial.print("Phosphorus: "); Serial.println(p);
  Serial.print("Potassium: "); Serial.println(k);
  Serial.print("Temperature: "); Serial.println(temp);
  Serial.print("pH: "); Serial.println(phValue);
  Serial.print("Moisture: "); Serial.println(moisturePercent);
  Serial.println("-----------------------\n");

  delay(5000);
}