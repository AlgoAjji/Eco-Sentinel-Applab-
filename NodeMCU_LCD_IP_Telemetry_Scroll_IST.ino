#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <NTPClient.h>
#include <WiFiUdp.h>

// ============================================================================
// NODEMCU WIFI
// ============================================================================
const char* ssid = "Your_wifi";
const char* password = "password";

// ============================================================================
// LCD / SERVER
// ============================================================================
LiquidCrystal_I2C lcd(0x27, 16, 2);
ESP8266WebServer server(80);

WiFiUDP ntpUDP;

// India NTP pool is preferred. Fallbacks keep the clock working if a pool
// hostname is temporarily unavailable.
NTPClient timeClient(
  ntpUDP,
  "in.pool.ntp.org",
  19800,       // IST = UTC+05:30
  21600000UL   // resync every 6 hours
);

// ============================================================================
// STATE
// ============================================================================
unsigned long lastMsgTime = 0;
bool isUnoOnline = false;

String lastSpecies = "Waiting for UNO Q";
int lastBattery = 0;
float lastConfidence = 0.0;

// Scrolling state for the full bird name.
String scrollText = "";
int scrollPos = 0;
unsigned long lastScrollTime = 0;
const unsigned long SCROLL_INTERVAL_MS = 400;
bool scrolling = false;

// ============================================================================
// HELPERS
// ============================================================================

String currentISTTime() {
  timeClient.update();
  String t = timeClient.getFormattedTime();  // HH:MM:SS
  return t.substring(0, 5);                 // HH:MM
}

void startSpeciesScroll(const String& species) {
  lastSpecies = species;

  // Pad with spaces so the text leaves the display cleanly, then return.
  if (species.length() <= 10) {
    scrollText = species;
    scrolling = false;
    scrollPos = 0;
    return;
  }

  scrollText = "     " + species + "     ";
  scrollPos = 0;
  scrolling = true;
  lastScrollTime = millis();
}

void renderLCD() {
  String timeNow = currentISTTime();

  // -------- Row 0: bird name / scrolling bird name --------
  String row0;

  if (scrolling) {
    if (scrollPos + 16 <= (int)scrollText.length()) {
      row0 = scrollText.substring(scrollPos, scrollPos + 16);
    } else {
      row0 = scrollText.substring(scrollPos);
      while (row0.length() < 16) row0 += " ";
    }
  } else {
    row0 = scrollText;
    if (row0.length() > 10) row0 = row0.substring(0, 10);

    while (row0.length() < 10) row0 += " ";
    row0 += " ";
    row0 += timeNow;
  }

  while (row0.length() < 16) row0 += " ";
  if (row0.length() > 16) row0 = row0.substring(0, 16);

  // -------- Row 1: battery + confidence/status --------
  String row1;

  if (lastBattery >= 0) {
    row1 = "Bat:" + String(lastBattery) + "%";
  } else {
    row1 = "Bat:N/A";
  }

  if (lastConfidence > 0.0) {
    row1 += " C:" + String(lastConfidence, 1) + "%";
  } else {
    row1 += " " + timeNow;
  }

  while (row1.length() < 16) row1 += " ";
  if (row1.length() > 16) row1 = row1.substring(0, 16);

  lcd.setCursor(0, 0);
  lcd.print(row0);
  lcd.setCursor(0, 1);
  lcd.print(row1);
}

void handleTelemetry() {
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "Missing Data");
    return;
  }

  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, server.arg("plain"));

  if (err) {
    server.send(400, "text/plain", "Invalid JSON");
    return;
  }

  String receivedSpecies = doc["species"] | "Unknown";

  // The Python side sends confidence as percent.
  lastConfidence = doc["confidence"] | 0.0;

  if (doc["battery"].is<int>()) {
    lastBattery = doc["battery"].as<int>();
  }

  isUnoOnline = true;
  lastMsgTime = millis();

  startSpeciesScroll(receivedSpecies);
  renderLCD();

  server.send(200, "text/plain", "OK");
}

// ============================================================================
// WIFI + NTP
// ============================================================================
void connectWiFi() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi");

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("[Wi-Fi Connected]");
  Serial.print("NodeMCU IP: ");
  Serial.println(WiFi.localIP());

  timeClient.begin();

  // Try to get Indian network time immediately.
  bool timeOK = false;
  for (int i = 0; i < 5; i++) {
    if (timeClient.update()) {
      timeOK = true;
      break;
    }
    timeClient.forceUpdate();
    delay(250);
  }

  if (timeOK) {
    Serial.print("[TIME] India NTP synchronized: ");
    Serial.println(timeClient.getFormattedTime());
  } else {
    Serial.println("[TIME] Initial NTP update failed; continuing with client clock.");
  }
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("[LoLin NodeMCU] Booting wildlife telemetry node...");

  Wire.begin();

  lcd.init();
  lcd.backlight();

  connectWiFi();

  isUnoOnline = false;
  lastMsgTime = millis() - 30001;

  startSpeciesScroll("Waiting for UNO Q");
  renderLCD();

  server.on("/telemetry", HTTP_POST, handleTelemetry);
  server.begin();

  Serial.println("[SERVER] HTTP telemetry endpoint active: /telemetry");
}

// ============================================================================
// LOOP
// ============================================================================
void loop() {
  server.handleClient();

  // Keep time synchronized to Indian NTP pool.
  timeClient.update();

  // Scroll full species name across the first LCD line.
  if (scrolling && millis() - lastScrollTime >= SCROLL_INTERVAL_MS) {
    lastScrollTime = millis();

    scrollPos++;

    // End of message: pause briefly at the end, then restart.
    if (scrollPos > (int)scrollText.length() - 16) {
      scrollPos = 0;
    }

    renderLCD();
  }

  // UNO Q is considered offline only after 30 seconds without telemetry.
  if (isUnoOnline && (millis() - lastMsgTime > 30000UL)) {
    isUnoOnline = false;

    scrolling = false;
    scrollText = "[UNO Q OFFLINE]";
    scrollPos = 0;

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("[UNO Q OFFLINE]");
    lcd.setCursor(0, 1);
    lcd.print("IST " + currentISTTime());

    Serial.println("[ALERT] UNO Q telemetry timeout.");
  }
}
