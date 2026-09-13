#include "Arduino_RouterBridge.h"
#include <Arduino_LED_Matrix.h>

Arduino_LED_Matrix matrix;

// 8x13 pixel frames packed in the format expected by Arduino_LED_Matrix.
// Each row contains 13 pixels. Two frames are used for each animation.
const uint32_t LISTEN_1[4] = {0x00001001, 0xC01F0070, 0x01000000, 0x00000000};
const uint32_t LISTEN_2[4] = {0x00002003, 0xC03F00F0, 0x02000000, 0x00000000};

const uint32_t CAND_1[4] = {0x00008802, 0x800800A0, 0x08800000, 0x00000000};
const uint32_t CAND_2[4] = {0x00000004, 0x40140040, 0x05004400, 0x00000000};

const uint32_t BIRD_1[4] = {0x00001801, 0xE09F8FFF, 0x27E01200, 0xA0000000};
const uint32_t BIRD_2[4] = {0x00002403, 0xF0BFCFFF, 0x1FE02201, 0x20000000};

const uint32_t CROW_1[4] = {0x0000C00F, 0x81FF07FE, 0x1FE02402, 0x40000000};
const uint32_t CROW_2[4] = {0x0000C00F, 0x81FF07FE, 0x1FE01801, 0x20000000};

String matrixState = "idle";
uint32_t nextFrameAt = 0;
uint8_t animationFrame = 0;

void setBirdMatrixState(String state) {
  matrixState = state;
  animationFrame = 0;
  nextFrameAt = 0;

  if (matrixState == "offline") {
    matrix.clear();
  }
}

void setup() {
  Serial.begin(115200);

  matrix.begin();
  matrix.clear();

  Bridge.begin();
  Bridge.provide("set_bird_matrix_state", setBirdMatrixState);

  Serial.println("[MATRIX] Bird animation controller ready.");
}

void showFrame(const uint32_t frame[4]) {
  matrix.loadFrame(frame);
}

void loop() {
  const uint32_t now = millis();

  if (matrixState == "idle") {
    matrix.clear();
    delay(50);
    return;
  }

  // Confirmed bird: energetic wing flap, then return to listening.
  if (matrixState == "bird") {
    if (now >= nextFrameAt) {
      if (animationFrame == 0) {
        showFrame(BIRD_1);
        animationFrame = 1;
        nextFrameAt = now + 180;
      } else if (animationFrame == 1) {
        showFrame(BIRD_2);
        animationFrame = 2;
        nextFrameAt = now + 180;
      } else if (animationFrame == 2) {
        showFrame(BIRD_1);
        animationFrame = 3;
        nextFrameAt = now + 180;
      } else {
        matrixState = "listening";
        animationFrame = 0;
        nextFrameAt = now;
      }
    }
    delay(5);
    return;
  }

  // House Crow: slightly different two-frame silhouette.
  if (matrixState == "crow") {
    if (now >= nextFrameAt) {
      if (animationFrame == 0) {
        showFrame(CROW_1);
        animationFrame = 1;
        nextFrameAt = now + 190;
      } else if (animationFrame == 1) {
        showFrame(CROW_2);
        animationFrame = 2;
        nextFrameAt = now + 190;
      } else if (animationFrame == 2) {
        showFrame(CROW_1);
        animationFrame = 3;
        nextFrameAt = now + 190;
      } else {
        matrixState = "listening";
        animationFrame = 0;
        nextFrameAt = now;
      }
    }
    delay(5);
    return;
  }

  // Weak low-SNR candidate: gentle pulsing/listening pattern.
  if (matrixState == "candidate") {
    if (now >= nextFrameAt) {
      if (animationFrame == 0) {
        showFrame(CAND_1);
        animationFrame = 1;
        nextFrameAt = now + 300;
      } else {
        showFrame(CAND_2);
        animationFrame = 0;
        nextFrameAt = now + 300;
      }
    }
    delay(5);
    return;
  }

  // Default/listening heartbeat.
  if (matrixState == "listening") {
    if (now >= nextFrameAt) {
      if (animationFrame == 0) {
        showFrame(LISTEN_1);
        animationFrame = 1;
        nextFrameAt = now + 500;
      } else {
        showFrame(LISTEN_2);
        animationFrame = 0;
        nextFrameAt = now + 700;
      }
    }
    delay(5);
    return;
  }

  if (matrixState == "offline") {
    matrix.clear();
    delay(100);
    return;
  }

  // Unknown state: fail safely to listening.
  matrixState = "listening";
  animationFrame = 0;
  nextFrameAt = now;
}
