# Hand Gesture Recognition Project

## Overview
This project utilizes a hand gesture recognition system to interact with your computer through hand movements. The system leverages computer vision techniques with the help of MediaPipe and PyAutoGUI to detect gestures and perform actions like switching applications or closing tabs.

## Features
- Detect hand movements and gestures using the webcam.
- Recognize specific gestures for controlling computer actions (e.g., switching applications).
- Smooth and consistent gesture detection to minimize false positives.
- Incremental switching between open applications using hand waves.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)

## Installation
To get started with the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd hand-gesture-recognition
    ```

2. Install required Python packages:
    ```bash
    pip install mediapipe opencv-python pyautogui numpy
    ```

## Usage
### Running the Gesture Detection Scripts
1. **Wave Gesture Detection**:
   Use the following command to start the application:
   ```bash
   python wave_gesture_detection.py
