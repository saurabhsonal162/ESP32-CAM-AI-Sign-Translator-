# AI-Based Sign Language Translator (ESP32-CAM)

## Overview
This project implements a real-time sign language translator using an ESP32-CAM module and machine learning. The system captures hand gestures through the camera, processes them using a trained AI model, and converts them into text output, bridging communication barriers for sign language users.

## Features
- **Real-time Translation**: Converts sign language gestures to text with minimal latency.
- **Portable & Affordable**: Built on the low-cost ESP32-CAM platform.
- **Edge Computing**: Performs gesture recognition directly on the ESP32 without requiring constant internet connectivity.
- **User-friendly Interface**: Displays recognized gestures on the connected screen.
- **Expandable Dictionary**: Uses a trained model that can be updated with additional signs.

## Hardware Requirements
- **ESP32-CAM module** (for image capture & AI-based recognition)
- **FTDI USB-to-Serial Adapter** (for programming the ESP32-CAM)
- **Jumper Wires** (for connections)
- **5V Power Supply**

## Software Dependencies
- OpenCV (for image processing)
- TensorFlow Lite (for AI inference on ESP32)
- NumPy (for data handling)
- Python (for model training and testing)

## Installation

### Hardware Setup
1. Connect the ESP32-CAM to your computer using the FTDI USB-to-Serial Adapter.
2. For flashing mode, connect GPIO0 to GND before powering up.
3. Power the ESP32-CAM with a stable 5V supply.

### Software Setup
1. Install Python and required dependencies:
   ```bash
   pip install opencv-python numpy tensorflow
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/esp32-sign-language-translator.git
   ```
3. Upload the ESP32 firmware using Arduino IDE or PlatformIO.
4. Run the Python script to process video feed and perform sign recognition.

## Usage

### Running the Translator
1. Power up the ESP32-CAM and ensure it is connected to the network.
2. Run the Python script to capture and process frames from the ESP32-CAM stream.
3. Perform sign language gestures in front of the camera.
4. View the predicted sign displayed in the console or on a connected display.

## Model Training
The sign language recognition model was trained using:
- **Dataset**: Sign Language MNIST dataset.
- **Model Architecture**: Convolutional Neural Network (CNN) converted to TensorFlow Lite.
- **Training Environment**: TensorFlow with Python.

To retrain the model with custom gestures:
1. Collect gesture images using the ESP32-CAM.
2. Train the model using TensorFlow.
3. Convert the trained model to TensorFlow Lite format.
4. Deploy the new model to the ESP32-CAM.

## Troubleshooting
- **Camera Not Detected**: Check connections and power supply.
- **Low Recognition Accuracy**: Ensure proper lighting and clear hand positioning.
- **Memory Issues**: Reduce image resolution or simplify model architecture.
- **Connectivity Problems**: Reset the module and reconnect to the network.

## Future Improvements
- Support for two-handed signs.
- Battery-powered operation with power optimization.
- Integration with smart home and accessibility technologies.

## Contributing
Contributions are welcome! Feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- TensorFlow Lite for Microcontrollers team.
- The deaf and hard of hearing community for feedback and testing.

