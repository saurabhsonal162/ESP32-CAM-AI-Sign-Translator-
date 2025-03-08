import cv2
import numpy as np
import tensorflow as tf
import urllib.request
import sys
import os

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)
print("TensorFlow version:", tf.__version__)

# Check if model file exists
model_path = "sign_language_model.tflite"
if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found!")
    model_files = [f for f in os.listdir(".") if f.endswith(".tflite")]
    if model_files:
        model_path = model_files[0]
        print(f"Using '{model_path}' instead.")
    else:
        print("No .tflite files found in current directory.")
        sys.exit(1)

try:
    # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    print(f"Model loaded successfully. Input shape: {input_shape}")
except Exception as e:
    print(f"ERROR loading model: {str(e)}")
    sys.exit(1)

# Sign Language Mapping
letters = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

def preprocess_image(frame):
    """Preprocess the frame for model input"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # Define Region of Interest (ROI)
        roi_x, roi_y = w // 4, h // 4
        roi_w, roi_h = w // 2, h // 2
        roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Draw ROI on the frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
        
        # Resize and normalize
        resized = cv2.resize(roi, (28, 28))
        normalized = resized / 255.0
        input_data = normalized.reshape(1, 28, 28, 1).astype(np.float32)
        
        return input_data, frame
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None, frame

def predict_sign(input_data):
    """Perform prediction using the TFLite model"""
    if input_data is None:
        return None, 0.0
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = np.argmax(output_data)
        confidence = output_data[0][predicted_class]

        return predicted_class, confidence
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, 0.0

def main():
    """Main function to fetch video stream and process frames"""
    print("Starting Sign Language Translator")
    esp32_ip = input("Enter ESP32-CAM IP address (e.g., 192.168.1.100): ")
    stream_url = f"http://{esp32_ip}/stream"

    try:
        print(f"Connecting to {stream_url}...")
        stream = urllib.request.urlopen(stream_url)
    except Exception as e:
        print(f"Failed to open stream: {e}")
        return

    print("Stream connected! Processing frames...")

    # Buffer to hold incoming stream data
    buffer = b""
    last_predictions = []
    prediction_window = 5
    current_prediction = None

    try:
        while True:
            buffer += stream.read(4096)  # Read bytes from stream
            start = buffer.find(b'\xff\xd8')  # JPEG start marker
            end = buffer.find(b'\xff\xd9')    # JPEG end marker

            if start != -1 and end != -1:
                jpg = buffer[start:end+2]  # Extract JPEG
                buffer = buffer[end+2:]  # Remove processed bytes from buffer

                # Convert byte array to OpenCV image
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    continue
                
                input_data, frame = preprocess_image(frame)

                if input_data is not None:
                    predicted_class, confidence = predict_sign(input_data)

                    if predicted_class is not None:
                        last_predictions.append(predicted_class)
                        if len(last_predictions) > prediction_window:
                            last_predictions.pop(0)

                        # Get most common prediction
                        from collections import Counter
                        prediction_counts = Counter(last_predictions)
                        if prediction_counts:
                            smooth_prediction = prediction_counts.most_common(1)[0][0]
                            if confidence > 0.5:
                                current_prediction = smooth_prediction

                # Display the prediction
                if current_prediction is not None:
                    letter = letters.get(current_prediction, '?')
                    cv2.putText(frame, f"Prediction: {letter} ({confidence:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Show frame
                cv2.imshow("ESP32-CAM Sign Language Translator", frame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuitting application")
                    break

    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    finally:
        print("Cleaning up resources...")
        cv2.destroyAllWindows()
        print("Application terminated.")

if __name__ == "__main__":
    main()
