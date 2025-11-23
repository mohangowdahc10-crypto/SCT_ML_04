import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide INFO messages
import tensorflow as tf
print('TF version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# --- Configuration ---
MODEL_PATH = 'gesture_model.h5'
IMAGE_SIZE = (64, 64)
# Define your actual gesture classes here
CLASS_NAMES = ['open hand','fist','peace sign','thumbs up','index finger'] 
NUM_CLASSES = len(CLASS_NAMES)

def create_and_save_dummy_model():
    """
    Creates a simple CNN structure and saves it as a placeholder.
    """

    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}. Skipping dummy creation.")
        return

    print("--- Creating DUMMY Model Structure ---")
    model = Sequential([
        # Input Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(128, activation='relu'),
        # Output Layer
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    
    # Compile with dummy settings (not for real training)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save(MODEL_PATH)
    print(f"DUMMY Model saved to {MODEL_PATH}. Please train a real model later!")

def load_gesture_model():
    """Loads the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run the dummy model creation first.")
        return None
    try:
        model = load_model(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def preprocess_roi(roi_img):
    """Resizes and normalizes the Region of Interest (ROI) image for prediction."""
    # Resize to the target size (e.g., 64x64)
    img_resized = cv2.resize(roi_img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    # Convert to float and normalize (0-1)
    img_normalized = img_resized.astype('float32') / 255.0
    # Add batch dimension (1, 64, 64, 3)
    img_preprocessed = np.expand_dims(img_normalized, axis=0)
    return img_preprocessed

def real_time_recognition():
    """Initializes webcam and runs the real-time prediction loop."""
    
    # Ensure a model file exists
    create_and_save_dummy_model()
    model = load_gesture_model()
    if model is None:
        return

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define the Region of Interest (ROI) area coordinates (fixed in the center)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define a 200x200 square ROI in the center of the screen
    roi_w, roi_h = 200, 200
    x1 = (W // 2) - (roi_w // 2)
    y1 = (H // 2) - (roi_h // 2)
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    
    print("\nStarting Real-Time Recognition. Place your hand in the green box.")
    print("Press 'q' to quit.")

    while True:
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more natural mirror effect
        frame = cv2.flip(frame, 1)

        # 1. Extract the ROI (Region of Interest)
        roi = frame[y1:y2, x1:x2]

        # 2. Preprocess the ROI for the model
        processed_roi = preprocess_roi(roi)

        # 3. Predict the gesture
        predictions = model.predict(processed_roi, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Determine the text to display
        predicted_gesture = CLASS_NAMES[predicted_index]
        prediction_text = f"Gesture: {predicted_gesture}"
        confidence_text = f"Conf: {confidence:.2f}%"

        # 4. Display the ROI box and Prediction Text
        
        # Draw the ROI box (Green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display Prediction Text above the ROI
        cv2.putText(frame, prediction_text, (x1, y1 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the final frame
        cv2.imshow('Hand Gesture Recognizer (Press Q to quit)', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If you have a fully trained model, comment out the line below.
    # Otherwise, run it once to create the placeholder model file.
    create_and_save_dummy_model()
    
    real_time_recognition()