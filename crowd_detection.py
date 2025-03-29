import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the emotion detection model (Use any pre-trained model for FER2013)
model = load_model("emotion_model.h5")  # Replace with your model file

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion labels for the model
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Open the camera (0 for webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Extract face region
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48))  # FER model input size
        face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
        face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension
        face_resized = face_resized / 255.0  # Normalize pixel values

        # Predict emotion
        emotion_idx = np.argmax(model.predict(face_resized))
        emotion = emotion_labels[emotion_idx]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted emotion
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display people count
    cv2.putText(frame, f"People Count: {len(faces)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video feed
    cv2.imshow("Crowd and Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
