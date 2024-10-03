import cv2
import numpy as np
from keras.models import load_model # type: ignore

# Load the pre-trained emotion recognition model
model_path = 'C:\\Users\\Deepika\\OneDrive\\Documents\\Emotion-recognition-master\\models\\_mini_XCEPTION.102-0.66.hdf5'
emotion_model = load_model(model_path, compile=False)  # Update here

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load Haarcascade face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess the face for emotion prediction
        face = gray_frame[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))  # Resize to 64x64 pixels
        face = face.astype('float32') / 255.0  # Normalize to [0, 1]
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=-1)  # Add channel dimension

        # Predict the emotion
        predictions = emotion_model.predict(face)
        max_index = np.argmax(predictions[0])  # Get the index of the highest probability
        predicted_emotion = emotion_labels[max_index]
        probability = predictions[0][max_index]

        # Display the predicted emotion and probability on the frame
        label = f"{predicted_emotion} ({probability:.2f})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the video frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
