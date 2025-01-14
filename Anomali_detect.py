import cv2
import numpy as np
from keras.models import load_model

# Load the saved trained model
model = load_model('Suspicious_Human_Activity_Detection_LRCN_Model.h5')  # Specify the correct path to your saved model

# Load the video file
video_path = 't_w010_converted.mp4'  # Input video path
cap = cv2.VideoCapture(video_path)

# Get video properties (needed for saving the output video)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
output_video = cv2.VideoWriter('output3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Function to preprocess each frame (modify as per your model's input requirements)
def preprocess_frame(frame):
    # Resize the frame to match the input shape of your model (assuming 224x224 here)
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize the pixel values (if your model requires it)
    normalized_frame = resized_frame / 255.0
    # Add batch dimension (1, 224, 224, 3)
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read each frame
    if not ret:
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Make predictions
    prediction = model.predict(input_frame)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming output is one-hot encoded
    
    # Add detection result on the frame
    if predicted_class == 1:  # Assuming 1 indicates a crime detected
        label = "Crime Detected"
        color = (0, 0, 255)  # Red color for crime detection
    else:
        label = "No Crime"
        color = (0, 255, 0)  # Green color for no crime
    
    # Draw the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Save the frame to the output video
    output_video.write(frame)

# Release the resources
cap.release()
output_video.release()
