# VisionGuard: Detecting Anomalies, Securing Lives

## Overview
VisionGuard is a real-time anomaly detection system designed to ensure public safety by analyzing CCTV footage for suspicious human activities. Utilizing state-of-the-art deep learning models, VisionGuard identifies potential crimes and weapons, alerts authorities, and provides detailed insights about the location and time of detection.

## Features
- **Real-Time Anomaly Detection**: Detect suspicious human activities and potential weapon use.
- **Automated Alerts**: Sends instant notifications with detailed location and time of detection via Twilio.
- **Video Processing**: Processes uploaded CCTV footage and provides annotated output.
- **Geolocation Support**: Automatically retrieves the current location of detection using geocoder.
- **User-Friendly Interface**: Intuitive Streamlit-based application for easy interaction.

## Tech Stack
- **Backend**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Frontend**: Streamlit
- **Libraries**: OpenCV, NumPy, Geocoder, Twilio
- **Notification Service**: Twilio API

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Upload a CCTV video file for analysis.
3. View the real-time detection results and receive notifications if suspicious activities are detected.
4. Processed videos with annotations will be saved locally.

## How It Works
1. **Video Preprocessing**: Uploaded video frames are resized and normalized for model input.
2. **Anomaly Detection**: The deep learning model predicts the likelihood of crimes and weapon detection for each frame.
3. **Notification System**: If suspicious activity is detected, an alert is sent to the configured recipient using Twilio.
4. **Annotated Output**: Processed videos are saved with detection labels and bounding boxes for visualization.

## Configuration
- Update the Twilio credentials in the script:
  ```python
  twilio_account_sid = 'your_account_sid'
  twilio_auth_token = 'your_auth_token'
  twilio_sender_number = 'your_twilio_phone_number'
  twilio_recipient_number = 'recipient_phone_number'
  ```
- Adjust model thresholds (default: 0.5) for detection probabilities in the script as needed.

## Future Enhancements
- **Multi-Camera Support**: Extend the system to handle inputs from multiple CCTV cameras, enabling broader coverage in real-time surveillance.
- **Real-Time Video Streaming**: Implement real-time video streaming for live monitoring, with continuous anomaly detection on the fly.
- **Enhanced Crime Classification**: Train the model to recognize and classify different types of crimes (e.g., theft, assault) to provide more detailed alerts.
- **Integration with Security Systems**: Allow integration with existing security infrastructure, such as automated alarms, sirens, or police notifications, upon detection of a crime.
- **User Dashboard**: Create a user dashboard to display past incidents, real-time data, and video analysis history, allowing authorities to track and manage alerts more efficiently.
- **Cloud Storage and Analysis**: Enable cloud-based storage of video data and historical alerts, allowing for easier access and analysis across different devices and locations.
- **Object Tracking**: Implement object tracking for more accurate identification of individuals and activities over time.
- **Voice Alerts**: Incorporate voice alerts along with text notifications for situations where immediate human intervention is necessary.

### Co-Owners
- [Shruthika Sunku](https://github.com/Shruthika-s)
- [Suhas Uppala](https://github.com/Suhas-Uppala)

## Acknowledgments
- TensorFlow and Keras for providing robust deep learning tools.
- Twilio for the notification service.
- OpenCV for video processing.
