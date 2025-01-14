import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from twilio.rest import Client
import geocoder
from tempfile import NamedTemporaryFile
import os
from datetime import datetime

# Load the pre-trained model
anomaly_model = load_model('Suspicious_Human_Activity_Detection_LRCN_Model.h5')

# Twilio configuration
twilio_account_sid = 'AC4e7eb78b90d176256ebfd74b912314ff'
twilio_auth_token = '7666260ba3947ddf32458015571359b0'
twilio_sender_number = '+16095282776'
twilio_recipient_number = '+917989665270'

twilio_client = Client(twilio_account_sid, twilio_auth_token)

def get_current_location():
    geo_info = geocoder.ip('me')
    coordinates = geo_info.latlng
    if coordinates:
        detailed_location = geocoder.osm(coordinates, method='reverse')
        if detailed_location and detailed_location.address:
            return detailed_location.address
        else:
            return f"Coordinates: {coordinates}. Detailed address unavailable."
    else:
        return "Unable to determine location."

def send_alert_notification():
    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location_info = get_current_location()
    alert_message = (f"ALERT: Suspicious activity detected! Location: {location_info}. "
                     f"Detection time: {detection_time}. Please take immediate action.")
    message = twilio_client.messages.create(
        body=alert_message,
        from_=twilio_sender_number,
        to=twilio_recipient_number
    )
    st.write(f"Alert sent successfully to {twilio_recipient_number}.")

def preprocess_video_frame(frame):
    target_height, target_width = 224, 224
    resized_frame = cv2.resize(frame, (target_width, target_height))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0)

st.markdown(
    """
    <style>
    .title {
        color: #FF5733;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="title">VisionGuard: Detecting Anomalies, Securing Lives</p>', unsafe_allow_html=True)
st.title('VisionGuard: Detecting Anomalies, Securing Lives')

# Initialize session state
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = False

if 'temp_video_path' not in st.session_state:
    st.session_state.temp_video_path = None

if 'is_crime_detected' not in st.session_state:
    st.session_state.is_crime_detected = False

if 'output_video_filename' not in st.session_state:
    st.session_state.output_video_filename = None

if 'video_writer' not in st.session_state:
    st.session_state.video_writer = None

def reset_session_state():
    st.session_state.is_processed = False
    st.session_state.is_crime_detected = False
    if st.session_state.temp_video_path:
        if os.path.exists(st.session_state.temp_video_path):
            os.remove(st.session_state.temp_video_path)
        st.session_state.temp_video_path = None
    if st.session_state.output_video_filename:
        if os.path.exists(st.session_state.output_video_filename):
            os.remove(st.session_state.output_video_filename)
        st.session_state.output_video_filename = None
    if st.session_state.video_writer:
        st.session_state.video_writer.release()
        st.session_state.video_writer = None
    st.write("Session reset and history cleared.")

uploaded_video = st.file_uploader("Upload a video file for analysis", type=['mp4'])

if uploaded_video and not st.session_state.is_processed:
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        st.session_state.temp_video_path = temp_video.name

    video_capture = cv2.VideoCapture(st.session_state.temp_video_path)

    video_width = int(video_capture.get(3))
    video_height = int(video_capture.get(4))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)

    video_codec = cv2.VideoWriter_fourcc(*'XVID')

    display_frame = st.empty()

    crime_frame_counter = 0
    is_alert_sent = False

    while video_capture.isOpened():
        ret, video_frame = video_capture.read()
        if not ret:
            break

        processed_frame = preprocess_video_frame(video_frame)
        predictions = anomaly_model.predict(processed_frame)

        crime_probability = predictions[0][0]
        weapon_probability = predictions[0][1]

        detection_label = 'Normal Activity'
        bounding_box_color = (0, 255, 0)

        if crime_probability > 0.5:
            st.session_state.is_crime_detected = True
            crime_frame_counter += 1

            if crime_frame_counter == 5 and not is_alert_sent:
                send_alert_notification()
                is_alert_sent = True

            if weapon_probability > 0.5:
                detection_label = 'Crime & Weapon Detected'
            else:
                detection_label = 'Crime Detected'
            bounding_box_color = (0, 0, 255)

        cv2.putText(video_frame, detection_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, bounding_box_color, 2, cv2.LINE_AA)

        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        display_frame.image(rgb_frame, channels="RGB", use_column_width=True)

        if st.session_state.video_writer is None:
            output_filename = 'crime_detected.avi' if st.session_state.is_crime_detected else 'normal_detection.avi'
            st.session_state.output_video_filename = output_filename
            st.session_state.video_writer = cv2.VideoWriter(output_filename, video_codec, video_fps, (video_width, video_height))

        st.session_state.video_writer.write(video_frame)

    video_capture.release()
    if st.session_state.video_writer:
        st.session_state.video_writer.release()
    st.write("Video processing completed.")
    st.session_state.is_processed = True

if st.session_state.is_processed:
    if st.button('Clear History'):
        reset_session_state()
