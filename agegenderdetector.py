import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Hide Streamlit's main menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Function to detect face boxes using the pre-trained network
def get_face_box(net, frame, conf_threshold=0.7):
    opencv_dnn_frame = frame.copy()
    frame_height = opencv_dnn_frame.shape[0]
    frame_width = opencv_dnn_frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob_img)
    detections = net.forward()
    b_boxes_detect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return opencv_dnn_frame, b_boxes_detect

# Title and introduction for the Streamlit app
st.write("# Age & Gender Detection System")
st.write("#### By Shreya Shetty")
st.write("## Upload a Picture to check Age and Gender")

# Upload image
uploaded_file = st.file_uploader("Choose a file for Detection: ")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cap = np.array(image)

    # Check the number of channels in the image
    if cap.ndim == 2:  # Grayscale image (2 dimensions)
        cap = cv2.cvtColor(cap, cv2.COLOR_GRAY2BGR)
    elif cap.shape[2] == 4:  # RGBA image (4 channels)
        cap = cv2.cvtColor(cap, cv2.COLOR_RGBA2RGB)
    
    # Proceed with further processing
    face_txt_path = "opencv_face_detector.pbtxt"
    face_model_path = "opencv_face_detector_uint8.pb"

    age_txt_path = "age_deploy.prototxt"
    age_model_path = "age_net.caffemodel"

    gender_txt_path = "gender_deploy.prototxt"
    gender_model_path = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_classes = ['~1-2', '~3-5', '~6-14', '~16-22', '~25-30', '~32-40', '~45-50', 'Above 60']
    gender_classes = ['Male', 'Female']

    age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
    gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
    face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

    padding = 20
    t = time.time()
    frameFace, b_boxes = get_face_box(face_net, cap)
    
    if not b_boxes:
        st.write("No face detected, checking next frame.")
    
    for bbox in b_boxes:
        face = cap[max(0, bbox[1] - padding): min(bbox[3] + padding, cap.shape[0] - 1),
                   max(0, bbox[0] - padding): min(bbox[2] + padding, cap.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender prediction
        gender_net.setInput(blob)
        gender_pred_list = gender_net.forward()
        gender = gender_classes[gender_pred_list[0].argmax()]
        genderPercent = gender_pred_list[0].max() * 100
        st.write("Gender: ", gender, "      Confidence: ", "{:.2f}".format(genderPercent), "%")

        # Age prediction
        age_net.setInput(blob)
        age_pred_list = age_net.forward()
        age = age_classes[age_pred_list[0].argmax()]
        agePercent = age_pred_list[0].max() * 100
        st.write("Age: ", age, "      Confidence: ", "{:.2f}".format(agePercent), "%")

        label = "{},{}".format(gender, age)
        cv2.putText(
            frameFace,
            label,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA)

        # Display the image with bounding boxes and labels
        st.image(frameFace)
