
import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os

# Function to blur detected faces in a video
def blur_faces_in_video(input_video_path, output_video_path):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'WMV2')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    def blur_face(frame, x1, y1, x2, y2):
        face_roi = frame[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)
        frame[y1:y2, x1:x2] = blurred_face
        return frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                x2 = x1 + int(bboxC.width * iw)
                y2 = y1 + int(bboxC.height * ih)
                frame = blur_face(frame, x1, y1, x2, y2)

        out.write(frame)

    cap.release()
    out.release()

# Streamlit app logic
def main():
    st.title("Blur Faces in Video")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_file:
            temp_input_file.write(uploaded_file.read())
            input_video_path = temp_input_file.name
        
        output_video_path = input_video_path.replace('.mp4', '_blurred.wmv')

        with st.spinner("Processing..."):
            blur_faces_in_video(input_video_path, output_video_path)

        st.success("Processing complete!")

        with open(output_video_path, 'rb') as output_video_file:
            video_bytes = output_video_file.read()
            st.download_button(label="Download blurred video", data=video_bytes, file_name="blurred_faces_output.wmv", mime='video/wmv')

        os.remove(input_video_path)
        os.remove(output_video_path)

if __name__ == "__main__":
    main()
