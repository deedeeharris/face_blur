import streamlit as st
import cv2
import tempfile
import os
from deepface import DeepFace

# Function to blur detected faces in a video using DeepFace
def blur_faces_in_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'WMV2')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Detect faces using DeepFace with RetinaFace as the backend
            face_objs = DeepFace.extract_faces(frame, detector_backend='retinaface')
            
            for face_obj in face_objs:
                face = face_obj['face']
                facial_area = face_obj['facial_area']

                # Extract coordinates
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Apply Gaussian blur to the detected face
                blurred_face = cv2.GaussianBlur(face, (99, 99), 30)

                # Replace the original face area with the blurred face
                frame[y:y+h, x:x+w] = blurred_face

        except Exception as e:
            # No faces detected in this frame
            pass

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
