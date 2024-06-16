#Interface Usuario

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Detecção de Jogadores de Futebol")
run = st.checkbox('Abrir Câmera')

if run:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        st.image(frame, channels="BGR")
    cap.release()

# Função para detectar e reconhecer faces
def detect_and_recognize_faces(image, model):
    # Aqui vai o código para detectar e reconhecer as faces
    pass

if run:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        faces, stats = detect_and_recognize_faces(frame, model)
        for (x, y, w, h), stat in zip(faces, stats):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, stat, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        st.image(frame, channels="BGR")
    cap.release()



def detect_and_recognize_faces(image, model):
    # Detectar faces
    faces = face_detector(image)
    stats = []
    for face in faces:
        # Reconhecer face
        player_id = recognize_face(face, model)
        if player_id is not None:
            stats.append(get_player_stats(player_id))
        else:
            stats.append("Unknown")
    return faces, stats


