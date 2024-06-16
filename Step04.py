"""Função para Detecção e Reconhecimento de Faces"""

import cv2
import face_recognition
import os
import numpy as np

train_encodings = np.load('train_encodings.npy')
train_labels = np.load('train_labels.npy')

def detect_and_recognize_faces(image, train_encodings, train_labels):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    faces = []
    stats = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(train_encodings, face_encoding)
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]

        # Definir um limiar para reconhecimento
        if min_distance < 0.6:  # Este limiar pode ser ajustado
            name = train_labels[min_distance_index]
            stats.append(name)
        else:
            stats.append("Unknown")

        faces.append((left, top, right - left, bottom - top))

    return faces, stats

# Exemplo de uso da função
image_path = '/Users/andre/Desktop/App_Closer/train_data/Pepe/Pepe14.png'
image = cv2.imread(image_path)
faces, stats = detect_and_recognize_faces(image, train_encodings, train_labels)

for (x, y, w, h), stat in zip(faces, stats):
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, stat, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

cv2.imshow("Recognized Faces", image)
cv2.waitKey(3000)  # Espera 3 segundos (3000 milissegundos)
cv2.destroyAllWindows()

