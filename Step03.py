"""Treinar o Modelo de Reconhecimento Facial"""

import os
import numpy as np
import cv2
import face_recognition

def create_face_encodings(data_dir):
    encodings = []
    labels = []

    for label in os.listdir(data_dir):
        player_dir = os.path.join(data_dir, label)
        if os.path.isdir(player_dir):  # Verifica se é um diretório
            for img_path in os.listdir(player_dir):
                img_full_path = os.path.join(player_dir, img_path)
                if os.path.isfile(img_full_path):  # Verifica se é um arquivo
                    image = face_recognition.load_image_file(img_full_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        encodings.append(face_encodings[0])
                        labels.append(label)
    return np.array(encodings), np.array(labels)

# Diretórios com os dados de treino e teste
train_data_dir = 'train_data'
test_data_dir = 'test_data'

# Criar as codificações de faces para os dados de treino
train_encodings, train_labels = create_face_encodings(train_data_dir)

# Salvar os encodings e labels para uso posterior
np.save('train_encodings.npy', train_encodings)
np.save('train_labels.npy', train_labels)

