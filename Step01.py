""" Redimensionar e Normalizar Imagens """

import cv2
import os

def preprocess_images(input_dir, output_dir, size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if os.path.isfile(img_path):  # Certifica-se de que é um arquivo
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, size)
                normalized_img = resized_img / 255.0  # Normalização
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, normalized_img * 255)

base_dir = 'Assets'
processed_dir = 'Assets/processed_data'

for jogador in os.listdir(base_dir):
    input_dir = os.path.join(base_dir, jogador)
    output_dir = os.path.join(processed_dir, jogador)
    if os.path.isdir(input_dir) and not jogador.startswith('.'):  
        print(jogador)
        preprocess_images(input_dir, output_dir)

print("Process Done.")
