"""Dividir os Dados em Conjuntos de Treinamento e Teste:"""

import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(base_dir, train_dir, test_dir, test_size=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for jogador in os.listdir(base_dir):
        jogador_dir = os.path.join(base_dir, jogador)
        images = os.listdir(jogador_dir)
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

        os.makedirs(os.path.join(train_dir, jogador), exist_ok=True)
        os.makedirs(os.path.join(test_dir, jogador), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(jogador_dir, img), os.path.join(train_dir, jogador, img))

        for img in test_images:
            shutil.copy(os.path.join(jogador_dir, img), os.path.join(test_dir, jogador, img))

processed_dir = 'Assets/processed_data'
train_dir = 'train_data'
test_dir = 'test_data'

split_data(processed_dir, train_dir, test_dir)

print("Process Done.")