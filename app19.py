import streamlit as st
import cv2
import numpy as np
import face_recognition
from PIL import Image
import os
import random
import pandas as pd
import base64
import pygame

# Inicializar o mixer do pygame
pygame.mixer.init()

# Carregar o som
pygame.mixer.music.load('./Assets/boop.mp3')

# Carregar os encodings e labels
train_encodings = np.load('train_encodings.npy')
train_labels = np.load('train_labels.npy')

# Carregar estatísticas dos jogadores
stats_df = pd.read_excel('stats.xlsx')

# Selecionar as colunas desejadas (ajuste conforme necessário)
selected_columns = ['Player', 'Pos', 'Age', '90s', 'Gls', 'Ast']
stats_df = stats_df[selected_columns]

# Definir CSS para a imagem de fundo com transparência
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.3)), url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .centered {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }}
    .centered1 {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50%;
    }}
    .hidden {{
        display: none;
    }}
    .fade {{
        animation: fadein 2s;
    }}
    @keyframes fadein {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    .right-align {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Carregar a imagem de fundo
background_image_path = './Assets/wallpaper.png'
set_background(background_image_path)

# Função para carregar uma imagem aleatória de um jogador
def load_random_player_image(player_name):
    player_dir = os.path.join('player_images', player_name)
    if not os.path.isdir(player_dir):
        return None
    # Filtrar apenas arquivos de imagem
    image_files = [f for f in os.listdir(player_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not image_files:
        return None
    random_image_path = os.path.join(player_dir, random.choice(image_files))
    return Image.open(random_image_path).convert("RGB")  # Garantir que a imagem esteja em RGB

def get_player_stats(player_name):
    player_stats = stats_df[stats_df['Player'] == player_name]
    if player_stats.empty:
        return None
    return player_stats.iloc[0].to_dict()

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
            stats.append("")

        faces.append((left, top, right - left, bottom - top))

    return faces, stats

# Função para sobrepor a imagem do jogador e estatísticas
def overlay_player_info(frame, player_image, player_stats, position):
    if player_image is None:
        return frame

    # Redimensionar a imagem do jogador
    player_image = player_image.resize((100, 100))
    player_image_np = np.array(player_image)

    x, y = position
    y1, y2 = y, y + player_image_np.shape[0]
    x1, x2 = x, x + player_image_np.shape[1]

    # Verificar se a sobreposição está dentro dos limites do frame
    if x2 > frame.shape[1]:
        x2 = frame.shape[1]
    if y2 > frame.shape[0]:
        y2 = frame.shape[0]

    if x1 < 0 or y1 < 0:
        return frame

    # Ajustar as dimensões da imagem do jogador se necessário
    player_image_np = player_image_np[:y2-y1, :x2-x1]

    # Converter a imagem do jogador de RGB para BGR
    player_image_np = cv2.cvtColor(player_image_np, cv2.COLOR_RGB2BGR)

    frame[y1:y2, x1:x2] = player_image_np

    # Exibir estatísticas do jogador ao lado da imagem
    if player_stats:
        stats_text = f"Pos: {player_stats['Pos']}\nAge: {player_stats['Age']}\n90s: {player_stats['90s']}\nGls: {player_stats['Gls']}\nAst: {player_stats['Ast']}"
        for i, line in enumerate(stats_text.split('\n')):
            y_offset = y2 + 20 + (i * 20)
            cv2.putText(frame, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

# Função para processar a imagem
def process_image(image):
    faces, stats = detect_and_recognize_faces(image, train_encodings, train_labels)
    player_detected = False

    for (x, y, w, h), stat in zip(faces, stats):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, stat, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if stat != "":
            player_detected = True
            player_image = load_random_player_image(stat)
            player_stats = get_player_stats(stat)
            if player_image and player_stats:
                image = overlay_player_info(image, player_image, player_stats, (x+w+10, y))

    if player_detected:
        pygame.mixer.music.play()

    return image

st.title("🇵🇹 Deteção e Reconhecimento de Faces de Jogadores da Nossa Seleção 🇨🇿")

# Estado para controlar se a câmera está ativa
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# Espaço reservado para o conteúdo de boas-vindas e botão
placeholder = st.empty()

if not st.session_state.run_camera:
    
    with placeholder.container():
        st.image("./Assets/closer.png", use_column_width=True)
        st.write("<div style='text-align: center;'>⚽ Bem-vindo ao Evento Closer: Unleashing the Power of Data & AI ⚽</div>", unsafe_allow_html=True)
        if st.button("Iniciar Câmera", key="start_button"):
            st.session_state.run_camera = True
            placeholder.empty()

# Iniciar a câmera se o botão for clicado
if st.session_state.run_camera:
    cap = cv2.VideoCapture(1)
    stframe = st.empty()
    no_faces_placeholder = st.empty()
    
    # Definir resolução da câmera (opcional, ajuste conforme necessário)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Falha ao acessar a câmera")
            break
        # Processar a imagem
        processed_frame = process_image(frame)
        # Mostrar a imagem
        stframe.image(processed_frame, channels="BGR", use_column_width=True)
        
        # Feedback do usuário
        faces, _ = detect_and_recognize_faces(frame, train_encodings, train_labels)
        if len(faces) == 0:
            no_faces_placeholder.write("Nenhuma face detectada")
        else:
            no_faces_placeholder.empty()
        
        # Limitar o tempo de espera
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
