import cv2
import numpy as np
import tensorflow as tf
import os
import random
import pygame
import tkinter as tk
from tkinter import Button, Canvas
from PIL import Image, ImageTk
import time

# Paths
model_path = "Models/retrained_graph.pb"  # Trained model path
labels_path = "Models/retrained_labels.txt"  # Emotion labels path
songs_dir = "Data/Songs/"  # Directory with categorized music
background_image_path = "background.jpg"  # Background image for GUI

# Load emotion labels
with open(labels_path, "r") as f:
    emotion_labels = [line.strip() for line in f.readlines()]

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Initialize music player
pygame.mixer.init()
current_song = None
song_list = []
current_song_index = -1
current_emotion = None


# Function to preprocess the input image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (48, 48))  # Resize to match model input
    normalized = resized / 255.0  # Normalize pixel values
    reshaped = np.reshape(normalized, (1, 48, 48, 1))  # Reshape for model
    return reshaped


# Function to load song list based on emotion
def load_song_list(emotion):
    global song_list, current_song_index, current_song
    emotion_folder = os.path.join(songs_dir, emotion)
    if os.path.exists(emotion_folder) and os.listdir(emotion_folder):
        song_list = os.listdir(emotion_folder)
        song_list = [os.path.join(emotion_folder, song) for song in song_list]
        random.shuffle(song_list)
        current_song_index = 0
        current_song = song_list[current_song_index]
    else:
        song_list = []
        current_song_index = -1
        current_song = None


# Function to play the current song
def play_current_song():
    global current_song
    if current_song and os.path.exists(current_song):
        pygame.mixer.music.load(current_song)
        pygame.mixer.music.play()
        print(f"Now Playing: {current_song}")
    else:
        print("No song to play!")


# Function to play a song based on detected emotion
def play_song(emotion):
    global current_emotion
    current_emotion = emotion
    load_song_list(emotion)
    if song_list:
        play_current_song()
    else:
        print(f"No songs found for emotion: {emotion}")


# Function to play the next song
def play_next_song():
    global current_song_index, current_song
    if song_list and current_song_index < len(song_list) - 1:
        current_song_index += 1
    else:
        current_song_index = 0  # Loop to the first song
    current_song = song_list[current_song_index]
    play_current_song()


# Function to play the previous song
def play_previous_song():
    global current_song_index, current_song
    if song_list and current_song_index > 0:
        current_song_index -= 1
    else:
        current_song_index = len(song_list) - 1  # Loop to the last song
    current_song = song_list[current_song_index]
    play_current_song()


# Function to stop the current song
def stop_song():
    pygame.mixer.music.stop()


# Face detection and emotion prediction
def detect_emotion():
    cap = cv2.VideoCapture(1)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    detected_emotion = "None"
    start_time = time.time()

    while time.time() - start_time < 5:  # Detect face for 5 seconds
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            processed_face = preprocess_image(face)
            predictions = model.predict(processed_face)
            emotion_index = np.argmax(predictions)
            detected_emotion = emotion_labels[emotion_index]

            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Detecting Emotion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Detected Emotion: {detected_emotion}")
    play_song(detected_emotion)


# GUI Setup
root = tk.Tk()
root.title("ZoniQ - Emotion-Based Music Player")
root.geometry("800x500")
root.resizable(False, False)

# Background Image
background_image = Image.open(background_image_path).resize((800, 500))
background_photo = ImageTk.PhotoImage(background_image)
canvas = Canvas(root, width=800, height=500)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=background_photo, anchor="nw")

# Title and Motto
canvas.create_text(400, 40, text="🎵 ZONIQ 🎵", font=("Arial Black", 28, "bold"), fill="white")
canvas.create_text(400, 80, text="Let Your Emotion Play the Perfect Track", font=("Arial", 16, "italic"), fill="white")

# Buttons
button_style = {"font": ("Arial", 12), "bg": "#ffffff", "fg": "black", "width": 20}

btn_detect = Button(root, text="Detect Emotion", command=detect_emotion, **button_style)
btn_detect.place(relx=0.5, rely=0.4, anchor="center")

btn_next = Button(root, text="Next Song", command=play_next_song, **button_style)
btn_next.place(relx=0.5, rely=0.5, anchor="center")

btn_prev = Button(root, text="Previous Song", command=play_previous_song, **button_style)
btn_prev.place(relx=0.5, rely=0.6, anchor="center")

btn_stop = Button(root, text="Stop Song", command=stop_song, **button_style)
btn_stop.place(relx=0.5, rely=0.7, anchor="center")

btn_exit = Button(root, text="Exit", command=root.destroy, **button_style)
btn_exit.place(relx=0.5, rely=0.8, anchor="center")

# Footer
canvas.create_text(400, 470, text="Developed by Team Futuristic 4", font=("Arial", 12, "italic"), fill="white")

# Run the application
root.mainloop()
