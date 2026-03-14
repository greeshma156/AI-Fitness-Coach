import cv2
import numpy as np
import mediapipe as mp
import time
import pyttsx3
import threading
import random
import os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from playsound import playsound

# --- 1. AI SETUP (Moved here to prevent window lag) ---
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)

# --- 2. PERSISTENT STATS LOADING ---
# Load highscore from file at startup so it can be displayed immediately
highscore = 0
if os.path.exists("highscore.txt"):
    try:
        with open("highscore.txt", "r") as f:
            highscore = int(f.read().strip())
    except:
        highscore = 0

# --- 3. PHRASE BANKS ---
SWITCH_PHRASES = ["Great job! Let's move to squats.", "Curls done! Time for ten squats."]
LUNGE_START = ["Squats complete. Now for ten lunges.", "Excellent. Now for lunges."]
PLANK_START = ["Lunges done! Final challenge: 30 second plank. Get down!"]
FINISH_PHRASES = ["Workout completed!", "Session finished. You are a champion!"]

# --- 4. VOICE & MUSIC ENGINE ---
def speak(text):
    def run_speech():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 210) 
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            engine.say(text)
            engine.runAndWait()
        except: pass
    threading.Thread(target=run_speech, daemon=True).start()

def play_music():
    music_dir = "music"
    if not os.path.exists(music_dir):
        print(f"Error: '{music_dir}' folder not found.")
        return
    songs = [f for f in os.listdir(music_dir) if f.endswith(".mp3")]
    if not songs:
        print("Error: No .mp3 files found.")
        return
    random_song = random.choice(songs)
    song_path = os.path.join(music_dir, random_song)
    def run_music():
        try:
            # Lowering volume by adding a placeholder for potential OS-level volume control
            # Note: playsound doesn't natively support volume, so we rely on the system mixer 
            # or pre-processed low-volume audio files.
            playsound(song_path, block=False)
        except: pass
    threading.Thread(target=run_music, daemon=True).start()

# --- 5. MATH ENGINE ---
def calculate_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    v1, v2 = a - b, c - b
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# --- 6. GLOBAL STATE ---
exercise = "CURLS"
current_side = "LEFT" 
reps_left, reps_right, reps_squats, reps_lunges = 0, 0, 0, 0
total_sets = 5
current_set = 1
plank_duration = 30
plank_accumulated_time = 0
plank_last_active_time = None
last_announced_second = 0
stage = "DOWN" # Start in DOWN stage to ensure a full range of motion
last_count_time = 0
angle_history = deque(maxlen=8)
workout_finished = False
font = cv2.FONT_HERSHEY_DUPLEX 
form_correct = True 
start_time = time.time()
form_checks, correct_checks = 0, 0 

# --- 7. CAMERA SETUP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow('AI Fitness Coach - V21.3', cv2.WINDOW_NORMAL)

speak(f"Set 1 starting. Begin left bicep curl.")
play_music() 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks and not workout_finished:
        landmarks = detection_result.pose_landmarks[0]
        current_time = time.time()

        if exercise == "CURLS":
            side_map = {"LEFT": {"s": 12, "e": 14, "w": 16}, "RIGHT": {"s": 11, "e": 13, "w": 15}}
            idx = side_map[current_side]
            p1 = [landmarks[idx["s"]].x, landmarks[idx["s"]].y] # Shoulder
            p2 = [landmarks[idx["e"]].x, landmarks[idx["e"]].y] # Elbow
            p3 = [landmarks[idx["w"]].x, landmarks[idx["w"]].y] # Wrist
            
            angle = calculate_angle(p1, p2, p3)
            angle_history.append(angle)
            smooth_angle = sum(angle_history)/len(angle_history)

            # Accuracy Fix: Reset stage only when arm is straight
            if smooth_angle > 150: 
                stage = "DOWN"
            
            # Accuracy Fix: Count only if arm is bent AND wrist is higher than elbow
            if smooth_angle < 45 and stage == "DOWN":
                if (current_time - last_count_time) > 1.2 and p3[1] < p2[1]:
                    last_count_time = current_time
                    stage = "UP"
                    if current_side == "LEFT":
                        reps_left += 1
                        if reps_left == 10:
                            speak("Switching side")
                            current_side = "RIGHT"
                            angle_history.clear()
                        else: speak(str(reps_left))
                    else:
                        reps_right += 1
                        if reps_right == 10:
                            speak(random.choice(SWITCH_PHRASES))
                            exercise, stage = "SQUATS", "DOWN"
                            angle_history.clear()
                        else: speak(str(reps_right))

        elif exercise == "SQUATS":
            p1, p2, p3 = [landmarks[24].x, landmarks[24].y], [landmarks[26].x, landmarks[26].y], [landmarks[28].x, landmarks[28].y]
            angle = calculate_angle(p1, p2, p3)
            angle_history.append(angle)
            smooth_angle = sum(angle_history)/len(angle_history)
            if smooth_angle > 160: stage = "DOWN"
            if smooth_angle < 125 and stage == "DOWN":
                if (current_time - last_count_time) > 1.5:
                    reps_squats += 1
                    last_count_time, stage = current_time, "UP"
                    if reps_squats == 10:
                        speak(random.choice(LUNGE_START))
                        exercise, current_side, stage = "LUNGES", "LEFT", "DOWN"
                        angle_history.clear()
                    else: speak(str(reps_squats))

        elif exercise == "LUNGES":
            side_map = {"LEFT": {"h": 24, "k": 26, "a": 28}, "RIGHT": {"h": 23, "k": 25, "a": 27}}
            idx = side_map[current_side]
            p1, p2, p3 = [landmarks[idx["h"]].x, landmarks[idx["h"]].y], [landmarks[idx["k"]].x, landmarks[idx["k"]].y], [landmarks[idx["a"]].x, landmarks[idx["a"]].y]
            angle = calculate_angle(p1, p2, p3)
            angle_history.append(angle)
            smooth_angle = sum(angle_history)/len(angle_history)
            if smooth_angle > 155: stage = "DOWN"
            if smooth_angle < 120 and stage == "DOWN":
                if (current_time - last_count_time) > 1.8:
                    reps_lunges += 1
                    last_count_time, stage = current_time, "UP"
                    if reps_lunges == 5:
                        speak("Switch legs.")
                        current_side = "RIGHT"
                        angle_history.clear()
                    elif reps_lunges == 10:
                        speak(PLANK_START[0])
                        exercise, stage = "PLANK", "GET_READY"
                        angle_history.clear()
                    else: speak(str(reps_lunges))

        elif exercise == "PLANK":
            p1, p2, p3 = [landmarks[12].x, landmarks[12].y], [landmarks[24].x, landmarks[24].y], [landmarks[28].x, landmarks[28].y]
            angle = calculate_angle(p1, p2, p3)
            is_horizontal = abs(landmarks[12].y - landmarks[24].y) < 0.20
            form_checks += 1
            if angle > 150:
                correct_checks += 1
                form_correct = True
            else:
                form_correct = False
            if angle > 160 and is_horizontal:
                if plank_last_active_time is None:
                    plank_last_active_time = time.time()
                    if plank_accumulated_time == 0: speak("Plank started. Hold it!")
                else:
                    plank_accumulated_time += (time.time() - plank_last_active_time)
                    plank_last_active_time = time.time()
                remaining = int(plank_duration - plank_accumulated_time)
                if remaining != last_announced_second:
                    if remaining <= 5 and remaining > 0: speak(str(remaining))
                    elif remaining % 5 == 0 and remaining > 0: speak(str(remaining))
                    last_announced_second = remaining
                if plank_accumulated_time >= plank_duration:
                    if current_set < total_sets:
                        speak(f"Set {current_set} completed. Great effort!")
                        current_set += 1
                        speak(f"Starting set {current_set}. Get ready for bicep curls.")
                        # Reset stats for next set
                        exercise, current_side, stage = "CURLS", "LEFT", "DOWN"
                        reps_left, reps_right, reps_squats, reps_lunges = 0, 0, 0, 0
                        plank_accumulated_time = 0
                        plank_last_active_time = None
                        angle_history.clear()
                    else:
                        speak(f"Set {current_set} completed.")
                        speak(random.choice(FINISH_PHRASES))
                        
                        # Update highscore file if new session is better
                        current_accuracy = int((correct_checks / form_checks * 100)) if form_checks > 0 else 0
                        if current_accuracy > highscore:
                            with open("highscore.txt", "w") as f:
                                f.write(str(current_accuracy))
                        
                        workout_finished = True
            else:
                plank_last_active_time = None 

        cv2.polylines(frame, [np.array([(int(p[0]*w), int(p[1]*h)) for p in [p1,p2,p3]])], False, (0, 255, 0), 4)

    # --- HUD RENDER (Always Visible) ---
    cv2.rectangle(frame, (w-280, 0), (w, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"HIGHSCORE: {highscore}%", (w-265, 35), font, 0.7, (0, 255, 255), 2)
    current_accuracy = int((correct_checks / form_checks * 100)) if form_checks > 0 else 0
    cv2.putText(frame, f"CURRENT: {current_accuracy}%", (w-265, 75), font, 0.7, (0, 255, 0), 2)

    if not workout_finished:
        cv2.rectangle(frame, (0,0), (450, 210), (0, 0, 0), -1)
        cv2.putText(frame, f"SET: {current_set}/{total_sets}", (15, 30), font, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"CURLS L/R: {reps_left} | {reps_right}", (15, 65), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"SQUATS: {reps_squats}/1 | LUNGES: {reps_lunges}/2", (15, 95), font, 0.6, (255, 255, 255), 1)
        if exercise == "PLANK":
            display_time = max(0, int(plank_duration - plank_accumulated_time))
            cv2.putText(frame, f"PLANK TIME: {display_time}s", (15, 125), font, 0.7, (0, 255, 255), 2)
            if not form_correct:
                cv2.putText(frame, "FIX FORM!", (250, 160), font, 0.9, (0, 0, 255), 3)
        cv2.putText(frame, f"ACTIVE: {exercise}", (15, 165), font, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "WORKOUT COMPLETE", (w//2-150, h//2), font, 1, (0, 255, 0), 2)

    cv2.imshow('AI Fitness Coach - v1.0.0', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()