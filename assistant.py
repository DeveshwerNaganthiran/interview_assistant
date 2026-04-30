import os
import sys
import time
import base64
import wave
import threading
import json
import platform
import ctypes
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import requests
from PIL import Image, ImageTk
from dotenv import load_dotenv

import tkinter as tk
from tkinter import scrolledtext

# Load environment variables
load_dotenv()

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MSI_HOST = os.getenv("MSI_HOST")
MSI_API_KEY = os.getenv("MSI_API_KEY")
MSI_MODEL = os.getenv("MSI_MODEL")
MSI_USER_ID = os.getenv("MSI_USER_ID")
MSI_DATASTORE_ID = os.getenv("MSI_DATASTORE_ID")

RESUME_PATH = "resume.txt"
JOB_DESC_PATH = "job_description.txt"

# Webcam indices (adjust to your setup)
CAMERA_PAPER = 0   # webcam aimed at the problem sheet
CAMERA_FACE = None # webcam for your face (optional, can be None)

AUDIO_SAMPLERATE = 16000
AUDIO_CHANNELS = 1
RECORD_FILENAME = "last_question.wav"

# ----------------------------------------------------------------------
# Load static context
# ----------------------------------------------------------------------
def load_text(path):
    """Load plain text from file, return empty string if missing."""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""

RESUME = load_text(RESUME_PATH)
JOB_DESC = load_text(JOB_DESC_PATH)

if not RESUME:
    print("WARNING: resume.txt is empty or missing.")
if not JOB_DESC:
    print("WARNING: job_description.txt is empty or missing.")

# ----------------------------------------------------------------------
# MSI GenAI API client
# ----------------------------------------------------------------------
class InterviewAI:
    def __init__(self):
        self.host = MSI_HOST
        self.api_key = MSI_API_KEY
        self.model = MSI_MODEL
        self.user_id = MSI_USER_ID
        self.datastore_id = MSI_DATASTORE_ID
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-User-Id": self.user_id,
            "X-Datastore-Id": self.datastore_id,
        }
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        parts = [
            "You are an expert interview assistant.",
            "You help the candidate (the user) answer interview questions.",
            "Use the provided resume and job description to tailor answers.",
            "",
            "=== CANDIDATE RESUME ===",
            RESUME if RESUME else "(not provided)",
            "",
            "=== JOB DESCRIPTION ===",
            JOB_DESC if JOB_DESC else "(not provided)",
            "",
            "When the user asks a question, respond with a concise yet complete answer",
            "that highlights relevant experience and skills.",
            "For coding problems, provide explanation and code if appropriate.",
            "If the question is a behavioral one, use the STAR method based on the resume.",
        ]
        return "\n".join(parts)

    def ask(self, question: str, image_bgr = None) -> str:
        content = [{"type": "text", "text": question}]

        if image_bgr is not None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        try:
            resp = requests.post(
                f"{self.host}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            if resp.status_code != 200:
                return f"API error {resp.status_code}: {resp.text[:300]}"
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            return answer.strip()
        except Exception as e:
            return f"Request failed: {str(e)}"

# ----------------------------------------------------------------------
# Audio recorder
# ----------------------------------------------------------------------
class AudioRecorder:
    def __init__(self, filename=RECORD_FILENAME):
        self.filename = filename
        self.frames = []
        self.recording = False
        self._thread = None

    def record(self):
        if self.recording: return
        self.frames = []
        self.recording = True
        self._thread = threading.Thread(target=self._record_loop)
        self._thread.start()

    def stop(self):
        if not self.recording: return
        self.recording = False
        if self._thread:
            self._thread.join()
        self._save()

    def _record_loop(self):
        def callback(indata, frames, time_info, status):
            if self.recording:
                self.frames.append(indata.copy())
        with sd.InputStream(samplerate=AUDIO_SAMPLERATE, channels=AUDIO_CHANNELS,
                            callback=callback, dtype='float32'):
            while self.recording:
                sd.sleep(100)

    def _save(self):
        if not self.frames: return
        audio = np.concatenate(self.frames, axis=0)
        audio_int16 = np.int16(audio * 32767)
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLERATE)
            wf.writeframes(audio_int16.tobytes())

# ----------------------------------------------------------------------
# Speech‑to‑text
# ----------------------------------------------------------------------
def transcribe_audio(filename, language="en-US"):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language=language)
    except:
        return ""

# ----------------------------------------------------------------------
# GUI Application Main Class
# ----------------------------------------------------------------------
class InterviewAssistantApp:
    def __init__(self, root, ai, recorder, cap_paper, cap_face):
        self.root = root
        
        # 1. Disguise the window title so it looks like a normal background process
        self.root.title("Windows Input Experience")
        self.root.geometry("1100x650")
        
        # 2. Keep the window always on top so you never lose it behind your browser
        self.root.wm_attributes("-topmost", True)

        # 3. Make the window invisible to screen sharing (Windows 10/11 only)
        if platform.system() == "Windows":
            try:
                # Get the window handle (HWND)
                hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                # 17 (0x11) is WDA_EXCLUDEFROMCAPTURE: Completely hides the window from screen capture tools
                ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 17)
            except Exception as e:
                print(f"Could not enable stealth mode: {e}")
        
        self.ai = ai
        self.recorder = recorder
        self.cap_paper = cap_paper
        self.cap_face = cap_face

        self.recording_mode = None  # 'r' or 'c'
        self.last_paper_frame = None

        self.setup_ui()
        self.update_video()

        # Keyboard bindings (Requires window to be in focus)
        self.root.bind('<r>', self.toggle_audio_record)
        self.root.bind('<c>', self.toggle_capture_record)
        self.root.bind('<q>', self.on_closing)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Left Panel (Video Feeds) - Anchored to the top
        self.video_frame = tk.Frame(self.root, bg="#2b2b2b")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

        self.paper_label = tk.Label(self.video_frame, text="Problem / Paper Camera", fg="white", bg="#2b2b2b", font=("Arial", 12, "bold"))
        self.paper_label.pack(pady=(10, 5), anchor=tk.N)
        
        self.paper_vid_lbl = tk.Label(self.video_frame, bg="black")
        self.paper_vid_lbl.pack(anchor=tk.N)

        if self.cap_face and self.cap_face.isOpened():
            self.face_label = tk.Label(self.video_frame, text="Your Face", fg="white", bg="#2b2b2b", font=("Arial", 12, "bold"))
            self.face_label.pack(pady=(20, 5), anchor=tk.N)
            self.face_vid_lbl = tk.Label(self.video_frame, bg="black")
            self.face_vid_lbl.pack(anchor=tk.N)

        # Right Panel (Chat / Output)
        self.text_frame = tk.Frame(self.root)
        self.text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status Bar - Added wraplength and \n to prevent cutoff!
        self.status_lbl = tk.Label(
            self.text_frame, 
            text="🟢 READY\nPress 'r' (Record Audio) | Press 'c' (Record Audio + Capture Screen)", 
            fg="green", font=("Arial", 12, "bold"),
            justify=tk.CENTER,
            wraplength=500 # Forces the text to wrap instead of going off-screen
        )
        self.status_lbl.pack(pady=(0, 10))

        # Scrolling Text Display
        self.chat_display = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD, font=("Consolas", 11), state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display.tag_config('interviewer', foreground='#00008B', font=("Consolas", 12, "bold"))
        self.chat_display.tag_config('ai', foreground='#006400', font=("Consolas", 12))
        self.chat_display.tag_config('system', foreground='gray')

    def log_chat(self, speaker, text, tag):
        """Helper to append text to the GUI chat window safely"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"[{speaker}]\n", tag)
        self.chat_display.insert(tk.END, f"{text}\n\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def update_video(self):
        """Loop that continually grabs frames from OpenCV and puts them in Tkinter"""
        ret, frame = self.cap_paper.read()
        if ret:
            self.last_paper_frame = frame.copy()
            # Resize for UI
            frame_resized = cv2.resize(frame, (480, 360))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.paper_vid_lbl.imgtk = imgtk
            self.paper_vid_lbl.configure(image=imgtk)

        if self.cap_face and self.cap_face.isOpened():
            ret_f, frame_f = self.cap_face.read()
            if ret_f:
                frame_f_resized = cv2.resize(frame_f, (320, 240))
                frame_f_rgb = cv2.cvtColor(frame_f_resized, cv2.COLOR_BGR2RGB)
                img_f = Image.fromarray(frame_f_rgb)
                imgtk_f = ImageTk.PhotoImage(image=img_f)
                self.face_vid_lbl.imgtk = imgtk_f
                self.face_vid_lbl.configure(image=imgtk_f)

        # Call this function again in 30ms
        self.root.after(30, self.update_video)

    def toggle_audio_record(self, event=None):
        if self.recording_mode == 'c': return # Prevent overlapping commands

        if self.recording_mode is None:
            self.recording_mode = 'r'
            self.recorder.record()
            self.status_lbl.config(text="🔴 RECORDING AUDIO...\nPress 'r' again to stop and send.", fg="red")
        
        elif self.recording_mode == 'r':
            self.status_lbl.config(text="⏳ Processing...\nTranscribing audio & generating AI Response", fg="orange")
            self.recorder.stop()
            self.recording_mode = None
            threading.Thread(target=self.process_audio_only, daemon=True).start()

    def toggle_capture_record(self, event=None):
        if self.recording_mode == 'r': return 

        if self.recording_mode is None:
            self.recording_mode = 'c'
            self.recorder.record()
            self.status_lbl.config(text="🔴 RECORDING AUDIO + CAPTURE...\nPress 'c' again to send.", fg="red")
            
        elif self.recording_mode == 'c':
            self.status_lbl.config(text="⏳ Processing...\nAnalyzing Image, Transcribing & generating AI Response", fg="orange")
            self.recorder.stop()
            self.recording_mode = None
            
            captured_frame = self.last_paper_frame.copy() if self.last_paper_frame is not None else None
            threading.Thread(target=self.process_audio_and_capture, args=(captured_frame,), daemon=True).start()

    def process_audio_only(self):
        question_text = transcribe_audio(RECORD_FILENAME)
        if question_text:
            self.root.after(0, self.log_chat, "Interviewer", question_text, 'interviewer')
            answer = self.ai.ask(question_text)
            self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
        else:
            self.root.after(0, self.log_chat, "System", "No speech detected. Please try again.", 'system')
            
        self.root.after(0, lambda: self.status_lbl.config(text="🟢 READY\nPress 'r' (Record Audio) | Press 'c' (Record Audio + Capture Screen)", fg="green"))

    def process_audio_and_capture(self, frame):
        question_text = transcribe_audio(RECORD_FILENAME)
        
        if question_text:
            prompt = f"The interviewer asked this question: '{question_text}'. Please look at the provided image and answer accordingly."
            display_text = f"📷 (Image Attached)\n\"{question_text}\""
        else:
            prompt = "The image shows a problem I need to solve. Please read it and provide a solution."
            display_text = "📷 (Image Attached)\n[No verbal question detected]"

        self.root.after(0, self.log_chat, "Interviewer", display_text, 'interviewer')
        answer = self.ai.ask(prompt, image_bgr=frame)
        self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
        
        self.root.after(0, lambda: self.status_lbl.config(text="🟢 READY\nPress 'r' (Record Audio) | Press 'c' (Record Audio + Capture Screen)", fg="green"))

    def on_closing(self, event=None):
        """Cleanup cameras and gracefully exit"""
        print("Shutting down...")
        self.cap_paper.release()
        if self.cap_face:
            self.cap_face.release()
        self.root.quit()

# ----------------------------------------------------------------------
# Bootstrapper
# ----------------------------------------------------------------------
def main():
    ai = InterviewAI()
    recorder = AudioRecorder()

    cap_paper = cv2.VideoCapture(CAMERA_PAPER, cv2.CAP_DSHOW)
    cap_face = None
    if CAMERA_FACE is not None and CAMERA_FACE >= 0:
        cap_face = cv2.VideoCapture(CAMERA_FACE, cv2.CAP_DSHOW)

    if not cap_paper.isOpened():
        print("Could not open paper camera. Check your CAMERA_PAPER index.")
        sys.exit(1)

    root = tk.Tk()
    app = InterviewAssistantApp(root, ai, recorder, cap_paper, cap_face)
    
    # Start the GUI Loop
    root.mainloop()

if __name__ == "__main__":
    main()