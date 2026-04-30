import os
import sys
import time
import base64
import wave
import threading
import platform
import ctypes
from pathlib import Path
from io import BytesIO

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import requests
from PIL import Image, ImageGrab
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

AUDIO_SAMPLERATE = 16000
AUDIO_CHANNELS = 1
RECORD_FILENAME = "last_question.wav"

# ----------------------------------------------------------------------
# Load static context
# ----------------------------------------------------------------------
def load_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""

RESUME = load_text(RESUME_PATH)
JOB_DESC = load_text(JOB_DESC_PATH)

# ----------------------------------------------------------------------
# MSI GenAI API client
# ----------------------------------------------------------------------
class InterviewAI:
    def __init__(self):
        self.host = os.getenv('MSI_HOST', "https://genai-service.stage.commandcentral.com/app-gateway/api/v2")
        self.api_key = os.getenv('MSI_API_KEY', "(I9ZpcAsjzv*aSwdRHxc3nOnuZR1LY!aNkxPG~e9")
        self.user_id = os.getenv('MSI_USER_ID', "rnj673@motorolasolutions.com")
        self.datastore_id = os.getenv('MSI_DATASTORE_ID', "1579319e-2b48-4bad-9825-4a7dd10ac0ef")
        self.model = os.getenv('MSI_MODEL', "ChatGPT4o-mini")
        
        self.chat_url = self.host + "/chat"
 
        self.http = requests.Session()
        self.http.headers.update({
            "Connection": "close", 
            "Cache-Control": "no-cache", 
            "Pragma": "no-cache",
            "Content-Type": "application/json",
            "x-msi-genai-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "X-User-Id": self.user_id,
            "X-Datastore-Id": self.datastore_id
        })
        
        self.max_image_dim = int(os.getenv("MSI_MAX_IMAGE_DIM", "1024"))
        self.jpeg_quality = int(os.getenv("MSI_JPEG_QUALITY", "85"))
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        parts = [
            "You are an expert interview assistant.",
            "You help the candidate (the user) answer interview questions.",
            "Use the provided resume and job description to tailor answers.",
            "If given an image, assume it is a coding problem or architecture diagram the candidate needs to solve.",
            "",
            "=== CANDIDATE RESUME ===",
            RESUME if RESUME else "(not provided)",
            "",
            "=== JOB DESCRIPTION ===",
            JOB_DESC if JOB_DESC else "(not provided)",
        ]
        return "\n".join(parts)

    def ask(self, question: str, image_pil = None) -> str:
        content = [{"type": "text", "text": question}]

        if image_pil is not None:
            # Convert to RGB (in case screenshot is RGBA)
            image_pil = image_pil.convert('RGB')
            
            # Resize image if it exceeds the max dimension
            w, h = image_pil.size
            if max(h, w) > self.max_image_dim:
                scale = self.max_image_dim / max(h, w)
                image_pil = image_pil.resize((int(w * scale), int(h * scale)))

            # Save to memory buffer
            buf = BytesIO()
            image_pil.save(buf, format="JPEG", quality=self.jpeg_quality)
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
            "max_tokens": 800,
            "temperature": 0.5,
        }

        try:
            resp = self.http.post(self.chat_url, json=payload, timeout=30)
            if resp.status_code != 200:
                return f"API error {resp.status_code}: {resp.text[:300]}"
            
            data = resp.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "Could not parse response")
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
    def __init__(self, root, ai, recorder):
        self.root = root
        
        # Disguise the window title
        self.root.title("Windows Input Experience")
        
        # Make the window much smaller since we removed the cameras
        self.root.geometry("500x700")
        
        # Keep the window always on top
        self.root.wm_attributes("-topmost", True)

        # Hide from Taskbar
        self.root.attributes('-toolwindow', True)

        # Lowered transparency to 0.70 so you can easily see VS Code behind it
        self.root.attributes('-alpha', 0.70)

        # Force draw before stealth
        self.root.update()

        # Stealth Mode (Invisible to Screen Share)
        if platform.system() == "Windows":
            try:
                hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 17)
            except Exception as e:
                print(f"Could not enable stealth mode: {e}")
        
        self.ai = ai
        self.recorder = recorder
        self.recording_mode = None  
        self.last_screenshot = None

        self.setup_ui()

        # Keyboard bindings 
        self.root.bind('<r>', self.toggle_audio_record)
        self.root.bind('<c>', self.toggle_capture_record)
        self.root.bind('<q>', self.on_closing)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        self.text_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status_lbl = tk.Label(
            self.text_frame, 
            text="🟢 READY\nPress 'r' (Audio Only) | Press 'c' (Audio + Screenshot)", 
            fg="#00ff00", bg="#1e1e1e", font=("Arial", 10, "bold"),
            justify=tk.CENTER, wraplength=450
        )
        self.status_lbl.pack(pady=(5, 5))

        self.chat_display = scrolledtext.ScrolledText(
            self.text_frame, wrap=tk.WORD, font=("Consolas", 11), 
            bg="#2b2b2b", fg="#ffffff", state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display.tag_config('interviewer', foreground='#569CD6', font=("Consolas", 11, "bold")) # VS Code Blue
        self.chat_display.tag_config('ai', foreground='#4CAF50', font=("Consolas", 11)) # Green
        self.chat_display.tag_config('system', foreground='gray')

    def log_chat(self, speaker, text, tag):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"[{speaker}]\n", tag)
        self.chat_display.insert(tk.END, f"{text}\n\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def toggle_audio_record(self, event=None):
        if self.recording_mode == 'c': return 

        if self.recording_mode is None:
            self.recording_mode = 'r'
            self.recorder.record()
            self.status_lbl.config(text="🔴 RECORDING AUDIO...\nPress 'r' again to stop and send.", fg="red")
        
        elif self.recording_mode == 'r':
            self.status_lbl.config(text="⏳ Processing...\nTranscribing & generating AI Response", fg="orange")
            self.recorder.stop()
            self.recording_mode = None
            threading.Thread(target=self.process_audio_only, daemon=True).start()

    def toggle_capture_record(self, event=None):
        if self.recording_mode == 'r': return 

        if self.recording_mode is None:
            self.recording_mode = 'c'
            
            # Take screenshot instantly and store it in RAM
            # Note: This captures ALL monitors. 
            self.last_screenshot = ImageGrab.grab() 
            
            self.recorder.record()
            self.status_lbl.config(text="🔴 SCREENSHOT SAVED + RECORDING AUDIO...\nPress 'c' again to send both.", fg="red")
            
        elif self.recording_mode == 'c':
            self.status_lbl.config(text="⏳ Processing...\nAnalyzing Screen & Generating AI Response", fg="orange")
            self.recorder.stop()
            self.recording_mode = None
            
            captured_image = self.last_screenshot
            threading.Thread(target=self.process_audio_and_capture, args=(captured_image,), daemon=True).start()

    def process_audio_only(self):
        question_text = transcribe_audio(RECORD_FILENAME)
        if question_text:
            self.root.after(0, self.log_chat, "Interviewer", question_text, 'interviewer')
            answer = self.ai.ask(question_text)
            self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
        else:
            self.root.after(0, self.log_chat, "System", "No speech detected. Please try again.", 'system')
            
        self.root.after(0, lambda: self.status_lbl.config(text="🟢 READY\nPress 'r' (Audio Only) | Press 'c' (Audio + Screenshot)", fg="#00ff00"))

    def process_audio_and_capture(self, image_pil):
        question_text = transcribe_audio(RECORD_FILENAME)
        
        if question_text:
            prompt = f"The interviewer asked this question: '{question_text}'. Please look at the provided screenshot of my coding environment and answer accordingly."
            display_text = f"📷 (Screenshot Attached)\n\"{question_text}\""
        else:
            prompt = "The screenshot shows a coding problem or environment I need to solve. Please read it and provide a solution or guidance."
            display_text = "📷 (Screenshot Attached)\n[No verbal question detected]"

        self.root.after(0, self.log_chat, "Interviewer", display_text, 'interviewer')
        answer = self.ai.ask(prompt, image_pil=image_pil)
        self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
        
        self.root.after(0, lambda: self.status_lbl.config(text="🟢 READY\nPress 'r' (Audio Only) | Press 'c' (Audio + Screenshot)", fg="#00ff00"))

    def on_closing(self, event=None):
        self.root.quit()

# ----------------------------------------------------------------------
# Bootstrapper
# ----------------------------------------------------------------------
def main():
    ai = InterviewAI()
    recorder = AudioRecorder()

    root = tk.Tk()
    app = InterviewAssistantApp(root, ai, recorder)
    
    root.mainloop()

if __name__ == "__main__":
    main()