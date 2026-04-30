import os
import sys
import time
import base64
import wave
import threading
import platform
import ctypes
import json
from pathlib import Path
from io import BytesIO

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import requests
from PIL import Image, ImageGrab
from dotenv import load_dotenv

# Enforce override to ensure the correct API key is grabbed
load_dotenv(override=True)

import tkinter as tk
from tkinter import scrolledtext

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
# MSI GenAI API client - Session-Based Workflow
# ----------------------------------------------------------------------
class InterviewAI:
    def __init__(self):
        self.host = os.getenv('MSI_HOST', "https://genai-service.stage.commandcentral.com/app-gateway/api/v2")
        self.api_key = os.getenv('MSI_API_KEY', "(I9ZpcAsjzv*aSwdRHxc3nOnuZR1LY!aNkxPG~e9")
        self.user_id = os.getenv('MSI_USER_ID', "rnj673@motorolasolutions.com")
        self.datastore_id = os.getenv('MSI_DATASTORE_ID', "1579319e-2b48-4bad-9825-4a7dd10ac0ef")
        self.model = os.getenv('MSI_MODEL', "Claude-Sonnet-3_7") # Ensure Claude is set here if .env fails
        
        self.chat_url = self.host + "/chat"
        self.upload_url = self.host + "/upload"
 
        self.http = requests.Session()
        self.http.headers.update({
            "Connection": "close", 
            "Cache-Control": "no-cache", 
            "Pragma": "no-cache"
        })
        
        self.max_image_dim = int(os.getenv("MSI_MAX_IMAGE_DIM", "1024"))
        self.jpeg_quality = int(os.getenv("MSI_JPEG_QUALITY", "85"))
        
        # SPEED OPTIMIZATION: Cache the session so we don't recreate it every question
        self.active_session_id = None

    def get_or_create_session(self) -> str:
        """Returns the active session or creates a new one if it doesn't exist."""
        if self.active_session_id:
            return self.active_session_id

        headers = {
            "Content-Type": "application/json",
            "x-msi-genai-api-key": self.api_key
        }
        payload = {
            "userId": self.user_id,
            "model": self.model,
            "datastoreId": self.datastore_id,
            "prompt": "init"
        }
        
        # Increased timeout to 45s for heavy models like Claude
        response = self.http.post(self.chat_url, headers=headers, json=payload, timeout=45)
        if response.status_code >= 400:
            raise RuntimeError(f"Session init failed {response.status_code}: {response.text}")
        
        response_data = response.json()
        if response_data.get("status") and "sessionId" in response_data:
            self.active_session_id = response_data["sessionId"]
            return self.active_session_id
        else:
            raise RuntimeError(f"Invalid session response: {response_data}")

    def upload_image(self, session_id: str, image_pil) -> bool:
        headers = {"x-msi-genai-api-key": self.api_key}
        image_pil = image_pil.convert('RGB')
        w, h = image_pil.size
        if max(h, w) > self.max_image_dim:
            scale = self.max_image_dim / max(h, w)
            image_pil = image_pil.resize((int(w * scale), int(h * scale)))

        buf = BytesIO()
        image_pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        image_bytes = buf.getvalue()
        
        url = f"{self.upload_url}/{session_id}?userId={self.user_id}"
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        
        # Increased timeout to 45s
        response = self.http.post(url, headers=headers, files=files, timeout=45)
        
        if response.status_code != 200:
            raise RuntimeError(f"Upload failed {response.status_code}: {response.text}")
        return True

    def extract_text_from_response(self, response_data: dict) -> str:
        try:
            ans = ""
            if "data" in response_data and isinstance(response_data["data"], dict):
                data = response_data["data"]
                for k in ["text", "message", "response", "output", "msg"]:
                    if k in data and isinstance(data[k], str) and data[k].strip():
                        ans = data[k].strip()
                        break

            if not ans:
                for k in ["message", "response", "text", "msg"]:
                    if k in response_data and isinstance(response_data[k], str) and response_data[k].strip():
                        ans = response_data[k].strip()
                        break
            
            if not ans:
                ans = json.dumps(response_data)
                
            ans = ans.replace("Unsupported media type found. Ignoring non-image media.", "")
            ans = ans.replace("Unsupported media type found. Ignoring non-image media. \n", "")
            return ans.strip()
        except Exception:
            return ""

    def ask(self, question: str, image_pil = None) -> str:
        try:
            # SPEED OPTIMIZATION: Uses cached session instead of creating a new one
            session_id = self.get_or_create_session()
            
            if image_pil is not None:
                self.upload_image(session_id, image_pil)

            # 3. Combine context into a single string with STRICT conversational constraints
            # 3. Combine context into a single string with STRICT conversational constraints
            # 3. Combine context into a single string with STRICT conversational constraints
            full_prompt = (
                "CRITICAL PERSONA: You are a candidate in a software engineering job interview. "
                "Speak EXACTLY as you would out loud to the interviewer. Be confident, casual, direct, and highly observant. "
                "NEVER act like a tutor, an AI, or a senior developer. "
                "You MUST sound like a normal human engineer talking out loud.\n\n"
                "BANNED PHRASES (DO NOT USE): 'I see several issues', 'Here is how I would correct', 'Let me go through them', 'The code snippet presented', 'This should resolve the errors'.\n\n"
                "CODE ERROR INSTRUCTIONS: If shown an image of code, you MUST act like a strict compiler. Do these 3 things in order:\n"
                "1. Mentally trace the image TOP-TO-BOTTOM, character-by-character. Identify ALL REAL errors. DO NOT STOP AT JUST ONE! You must catch everything.\n"
                "CRITICAL VISION RULE: DO NOT USE LINE NUMBERS (e.g., never say 'On line 4'). OCR often misaligns them. Instead, quote the exact broken code directly.\n"
                "CRITICAL ORDERING RULE: You MUST explain the errors in the EXACT top-to-bottom chronological order they appear in the image. First bug first, second bug second, etc. Do not jump around.\n"
                "CRITICAL VISION CHECKS:\n"
                "- Check EVERY SINGLE variable assignment (is it using `==` instead of `=`?).\n"
                "- Check EVERY SINGLE function definition (missing `()` or `:`?).\n"
                "- Check EVERY SINGLE method call (hyphens `-` used instead of dots `.`).\n"
                "- Check EVERY SINGLE variable name for spelling typos.\n"
                "2. Casually explain what's wrong with the lines you found, strictly following the top-to-bottom order.\n"
                "3. Provide the exact corrected code block containing ALL your fixes. Use `// ...` or `# ...` to skip unchanged code.\n\n"
                "GOOD EXAMPLE OF HOW TO SPEAK (Mimic this exact tone and ordering):\n"
                "- 'Ah, I spot a few bugs here. Going top to bottom: First, when assigning the `name`, `gift`, and `option` variables, you're using double equals `==` which is for comparison, not assignment. Next, down inside the if statement, it says `christmaslist-append` with a hyphen instead of a dot. Finally, right below that, the variable `christmaslist` is misspelled. I'd fix those lines like this:\n"
                "```python\n"
                "name = input(\"Hello what is your name?\")\n"
                "# ...\n"
                "gift = input(\"\")\n"
                "# ...\n"
                "option = input(\"\")\n"
                "if option == \"Y\":\n"
                "    christmaslist.append(gift)\n"
                "    print(christmaslist)\n"
                "```'\n\n"
                "If asked a behavioral question, answer naturally in the FIRST PERSON ('I', 'my') using your RESUME below. Tailor it to the JOB DESCRIPTION.\n\n"
                "CRITICAL CONTEXT: The question is transcribed via Speech-to-Text. "
                "Expect phonetic mistakes. Infer the INTENDED question.\n\n"
                "CRITICAL FORMATTING INSTRUCTION: Always wrap key technical concepts in **double asterisks**. "
                "ALWAYS wrap code blocks in triple backticks (```) and inline code in single backticks (`).\n\n"
                f"=== CANDIDATE RESUME ===\n{RESUME if RESUME else '(not provided)'}\n\n"
                f"=== JOB DESCRIPTION ===\n{JOB_DESC if JOB_DESC else '(not provided)'}\n\n"
                f"USER QUESTION: {question}"
            )

            headers = {
                "Content-Type": "application/json",
                "x-msi-genai-api-key": self.api_key
            }
            payload = {
                "userId": self.user_id,
                "model": self.model,
                "datastoreId": self.datastore_id,
                "sessionId": session_id,
                "prompt": full_prompt
            }

            # Increased timeout to 60s to give Claude time to think
            response = self.http.post(self.chat_url, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                # If the session expired on the server, reset it for the next try
                if response.status_code in [401, 403, 404]:
                    self.active_session_id = None 
                return f"API error {response.status_code}: {response.text[:300]}"
            
            return self.extract_text_from_response(response.json())
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
        self.root.title("Windows Input Experience")
        self.root.geometry("440x650")
        self.root.wm_attributes("-topmost", True)
        self.root.attributes('-toolwindow', True)
        self.root.attributes('-alpha', 0.85)
        self.root.config(bg="#151821")
        self.root.update()

        if platform.system() == "Windows":
            try:
                hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 17)
            except Exception: pass
        
        self.ai = ai
        self.recorder = recorder
        self.recording_mode = None  
        self.last_screenshot = None

        self.setup_ui()

        self.root.bind('<r>', self.toggle_audio_record)
        self.root.bind('<c>', self.toggle_capture_record)
        self.root.bind('<q>', self.on_closing)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        BG_COLOR = "#151821"        
        CYAN = "#00E5FF"            
        WHITE = "#F5F5F5"           
        GREY_BLUE = "#8B949E"       
        
        main_font = ("Segoe UI", 11)
        bold_font = ("Segoe UI", 11, "bold")
        name_font = ("Segoe UI", 10, "bold")
        
        # New Fonts for Code
        code_font = ("Consolas", 11)

        self.text_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.status_lbl = tk.Label(
            self.text_frame, 
            text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", 
            fg=GREY_BLUE, bg=BG_COLOR, font=("Segoe UI", 10),
            justify=tk.LEFT, anchor="w", pady=10
        )
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X)

        self.chat_display = scrolledtext.ScrolledText(
            self.text_frame, wrap=tk.WORD, font=main_font, 
            bg=BG_COLOR, fg=WHITE, state=tk.DISABLED, 
            bd=0, highlightthickness=0, insertbackground=WHITE
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Regular tags
        self.chat_display.tag_config('ai_name', foreground=CYAN, font=name_font)
        self.chat_display.tag_config('ai_text', foreground=WHITE, font=main_font, spacing3=5)
        self.chat_display.tag_config('cyan_highlight', foreground=CYAN, font=bold_font)
        self.chat_display.tag_config('interviewer_name', foreground=GREY_BLUE, font=name_font)
        self.chat_display.tag_config('interviewer_text', foreground=GREY_BLUE, font=main_font, spacing3=10)
        self.chat_display.tag_config('system', foreground='#E2B714', font=main_font, spacing3=10) 
        
        # --- NEW CODE FORMATTING TAGS ---
        # Inline code: slight background, distinct color
        self.chat_display.tag_config('inline_code', foreground="#FFB86C", background="#222633", font=code_font)
        # Code block: darker indented background, standard IDE yellow/green text
        self.chat_display.tag_config('code_block', foreground="#DCDCAA", background="#0D1017", font=code_font, lmargin1=10, lmargin2=10)

    def log_chat(self, speaker, text, tag):
        self.chat_display.config(state=tk.NORMAL)
        
        # Mark the exact line where this new message is starting
        start_index = self.chat_display.index(tk.END + "-1c")

        if speaker == "AI Assistant":
            self.chat_display.insert(tk.END, f"✨ {speaker}\n", 'ai_name')
        elif speaker == "Interviewer":
            self.chat_display.insert(tk.END, f"👤 {speaker}\n", 'interviewer_name')
        else:
            self.chat_display.insert(tk.END, f"⚙️ {speaker}\n", 'system')

        # CUSTOM MARKDOWN PARSER FOR CODE AND BOLD
        if tag == 'ai':
            blocks = text.split("```")
            for b_idx, block in enumerate(blocks):
                if b_idx % 2 == 1:
                    lines = block.split('\n', 1)
                    if len(lines) > 1 and lines[0].strip().isalpha():
                        clean_block = lines[1]
                    else:
                        clean_block = block
                    self.chat_display.insert(tk.END, f"\n{clean_block.rstrip()}\n\n", 'code_block')
                else:
                    inline_parts = block.split("`")
                    for i_idx, inline_part in enumerate(inline_parts):
                        if i_idx % 2 == 1:
                            self.chat_display.insert(tk.END, inline_part, 'inline_code')
                        else:
                            bold_parts = inline_part.split("**")
                            for bold_idx, bold_part in enumerate(bold_parts):
                                if bold_idx % 2 == 1:
                                    self.chat_display.insert(tk.END, bold_part, 'cyan_highlight')
                                else:
                                    self.chat_display.insert(tk.END, bold_part, 'ai_text')
            self.chat_display.insert(tk.END, "\n\n")
        else:
            if tag == 'interviewer':
                self.chat_display.insert(tk.END, f"{text}\n\n", 'interviewer_text')
            else:
                self.chat_display.insert(tk.END, f"{text}\n\n", 'system')

        # --- THE SCROLL FIX ---
        # 1. Force the UI to scroll to the absolute bottom first
        self.chat_display.see(tk.END)
        self.chat_display.update_idletasks() # Force Tkinter to refresh the screen
        
        # 2. If it's the AI, snap the view back to the start of the answer
        if speaker == "AI Assistant":
            self.chat_display.see(start_index)

        self.chat_display.config(state=tk.DISABLED)

    def toggle_audio_record(self, event=None):
        if self.recording_mode == 'c': return 

        if self.recording_mode is None:
            self.recording_mode = 'r'
            self.recorder.record()
            self.status_lbl.config(text="🔴 Recording audio... Press 'R' to send.", fg="#FF5555")
        
        elif self.recording_mode == 'r':
            self.status_lbl.config(text="⏳ Thinking...", fg="#00E5FF")
            self.recorder.stop()
            self.recording_mode = None
            threading.Thread(target=self.process_audio_only, daemon=True).start()

    def toggle_capture_record(self, event=None):
        if self.recording_mode == 'r': return 

        if self.recording_mode is None:
            self.recording_mode = 'c'
            self.last_screenshot = ImageGrab.grab() 
            self.recorder.record()
            self.status_lbl.config(text="🔴 Screen captured. Recording audio... Press 'C' to send.", fg="#FF5555")
            
        elif self.recording_mode == 'c':
            self.status_lbl.config(text="⏳ Analyzing screen and thinking...", fg="#00E5FF")
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
            
        self.root.after(0, lambda: self.status_lbl.config(text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", fg="#8B949E"))

    def process_audio_and_capture(self, image_pil):
        question_text = transcribe_audio(RECORD_FILENAME)
        
        if question_text:
            prompt = f"The interviewer asked this question: '{question_text}'. Please look at the provided screenshot and answer."
            display_text = f"📷 (Screenshot Attached)\n\"{question_text}\""
        else:
            prompt = "The screenshot shows a coding problem or environment. Please analyze it and provide a solution."
            display_text = "📷 (Screenshot Attached)\n[No verbal question detected]"

        self.root.after(0, self.log_chat, "Interviewer", display_text, 'interviewer')
        answer = self.ai.ask(prompt, image_pil=image_pil)
        self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
        
        self.root.after(0, lambda: self.status_lbl.config(text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", fg="#8B949E"))

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