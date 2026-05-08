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
import mss
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
        self.api_key = os.getenv('MSI_API_KEY', "6HvdDU8.71Q(~C:Tb-jm2pKbdreja4qQ!Vu6YhBg")
        self.user_id = os.getenv('MSI_USER_ID', "bgvk38@motorolasolutions.com")
        self.datastore_id = os.getenv('MSI_DATASTORE_ID', "1579319e-2b48-4bad-9825-4a7dd10ac0ef")
        
        self.chat_url = self.host + "/chat"
        self.upload_url = self.host + "/upload"
 
        self.http = requests.Session()
        self.http.headers.update({
            "Connection": "keep-alive", 
            "Cache-Control": "no-cache", 
            "Pragma": "no-cache"
        })
        
        # --- FIX 1: Dictionary to hold separate sessions for GPT and Claude ---
        self.active_sessions = {}

    def get_or_create_session(self, model_name: str) -> str:
        """Creates a unique session for EACH model to prevent 403 Forbidden errors."""
        if model_name in self.active_sessions:
            return self.active_sessions[model_name]

        headers = {
            "Content-Type": "application/json",
            "x-msi-genai-api-key": self.api_key
        }
        payload = {
            "userId": self.user_id,
            "model": model_name,
            "datastoreId": self.datastore_id,
            "prompt": "init"
        }
        
        response = self.http.post(self.chat_url, headers=headers, json=payload, timeout=45)
        if response.status_code >= 400:
            print(f"Warning: Session init failed for {model_name}. Error {response.status_code}")
            return "" # Fallback gracefully
        
        response_data = response.json()
        if response_data.get("status") and "sessionId" in response_data:
            self.active_sessions[model_name] = response_data["sessionId"]
            return self.active_sessions[model_name]
        return ""

    def ask_quick_intro(self, question: str) -> str:
        # (Keep your highly optimized version here without datastoreId!)
        try:
            intro_prompt = (
                "Roleplay Context: You are a Software Engineer in a job interview. "
                "NEVER mention you are an AI. Speak naturally in the first person ('I', 'my').\n\n"
                f"=== YOUR RESUME ===\n{RESUME if RESUME else '(not provided)'}\n\n"
                f"Interviewer asked: '{question}'.\n"
                "Provide EXACTLY one short, conversational opening sentence to answer this. "
                "Keep it under 15 words. No code. No pleasantries."
            )
            headers = {
                "Content-Type": "application/json",
                "x-msi-genai-api-key": self.api_key
            }
            payload = {
                "userId": self.user_id,
                "model": "ChatGPT4o-mini", # Super fast model for intro
                "prompt": intro_prompt,
                "stream": False 
            }
            response = requests.post(self.chat_url, headers=headers, json=payload, timeout=5)
            if response.status_code == 200:
                return self.extract_text_from_response(response.json())
        except Exception:
            pass
        return "..."

    def upload_image(self, session_id: str, image_pil) -> bool:
        # (Keep your existing upload_image code here exactly as it is)
        headers = {"x-msi-genai-api-key": self.api_key}
        image_pil = image_pil.convert('RGB')
        
        w, h = image_pil.size
        max_dim = 4096 
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image_pil = image_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = BytesIO()
        image_pil.save(buf, format="PNG") 
        image_bytes = buf.getvalue()
        
        url = f"{self.upload_url}/{session_id}?userId={self.user_id}"
        files = {"file": ("image.png", image_bytes, "image/png")} 
        
        response = self.http.post(url, headers=headers, files=files, timeout=45)
        if response.status_code != 200:
            raise RuntimeError(f"Upload failed {response.status_code}")
        return True

    def extract_text_from_response(self, response_data: dict) -> str:
        # (Keep your existing extract_text_from_response code here)
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

    def ask(self, question: str, image_pil=None) -> str:
        try:
            # --- FIX 2: Dynamic Model Routing ---
            if image_pil is None:
                # 'R' pressed (Audio Only) -> Use GPT
                current_model = "ChatGPT4o-mini" 
            else:
                # 'C' pressed (Image Capture) -> Use Claude
                current_model = "ChatGPT4o-mini" 
                print(f"📷 Image detected! Routing to {current_model}...")

            # Get the correct session for the chosen model to avoid 403 Forbidden
            session_id = self.get_or_create_session(current_model)
            
            if image_pil is not None and session_id:
                self.upload_image(session_id, image_pil)
                import time
                time.sleep(1.5) # Give gateway time to process the image

            full_prompt = (
                "- CRITICAL: The user is speaking into a microphone. ALWAYS correct phonetic typos in your head before answering.\n"
                "Roleplay Context: You are a Senior Software Engineer taking a technical interview. "
                "Speak naturally out loud to the interviewer, but be highly technical and extremely thorough. "
                "Please stay in character as a human engineer talking out loud and avoid referencing yourself as an AI.\n\n"
                
                "Problem Solving Instructions (STRICT ADHERENCE REQUIRED):\n"
                "1. If the user gives you a coding scenario, DO NOT jump straight to writing code.\n"
                "2. DATA DRY-RUN (MANDATORY): explicitly write out a 'dry-run' of the provided example input. Map out exact array indices, variable values, and edge cases out loud.\n"
                "3. Follow the requirements literally.\n"
                "4. Provide the full code solution wrapped in triple backticks (```[language] ... ```).\n"
                "5. After the code, briefly explain the Time Complexity (Big-O) and Space Complexity of your solution.\n"
                "6. MULTI-PART QUESTIONS (CRITICAL): Read the user's prompt carefully to the very end and answer follow-up questions.\n\n"
                
                f"=== CANDIDATE RESUME ===\n{RESUME if RESUME else '(not provided)'}\n\n"
                f"=== JOB DESCRIPTION ===\n{JOB_DESC if JOB_DESC else '(not provided)'}\n\n"
                f"USER QUESTION: {question}"
            )

            headers = {
                "Content-Type": "application/json",
                "x-msi-genai-api-key": self.api_key
            }
            
            # If session_id failed to generate, we pass it without a sessionId
            payload = {
                "userId": self.user_id,
                "model": current_model,
                "datastoreId": self.datastore_id,
                "prompt": full_prompt,
                "stream": False 
            }
            if session_id:
                payload["sessionId"] = session_id

            # --- FIX 3: INCREASE TIMEOUT TO 300 SECONDS (5 MINUTES) ---
            # timeout=(connect_timeout, read_timeout)
            # This ensures the connection stays open while the AI writes massive code blocks!
            response = self.http.post(self.chat_url, headers=headers, json=payload, timeout=(15, 300))
            
            if response.status_code != 200:
                if response.status_code in [401, 403, 404]:
                    # Clear the bad session so it recreates next time
                    if current_model in self.active_sessions:
                        del self.active_sessions[current_model]
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
        self.actual_samplerate = 16000 # Default, will be updated dynamically

    def get_audio_buffer(self):
        if not self.frames: 
            print("❌ Error: No audio frames captured. Microphone thread may have crashed.", flush=True)
            return None
            
        audio = np.concatenate(self.frames, axis=0)
        audio_int16 = np.int16(audio * 32767)
        
        buf = BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(int(self.actual_samplerate))
            wf.writeframes(audio_int16.tobytes())
        buf.seek(0)
        return buf

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
        try:
            # 1. Ask Windows what the default microphone is and its required sample rate
            device_info = sd.query_devices(None, 'input')
            self.actual_samplerate = int(device_info['default_samplerate'])
            print(f"\n✅ USING MICROPHONE: {device_info['name']} @ {self.actual_samplerate}Hz", flush=True)

            # 2. Callback to store audio frames and print a live volume meter
            def callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio Status Warning: {status}", flush=True)
                if self.recording:
                    self.frames.append(indata.copy())
                    
                    # Optional: Print a tiny volume meter in the terminal to prove it hears you
                    volume_norm = np.linalg.norm(indata) * 10
                    if volume_norm > 1.0:
                        print("🔊" + "|" * int(volume_norm), flush=True)

            # 3. Start listening using the dynamic sample rate
            with sd.InputStream(samplerate=self.actual_samplerate, channels=AUDIO_CHANNELS,
                                callback=callback, dtype='float32'):
                while self.recording:
                    sd.sleep(100)
                    
        except Exception as e:
            print(f"\n❌ FATAL MICROPHONE ERROR: {e}", flush=True)
            self.recording = False

    def _save(self):
        if not self.frames: return
        audio = np.concatenate(self.frames, axis=0)
        audio_int16 = np.int16(audio * 32767)
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(int(self.actual_samplerate))
            wf.writeframes(audio_int16.tobytes())
# ----------------------------------------------------------------------
# Speech‑to‑text
# ----------------------------------------------------------------------
# Replace your current transcribe_audio function with this
def transcribe_audio(audio_source, language="en-US"):
    recognizer = sr.Recognizer()
    try:
        # Check if it's a memory buffer or a string filename
        if isinstance(audio_source, BytesIO):
            with sr.AudioFile(audio_source) as source:
                audio_data = recognizer.record(source)
        else:
            with sr.AudioFile(audio_source) as source:
                audio_data = recognizer.record(source)
                
        # NOTE: For sub-second transcription, consider replacing recognize_google 
        # with a faster cloud API (like Groq Whisper) or a local model (faster-whisper)
        return recognizer.recognize_google(audio_data, language=language)
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# ----------------------------------------------------------------------
# GUI Application Main Class
# ----------------------------------------------------------------------
class InterviewAssistantApp:
    def __init__(self, root, ai, recorder):
        self.root = root
        self.root.title("Service Host: Windows Equatorial Input")
        self.root.geometry("440x650-20+20") 
        self.root.wm_attributes("-topmost", True)
        
        # --- NEW: Remove the native Windows title bar ---
        self.root.overrideredirect(True)
         
        
        self.root.attributes('-alpha', 0.92)
        self.root.config(bg="#282C3A")
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
        self.is_ghost_mode = False 
        
        
        self.image_buffer = []
        self.root.bind('<a>', self.add_to_image_buffer) # Press A to queue an image 
        
        # --- NEW: Variables to track dragging ---
        self._offsetx = 0
        self._offsety = 0
        

        self.setup_ui()

        # --- NEW: Bind mouse clicks for custom dragging ---
        self.root.bind('<Button-1>', self.click_window)
        self.root.bind('<B1-Motion>', self.drag_window)

        
        self.root.bind('<r>', self.toggle_audio_record)
        self.root.bind('<c>', self.toggle_capture_record)
        self.root.bind('<t>', self.toggle_ghost_mode) 
        self.root.bind('<q>', self.on_closing)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _start_stream_ui(self, speaker):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"✨ {speaker}\n", 'ai_name')
        
        # Mark where the stream starts so we can re-format it later
        self.stream_start_index = self.chat_display.index(tk.END)
        self.chat_display.see(tk.END)
        self.chat_display.update_idletasks()

    def _append_stream_chunk(self, chunk):
        # Append plain text in real-time
        self.chat_display.insert(tk.END, chunk, 'ai_text')
        self.chat_display.see(tk.END)
        
        # Hard force the UI to draw this exact word right now
        self.chat_display.update()

    def _finalize_stream_ui(self, full_text):
        # Optional: Once the stream is done, erase the plain text and run it 
        # through your existing Markdown/Code highlighter logic so it looks pretty!
        self.chat_display.delete(self.stream_start_index, tk.END)
        self.log_chat("AI Assistant", full_text, 'ai') # Re-uses your markdown formatter


    def click_window(self, event):
        # Ignore dragging if the user clicks the text box or scrollbar
        if event.widget.winfo_class() in ["Text", "Scrollbar"]:
            return
        
        # Record the exact spot you clicked inside the window
        self._offsetx = event.x
        self._offsety = event.y

    def drag_window(self, event):
        # Ignore dragging if the user clicks the text box or scrollbar
        if event.widget.winfo_class() in ["Text", "Scrollbar"]:
            return
            
        # Calculate new position and move the window smoothly
        x = self.root.winfo_pointerx() - self._offsetx
        y = self.root.winfo_pointery() - self._offsety
        self.root.geometry(f"+{x}+{y}")

    def toggle_ghost_mode(self, event=None):
        if self.is_ghost_mode:
            self.root.attributes('-alpha', 0.92) # Higher number = more solid/readable # Normal opacity
            self.is_ghost_mode = False
        else:
            self.root.attributes('-alpha', 0.15) # Almost fully transparent
            self.is_ghost_mode = True

    def copy_text(self, event=None):
        try:
            # Grab the text that is currently highlighted
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            # Clear the clipboard and append the new text
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.status_lbl.config(text="✅ Copied to clipboard!", fg="#00FF00")
        except tk.TclError:
            # This happens if you press Ctrl+C without highlighting anything
            pass
        return "break"

    def add_to_image_buffer(self, event=None):
        if self.recording_mode is not None: return

        # Hide UI, capture, bring back
        self.root.attributes('-alpha', 0.0)
        self.root.update()
        time.sleep(0.15)

        win_x = self.root.winfo_x()
        win_y = self.root.winfo_y()

        with mss.mss() as sct:
            target_monitor = sct.monitors[1]
            for monitor in sct.monitors[1:]:
                if (monitor["left"] <= win_x < monitor["left"] + monitor["width"] and 
                    monitor["top"] <= win_y < monitor["top"] + monitor["height"]):
                    target_monitor = monitor
                    break
            sct_img = sct.grab(target_monitor)
            pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # Add image to our list
        self.image_buffer.append(pil_img)

        self.root.attributes('-alpha', 0.15 if self.is_ghost_mode else 0.85)
        self.status_lbl.config(text=f"📸 Buffered {len(self.image_buffer)} image(s). Scroll & press 'A' again, or 'C' to send.", fg="#00FF00")

    def setup_ui(self):
        # --- NEW LIGHTER & HIGH-CONTRAST COLORS ---
        BG_COLOR = "#282C3A"        # Lighter, softer grey (was almost black)
        CYAN = "#00FFFF"            # Brighter cyan for highlights
        WHITE = "#FFFFFF"           # Pure white for maximum readability
        GREY_BLUE = "#AAB4C8"       # Brighter grey for interviewer text
        
        main_font = ("Segoe UI", 12) # Bumped font size up from 11 to 12
        bold_font = ("Segoe UI", 12, "bold")
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
        # Inline code: brighter orange, lighter background
        self.chat_display.tag_config('inline_code', foreground="#FFC785", background="#3A3F58", font=code_font)
        # Code block: lighter indented background
        self.chat_display.tag_config('code_block', foreground="#DCDCAA", background="#1E2233", font=code_font, lmargin1=10, lmargin2=10)
        
        # --- NEW CODE FORMATTING TAGS ---
        # Inline code: slight background, distinct color
        self.chat_display.tag_config('inline_code', foreground="#FFB86C", background="#222633", font=code_font)
        # Code block: darker indented background, standard IDE yellow/green text
        self.chat_display.tag_config('code_block', foreground="#DCDCAA", background="#0D1017", font=code_font, lmargin1=10, lmargin2=10)
        # --- ENABLE COPYING ---
        self.chat_display.bind("<Control-c>", self.copy_text)
        self.chat_display.bind("<Command-c>", self.copy_text) # For Mac users

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
            
            # --- Hide UI to take the final screenshot ---
            self.root.attributes('-alpha', 0.0) 
            self.root.update()
            time.sleep(0.15) 
            
            win_x = self.root.winfo_x()
            win_y = self.root.winfo_y()

            with mss.mss() as sct:
                target_monitor = sct.monitors[1]
                for monitor in sct.monitors[1:]:
                    if (monitor["left"] <= win_x < monitor["left"] + monitor["width"] and 
                        monitor["top"] <= win_y < monitor["top"] + monitor["height"]):
                        target_monitor = monitor
                        break
                sct_img = sct.grab(target_monitor)
                final_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Add the final image to the buffer
            self.image_buffer.append(final_img)

            # --- STITCH IMAGES VERTICALLY ---
            widths, heights = zip(*(i.size for i in self.image_buffer))
            total_height = sum(heights)
            max_width = max(widths)

            stitched_image = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for im in self.image_buffer:
                stitched_image.paste(im, (0, y_offset))
                y_offset += im.size[1] # move down for the next image

            # Save the stitched image and clear the buffer for next time
            self.last_screenshot = stitched_image
            self.image_buffer = [] 

            # --- Bring UI back ---
            self.root.attributes('-alpha', 0.15 if self.is_ghost_mode else 0.85)
            self.root.update()
            
            self.recorder.record()
            self.status_lbl.config(text="🔴 Screen(s) captured. Recording audio... Press 'C' to send.", fg="#FF5555")
            
        elif self.recording_mode == 'c':
            self.status_lbl.config(text="⏳ Analyzing screen and thinking...", fg="#00E5FF")
            self.recorder.stop()
            self.recording_mode = None
            
            captured_image = self.last_screenshot
            threading.Thread(target=self.process_audio_and_capture, args=(captured_image,), daemon=True).start()

    def process_audio_only(self):
        # Let the user know the audio is being transcribed (takes ~2-3 secs)
        self.root.after(0, lambda: self.status_lbl.config(text="🎙️ Transcribing audio...", fg="#E2B714"))
        
        audio_buffer = self.recorder.get_audio_buffer()
        question_text = transcribe_audio(audio_buffer)
        
        if question_text:
            self.root.after(0, self.log_chat, "Interviewer", question_text, 'interviewer')
            self.root.after(0, lambda: self.status_lbl.config(text="⚡ Fetching Quick Intro...", fg="#00E5FF"))
            
            # --- TWO-STEP PARALLEL EXECUTION ---
            
            def fetch_quick_intro():
                # This should now finish in ~1 to 2 seconds flat
                intro = self.ai.ask_quick_intro(question_text)
                if intro and intro != "...":
                    # Put it on screen
                    self.root.after(0, self.log_chat, "AI Assistant", f"💡 *Quick thought:* {intro}", 'ai')
                    
                    # 🛑 TRICK 3: Force the UI window to hard-refresh immediately!
                    self.root.after(50, self.root.update)

            def fetch_deep_dive():
                # Does the heavy database search + full code generation
                answer = self.ai.ask(question_text)
                self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
                self.root.after(0, lambda: self.status_lbl.config(text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", fg="#8B949E"))

            # Start the quick intro immediately
            threading.Thread(target=fetch_quick_intro, daemon=True).start()
            
            # Give the gateway a tiny breather before slamming it with the heavy request
            self.root.after(500, lambda: threading.Thread(target=fetch_deep_dive, daemon=True).start())

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