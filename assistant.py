import os
import sys
import time
import base64
import threading
import platform
import ctypes
import json
import queue
from pathlib import Path
from io import BytesIO

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import requests
import mss
from PIL import Image
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
        self.active_sessions = {}

    def get_or_create_session(self, model_name: str) -> str:
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
            return ""
        
        response_data = response.json()
        if response_data.get("status") and "sessionId" in response_data:
            self.active_sessions[model_name] = response_data["sessionId"]
            return self.active_sessions[model_name]
        return ""

    def ask_quick_intro(self, question: str) -> str:
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
                "model": "ChatGPT4o-mini",
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

    def ask(self, question: str, image_pil=None, chunk_callback=None) -> str:
        try:
            if image_pil is None:
                current_model = "ChatGPT4o-mini" 
            else:
                current_model = "ChatGPT4o-mini" 
                print(f"📷 Image detected! Routing to {current_model}...")

            session_id = self.get_or_create_session(current_model)
            
            if image_pil is not None and session_id:
                self.upload_image(session_id, image_pil)
                time.sleep(1.5) 

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
            
            payload = {
                "userId": self.user_id,
                "model": current_model,
                "datastoreId": self.datastore_id,
                "prompt": full_prompt,
                "stream": chunk_callback is not None # 🔴 NEW: Tell API to stream if callback exists
            }
            if session_id:
                payload["sessionId"] = session_id

            # 🔴 NEW: stream parameter added to requests
            response = self.http.post(self.chat_url, headers=headers, json=payload, timeout=(15, 300), stream=(chunk_callback is not None))
            
            if response.status_code != 200:
                if response.status_code in [401, 403, 404]:
                    if current_model in self.active_sessions:
                        del self.active_sessions[current_model]
                return f"API error {response.status_code}: {response.text[:300]}"
            
            # 🔴 NEW: Streaming chunk processor
            if chunk_callback:
                full_answer = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            decoded_line = decoded_line[6:]
                        if decoded_line == '[DONE]':
                            break
                        try:
                            data = json.loads(decoded_line)
                            chunk = ""
                            
                            # Safely extract text depending on how the gateway formats it
                            if "choices" in data and len(data["choices"]) > 0:
                                chunk = data["choices"][0].get("delta", {}).get("content", "")
                            elif "data" in data and isinstance(data["data"], dict) and "text" in data["data"]:
                                chunk = data["data"]["text"]
                            elif "text" in data:
                                chunk = data["text"]
                                
                            if chunk:
                                # Smart append: prevents doubling up if API sends cumulative string
                                if chunk.startswith(full_answer) and len(chunk) > len(full_answer):
                                    new_text = chunk[len(full_answer):]
                                    full_answer = chunk
                                    chunk_callback(new_text)
                                else:
                                    full_answer += chunk
                                    chunk_callback(chunk)
                        except json.JSONDecodeError:
                            pass
                return full_answer
            else:
                return self.extract_text_from_response(response.json())
        except Exception as e:
            return f"Request failed: {str(e)}"
        
# ----------------------------------------------------------------------
# NEW: Real-Time Vosk Audio Recorder
# ----------------------------------------------------------------------
class AudioRecorder:
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.q = queue.Queue()
        self.recording = False
        self._thread = None
        self.actual_samplerate = 16000
        self.on_text_update = None # UI callback
        
        print(f"Loading Vosk model from '{model_path}'... This takes a second.", flush=True)
        try:
            self.model = Model(model_path)
            print("✅ Vosk model loaded successfully!", flush=True)
        except Exception as e:
            print(f"❌ Failed to load Vosk model: {e}")
            print(f"Make sure you downloaded the model and extracted it to the folder '{model_path}' next to this script.")
            sys.exit(1)

        self.final_text = ""
        self.live_text = ""

    def record(self):
        if self.recording: return
        
        # Clear any old audio bits left in the queue
        while not self.q.empty():
            self.q.get_nowait()
            
        self.recording = True
        self.final_text = ""
        self.live_text = ""
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.recording: return ""
        self.recording = False
        if self._thread:
            self._thread.join()
            
        # Combine the confirmed text + any trailing partial text
        full_text = (self.final_text + " " + self.live_text).strip()
        return full_text

    def _record_loop(self):
        try:
            # 🔴 CRITICAL FIX: Force exactly 16000 Hz instead of your system's 44100 Hz
            self.actual_samplerate = 16000 
            rec = KaldiRecognizer(self.model, self.actual_samplerate)

            def callback(indata, frames, time_info, status):
                if status:
                    pass 
                if self.recording:
                    # Convert raw audio bytes to measure volume
                    audio_data = np.frombuffer(indata, dtype=np.int16)
                    rms_volume = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                    
                    # 🔴 NOISE GATE: Based on your test, your room is < 5.0 and your voice is > 1000.
                    # Anything under 50 is ignored to prevent hallucinations.
                    if rms_volume > 50.0:
                        self.q.put(bytes(indata))

            # Open stream specifically locked to 16000 Hz
            with sd.RawInputStream(samplerate=self.actual_samplerate, blocksize=8000, 
                                   dtype='int16', channels=1, callback=callback):
                while self.recording:
                    try:
                        # Grab audio chunks from microphone
                        data = self.q.get(timeout=0.1)
                        if rec.AcceptWaveform(data):
                            # A full phrase was completed
                            result = json.loads(rec.Result())
                            text = result.get("text", "")
                            if text:
                                self.final_text += text + " "
                                if self.on_text_update:
                                    self.on_text_update(self.final_text)
                        else:
                            # A phrase is currently being spoken (partial/live)
                            partial = json.loads(rec.PartialResult())
                            partial_text = partial.get("partial", "")
                            if partial_text:
                                self.live_text = partial_text
                                if self.on_text_update:
                                    self.on_text_update(self.final_text + partial_text)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"\n❌ FATAL MICROPHONE ERROR: {e}", flush=True)
            self.recording = False

# ----------------------------------------------------------------------
# GUI Application Main Class
# ----------------------------------------------------------------------
class InterviewAssistantApp:
    def __init__(self, root, ai, recorder):
        self.root = root
        self.root.title("Service Host: Windows Equatorial Input")
        self.root.geometry("440x650-20+20") 
        self.root.wm_attributes("-topmost", True)
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
        
        # Link the recorder to update our UI in real-time
        self.recorder.on_text_update = self.update_live_transcription
        
        self.recording_mode = None  
        self.last_screenshot = None
        self.is_ghost_mode = False 
        
        self.image_buffer = []
        self.root.bind('<a>', self.add_to_image_buffer)
        
        self._offsetx = 0
        self._offsety = 0

        self.setup_ui()

        self.root.bind('<Button-1>', self.click_window)
        self.root.bind('<B1-Motion>', self.drag_window)
        
        self.root.bind('<r>', self.toggle_audio_record)
        self.root.bind('<c>', self.toggle_capture_record)
        self.root.bind('<t>', self.toggle_ghost_mode) 
        self.root.bind('<q>', self.on_closing)
        
        # 🔴 NEW: Press Escape to stop typing and re-enable shortcuts
        self.root.bind('<Escape>', lambda e: self.root.focus_set())
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_live_transcription(self, text):
        """Called constantly by the audio thread while you speak."""
        display_text = f"🎙️ {text}..." if text else "🎙️ Listening..."
        # Update the UI status label securely from the main thread
        self.root.after(0, lambda: self.status_lbl.config(text=display_text, fg="#00FF00"))

    def click_window(self, event):
        # 🔴 NEW: Added "Entry" and "Button" so it doesn't steal focus when you click the text box!
        if event.widget.winfo_class() in ["Text", "Scrollbar", "Entry", "Button"]:
            return
            
        # If you click the background, drop focus from the entry box
        self.root.focus_set()
        
        self._offsetx = event.x
        self._offsety = event.y

    def drag_window(self, event):
        if event.widget.winfo_class() in ["Text", "Scrollbar"]:
            return
        x = self.root.winfo_pointerx() - self._offsetx
        y = self.root.winfo_pointery() - self._offsety
        self.root.geometry(f"+{x}+{y}")

    def toggle_ghost_mode(self, event=None):
        if event and event.widget.winfo_class() == 'Entry': return
        if self.is_ghost_mode:
            self.root.attributes('-alpha', 0.92)
            self.is_ghost_mode = False
        else:
            self.root.attributes('-alpha', 0.15) 
            self.is_ghost_mode = True

    def copy_text(self, event=None):
        try:
            selected_text = self.chat_display.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.status_lbl.config(text="✅ Copied to clipboard!", fg="#00FF00")
        except tk.TclError:
            pass
        return "break"

    def add_to_image_buffer(self, event=None):
        if event and event.widget.winfo_class() == 'Entry': return
        if self.recording_mode is not None: return

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

        self.image_buffer.append(pil_img)

        self.root.attributes('-alpha', 0.15 if self.is_ghost_mode else 0.85)
        self.status_lbl.config(text=f"📸 Buffered {len(self.image_buffer)} image(s). Scroll & press 'A' again, or 'C' to send.", fg="#00FF00")

    def setup_ui(self):
        BG_COLOR = "#282C3A"        
        CYAN = "#00FFFF"            
        WHITE = "#FFFFFF"           
        GREY_BLUE = "#AAB4C8"       
        
        main_font = ("Segoe UI", 12) 
        bold_font = ("Segoe UI", 12, "bold")
        name_font = ("Segoe UI", 10, "bold")
        code_font = ("Consolas", 11)

        self.text_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.status_lbl = tk.Label(
            self.text_frame, 
            text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", 
            fg=GREY_BLUE, bg=BG_COLOR, font=("Segoe UI", 10),
            justify=tk.LEFT, anchor="w", pady=10, wraplength=400
        )
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X)

        self.chat_display = scrolledtext.ScrolledText(
            self.text_frame, wrap=tk.WORD, font=main_font, 
            bg=BG_COLOR, fg=WHITE, state=tk.DISABLED, 
            bd=0, highlightthickness=0, insertbackground=WHITE
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display.tag_config('ai_name', foreground=CYAN, font=name_font)
        self.chat_display.tag_config('ai_text', foreground=WHITE, font=main_font, spacing3=5)
        self.chat_display.tag_config('cyan_highlight', foreground=CYAN, font=bold_font)
        self.chat_display.tag_config('interviewer_name', foreground=GREY_BLUE, font=name_font)
        self.chat_display.tag_config('interviewer_text', foreground=GREY_BLUE, font=main_font, spacing3=10)
        self.chat_display.tag_config('system', foreground='#E2B714', font=main_font, spacing3=10) 
        
        self.chat_display.tag_config('inline_code', foreground="#FFB86C", background="#222633", font=code_font)
        self.chat_display.tag_config('code_block', foreground="#DCDCAA", background="#0D1017", font=code_font, lmargin1=10, lmargin2=10)
        
        self.chat_display.bind("<Control-c>", self.copy_text)
        self.chat_display.bind("<Command-c>", self.copy_text) 

        self.input_frame = tk.Frame(self.text_frame, bg=BG_COLOR)
        self.input_frame.pack(fill=tk.X, pady=(10, 0))

        self.message_entry = tk.Entry(
            self.input_frame, 
            font=main_font, 
            bg="#1E2233", fg=WHITE, 
            insertbackground=WHITE, 
            relief=tk.FLAT
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 10))
        
        self.send_btn = tk.Button(
            self.input_frame, text="Send", 
            bg=CYAN, fg="#000000", 
            font=name_font,
            relief=tk.FLAT,
            command=self.send_text_message
        )
        self.send_btn.pack(side=tk.RIGHT, ipadx=10, ipady=2)
        
        self.message_entry.bind("<Return>", lambda e: self.send_text_message())

    def log_chat(self, speaker, text, tag):
        self.chat_display.config(state=tk.NORMAL)
        start_index = self.chat_display.index(tk.END + "-1c")

        if speaker == "AI Assistant":
            self.chat_display.insert(tk.END, f"✨ {speaker}\n", 'ai_name')
        elif speaker == "Interviewer":
            self.chat_display.insert(tk.END, f"👤 {speaker}\n", 'interviewer_name')
        else:
            self.chat_display.insert(tk.END, f"⚙️ {speaker}\n", 'system')

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

        self.chat_display.see(tk.END)
        self.chat_display.update_idletasks() 
        
        if speaker == "AI Assistant":
            self.chat_display.see(start_index)

        self.chat_display.config(state=tk.DISABLED)

    def toggle_audio_record(self, event=None):
        if event and event.widget.winfo_class() == 'Entry': return
        if self.recording_mode == 'c': return 

        if self.recording_mode is None:
            self.recording_mode = 'r'
            self.recorder.record()
            self.status_lbl.config(text="🎙️ Listening...", fg="#00FF00")
        
        elif self.recording_mode == 'r':
            self.status_lbl.config(text="⏳ Thinking...", fg="#00E5FF")
            # This instantly gives us the transcribed text
            final_text = self.recorder.stop()
            self.recording_mode = None
            threading.Thread(target=self.process_audio_only, args=(final_text,), daemon=True).start()

    def toggle_capture_record(self, event=None):
        if event and event.widget.winfo_class() == 'Entry': return
        if self.recording_mode == 'r': return 

        if self.recording_mode is None:
            self.recording_mode = 'c'
            
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
            
            self.image_buffer.append(final_img)

            widths, heights = zip(*(i.size for i in self.image_buffer))
            total_height = sum(heights)
            max_width = max(widths)

            stitched_image = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for im in self.image_buffer:
                stitched_image.paste(im, (0, y_offset))
                y_offset += im.size[1] 

            self.last_screenshot = stitched_image
            self.image_buffer = [] 

            self.root.attributes('-alpha', 0.15 if self.is_ghost_mode else 0.85)
            self.root.update()
            
            self.recorder.record()
            self.status_lbl.config(text="🔴 Screen(s) captured. Speak now... Press 'C' to send.", fg="#FF5555")
            
        elif self.recording_mode == 'c':
            self.status_lbl.config(text="⏳ Analyzing screen and thinking...", fg="#00E5FF")
            
            # This instantly gives us the transcribed text
            final_text = self.recorder.stop()
            self.recording_mode = None
            
            captured_image = self.last_screenshot
            threading.Thread(target=self.process_audio_and_capture, args=(captured_image, final_text), daemon=True).start()

    def process_audio_only(self, question_text):
        if question_text and question_text.strip():
            self.root.after(0, self.log_chat, "Interviewer", question_text, 'interviewer')
            self.root.after(0, lambda: self.status_lbl.config(text="⚡ Fetching Quick Intro...", fg="#00E5FF"))
            
            def fetch_quick_intro():
                intro = self.ai.ask_quick_intro(question_text)
                if intro and intro != "...":
                    self.root.after(0, self.log_chat, "AI Assistant", f"💡 *Quick thought:* {intro}", 'ai')
                    self.root.after(50, self.root.update)

            def fetch_deep_dive():
                answer = self.ai.ask(question_text)
                self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
                self.root.after(0, lambda: self.status_lbl.config(text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", fg="#8B949E"))

            threading.Thread(target=fetch_quick_intro, daemon=True).start()
            self.root.after(500, lambda: threading.Thread(target=fetch_deep_dive, daemon=True).start())

        else:
            self.root.after(0, self.log_chat, "System", "No speech detected. Please try again.", 'system')
            self.root.after(0, lambda: self.status_lbl.config(text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", fg="#8B949E"))

    def process_audio_and_capture(self, image_pil, question_text):
        if question_text and question_text.strip():
            prompt = f"The interviewer asked this question: '{question_text.strip()}'. Please look at the provided screenshot and answer."
            display_text = f"📷 (Screenshot Attached)\n\"{question_text.strip()}\""
        else:
            prompt = "The screenshot shows a coding problem or environment. Please analyze it and provide a solution."
            display_text = "📷 (Screenshot Attached)\n[No verbal question detected]"

        self.root.after(0, self.log_chat, "Interviewer", display_text, 'interviewer')
        answer = self.ai.ask(prompt, image_pil=image_pil)
        self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
        
        self.root.after(0, lambda: self.status_lbl.config(text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", fg="#8B949E"))

    def send_text_message(self):
        text = self.message_entry.get().strip()
        if not text: return
        self.message_entry.delete(0, tk.END)
        
        # 🔴 NEW: Remove focus from the text box so 'R' and 'C' work again!
        self.root.focus_set()
        
        self.log_chat("Interviewer", text, 'interviewer')
        self.status_lbl.config(text="⏳ Thinking...", fg="#00E5FF")
        
        threading.Thread(target=self._process_text_only, args=(text,), daemon=True).start()

    def _process_text_only(self, text):
        self.root.after(0, lambda: self.status_lbl.config(text="⚡ Fetching Quick Intro...", fg="#00E5FF"))
        
        def fetch_quick_intro():
            intro = self.ai.ask_quick_intro(text)
            if intro and intro != "...":
                self.root.after(0, self.log_chat, "AI Assistant", f"💡 *Quick thought:* {intro}", 'ai')
                self.root.after(50, self.root.update)

        def fetch_deep_dive():
            answer = self.ai.ask(text)
            self.root.after(0, self.log_chat, "AI Assistant", answer, 'ai')
            self.root.after(0, lambda: self.status_lbl.config(text="Type message...\n(Press 'R' for Audio | 'C' for Screenshot)", fg="#8B949E"))

        threading.Thread(target=fetch_quick_intro, daemon=True).start()
        self.root.after(500, lambda: threading.Thread(target=fetch_deep_dive, daemon=True).start())

    def on_closing(self, event=None):
        if event and event.widget.winfo_class() == 'Entry': return
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