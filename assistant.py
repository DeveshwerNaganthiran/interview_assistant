import os
import sys
import time
import base64
import wave
import threading
import json
from pathlib import Path
from io import BytesIO


import cv2
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import requests
from PIL import Image
from dotenv import load_dotenv


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
CAMERA_FACE   = None   # webcam for your face (optional, can be None)


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
    """
    Handles communication with the MSI GenAI service.
    Supports text-only and image+text messages.
    """
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
            # Add any other required headers your MSI instance expects
        }
        self.system_prompt = self._build_system_prompt()


    def _build_system_prompt(self) -> str:
        """Create the main system instruction from resume and JD."""
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
        """
        Send a question (text) plus optional image to the model.
        Returns the model's answer as a string.
        """
        # Build message content list
        content = [{"type": "text", "text": question}]


        # If an image is provided, encode it as base64 data URL
        if image_bgr is not None:
            # Convert BGR (OpenCV) to RGB PIL
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
# Audio recorder (start/stop via threading)
# ----------------------------------------------------------------------
class AudioRecorder:
    """Records audio to a WAV file when activated."""
    def __init__(self, filename=RECORD_FILENAME):
        self.filename = filename
        self.frames = []
        self.recording = False
        self._thread = None


    def record(self):
        """Start recording in a background thread."""
        if self.recording:
            return
        self.frames = []
        self.recording = True
        self._thread = threading.Thread(target=self._record_loop)
        self._thread.start()


    def stop(self):
        """Stop recording and save the file."""
        if not self.recording:
            return
        self.recording = False
        if self._thread:
            self._thread.join()
        self._save()


    def _record_loop(self):
        """Internal loop that captures audio chunks."""
        def callback(indata, frames, time_info, status):
            if self.recording:
                self.frames.append(indata.copy())


        with sd.InputStream(samplerate=AUDIO_SAMPLERATE, channels=AUDIO_CHANNELS,
                            callback=callback, dtype='float32'):
            while self.recording:
                sd.sleep(100)


    def _save(self):
        """Write recorded audio to WAV file."""
        if not self.frames:
            print("No audio recorded.")
            return
        audio = np.concatenate(self.frames, axis=0)
        # Convert float32 [-1, 1] to int16
        audio_int16 = np.int16(audio * 32767)
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(AUDIO_SAMPLERATE)
            wf.writeframes(audio_int16.tobytes())
        print(f"Audio saved to {self.filename}")


# ----------------------------------------------------------------------
# Speech‑to‑text using Google Web Speech (free, requires internet)
# ----------------------------------------------------------------------
def transcribe_audio(filename, language="en-US"):
    """Transcribe WAV file to text. Returns empty string on failure."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Speech service error: {e}")
        return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("INTERVIEW ASSISTANT")
    print("Press 'r' to start/stop recording a question")
    print("Press 'c' to capture the paper/screen (no voice)")
    print("Press 'q' to quit")
    print("=" * 60)


    # Initialize components
    ai = InterviewAI()
    recorder = AudioRecorder()


    # Open webcams
    cap_paper = cv2.VideoCapture(CAMERA_PAPER, cv2.CAP_DSHOW)
    cap_face = None
    if CAMERA_FACE is not None and CAMERA_FACE >= 0:
        cap_face = cv2.VideoCapture(CAMERA_FACE, cv2.CAP_DSHOW)


    if not cap_paper.isOpened():
        print("Could not open paper camera.")
        sys.exit(1)


    # State
    recording = False
    last_paper_frame = None


    try:
        while True:
            # Read paper camera (always)
            ret_paper, frame_paper = cap_paper.read()
            if not ret_paper:
                print("Paper camera lost.")
                break
            last_paper_frame = frame_paper.copy()


            # Show paper camera window
            cv2.imshow("Paper / Problem View", frame_paper)


            # Show face camera if available
            if cap_face and cap_face.isOpened():
                ret_face, frame_face = cap_face.read()
                if ret_face:
                    cv2.imshow("Your Face", frame_face)


            key = cv2.waitKey(1) & 0xFF


            # ---- Recording toggle ----
            if key == ord('r'):
                if not recording:
                    recording = True
                    recorder.record()
                    print("[REC] Recording started... Press 'r' again to stop.")
                else:
                    recording = False
                    recorder.stop()
                    print("[REC] Recording stopped.")


                    # Transcribe the recorded question
                    question_text = transcribe_audio(RECORD_FILENAME)
                    if question_text:
                        print(f"[Q] {question_text}")
                        # Ask the AI (no image)
                        answer = ai.ask(question_text)
                        print(f"[A] {answer}\n")
                    else:
                        print("No question detected. Try again.")


            # ---- Capture paper / screen (no voice) ----
            if key == ord('c'):
                print("[CAP] Capturing problem from camera...")
                question = "The image shows a problem I need to solve. Please read it and provide a solution."
                answer = ai.ask(question, image_bgr=last_paper_frame)
                print(f"[A] {answer}\n")


            # Quit
            if key == ord('q'):
                break


    finally:
        cap_paper.release()
        if cap_face:
            cap_face.release()
        cv2.destroyAllWindows()
        print("Assistant closed.")


if __name__ == "__main__":
    main()
