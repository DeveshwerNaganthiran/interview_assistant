import sounddevice as sd
import numpy as np

# 1. Check which microphone Python thinks is the default
default_mic = sd.query_devices(None, 'input')
print(f"\n🎤 Python is trying to use: {default_mic['name']}")
print(f"Sample Rate: {default_mic['default_samplerate']} Hz\n")

def callback(indata, frames, time, status):
    if status:
        print(f"Status Warning: {status}")
    # Calculate and print the volume
    audio_data = np.frombuffer(indata, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
    print(f"Mic Volume: {rms:.2f}", flush=True)

print("Listening... Speak into your mic! (Press Ctrl+C to stop)")

try:
    # 2. Start a simple raw stream
    with sd.RawInputStream(channels=1, dtype='int16', callback=callback):
        sd.sleep(10000)  # Listen for 10 seconds
except Exception as e:
    print(f"\n❌ ERROR STARTING STREAM: {e}")