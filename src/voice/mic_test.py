"""
Microphone Diagnostic — run this to check your mic levels.
Usage: python -m src.voice.mic_test
"""
import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 16000
CHUNK_SECS  = 0.1
DURATION    = 5  # seconds to test

print("=" * 60)
print("  PANACEA — Microphone Diagnostic")
print("=" * 60)

# Show available devices
print("\nAvailable audio devices:")
print(sd.query_devices())
print(f"\nDefault input device: {sd.query_devices(kind='input')['name']}")
print(f"\nRecording for {DURATION} seconds... SPEAK NOW!\n")

chunk_samples = int(SAMPLE_RATE * CHUNK_SECS)
rms_values = []

def callback(indata, frames, time_info, status):
    rms = float(np.sqrt(np.mean(indata ** 2)))
    rms_values.append(rms)
    # Visual bar
    bar_len = int(min(rms * 500, 50))
    bar = "█" * bar_len
    label = "SPEECH" if rms > 0.03 else "QUIET " if rms > 0.01 else "SILENT"
    print(f"  RMS={rms:.4f}  [{label}]  {bar}", flush=True)

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                     blocksize=chunk_samples, callback=callback):
    time.sleep(DURATION)

print("\n" + "=" * 60)
print(f"  Results ({len(rms_values)} samples)")
print(f"  Min RMS:  {min(rms_values):.5f}")
print(f"  Max RMS:  {max(rms_values):.5f}")
print(f"  Mean RMS: {np.mean(rms_values):.5f}")
print(f"  Ambient (first 10 chunks): {np.mean(rms_values[:10]):.5f}")
print("=" * 60)

ambient = np.mean(rms_values[:10])
recommended = round(ambient * 2.5, 4)
print(f"\n  Recommended SILENCE_THRESH for your mic: {recommended}")
print(f"  Current threshold: 0.03")

if max(rms_values) < 0.01:
    print("\n  ⚠️  Your microphone may not be working or is muted!")
    print("     Check Windows Settings → Privacy → Microphone")
elif max(rms_values) < 0.03:
    print(f"\n  ⚠️  Your speech peaks at {max(rms_values):.4f} which is BELOW 0.03!")
    print(f"     Set SILENCE_THRESH={recommended} in your .env file")
    print(f"     OR use the auto-calibration fix (already being applied)")
else:
    print(f"\n  ✅  Your mic looks good! Speech peaks at {max(rms_values):.4f}")
