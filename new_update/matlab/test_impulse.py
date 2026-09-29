import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from ml.audio.inference.predict import AudioPredictor

wav_path = 'datasets/audio/impulse/impulse_0001.wav'
sr_raw, data_raw = wavfile.read(wav_path)
if data_raw.dtype.kind == 'i':
    data_raw = data_raw.astype(np.float64) / np.iinfo(data_raw.dtype).max

print('=' * 60)
print('SPANDHAN Audio Classification: impulse_0001')
print('=' * 60)
print(f'Input File         : {wav_path}')
print(f'Original Sampling  : {sr_raw} Hz')
print(f'Original Length    : {len(data_raw)} samples ({len(data_raw)/sr_raw:.4f} s)')

# 1. Prediction on Raw WAV
predictor = AudioPredictor()
res_raw = predictor.predict_wav(wav_path)
print('\n[1] Inference on Input WAV File:')
print(f'  Detected Class   : {res_raw["class"].upper()}')
print(f'  Class ID         : {res_raw["class_id"]}')
print(f'  Confidence Score : {res_raw["confidence"]*100:.2f}%')
print('  Probabilities    :')
for k, v in res_raw['probabilities'].items():
    bar = '#' * int(v * 30)
    print(f'    {k:12s} : {v*100:6.2f}%  {bar}')

# 2. Simulated MATLAB Preprocessing (16kHz, 32000 samples, DC removal, normalize)
x = data_raw.mean(axis=1) if data_raw.ndim == 2 else data_raw
x = x - np.mean(x)
target_fs = 16000
target_samples = 32000
new_len = int(round(len(x) * target_fs / sr_raw))
x_res = resample(x, new_len)
if len(x_res) < target_samples:
    x_proc = np.pad(x_res, (0, target_samples - len(x_res)))
else:
    x_proc = x_res[:target_samples]
pk = np.max(np.abs(x_proc))
if pk > 0:
    x_proc = x_proc / pk

res_matlab = predictor.predict_signal(x_proc, target_fs)
print('\n[2] Inference on MATLAB Preprocessed Vector (16 kHz, 32,000 samples):')
print(f'  Preprocessed Size: {x_proc.shape} @ {target_fs} Hz')
print(f'  Detected Class   : {res_matlab["class"].upper()}')
print(f'  Class ID         : {res_matlab["class_id"]}')
print(f'  Confidence Score : {res_matlab["confidence"]*100:.2f}%')
print('  Probabilities    :')
for k, v in res_matlab['probabilities'].items():
    bar = '#' * int(v * 30)
    print(f'    {k:12s} : {v*100:6.2f}%  {bar}')

print('=' * 60)
