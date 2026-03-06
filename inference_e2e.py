from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
import librosa

# -----------------------------------------------------------------------------
# Safe cuDNN settings
# -----------------------------------------------------------------------------
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h = None

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]
    
def get_mel_from_audio(wav_path, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
    """
    Calculate mel spectrogram from a wav file and display it
    """
    y, orig_sr = librosa.load(wav_path, sr=None)
    if orig_sr != sr:
        print(orig_sr)
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels
    )
    mel_spec = np.log(np.clip(mel_spec, 1e-5, None))
    
    return mel_spec

# -----------------------------------------------------------------------------
# Inference function
# -----------------------------------------------------------------------------
def inference(a):
    torch.cuda.empty_cache()
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    os.makedirs(a.output_dir, exist_ok=True)
    filelist = sorted(os.listdir(a.input_mels_dir))

    with torch.no_grad():
        for filname in filelist:
            # Load MEL
            # for wav->mel->wive pipeline:
            x = get_mel_from_audio(os.path.join(a.input_mels_dir, filname))
            
            # x = np.load(os.path.join(a.input_mels_dir, filname)).astype(np.float32)
            # x = x.T[np.newaxis, :, :]  # [1, n_mels, T]
            x = x[np.newaxis, :, :]  # [1, n_mels, T]
            x = torch.from_numpy(x).to(device)
            
            x = x.cpu()
            generator = generator.cpu()

            # Predict audio
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze() * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            # Save WAV
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(f"Saved: {output_file}")

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='/home/ibrahim/english_tts/Ten_Wavs_HiFi_Dataset/12787_other')
    parser.add_argument('--output_dir', default='/home/ibrahim/english_tts/hifigan_test_demo_github/generated_wavs_hifi_dataset/12787_other')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    # Load config
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        json_config = json.load(f)

    global h
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(h.seed)
    # print(torch.cuda.memory_summary(device=None, abbreviated=True))
    inference(a)

if __name__ == '__main__':
    main()
