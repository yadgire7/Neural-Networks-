import torch
from torch.utils.data import Dataset
import torchaudio
import os
import pandas as pd

class UrbanSoundData(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annoatations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.annoatations)

    def __getitem__(self, index):
        audio_smaple_path = self._get_audio_sample_path(index)
        label = self.get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_smaple_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._truncate_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _truncate_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:,:self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        signal_len = signal.shape[1]
        if signal_len < self.num_samples:
            last_dim_padding = (0, self.num_samples - signal_len)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
            return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim= 0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annoatations.iloc[index,5]}"   #fold column is at the 5th index(starting with 0)
        file_path = os.path.join(self.audio_dir,fold,self.annoatations.iloc[index,0])  #file name is stored in the csv in the 1st column i.e. 0th index
        return file_path
    
    def get_audio_sample_label(self, index):
        return self.annoatations.iloc[index, 6]
    

if __name__ == '__main__':
    ANNOTATION_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "UrbanSound8K/audio"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft= 1024, hop_length= 512, n_mels= 64)

    usd = UrbanSoundData(annotations_file=ANNOTATION_FILE, audio_dir= AUDIO_DIR, transformation=mel_spectrogram, target_sample_rate=SAMPLE_RATE, num_samples=NUM_SAMPLES, device=device)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
