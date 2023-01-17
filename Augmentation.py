import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm_notebook as tqdm
import os

from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt

import os
from PIL import Image
import torch

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from librosa.core.constantq import audio
from pydub import AudioSegment

print(torch.__version__)
print(torchaudio.__version__)

# df=pd.read_csv('./ObjectFolder/objects.csv', names=["folder", "label", "scale", "material", "link"])
# train= df[:100]

import random
import torchaudio

import math
import os
import pathlib
import random
import torch


class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length
        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length-self.clip_length)
            audio_data = audio_data[offset:(offset+self.clip_length)]

        return self.vad(audio_data) # remove silences at the beggining/end

class RandomSpeedChange:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1, 0.5, 1.5])
        if speed_factor == 1.0: # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio

class CodecAugment:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        configs = [
            ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law")
            # ({"format": "gsm"}, "GSM-FR"),
            # ({"format": "mp3", "compression": -9}, "MP3"),
            # ({"format": "vorbis", "compression": -1}, "Vorbis"),
        ]
        for param, title in configs:
          transformed_audio = F.apply_codec(audio_data, sample_rate, **param)
          # plot_specgram(augmented, sample_rate, title=title)
          # play_audio(augmented, sample_rate)
        return transformed_audio
 


class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['remix', '1'], # convert to mono
            ['rate', str(self.sample_rate)], # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length-audio_length)
            noise = noise[..., offset:offset+audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise ) / 2
 
# noise_transform = RandomBackgroundNoise(sample_rate, './noises_directory')
# transformed_audio = noise_transform(audio_data)

class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data

def detect_leading_silence(sound, silence_threshold=-80.0, chunk_size=10):
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms


for subdir, dirs, files in os.walk( "/content/drive/MyDrive/ObjectFolder/Wavonly"):
  for file in files:
    l=(subdir.split("/"))[-1]
    f=os.path.join(subdir, file)
    augment="trim"
    print(f)
    sound = AudioSegment.from_file(f, format="wav")

    start_trim = detect_leading_silence(sound)
    # print(start_trim)
    trimmed_sound = sound[start_trim:]
    path = "/content/drive/MyDrive/ObjectFolder/CustomWavFiles/"+l+"/"+augment+file
    print(path)
    trimmed_sound.export(path, format="wav")
    # audio_data, sample_rate = torchaudio.load(f)
    # compose_transform = ComposeTransform([
    # RandomClip(sample_rate=sample_rate, clip_length=64000),
    # # RandomSpeedChange(sample_rate)
    # # CodecAugment(sample_rate)
    # # RandomBackgroundNoise(sample_rate, '/content/drive/MyDrive/ObjectFolder/background_sounds')
    # ])

    # wav = compose_transform(audio_data)
    


    # path = "/content/drive/MyDrive/ObjectFolder/CustomWavFiles/"+l+"/"+augment+file
    # print(path)
    # torchaudio.save(path, wav, sample_rate)


    # self.data.append(spec_to_image(get_melspectrogram_db(f))[np.newaxis,...])
    # self.labels.append(self.c2i[l])


