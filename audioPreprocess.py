# 11/1/2023
# compling
# description: This script is to preprocess audio dataset
# input:
# output:
import os
import pickle
import random

import librosa
import numpy as np
import matplotlib.pyplot as plt

ref_level = 4e-10 # for power spectrogram

def visualize_mel(mel: np.ndarray):
    '''
    This is to visualize and save a mel spectrogram
    :param mel: melspectrogram
    :return: None
    '''
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel, y_axis='mel', x_axis='time', ax=ax, vmin=0)
    ax.set_title('Mel spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

def wav_to_mel_spectrogram(sound,sr = 16000, n_fft = 1024, hop_length = 32, n_mels = 128, type = 'dB'):
    '''
    Convert sound to melspectrogram
    :param sound: sound sample points
    :param sr: sampling rate
    :param n_fft: for stft, recommend 1024 or (23ms*sr), higher it is , better frequency resulotion it gets
    :param hop_length: decide number of frames, recommend short length to get higher temporal resolution
    :param n_mels: the bins of mel spectrogram, normally 128
    :return: mel spectrogram
    '''

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=sound, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels = n_mels)
    if type == 'dB':
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=ref_level)
    return mel_spectrogram
def audios2mels(audios: list,sr = 16000,n_fft = 1024, hop_length = 64,n_mels = 128, using_pool = False, type = 'dB'):
    '''
    Convert a list of audios to melspectrograms
    :param audios: a list of audios
    :param sr: sampling rate
    :param n_fft: for stft, recommend 1024 or (23ms*sr), higher it is , better frequency resulotion it gets
    :param hop_length: decide number of frames, recommend short length to get higher temporal resolution
    :param n_mels: the bins of mel spectrogram, normally 128
    :param using_pool: False
    :return: a list of mels
    '''
    mels = []
    #Create a pool of processes
    if using_pool is True:
        from multiprocessing import Pool
        with Pool() as p:
           mels = p.starmap(wav_to_mel_spectrogram, [(sound,sr,n_fft,hop_length,n_mels) for sound in audios])
    else:
        mels = [wav_to_mel_spectrogram(audio,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,type=type) for audio in audios]

    return mels

def add_noise_audio(image, noise_factor=0.3,dtype = 'float32'):
    '''
    add noise to audio (spectrogram)
    :param image: mel spectrogram of audio
    :param noise_factor:
    :param dtype:
    :return: noisy audio mel spectrogram
    '''
    noise = noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=image.shape
    )
    noise = noise.astype(dtype)
    noisy_array = image + noise
    return np.clip(noisy_array,a_min=0,a_max=1)

def load_file(filename:os.PathLike) -> tuple[np.ndarray, int]:
    '''
    load audio file, return waveform and sample rate
    :param filename: audio file path
    :return: waveform, sample rate
    '''
    sound,sr = librosa.load(filename,sr=None)
    return sound,sr


def wav_to_spectrogram(wav, n_fft=1024, hop_length=32) -> np.ndarray:
    '''
    convert waveform to spectrogram
    :param wav: waveform
    :param n_fft: decide the window length of stft
    :param hop_length: the interval between two stft windows
    :return: spectrogram in dB
    '''

    S = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    S_power = np.abs(S) ** 2
    S_db = librosa.power_to_db(S_power, ref=ref_level)
    return S_db


def visualize_spectrogram(S_db):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, y_axis='linear', x_axis='time', ax=ax,vmin = 0)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")


def visualize_waveform(sound, sr):
    # Create a time variable for the x-axis
    time = librosa.times_like(sound, sr=sr)

    # Plot the waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(sound, sr=sr, alpha=0.9)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, len(sound) / sr)
    plt.ylim(-1,1)# Set x-axis limits
    plt.grid()
    plt.tight_layout()
    plt.show()

def lowpass_filter(audio,sr,cut_off_upper, cut_off_lower=0):
    # audio is data after reading in using tools like torchaudio.load or scipy.io.wavefile
    # sr is sample rate
    # cut_off_upper is upper limit of keep range
    # cut_off_lower is lower limit of keep range
    # work on single audio each time

    n = len(audio)
    dt = 1/sr
    y = np.reshape(audio,(len(audio,)))
    yf = np.fft.fft(y)/(n/2)
    freq = np.fft.fftfreq(n, dt)
    yf[(freq > cut_off_upper)] = 0
    yf[(freq < cut_off_lower)] = 0
    y = np.real(np.fft.ifft(yf)*n)
    return  y.astype("float32")

def padder_waveform(sound, sample_rate=16000, pad_len_ms=250, noise_level=0.001) -> np.ndarray:
    '''
    Random pad a waveform (mono) to fixed length
    :param sound: waveform
    :param sample_rate:
    :param pad_len_ms: target length of padded waveform
    :param noise_level: the sound will be padded with noise
    :return: padded waveform
    '''
    pad_len_frame = sample_rate // 1000 * pad_len_ms
    sound_len = len(sound)
    if sound_len > pad_len_frame:
        try:
            sig = sound[:pad_len_frame]
        except:
            print(sound)
            print(pad_len_frame)
            input()

    elif sound_len < pad_len_frame:
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, pad_len_frame - sound_len)
        pad_end_len = pad_len_frame - sound_len - pad_begin_len

        # Pad with 0s
        # pad_begin = torch.zeros((num_rows, pad_begin_len))
        # pad_end = torch.zeros((num_rows, pad_end_len))
        pad_begin = np.random.randn(pad_begin_len) * noise_level
        pad_end = np.random.randn(pad_end_len) * noise_level

        sig = np.concatenate((pad_begin, sound, pad_end))
    else:
        sig =sound
    return sig

