import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# Dataset definition
class SpeechDataset(Dataset):
    def __init__(self, data_dir, sample_rate=22050):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=80, n_fft=1024, hop_length=256)
        self.amplitude_to_db = AmplitudeToDB()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.data_dir, self.file_names[idx])
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        
        # Limit waveform length
        max_length = 3309120  # or any other suitable length
        if waveform.size(1) > max_length:
            waveform = waveform[:, :max_length]
        
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return waveform, mel_spec_db

# Specify your actual data directory here
data_dir = '/Users/gimhyeonbin/Desktop/chatbot/data'

# Check if data directory exists and contains files
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

dataset = SpeechDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Print mel spectrogram shape
for i, (waveform, mel_spec_db) in enumerate(dataloader):
    print("Mel spectrogram shape:", mel_spec_db.shape)
    break

# Tacotron2 Model definition
class Tacotron2(nn.Module):
    def __init__(self, mel_dim=80, hidden_dim=128, output_dim=1):
        super(Tacotron2, self).__init__()
        self.rnn = nn.GRU(input_size=mel_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, mel_spec):
        mel_spec = mel_spec.squeeze(1)  # Remove channel dimension if it exists
        mel_spec = mel_spec.transpose(1, 2)  # Swap freq_bins and time_steps
        rnn_output, _ = self.rnn(mel_spec)
        x = self.fc1(rnn_output)
        x = self.fc2(x)
        return x.transpose(1, 2), rnn_output  # Transpose back to (batch, channel, time)

# Initialize and train Tacotron2 model
model = Tacotron2()

# Ensure model has parameters
if list(model.parameters()):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
else:
    raise ValueError("Model has no parameters. Check your model definition.")

criterion = nn.MSELoss()

# Training function for Tacotron2
def train_tacotron2(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (waveform, mel_spec_db) in enumerate(dataloader):
            optimizer.zero_grad()
            output, _ = model(mel_spec_db)
            
            # Adjust waveform to match output size
            target = waveform[:, :, :output.size(2)]
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
                running_loss = 0.0

# Train Tacotron2 model
train_tacotron2(model, dataloader, optimizer, criterion)

# WaveGlow Model definition
class WaveGlow(nn.Module):
    def __init__(self, mel_dim=80):
        super(WaveGlow, self).__init__()
        self.conv1 = nn.Conv1d(mel_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32, 1)  # Output a single value for each time step

    def forward(self, mel_spec):
        # mel_spec shape: (batch_size, channels, freq_bins, time_steps)
        mel_spec = mel_spec.squeeze(1)  # Remove channel dimension
        # Now mel_spec shape: (batch_size, freq_bins, time_steps)
        x = self.conv1(mel_spec)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # Change to (batch_size, time_steps, features)
        x = self.fc(x)
        return x.transpose(1, 2)  # Change back to (batch_size, 1, time_steps)

# Initialize WaveGlow model
waveglow_model = WaveGlow()

# Ensure model has parameters
if list(waveglow_model.parameters()):
    waveglow_optimizer = optim.Adam(waveglow_model.parameters(), lr=0.001)
else:
    raise ValueError("WaveGlow model has no parameters. Check your model definition.")

waveglow_criterion = nn.MSELoss()

# Training function for WaveGlow
def train_waveglow(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (waveform, mel_spec_db) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(mel_spec_db)
            
            # Adjust waveform to match output size
            target = waveform[:, :, :output.size(2)]
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
                running_loss = 0.0

# Train WaveGlow model
train_waveglow(waveglow_model, dataloader, waveglow_optimizer, waveglow_criterion)