import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import IPython.display as ipd  # For displaying audio

# Dummy implementation for text_to_sequence
def text_to_sequence(text, cleaners):
    char_to_idx = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz")}
    sequence = [char_to_idx.get(char, 0) for char in text.lower()]
    return sequence

# Updated Dataset definition
class SpeechDataset(Dataset):
    def __init__(self, data_dir, sample_rate=22050):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=80, n_fft=1024, hop_length=256)
        self.amplitude_to_db = AmplitudeToDB()
        
        # Dummy text data (replace this with actual text data)
        self.text_data = ["dummy text" for _ in self.file_names]

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
        
        # Convert text to sequence
        text_sequence = torch.tensor(text_to_sequence(self.text_data[idx], ['korean_cleaners']), dtype=torch.long)
        
        return text_sequence, waveform, mel_spec_db

# Specify your actual data directory here
data_dir = '/Users/gimhyeonbin/Desktop/tts_make/ttsmake/data'

# Check if data directory exists and contains files
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

dataset = SpeechDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Updated Tacotron2 Model definition
class Tacotron2Model(nn.Module):
    def __init__(self, num_chars, mel_dim=80, hidden_dim=256):
        super(Tacotron2Model, self).__init__()
        self.embedding = nn.Embedding(num_chars, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(hidden_dim * 2 + mel_dim, hidden_dim)
        self.mel_linear = nn.Linear(hidden_dim, mel_dim)

    def forward(self, text_sequence):
        # Encoder
        embedded = self.embedding(text_sequence)
        encoder_outputs, _ = self.encoder(embedded)

        batch_size = text_sequence.size(0)

        # Decoder initialization
        decoder_input = torch.zeros(batch_size, self.mel_linear.out_features).to(text_sequence.device)
        hidden = torch.zeros(batch_size, self.decoder.hidden_size).to(text_sequence.device)
        cell = torch.zeros(batch_size, self.decoder.hidden_size).to(text_sequence.device)

        # List to store mel spectrogram frames
        mel_outputs = []

        # Decoder loop
        for i in range(encoder_outputs.size(1)):
            # Concatenate previous mel output with encoder output
            decoder_input = torch.cat([decoder_input, encoder_outputs[:, i, :]], dim=1)

            # LSTMCell forward pass
            hidden, cell = self.decoder(decoder_input, (hidden, cell))

            # Predict mel spectrogram frame
            mel_output = self.mel_linear(hidden)

            # Append mel output to mel_outputs list
            mel_outputs.append(mel_output.unsqueeze(2))

            # Update decoder input to the current mel output
            decoder_input = mel_output

        # Stack mel outputs along the time dimension
        mel_outputs = torch.cat(mel_outputs, dim=2)

        return mel_outputs

# Update the model initialization
num_chars = 256  # Adjust this based on your character set
tacotron2_model = Tacotron2Model(num_chars)

# Ensure model has parameters
if list(tacotron2_model.parameters()):
    optimizer = optim.Adam(tacotron2_model.parameters(), lr=0.001)
else:
    raise ValueError("Model has no parameters. Check your model definition.")

criterion = nn.MSELoss()

# Updated Training function for Tacotron2
def train_tacotron2(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (text_sequence, waveform, mel_spec_db) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(text_sequence)
            
            # Adjust mel_spec_db to match output size
            target = mel_spec_db.squeeze(1)[:, :, :output.size(2)]  # Adjust time dimension
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
                running_loss = 0.0

# Train Tacotron2 model
train_tacotron2(tacotron2_model, dataloader, optimizer, criterion)

# WaveGlow Model definition (updated)
class WaveGlowModel(nn.Module):
    def __init__(self, mel_dim=80):
        super(WaveGlowModel, self).__init__()
        self.conv1 = nn.Conv1d(mel_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32, 1)  # Output a single value for each time step

    def forward(self, mel_spec):
        # mel_spec shape: (batch_size, channels, freq_bins, time_steps)
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)  # Remove channel dimension if it exists
        # Now mel_spec shape: (batch_size, freq_bins, time_steps)
        x = self.conv1(mel_spec)
        x = self.conv2(x)  # Use x instead of mel_spec
        x = x.transpose(1, 2)  # Change to (batch_size, time_steps, features)
        x = self.fc(x)
        return x.transpose(1, 2)  # Change back to (batch_size, 1, time_steps)

# Initialize WaveGlow model
waveglow_model = WaveGlowModel()

# Ensure model has parameters
if list(waveglow_model.parameters()):
    waveglow_optimizer = optim.Adam(waveglow_model.parameters(), lr=0.001)
else:
    raise ValueError("WaveGlow model has no parameters. Check your model definition.")

waveglow_criterion = nn.MSELoss()

# Training function for WaveGlow model
def train_waveglow(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (_, _, mel_spec_db) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(mel_spec_db)
            
            # Adjust mel_spec_db to match output size
            target = mel_spec_db.squeeze(1)[:, :, :output.size(2)]  # Adjust time dimension
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
                running_loss = 0.0

# Train WaveGlow model
train_waveglow(waveglow_model, dataloader, waveglow_optimizer, waveglow_criterion)

# Text to mel spectrogram function
def text_to_mel_spectrogram(text, tacotron2_model):
    tacotron2_model.eval()
    with torch.no_grad():
        text_sequence = torch.tensor(text_to_sequence(text, ['korean_cleaners']), dtype=torch.long).unsqueeze(0)
        mel_spec = tacotron2_model(text_sequence)
    return mel_spec

# Mel spectrogram to waveform function
def mel_spectrogram_to_waveform(mel_spec, waveglow_model):
    waveglow_model.eval()
    with torch.no_grad():
        waveform = waveglow_model(mel_spec)
    return waveform

# Example usage
text_to_synthesize = "안녕? 나는 리월 칠성의 옥형, 각청이라고해 하늘이 잘 지내지? 그럼 안녕?"

# Convert text to mel spectrogram
mel_spec = text_to_mel_spectrogram(text_to_synthesize, tacotron2_model)

# Convert mel spectrogram to waveform
waveform = mel_spectrogram_to_waveform(mel_spec, waveglow_model)

# Ensure waveform is 2D (1, N)
if waveform.dim() == 3:
    waveform = waveform.squeeze(1)  # Remove the middle dimension if it's 3D

# Save waveform as .wav file
output_dir = './result'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'synthesized_speech.wav')
torchaudio.save(output_file, waveform.cpu(), 22050)

print(f"Generated audio saved at: {output_file}")

# Automatically play the saved audio
ipd.display(ipd.Audio(output_file))
