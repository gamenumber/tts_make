import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# Configuration for librosa mel spectrogram
frame_length = 0.025
frame_stride = 0.010

def Mel_S(wav_file):
    # Compute mel-spectrogram using librosa
    y, sr = librosa.load(wav_file, sr=16000)

    input_nfft = int(round(sr * frame_length))
    input_stride = int(round(sr * frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr, np.shape(S)))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram_example.png')
    plt.show()

    return S

# Define text_to_sequence function
def text_to_sequence(text, cleaners):
    char_to_idx = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz")}
    sequence = [char_to_idx.get(char, 0) for char in text.lower()]
    return sequence

# Define a SpeechDataset class for PyTorch
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

        # Normalize waveform
        waveform = waveform / torch.max(torch.abs(waveform))

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Convert text to sequence
        text_sequence = torch.tensor(text_to_sequence(self.text_data[idx], ['korean_cleaners']), dtype=torch.long)

        return text_sequence, waveform, mel_spec_db

# Define an Attention module
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        src_len = encoder_outputs.shape[1]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# Define the Tacotron2 model with output padding/clipping
class Tacotron2Model(nn.Module):
    def __init__(self, num_chars, mel_dim=80, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2, hidden_dim)
        self.decoder = nn.LSTMCell(hidden_dim * 2 + mel_dim, hidden_dim)
        self.mel_linear = nn.Linear(hidden_dim, mel_dim)

    def forward(self, text_sequence, target_mel_len, teacher_forcing_ratio=0.5):
        embedded = self.embedding(text_sequence)
        encoder_outputs, _ = self.encoder(embedded)

        batch_size = text_sequence.size(0)
        max_len = target_mel_len
        mel_dim = self.mel_linear.out_features

        decoder_input = torch.zeros(batch_size, mel_dim).to(text_sequence.device)
        hidden = torch.zeros(batch_size, self.decoder.hidden_size).to(text_sequence.device)
        cell = torch.zeros(batch_size, self.decoder.hidden_size).to(text_sequence.device)

        mel_outputs = []

        for t in range(max_len):
            attention_weights = self.attention(encoder_outputs, hidden)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            decoder_input = torch.cat([decoder_input, context], dim=1)
            hidden, cell = self.decoder(decoder_input, (hidden, cell))
            mel_output = self.mel_linear(hidden)
            mel_outputs.append(mel_output.unsqueeze(2))

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = mel_output if teacher_force else mel_output.detach()

        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Adjust output length using F.pad
        if mel_outputs.size(2) > target_mel_len:
            mel_outputs = mel_outputs[:, :, :target_mel_len]
        elif mel_outputs.size(2) < target_mel_len:
            mel_outputs = F.pad(mel_outputs, (0, target_mel_len - mel_outputs.size(2)))

        return mel_outputs

# Define a SimpleVocoder model
class SimpleVocoder(nn.Module):
    def __init__(self, mel_dim=80):
        super().__init__()
        self.conv1 = nn.Conv1d(mel_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, mel_spec):
        # Ensure input is of shape (batch_size, mel_dim, seq_length)
        if mel_spec.dim() == 4:  # input shape (batch_size, 1, mel_dim, seq_length)
            mel_spec = mel_spec.squeeze(1)  # change shape to (batch_size, mel_dim, seq_length)
        elif mel_spec.dim() == 2:  # input shape (mel_dim, seq_length)
            mel_spec = mel_spec.unsqueeze(0)  # add batch dimension
        elif mel_spec.dim() == 3 and mel_spec.size(1) == 1:  # input shape (batch_size, 1, seq_length)
            mel_spec = mel_spec.squeeze(1)  # remove the channel dimension
            mel_spec = mel_spec.permute(0, 2, 1)  # change shape to (batch_size, seq_length, mel_dim)

        x = self.relu(self.conv1(mel_spec))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def train_model(model, dataloader, optimizer, criterion, scheduler, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (text_sequence, waveform, mel_spec_db) in enumerate(dataloader):
            optimizer.zero_grad()

            if isinstance(model, Tacotron2Model):
                output = model(text_sequence, mel_spec_db.size(2))  # Pass mel_spec_db size for Tacotron2
                target = mel_spec_db.squeeze(1)[:, :, :output.size(2)]  # Adjust target size
            else:
                output = model(mel_spec_db)  # No need to pass target_mel_len for Vocoder
                target = mel_spec_db.squeeze(1)[:, :, :output.size(2)]  # Adjust target size if needed

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

    print("Training finished.")

# Function to generate mel spectrogram from text
def synthesize_text(model, text, max_len=100):
    model.eval()
    with torch.no_grad():
        text_sequence = torch.tensor(text_to_sequence(text, ['korean_cleaners']), dtype=torch.long).unsqueeze(0)
        mel_output = model(text_sequence, max_len)  # Pass max_len as target_mel_len
        return mel_output.squeeze(0).cpu().numpy()

# Function to convert mel spectrogram to waveform
def mel_to_audio(vocoder_model, mel_spec):
    vocoder_model.eval()
    with torch.no_grad():
        mel_spec = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        waveform = vocoder_model(mel_spec)
        return waveform.squeeze(0).cpu().numpy()

# Specify your actual data directory here
data_dir = '/Users/gimhyeonbin/Desktop/tts_make/ttsmake/data'

# Check if data directory exists and contains files
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

# Initialize datasets and dataloaders
dataset = SpeechDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize Tacotron2 model and ensure it has parameters
num_chars = 256  # Adjust this based on your character set
tacotron2_model = Tacotron2Model(num_chars)
if list(tacotron2_model.parameters()):
    optimizer_tacotron2 = optim.Adam(tacotron2_model.parameters(), lr=0.0001)
else:
    raise ValueError("Tacotron2 model has no parameters. Check your model definition.")

# Initialize Vocoder model and ensure it has parameters
vocoder_model = SimpleVocoder()
if list(vocoder_model.parameters()):
    optimizer_vocoder = optim.Adam(vocoder_model.parameters(), lr=0.0001)
else:
    raise ValueError("Vocoder model has no parameters. Check your model definition.")

# Loss function
criterion = nn.MSELoss()

# Learning rate scheduler
scheduler_tacotron2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_tacotron2, patience=5, factor=0.5)
scheduler_vocoder = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vocoder, patience=5, factor=0.5)

# Train Tacotron2 model
train_model(tacotron2_model, dataloader, optimizer_tacotron2, criterion, scheduler_tacotron2, epochs=10)

# Train Vocoder model
train_model(vocoder_model, dataloader, optimizer_vocoder, criterion, scheduler_vocoder, epochs=10)

# Example usage to synthesize text to audio
example_text = "안녕하세요. 텍스트를 음성으로 변환하는 예제입니다."
mel_spec = synthesize_text(tacotron2_model, example_text)
waveform = mel_to_audio(vocoder_model, mel_spec)

# Save waveform to file
# Ensure waveform is in the correct shape (1D or 2D)
# Assume waveform is a 1D numpy array
if waveform.ndim == 1:
    waveform = waveform.reshape(1, -1)  # Reshape to 2D tensor

# Convert waveform to numpy array if it's not already
if not isinstance(waveform, np.ndarray):
    waveform = waveform.numpy()

# Save the waveform
torchaudio.save('synthesized_audio.wav', torch.tensor(waveform), sample_rate=22050)

# Play the synthesized audio
ipd.Audio('synthesized_audio.wav')
