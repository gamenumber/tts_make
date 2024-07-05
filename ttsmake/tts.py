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

# 설정 개선
SAMPLE_RATE = 22050
MEL_DIM = 80
HIDDEN_DIM = 512
BATCH_SIZE = 1  # 배치 크기 증가
EPOCHS = 100  # 에폭 수 증가
LEARNING_RATE = 0.001  # 학습률 유지

def text_to_sequence(text, cleaners):
    char_to_idx = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ가나다라마바사아자차카타파하아야어여오요우유으이 .,!?-")}
    sequence = [char_to_idx.get(char, 0) for char in text.lower()]
    return sequence

class SpeechDataset(Dataset):
    def __init__(self, data_dir, sample_rate=SAMPLE_RATE):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=MEL_DIM, n_fft=1024, hop_length=256)
        self.amplitude_to_db = AmplitudeToDB()

        # 더미 텍스트 데이터 (실제 텍스트 데이터로 교체 필요)
        self.text_data = ["dummy text" for _ in self.file_names]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.data_dir, self.file_names[idx])
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # 모노로 변환
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        # 파형 길이 제한
        max_length = 3309120  # 또는 다른 적절한 길이
        if waveform.size(1) > max_length:
            waveform = waveform[:, :max_length]

        # 파형 정규화
        waveform = waveform / torch.max(torch.abs(waveform))

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # 텍스트를 시퀀스로 변환
        text_sequence = torch.tensor(text_to_sequence(self.text_data[idx], ['korean_cleaners']), dtype=torch.long)

        return text_sequence, waveform, mel_spec_db

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

class Tacotron2Model(nn.Module):
    def __init__(self, num_chars, mel_dim=MEL_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2, hidden_dim)
        self.decoder = nn.LSTMCell(hidden_dim * 2 + mel_dim, hidden_dim)
        self.mel_linear = nn.Linear(hidden_dim, mel_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text_sequence, target_mel_len, teacher_forcing_ratio=0.5):
        embedded = self.dropout(self.embedding(text_sequence))
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
            mel_output = self.mel_linear(self.dropout(hidden))
            mel_outputs.append(mel_output.unsqueeze(2))

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = mel_output if teacher_force else mel_output.detach()

        mel_outputs = torch.cat(mel_outputs, dim=2)

        if mel_outputs.size(2) > target_mel_len:
            mel_outputs = mel_outputs[:, :, :target_mel_len]
        elif mel_outputs.size(2) < target_mel_len:
            mel_outputs = F.pad(mel_outputs, (0, target_mel_len - mel_outputs.size(2)))

        return mel_outputs

class ImprovedVocoder(nn.Module):
    def __init__(self, mel_dim=MEL_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(mel_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.batch_norm5 = nn.BatchNorm1d(128)
        self.batch_norm6 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)

    def forward(self, mel_spec):
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)
        elif mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        elif mel_spec.dim() == 3 and mel_spec.size(1) == 1:
            mel_spec = mel_spec.squeeze(1)
            mel_spec = mel_spec.permute(0, 2, 1)

        x = self.dropout(self.relu(self.batch_norm1(self.conv1(mel_spec))))
        x = self.dropout(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(self.relu(self.batch_norm4(self.conv4(x))))
        x = self.dropout(self.relu(self.batch_norm5(self.conv5(x))))
        x = self.dropout(self.relu(self.batch_norm6(self.conv6(x))))
        x = self.conv7(x)
        
        return torch.tanh(x)

def train_model(model, dataloader, optimizer, criterion, scheduler, epochs=EPOCHS, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (text_sequence, waveform, mel_spec_db) in enumerate(dataloader):
            text_sequence, waveform, mel_spec_db = text_sequence.to(device), waveform.to(device), mel_spec_db.to(device)
            
            optimizer.zero_grad()

            if isinstance(model, Tacotron2Model):
                output = model(text_sequence, mel_spec_db.size(2))
                target = mel_spec_db.squeeze(1)[:, :, :output.size(2)]
            else:
                output = model(mel_spec_db)
                target = waveform[:, :, :output.size(2)]

            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

    print("Training finished.")

def synthesize_text(model, text, max_len=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        text_sequence = torch.tensor(text_to_sequence(text, ['korean_cleaners']), dtype=torch.long).unsqueeze(0).to(device)
        mel_output = model(text_sequence, max_len)
        return mel_output.squeeze(0).cpu().numpy()

def mel_to_audio(vocoder_model, mel_spec, device='cuda' if torch.cuda.is_available() else 'cpu'):
    vocoder_model.to(device)
    vocoder_model.eval()
    with torch.no_grad():
        mel_spec = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
        waveform = vocoder_model(mel_spec)
        return waveform.squeeze(0).cpu().numpy()

# 메인 실행
if __name__ == "__main__":
    data_dir = '/Users/gimhyeonbin/Desktop/tts_make/ttsmake/data'

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"데이터 디렉토리 '{data_dir}'를 찾을 수 없습니다.")

    dataset = SpeechDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_chars = 256
    tacotron2_model = Tacotron2Model(num_chars)
    vocoder_model = ImprovedVocoder()

    optimizer_tacotron2 = optim.Adam(tacotron2_model.parameters(), lr=LEARNING_RATE)
    optimizer_vocoder = optim.Adam(vocoder_model.parameters(), lr=LEARNING_RATE)

    criterion = nn.MSELoss()

    scheduler_tacotron2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_tacotron2, patience=10, factor=0.5)
    scheduler_vocoder = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vocoder, patience=10, factor=0.5)

    print("Tacotron2 훈련 시작...")
    train_model(tacotron2_model, dataloader, optimizer_tacotron2, criterion, scheduler_tacotron2)

    print("Vocoder 훈련 시작...")
    train_model(vocoder_model, dataloader, optimizer_vocoder, criterion, scheduler_vocoder)

    example_text = "안녕하세요. 텍스트를 음성으로 변환하는 예제입니다."
    print("텍스트 합성 중...")
    mel_spec = synthesize_text(tacotron2_model, example_text)
    print("멜 스펙트로그램에서 오디오 생성 중...")
    waveform = mel_to_audio(vocoder_model, mel_spec)

    waveform = np.clip(waveform, -1, 1)
    waveform_tensor = torch.tensor(waveform)

    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)
    elif waveform_tensor.dim() == 2:
        pass
    elif waveform_tensor.dim() == 3:
        waveform_tensor = waveform_tensor.squeeze(1)
    else:
        raise ValueError(f"예상치 못한 파형 형태: {waveform_tensor.shape}")

    print(f"최종 파형 텐서 형태: {waveform_tensor.shape}")

    print("합성된 오디오 저장 중...")
    torchaudio.save('synthesized_audio.wav', waveform_tensor, sample_rate=SAMPLE_RATE)

    print("합성된 오디오 재생 중...")
    ipd.Audio('synthesized_audio.wav')