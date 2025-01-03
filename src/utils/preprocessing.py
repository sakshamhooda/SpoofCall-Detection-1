import torch
import torchaudio
import numpy as np
import librosa
from scipy.signal import stft

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128):
        """
        Audio preprocessing utilities
        
        Args:
            sample_rate (int): Target sample rate
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of mel bins
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Initialize mel spectrogram transform
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
    def load_and_preprocess(self, audio_path, duration=None):
        """Load and preprocess audio file"""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Trim to duration if specified
        if duration:
            samples = int(self.sample_rate * duration)
            if waveform.shape[1] > samples:
                waveform = waveform[:, :samples]
            else:
                padding = samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
        return waveform
        
    def compute_spectrogram(self, waveform):
        """Compute mel spectrogram"""
        mel_spec = self.mel_spec_transform(waveform)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_spec_db
        
    def extract_features(self, waveform):
        """Extract additional audio features"""
        # Convert to numpy for librosa compatibility
        audio = waveform.numpy().squeeze()
        
        # Extract features
        features = {
            'mfcc': librosa.feature.mfcc(y=audio, sr=self.sample_rate),
            'chroma': librosa.feature.chroma_stft(y=audio, sr=self.sample_rate),
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio)
        }
        
        return features
        
    def compute_phase_vocoder(self, waveform, rate=1.0):
        """Phase vocoder for time stretching"""
        # Convert to numpy
        audio = waveform.numpy().squeeze()
        
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Apply phase vocoder
        D_stretched = librosa.phase_vocoder(D, rate, hop_length=self.hop_length)
        
        # Inverse STFT
        y_stretched = librosa.istft(D_stretched, hop_length=self.hop_length)
        
        return torch.from_numpy(y_stretched).unsqueeze(0)
        
    @staticmethod
    def apply_noise(waveform, noise_type='gaussian', level=0.005):
        """Apply different types of noise to the audio"""
        if noise_type == 'gaussian':
            noise = torch.randn_like(waveform) * level
            return waveform + noise
        elif noise_type == 'uniform':
            noise = torch.rand_like(waveform) * level * 2 - level
            return waveform + noise
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
    @staticmethod
    def apply_pitch_shift(waveform, sample_rate, n_steps):
        """Pitch shift the audio"""
        # Convert to numpy
        audio = waveform.numpy().squeeze()
        
        # Apply pitch shift
        audio_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
        
        return torch.from_numpy(audio_shifted).unsqueeze(0)

class VideoPreprocessor:
    def __init__(self, target_size=(224, 224)):
        """
        Video preprocessing utilities
        
        Args:
            target_size (tuple): Target frame size (height, width)
        """
        self.target_size = target_size
        
    def extract_faces(self, frame):
        """Extract faces from frame using dlib"""
        # TODO: Implement face extraction
        pass
        
    @staticmethod
    def adjust_brightness_contrast(frame, alpha=1.0, beta=0):
        """Adjust brightness and contrast of frame"""
        return np.clip(alpha * frame + beta, 0, 255).astype(np.uint8)
        
    @staticmethod
    def apply_color_jitter(frame, brightness=0, contrast=0, saturation=0, hue=0):
        """Apply color jittering to frame"""
        # Convert to float32
        frame = frame.astype(np.float32) / 255.0
        
        # Adjust brightness
        if brightness != 0:
            frame *= (1 + np.random.uniform(-brightness, brightness))
            
        # Adjust contrast
        if contrast != 0:
            mean = np.mean(frame, axis=(0, 1))
            frame = (frame - mean) * (1 + np.random.uniform(-contrast, contrast)) + mean
            
        # Adjust saturation
        if saturation != 0:
            gray = np.mean(frame, axis=2, keepdims=True)
            frame = frame * (1 + np.random.uniform(-saturation, saturation)) + \
                   gray * (1 - np.random.uniform(-saturation, saturation))
                   
        # Adjust hue
        if hue != 0:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-hue, hue) * 180) % 180
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        return np.clip(frame * 255, 0, 255).astype(np.uint8)