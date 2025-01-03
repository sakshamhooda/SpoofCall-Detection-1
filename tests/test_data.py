import unittest
import torch
import numpy as np
import tempfile
import os
import soundfile as sf
import cv2
from pathlib import Path
from src.data.data_loader import AudioDataset, VideoDataset
from src.utils.preprocessing import AudioPreprocessor, VideoPreprocessor
from torch.utils.data import DataLoader

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test audio file
        self.audio_path = os.path.join(self.temp_dir, 'test_audio.wav')
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        sf.write(self.audio_path, signal, sample_rate)
        
        # Create test video file
        self.video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (640, 480))
        for _ in range(30):  # 1 second video
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
    def test_audio_dataset(self):
        # Test AudioDataset
        dataset = AudioDataset(
            audio_paths=[self.audio_path],
            labels=[0],
            sample_rate=16000,
            duration=2
        )
        
        # Test length
        self.assertEqual(len(dataset), 1)
        
        # Test item retrieval
        spec, label = dataset[0]
        
        # Check shapes and types
        self.assertIsInstance(spec, torch.Tensor)
        self.assertEqual(len(spec.shape), 3)  # (channel, freq, time)
        self.assertEqual(label, 0)
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        batch_spec, batch_label = next(iter(dataloader))
        self.assertEqual(batch_spec.shape[0], 1)
        self.assertEqual(batch_label.shape[0], 1)
        
    def test_video_dataset(self):
        # Test VideoDataset
        dataset = VideoDataset(
            video_paths=[self.video_path],
            labels=[1],
            frame_count=16
        )
        
        # Test length
        self.assertEqual(len(dataset), 1)
        
        # Test item retrieval
        frames, label = dataset[0]
        
        # Check shapes and types
        self.assertIsInstance(frames, torch.Tensor)
        self.assertEqual(len(frames.shape), 4)  # (frames, channels, height, width)
        self.assertEqual(frames.shape[0], 16)  # Number of frames
        self.assertEqual(label, 1)
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        batch_frames, batch_label = next(iter(dataloader))
        self.assertEqual(batch_frames.shape[0], 1)
        self.assertEqual(batch_label.shape[0], 1)

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.audio_preprocessor = AudioPreprocessor(sample_rate=16000)
        self.video_preprocessor = VideoPreprocessor()
        
        # Create test data
        self.test_waveform = torch.randn(1, 16000)  # 1 second of audio
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    def test_audio_preprocessing(self):
        # Test spectrogram computation
        spec = self.audio_preprocessor.compute_spectrogram(self.test_waveform)
        self.assertIsInstance(spec, torch.Tensor)
        self.assertEqual(len(spec.shape), 3)
        
        # Test feature extraction
        features = self.audio_preprocessor.extract_features(self.test_waveform)
        required_features = ['mfcc', 'chroma', 'spectral_centroid', 
                           'spectral_rolloff', 'zero_crossing_rate']
        for feature in required_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], np.ndarray)
            
        # Test noise addition
        noisy_audio = self.audio_preprocessor.apply_noise(self.test_waveform)
        self.assertEqual(noisy_audio.shape, self.test_waveform.shape)
        self.assertFalse(torch.equal(noisy_audio, self.test_waveform))
        
        # Test pitch shifting
        shifted_audio = self.audio_preprocessor.apply_pitch_shift(
            self.test_waveform, 16000, n_steps=2
        )
        self.assertEqual(shifted_audio.shape, self.test_waveform.shape)
        
    def test_video_preprocessing(self):
        # Test brightness/contrast adjustment
        adjusted_frame = self.video_preprocessor.adjust_brightness_contrast(
            self.test_frame, alpha=1.2, beta=10
        )
        self.assertEqual(adjusted_frame.shape, self.test_frame.shape)
        self.assertFalse(np.array_equal(adjusted_frame, self.test_frame))
        
        # Test color jittering
        jittered_frame = self.video_preprocessor.apply_color_jitter(
            self.test_frame,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        self.assertEqual(jittered_frame.shape, self.test_frame.shape)
        self.assertFalse(np.array_equal(jittered_frame, self.test_frame))

class TestDataAugmentation(unittest.TestCase):
    def setUp(self):
        self.audio_preprocessor = AudioPreprocessor(sample_rate=16000)
        self.video_preprocessor = VideoPreprocessor()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
    def test_audio_augmentation_chain(self):
        # Create test audio file
        audio_path = os.path.join(self.temp_dir, 'test_augment.wav')
        sample_rate = 16000
        duration = 2
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t)
        sf.write(audio_path, signal, sample_rate)
        
        # Create dataset with augmentation
        dataset = AudioDataset(
            audio_paths=[audio_path],
            labels=[0],
            sample_rate=16000,
            duration=2,
            transform=lambda x: self.audio_preprocessor.apply_noise(
                self.audio_preprocessor.apply_pitch_shift(x, 16000, 2),
                level=0.01
            )
        )
        
        # Test augmentation in dataloader
        dataloader = DataLoader(dataset, batch_size=1)
        original_spec, _ = dataset[0]
        
        # Get multiple batches to test randomization
        specs = [next(iter(dataloader))[0] for _ in range(5)]
        
        # Verify all augmented spectrograms are different
        for i in range(len(specs)-1):
            self.assertFalse(torch.allclose(specs[i], specs[i+1]))
            
    def test_video_augmentation_chain(self):
        # Create test video file
        video_path = os.path.join(self.temp_dir, 'test_augment.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        for _ in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        # Create dataset with augmentation
        dataset = VideoDataset(
            video_paths=[video_path],
            labels=[1],
            frame_count=16,
            transform=lambda x: self.video_preprocessor.apply_color_jitter(
                self.video_preprocessor.adjust_brightness_contrast(x, 1.2, 10),
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        )
        
        # Test augmentation in dataloader
        dataloader = DataLoader(dataset, batch_size=1)
        original_frames, _ = dataset[0]
        
        # Get multiple batches to test randomization
        frames = [next(iter(dataloader))[0] for _ in range(5)]
        
        # Verify all augmented frames are different
        for i in range(len(frames)-1):
            self.assertFalse(torch.allclose(frames[i], frames[i+1]))
            
if __name__ == '__main__':
    unittest.main()