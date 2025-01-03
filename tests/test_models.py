import unittest
import torch
import numpy as np
from src.models.cycle_gan import CycleGAN
from src.models.star_gan import StarGAN
from src.models.spectrogram_cnn import SpectrogramCNN
from src.models.sequential_lstm import SequentialLSTM
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

class TestModels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.input_channels = 1
        self.seq_length = 128
        self.num_domains = 5
        
    def test_cyclegan(self):
        # Initialize model
        model = CycleGAN(input_channels=self.input_channels).to(self.device)
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.input_channels, 64, 64).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, x.shape)
        
        # Test training step
        model.set_optimizers()
        loss_dict = model.train_step(x, x)
        
        required_keys = ['loss_G', 'loss_D', 'loss_cycle', 'loss_identity']
        for key in required_keys:
            self.assertIn(key, loss_dict)
            
    def test_stargan(self):
        # Initialize model
        model = StarGAN(input_channels=self.input_channels, num_domains=self.num_domains).to(self.device)
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.input_channels, 64, 64).to(self.device)
        target_domain = torch.randint(0, self.num_domains, (self.batch_size,)).to(self.device)
        output = model(x, target_domain)
        
        self.assertEqual(output.shape, x.shape)
        
        # Test training step
        model.set_optimizers()
        source_domain = torch.randint(0, self.num_domains, (self.batch_size,)).to(self.device)
        loss_dict = model.train_step(x, source_domain, target_domain)
        
        required_keys = ['g_loss', 'd_loss', 'cycle_loss']
        for key in required_keys:
            self.assertIn(key, loss_dict)
            
    def test_spectrogram_cnn(self):
        # Initialize model
        model = SpectrogramCNN(input_channels=self.input_channels).to(self.device)
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.input_channels, 64, 64).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, 2))  # Binary classification
        
        # Test training step
        y = torch.randint(0, 2, (self.batch_size,)).to(self.device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        
        metrics = model.train_step(x, y, criterion, optimizer)
        
        required_keys = ['loss', 'accuracy']
        for key in required_keys:
            self.assertIn(key, metrics)
            
    def test_sequential_lstm(self):
        # Initialize model
        input_size = 128
        model = SequentialLSTM(input_size=input_size).to(self.device)
        
        # Test forward pass with variable length sequences
        x = torch.randn(self.batch_size, 50, input_size).to(self.device)  # 50 time steps
        lengths = torch.tensor([50, 40]).to(self.device)  # Different lengths for batch
        output = model(x, lengths)
        
        self.assertEqual(output.shape, (self.batch_size, 2))  # Binary classification
        
        # Test training step
        y = torch.randint(0, 2, (self.batch_size,)).to(self.device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        
        metrics = model.train_step(x, y, lengths, criterion, optimizer)
        
        required_keys = ['loss', 'accuracy']
        for key in required_keys:
            self.assertIn(key, metrics)
            
    def test_model_save_load(self):
        # Test save/load functionality for each model
        models = {
            'cyclegan': CycleGAN(input_channels=self.input_channels),
            'stargan': StarGAN(input_channels=self.input_channels, num_domains=self.num_domains),
            'cnn': SpectrogramCNN(input_channels=self.input_channels),
            'lstm': SequentialLSTM(input_size=128)
        }
        
        for name, model in models.items():
            # Save model
            model.save(f'test_{name}_model.pt')
            
            # Initialize new model and load weights
            new_model = type(model)(
                input_channels=self.input_channels,
                num_domains=self.num_domains if name == 'stargan' else None
            )
            new_model.load(f'test_{name}_model.pt')
            
            # Compare model parameters
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))
                
    def test_model_parameter_count(self):
        # Test parameter counting functionality
        models = {
            'cyclegan': CycleGAN(input_channels=self.input_channels),
            'stargan': StarGAN(input_channels=self.input_channels, num_domains=self.num_domains),
            'cnn': SpectrogramCNN(input_channels=self.input_channels),
            'lstm': SequentialLSTM(input_size=128)
        }
        
        for name, model in models.items():
            param_count = model.get_number_of_parameters()
            self.assertGreater(param_count, 0)
            
            # Manual count for verification
            manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertEqual(param_count, manual_count)
            
class TestModelInputValidation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_invalid_inputs(self):
        # Test model behavior with invalid inputs
        model = SpectrogramCNN(input_channels=1).to(self.device)
        
        # Test with wrong number of dimensions
        with self.assertRaises(RuntimeError):
            x = torch.randn(2, 64, 64).to(self.device)  # Missing channel dimension
            model(x)
            
        # Test with wrong channel dimension
        with self.assertRaises(RuntimeError):
            x = torch.randn(2, 3, 64, 64).to(self.device)  # 3 channels instead of 1
            model(x)
            
    def test_edge_cases(self):
        model = SequentialLSTM(input_size=128).to(self.device)
        
        # Test with zero-length sequence
        x = torch.randn(2, 0, 128).to(self.device)
        with self.assertRaises(RuntimeError):
            model(x)
            
        # Test with single time step
        x = torch.randn(2, 1, 128).to(self.device)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))
        
        # Test with very long sequence
        x = torch.randn(2, 1000, 128).to(self.device)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))

if __name__ == '__main__':
    unittest.main()