import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path

from models.cycle_gan import CycleGAN
from models.star_gan import StarGAN
from models.spectrogram_cnn import SpectrogramCNN
from models.sequential_lstm import SequentialLSTM
from data.data_loader import get_dataloaders
from training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train spoof detection models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['cyclegan', 'stargan', 'cnn', 'lstm'],
                       help='Model to train')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing dataset')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    if args.model == 'cyclegan':
        model = CycleGAN(
            input_channels=config['model']['input_channels']
        )
    elif args.model == 'stargan':
        model = StarGAN(
            input_channels=config['model']['input_channels'],
            num_domains=config['model']['num_domains']
        )
    elif args.model == 'cnn':
        model = SpectrogramCNN(
            input_channels=config['model']['input_channels'],
            hidden_channels=config['model']['hidden_channels']
        )
    elif args.model == 'lstm':
        model = SequentialLSTM(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    
    model = model.to(device)
    
    # Set up data paths
    data_dir = Path(args.data_dir)
    if args.model in ['cyclegan', 'stargan', 'lstm']:
        # Audio paths
        audio_paths = list(data_dir.glob('**/*.wav'))
        # Assuming labels are in a separate file or derived from directory structure
        labels = [1 if 'spoof' in str(p) else 0 for p in audio_paths]
        dataloaders = get_dataloaders(
            audio_paths=audio_paths,
            labels=labels,
            batch_size=config['training']['batch_size']
        )
    else:
        # Video paths
        video_paths = list(data_dir.glob('**/*.mp4'))
        labels = [1 if 'fake' in str(p) else 0 for p in video_paths]
        dataloaders = get_dataloaders(
            video_paths=video_paths,
            labels=labels,
            batch_size=config['training']['batch_size']
        )
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    
    if args.model in ['cyclegan', 'stargan']:
        model.set_optimizers(
            lr=config['training']['learning_rate']
        )
        optimizer = None  # Optimizers are handled within the GAN models
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['audio_train' if args.model in ['cyclegan', 'stargan', 'lstm'] else 'video_train'],
        val_loader=dataloaders['audio_val' if args.model in ['cyclegan', 'stargan', 'lstm'] else 'video_val'],
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # Train model
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    # Save training history
    torch.save(history, f'training_history_{args.model}.pt')
    
if __name__ == '__main__':
    main()