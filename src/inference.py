import torch
import argparse
import yaml
from pathlib import Path
import logging
import numpy as np
import json
import pandas as pd
import time
from tqdm import tqdm

from models.cycle_gan import CycleGAN
from models.star_gan import StarGAN
from models.spectrogram_cnn import SpectrogramCNN
from models.sequential_lstm import SequentialLSTM
from data.data_loader import AudioDataset, VideoDataset
from utils.preprocessing import AudioPreprocessor, VideoPreprocessor
from utils.visualization import Visualizer
from evaluation.metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on spoof detection models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['cyclegan', 'stargan', 'cnn', 'lstm'],
                       help='Model to use')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input file or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--save_features', action='store_true',
                       help='Save extracted features')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold')
    return parser.parse_args()

class Inferencer:
    def __init__(self, model_type, model_path, config, device='cuda'):
        self.model_type = model_type
        self.device = device
        self.config = config
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)
        
        # Initialize preprocessors
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=config['data']['sample_rate']
        )
        self.video_preprocessor = VideoPreprocessor()
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
        # Initialize metrics
        self.metrics = EvaluationMetrics()
        
    def _initialize_model(self):
        """Initialize model based on type"""
        if self.model_type == 'cyclegan':
            return CycleGAN(
                input_channels=self.config['model']['input_channels']
            )
        elif self.model_type == 'stargan':
            return StarGAN(
                input_channels=self.config['model']['input_channels'],
                num_domains=self.config['model']['num_domains']
            )
        elif self.model_type == 'cnn':
            return SpectrogramCNN(
                input_channels=self.config['model']['input_channels'],
                hidden_channels=self.config['model']['hidden_channels']
            )
        elif self.model_type == 'lstm':
            return SequentialLSTM(
                input_size=self.config['model']['input_size'],
                hidden_size=self.config['model']['hidden_size'],
                num_layers=self.config['model']['num_layers'],
                dropout=self.config['model']['dropout']
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def process_audio(self, audio_path):
        """Process single audio file"""
        # Load and preprocess audio
        waveform = self.audio_preprocessor.load_and_preprocess(
            audio_path, 
            duration=self.config['data']['duration']
        )
        
        # Compute spectrogram
        spec = self.audio_preprocessor.compute_spectrogram(waveform)
        
        # Extract additional features
        features = self.audio_preprocessor.extract_features(waveform)
        
        # Add batch dimension
        spec = spec.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            if self.model_type in ['cyclegan', 'stargan']:
                output = self.model(spec)
                # For GANs, we use discriminator score as spoofing score
                score = self.model.discriminator(output)[0].item()
            else:
                logits = self.model(spec)
                score = torch.softmax(logits, dim=1)[0, 1].item()
                
        return {
            'score': score,
            'prediction': 'Spoof' if score > 0.5 else 'Real',
            'spectrogram': spec.cpu().numpy(),
            'features': features,
            'waveform': waveform.cpu().numpy()
        }
        
    def process_video(self, video_path):
        """Process single video file"""
        # Create video dataset for single file
        dataset = VideoDataset([video_path], [0])  # Label doesn't matter for inference
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        # Get frames
        frames, _ = next(iter(dataloader))
        frames = frames.to(self.device)
        
        # Get prediction and features
        with torch.no_grad():
            if hasattr(self.model, 'extract_features'):
                features, logits = self.model.extract_features(frames, return_logits=True)
            else:
                logits = self.model(frames)
                features = None
                
            score = torch.softmax(logits, dim=1)[0, 1].item()
            
        return {
            'score': score,
            'prediction': 'Deepfake' if score > 0.5 else 'Real',
            'frames': frames.cpu().numpy(),
            'features': features.cpu().numpy() if features is not None else None
        }
        
    def process_batch(self, input_paths, output_dir, threshold=0.5, save_features=False):
        """Process multiple inputs"""
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize summary statistics
        stats = {
            'total_files': len(input_paths),
            'processed_files': 0,
            'detected_spoofs': 0,
            'processing_time': 0,
            'average_confidence': 0
        }
        
        for path in tqdm(input_paths, desc='Processing files'):
            path = Path(path)
            logger.info(f"Processing {path}")
            
            try:
                start_time = time.time()
                
                if path.suffix in ['.wav', '.mp3', '.flac']:
                    result = self.process_audio(str(path))
                elif path.suffix in ['.mp4', '.avi']:
                    result = self.process_video(str(path))
                else:
                    logger.warning(f"Unsupported file type: {path.suffix}")
                    continue
                
                processing_time = time.time() - start_time
                
                # Add file info and processing time
                result['file_path'] = str(path)
                result['processing_time'] = processing_time
                
                # Update summary statistics
                stats['processed_files'] += 1
                stats['processing_time'] += processing_time
                stats['average_confidence'] += result['score']
                if result['score'] > threshold:
                    stats['detected_spoofs'] += 1
                
                # Save results
                if save_features and 'features' in result:
                    feature_path = output_dir / f"{path.stem}_features.npy"
                    np.save(feature_path, result['features'])
                
                # Generate visualizations
                vis_dir = output_dir / 'visualizations'
                vis_dir.mkdir(exist_ok=True)
                
                if 'spectrogram' in result:
                    self.visualizer.plot_spectrogram(
                        result['spectrogram'][0],
                        title=f"Prediction: {result['prediction']} (Score: {result['score']:.3f})",
                        save_path=vis_dir / f"{path.stem}_spectrogram.png"
                    )
                
                if 'waveform' in result:
                    self.visualizer.plot_waveform(
                        result['waveform'],
                        title=f"Waveform Analysis",
                        save_path=vis_dir / f"{path.stem}_waveform.png"
                    )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                continue
        
        # Finalize statistics
        if stats['processed_files'] > 0:
            stats['average_confidence'] /= stats['processed_files']
            stats['average_processing_time'] = stats['processing_time'] / stats['processed_files']
        
        # Save results and statistics
        self._save_results(results, stats, output_dir)
        
        return results, stats
    
    def _save_results(self, results, stats, output_dir):
        """Save results and statistics to files"""
        # Save detailed results as JSON
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        
        # Save summary statistics
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Create and save a summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'file_path': result['file_path'],
                'prediction': result['prediction'],
                'confidence_score': result['score'],
                'processing_time': result['processing_time']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_dir / 'summary.csv', index=False)
        
        # Generate and save summary visualizations
        if len(results) > 0:
            plt_dir = output_dir / 'plots'
            plt_dir.mkdir(exist_ok=True)
            
            # Score distribution
            self.visualizer.plot_feature_importance(
                ['Score Distribution'],
                [r['score'] for r in results],
                save_path=plt_dir / 'score_distribution.png'
            )
            
            # Processing time distribution
            self.visualizer.plot_feature_importance(
                ['Processing Time'],
                [r['processing_time'] for r in results],
                save_path=plt_dir / 'processing_time_distribution.png'
            )

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize inferencer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inferencer = Inferencer(
        model_type=args.model,
        model_path=args.model_path,
        config=config,
        device=device
    )
    
    # Get input paths
    input_path = Path(args.input_path)
    if input_path.is_file():
        input_paths = [input_path]
    else:
        input_paths = list(input_path.glob('**/*'))
    
    # Process inputs
    results, stats = inferencer.process_batch(
        input_paths,
        args.output_dir,
        threshold=args.threshold,
        save_features=args.save_features
    )
    
    # Print summary
    logger.info("\nProcessing Summary:")
    logger.info(f"Total files processed: {stats['processed_files']}")
    logger.info(f"Detected spoofs: {stats['detected_spoofs']}")
    logger.info(f"Average confidence score: {stats['average_confidence']:.3f}")
    logger.info(f"Average processing time: {stats['average_processing_time']:.3f} seconds")

if __name__ == '__main__':
    main()