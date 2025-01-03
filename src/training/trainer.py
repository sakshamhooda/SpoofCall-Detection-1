import torch
import numpy as np
from tqdm import tqdm
from ..evaluation.metrics import EvaluationMetrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = EvaluationMetrics()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Record loss and predictions
                total_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'accuracy': np.mean(np.array(all_predictions) == np.array(all_targets))
                })
        
        # Calculate metrics
        metrics = self.metrics.calculate_classification_metrics(
            np.array(all_targets),
            np.array(all_predictions)
        )
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probas = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Record predictions
                probas = torch.softmax(output, dim=1)[:, 1]
                predictions = torch.argmax(output, dim=1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics.calculate_classification_metrics(
            np.array(all_targets),
            np.array(all_predictions)
        )
        
        # Add additional metrics
        spoofing_metrics = self.metrics.calculate_spoofing_metrics(
            np.array(all_targets),
            np.array(all_probas)
        )
        metrics.update(spoofing_metrics)
        
        # Calculate EER
        metrics['eer'] = self.metrics.calculate_eer(
            np.array(all_targets),
            np.array(all_probas)
        )
        
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self, num_epochs, early_stopping_patience=5):
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Record history
            history = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            }
            training_history.append(history)
            
            # Print metrics
            print(f'\nTraining metrics:')
            for metric, value in train_metrics.items():
                print(f'{metric}: {value:.4f}')
                
            print(f'\nValidation metrics:')
            for metric, value in val_metrics.items():
                print(f'{metric}: {value:.4f}')
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print('\nEarly stopping triggered!')
                    break
        
        return training_history