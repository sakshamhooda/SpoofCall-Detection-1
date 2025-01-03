import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
import torch
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

class Visualizer:
    @staticmethod
    def plot_spectrogram(spec, title=None, save_path=None):
        """Plot mel spectrogram"""
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spec.squeeze().numpy() if torch.is_tensor(spec) else spec,
            y_axis='mel',
            x_axis='time',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        if title:
            plt.title(title)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_waveform(waveform, sr=16000, title=None, save_path=None):
        """Plot audio waveform"""
        plt.figure(figsize=(10, 3))
        waveform = waveform.squeeze().numpy() if torch.is_tensor(waveform) else waveform
        times = np.arange(len(waveform)) / sr
        plt.plot(times, waveform)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        if title:
            plt.title(title)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training metrics over time"""
        metrics = list(history[0]['train'].keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
            
        epochs = [h['epoch'] for h in history]
        
        for ax, metric in zip(axes, metrics):
            train_values = [h['train'][metric] for h in history]
            val_values = [h['val'][metric] for h in history]
            
            ax.plot(epochs, train_values, label=f'Train {metric}')
            ax.plot(epochs, val_values, label=f'Val {metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels or ['Real', 'Spoof'],
            yticklabels=labels or ['Real', 'Spoof']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_roc_curve(y_true, y_score, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_precision_recall_curve(y_true, y_score, save_path=None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_attention_weights(attention_weights, save_path=None):
        """Plot attention weights"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Attention Weights Visualization')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def visualize_gan_outputs(real_samples, fake_samples, save_path=None):
        """Visualize real and generated samples from GAN"""
        n_samples = min(5, real_samples.shape[0])
        
        fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
        
        for i in range(n_samples):
            # Plot real samples
            axes[0, i].imshow(real_samples[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Real', pad=10)
                
            # Plot fake samples
            axes[1, i].imshow(fake_samples[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Generated', pad=10)
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def plot_feature_importance(feature_names, importance_scores, save_path=None):
        """Plot feature importance scores"""
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(importance_scores)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.barh(pos, importance_scores[sorted_idx])
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()