import torch
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_curve, auc

class EvaluationMetrics:
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred):
        """
        Calculate basic classification metrics:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    @staticmethod
    def calculate_eer(y_true, y_pred_proba):
        """
        Calculate Equal Error Rate (EER)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        fnr = 1 - tpr
        
        # Find the threshold where FAR = FRR
        eer = fpr[np.nanargmin(np.absolute(fpr - fnr))]
        
        return eer
    
    @staticmethod
    def calculate_frechet_inception_distance(real_features, generated_features):
        """
        Calculate Fréchet Inception Distance (FID)
        """
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        
        diff = mu1 - mu2
        covmean = np.sqrt(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return fid
    
    @staticmethod
    def calculate_inception_score(generated_features, n_split=10):
        """
        Calculate Inception Score (IS)
        """
        scores = []
        split_size = generated_features.shape[0] // n_split
        
        for i in range(n_split):
            split = generated_features[i * split_size:(i + 1) * split_size]
            
            # Calculate p(y|x)
            p_yx = torch.softmax(torch.tensor(split), dim=1)
            p_y = p_yx.mean(dim=0)
            
            # Calculate KL divergence
            kl_div = entropy(p_yx, p_y.unsqueeze(0), axis=1)
            scores.append(np.exp(np.mean(kl_div)))
            
        return np.mean(scores), np.std(scores)
    
    @staticmethod
    def calculate_wasserstein_distance(real_features, generated_features):
        """
        Calculate Wasserstein Distance
        """
        # Sort both distributions
        real_features = np.sort(real_features, axis=0)
        generated_features = np.sort(generated_features, axis=0)
        
        # Calculate Wasserstein distance
        return np.mean(np.abs(real_features - generated_features))
    
    @staticmethod
    def calculate_attention_distance(attention_weights):
        """
        Calculate average attention distance for Vision Transformers
        """
        n_heads = attention_weights.shape[1]
        seq_len = attention_weights.shape[2]
        
        # Create position matrix
        pos_i = np.arange(seq_len)
        pos_j = pos_i[:, np.newaxis]
        distance_matrix = np.abs(pos_i - pos_j)
        
        # Calculate average attention distance
        attention_dist = 0
        for h in range(n_heads):
            attention_dist += np.sum(attention_weights[:, h] * distance_matrix)
            
        return attention_dist / (n_heads * seq_len * seq_len)
    
    @staticmethod
    def calculate_spoofing_metrics(y_true, y_pred_proba):
        """
        Calculate voice spoofing specific metrics:
        - t-DCF (tandem Detection Cost Function)
        - SRR (Spoofing Recognition Rate)
        - LLRC (Log-Likelihood Ratio Cost)
        """
        # Calculate t-DCF
        c_miss = 1  # Cost of miss
        c_fa = 1    # Cost of false alarm
        p_target = 0.5  # Prior probability of target
        p_non_target = 1 - p_target
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        fnr = 1 - tpr
        
        t_dcf = c_miss * p_target * fnr + c_fa * p_non_target * fpr
        min_t_dcf = np.min(t_dcf)
        
        # Calculate SRR
        threshold = 0.5  # Can be adjusted based on requirements
        y_pred = (y_pred_proba >= threshold).astype(int)
        srr = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
        
        # Calculate LLRC
        eps = 1e-10  # Small constant to avoid log(0)
        llrc = -np.mean(y_true * np.log(y_pred_proba + eps) + 
                       (1 - y_true) * np.log(1 - y_pred_proba + eps))
        
        return {
            't_dcf': min_t_dcf,
            'srr': srr,
            'llrc': llrc
        }