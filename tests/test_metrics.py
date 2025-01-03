import unittest
import numpy as np
import torch
from src.evaluation.metrics import EvaluationMetrics

class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = EvaluationMetrics()
        
        # Create test data
        np.random.seed(42)
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        self.y_pred_proba = np.random.rand(8)
        
        # Create test features
        self.real_features = np.random.randn(100, 64)
        self.generated_features = np.random.randn(100, 64)
        self.attention_weights = np.random.rand(8, 4, 16, 16)  # batch, heads, seq_len, seq_len
        
    def test_classification_metrics(self):
        # Test basic classification metrics
        metrics = self.metrics.calculate_classification_metrics(
            self.y_true,
            self.y_pred
        )
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
            
    def test_equal_error_rate(self):
        # Test EER calculation
        eer = self.metrics.calculate_eer(self.y_true, self.y_pred_proba)
        
        self.assertIsInstance(eer, float)
        self.assertGreaterEqual(eer, 0.0)
        self.assertLessEqual(eer, 1.0)
        
    def test_frechet_inception_distance(self):
        # Test FID calculation
        fid = self.metrics.calculate_frechet_inception_distance(
            self.real_features,
            self.generated_features
        )
        
        self.assertIsInstance(fid, float)
        self.assertGreaterEqual(fid, 0.0)
        
    def test_inception_score(self):
        # Test IS calculation
        mean_score, std_score = self.metrics.calculate_inception_score(
            self.generated_features,
            n_split=2
        )
        
        self.assertIsInstance(mean_score, float)
        self.assertIsInstance(std_score, float)
        self.assertGreaterEqual(mean_score, 1.0)
        self.assertGreaterEqual(std_score, 0.0)
        
    def test_wasserstein_distance(self):
        # Test Wasserstein distance calculation
        w_dist = self.metrics.calculate_wasserstein_distance(
            self.real_features,
            self.generated_features
        )
        
        self.assertIsInstance(w_dist, float)
        self.assertGreaterEqual(w_dist, 0.0)
        
    def test_attention_metrics(self):
        # Test attention distance calculation
        attn_dist = self.metrics.calculate_attention_distance(
            self.attention_weights
        )
        
        self.assertIsInstance(attn_dist, float)
        self.assertGreaterEqual(attn_dist, 0.0)
        
    def test_spoofing_metrics(self):
        # Test spoofing-specific metrics
        metrics = self.metrics.calculate_spoofing_metrics(
            self.y_true,
            self.y_pred_proba
        )
        
        required_metrics = ['t_dcf', 'srr', 'llrc']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            
    def test_input_validation(self):
        # Test with invalid input shapes
        with self.assertRaises(ValueError):
            self.metrics.calculate_classification_metrics(
                np.array([0, 1]),
                np.array([0, 1, 2])
            )
            
        # Test with invalid input types
        with self.assertRaises(TypeError):
            self.metrics.calculate_frechet_inception_distance(
                [1, 2, 3],
                [4, 5, 6]
            )
            
        # Test with invalid probability values
        with self.assertRaises(ValueError):
            self.metrics.calculate_spoofing_metrics(
                self.y_true,
                np.array([0.5, 0.3, 1.2, 0.7, 0.1, 0.4, 0.8, 0.6])  # Invalid prob > 1
            )
            
        # Test with mismatched feature dimensions
        with self.assertRaises(ValueError):
            self.metrics.calculate_wasserstein_distance(
                np.random.randn(100, 64),
                np.random.randn(100, 32)
            )
            
    def test_metric_consistency(self):
        # Test consistency of metrics with known values
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])
        
        # Perfect predictions should give ideal scores
        metrics = self.metrics.calculate_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)
        
        # Test with known FID value for identical distributions
        n_samples = 1000
        n_dims = 10
        features = np.random.randn(n_samples, n_dims)
        fid = self.metrics.calculate_frechet_inception_distance(features, features)
        self.assertAlmostEqual(fid, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()