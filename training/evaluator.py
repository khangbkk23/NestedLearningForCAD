"""
Evaluation Pipeline for Continual Learning
Updated with Logit Masking for Task-Incremental Learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np


class Evaluator:
    """
    Evaluator for continual learning experiments.
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    @torch.no_grad()
    def evaluate_task(
        self,
        test_loader: DataLoader,
        task_id: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on a single task with Logit Masking.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        
        # [QUAN TRỌNG] Lấy danh sách class của task này
        # (Ví dụ: Task 0 -> [0, 1])
        active_classes = getattr(test_loader.dataset, 'task_classes', None)
        
        pbar = tqdm(test_loader, desc=f"Evaluating Task {task_id}") if verbose else test_loader
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # --- [ĐOẠN CODE MỚI] LOGIT MASKING ---
            # Mục tiêu: Gán -infinity cho các class không thuộc task này
            # để hàm max() không bao giờ chọn nhầm.
            if active_classes is not None:
                mask = torch.full_like(outputs, float('-inf'))
                mask[:, active_classes] = 0
                outputs = outputs + mask
            # -------------------------------------
            
            loss = self.criterion(outputs, labels)
            
            # Predictions
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            
            # Track metrics
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / labels.size(0):.2f}%'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        accuracy = 100. * total_correct / total_samples if total_samples > 0 else 0
        
        # Calculate per-class metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Precision, Recall, F1
        from sklearn.metrics import precision_recall_fscore_support
        precision_scores, recall_scores, f1_scores, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision_scores * 100,
            'recall': recall_scores * 100,
            'f1': f1_scores * 100
        }
    
    @torch.no_grad()
    def evaluate_all_tasks(
        self,
        test_loaders: List[DataLoader],
        verbose: bool = True
    ) -> Dict[int, Dict[str, float]]:
        results = {}
        
        for task_id, test_loader in enumerate(test_loaders):
            metrics = self.evaluate_task(test_loader, task_id, verbose=verbose)
            results[task_id] = metrics
            
            if verbose:
                print(f"Task {task_id}: Acc={metrics['accuracy']:.2f}%, F1={metrics['f1']:.2f}%")
        
        # Calculate average metrics
        if results:
            avg_metrics = {
                'avg_accuracy': np.mean([m['accuracy'] for m in results.values()]),
                'avg_f1': np.mean([m['f1'] for m in results.values()]),
                'avg_loss': np.mean([m['loss'] for m in results.values()])
            }
        else:
            avg_metrics = {'avg_accuracy': 0, 'avg_f1': 0, 'avg_loss': 0}
        
        if verbose:
            print(f"\nAverage: Acc={avg_metrics['avg_accuracy']:.2f}%, F1={avg_metrics['avg_f1']:.2f}%")
        
        results['average'] = avg_metrics
        
        return results
    
    @torch.no_grad()
    def calculate_forgetting(
        self,
        test_loaders: List[DataLoader],
        baseline_accuracies: Dict[int, float]
    ) -> Dict[str, float]:
        """Calculate forgetting metrics."""
        # Tắt verbose để không in lại quá nhiều
        current_results = self.evaluate_all_tasks(test_loaders, verbose=False)
        
        forgetting = {}
        for task_id in baseline_accuracies:
            if task_id in current_results:
                # Forgetting = Max_Acc_Ever - Current_Acc
                forgetting[task_id] = max(0, baseline_accuracies[task_id] - current_results[task_id]['accuracy'])
        
        avg_forgetting = np.mean(list(forgetting.values())) if forgetting else 0.0
        
        return {
            'per_task_forgetting': forgetting,
            'average_forgetting': avg_forgetting
        }