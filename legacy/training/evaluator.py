"""Evaluation utilities for continual learning experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


class Evaluator:
    """
    Evaluator for continual learning experiments.
    """
    def __init__(self, model: nn.Module, device: str = 'cuda', task_type: str = 'auto'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.image_criterion = nn.BCELoss()
        self.image_logit_criterion = nn.BCEWithLogitsLoss()
        self.task_type = task_type

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            images = batch['img']
            labels = batch['anomaly']
            masks = batch.get('img_mask')
            return images, labels, masks

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
            return images, labels, None

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _is_anomaly_output(self, outputs: Any) -> bool:
        return isinstance(outputs, dict) and (
            'image_score' in outputs or 'image_logit' in outputs
        )
        
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
        all_scores = []
        all_pixel_predictions = []
        all_pixel_labels = []
        anomaly_mode = False
        
        active_classes = getattr(test_loader.dataset, 'task_classes', None)
        
        pbar = tqdm(test_loader, desc=f"Evaluating Task {task_id}") if verbose else test_loader
        
        for batch in pbar:
            images, labels, masks = self._unpack_batch(batch)
            images, labels = images.to(self.device), labels.to(self.device)
            if torch.is_tensor(masks):
                masks = masks.to(self.device)
            
            outputs = self.model(images)

            if self.task_type != 'classification' and self._is_anomaly_output(outputs):
                anomaly_mode = True
                labels_float = labels.float().view(-1)
                image_logit = outputs.get('image_logit', None)
                image_score = outputs.get('image_score', None)

                if image_logit is not None:
                    image_logit = image_logit.view(-1)
                    loss = self.image_logit_criterion(image_logit, labels_float)
                    score_prob = torch.sigmoid(image_logit)
                elif image_score is not None:
                    score_prob = image_score.view(-1).clamp(1e-6, 1 - 1e-6)
                    loss = self.image_criterion(score_prob, labels_float)
                else:
                    raise ValueError("Anomaly output must provide image_score or image_logit")

                anomaly_map = outputs.get('anomaly_map', None)
                if torch.is_tensor(masks) and torch.is_tensor(anomaly_map):
                    target_masks = masks.float().clamp(0, 1)
                    if target_masks.ndim == 3:
                        target_masks = target_masks[:, None, :, :]
                    if target_masks.shape[-2:] != anomaly_map.shape[-2:]:
                        target_masks = F.interpolate(
                            target_masks,
                            size=anomaly_map.shape[-2:],
                            mode='nearest',
                        )

                    pixel_pred = (anomaly_map >= 0.5).long().view(-1)
                    pixel_true = (target_masks >= 0.5).long().view(-1)
                    all_pixel_predictions.extend(pixel_pred.cpu().numpy())
                    all_pixel_labels.extend(pixel_true.cpu().numpy())

                predicted = (score_prob >= 0.5).long()
                correct = predicted.eq(labels.long().view(-1)).sum().item()
                all_scores.extend(score_prob.detach().cpu().numpy())
            else:
                logits = outputs['logits'] if isinstance(outputs, dict) and 'logits' in outputs else outputs
                if not torch.is_tensor(logits):
                    raise TypeError("Model output must be a tensor or contain a 'logits' tensor")

                if active_classes is not None:
                    mask = torch.full_like(logits, float('-inf'))
                    mask[:, active_classes] = 0
                    logits = logits + mask
                
                loss = self.criterion(logits, labels)
                
                _, predicted = logits.max(1)
                correct = predicted.eq(labels).sum().item()
            
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
        
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        accuracy = 100. * total_correct / total_samples if total_samples > 0 else 0
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        if anomaly_mode:
            precision_scores, recall_scores, f1_scores, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='binary', zero_division=0
            )

            unique_labels = np.unique(all_labels)
            if len(unique_labels) > 1:
                auroc = roc_auc_score(all_labels, all_scores) * 100
                image_ap = average_precision_score(all_labels, all_scores) * 100
            else:
                auroc = 0.0
                image_ap = 0.0

            if all_pixel_labels:
                pixel_precision, pixel_recall, pixel_f1, _ = precision_recall_fscore_support(
                    np.array(all_pixel_labels),
                    np.array(all_pixel_predictions),
                    average='binary',
                    zero_division=0,
                )
                pixel_precision *= 100
                pixel_recall *= 100
                pixel_f1 *= 100
            else:
                pixel_precision = 0.0
                pixel_recall = 0.0
                pixel_f1 = 0.0

            return {
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision_scores * 100,
                'recall': recall_scores * 100,
                'f1': f1_scores * 100,
                'auroc': auroc,
                'image_ap': image_ap,
                'pixel_precision': pixel_precision,
                'pixel_recall': pixel_recall,
                'pixel_f1': pixel_f1,
            }

        precision_scores, recall_scores, f1_scores, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision_scores * 100,
            'recall': recall_scores * 100,
            'f1': f1_scores * 100,
            'auroc': 0.0,
            'image_ap': 0.0,
            'pixel_precision': 0.0,
            'pixel_recall': 0.0,
            'pixel_f1': 0.0,
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