import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from torchmetrics.classification import AveragePrecision


class Evaluator:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

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
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        all_scores = []
        pixel_ap_metric = AveragePrecision(task="binary", thresholds=256).to(self.device)
        pixel_ap_updates = 0
        
        active_classes = getattr(test_loader.dataset, 'task_classes', None)
        
        pbar = tqdm(test_loader, desc=f"Evaluating Task {task_id}") if verbose else test_loader
        
        for batch in pbar:
            images, labels, masks = self._unpack_batch(batch)
            images, labels = images.to(self.device), labels.to(self.device)
            if torch.is_tensor(masks):
                masks = masks.to(self.device)
            
            outputs = self.model(images)

            if self._is_anomaly_output(outputs):
                labels_float = labels.float().view(-1)
                image_logit = outputs.get('image_logit', None)
                image_score = outputs.get('image_score', None)

                if image_logit is not None:
                    image_logit = image_logit.view(-1)
                    loss = F.binary_cross_entropy_with_logits(image_logit, labels_float)
                    score_prob = torch.sigmoid(image_logit)
                elif image_score is not None:
                    score_prob = image_score.view(-1).clamp(1e-6, 1 - 1e-6)
                    loss = F.binary_cross_entropy(score_prob, labels_float)
                else:
                    raise ValueError("Anomaly output must provide image_score or image_logit")

                predicted = (score_prob >= 0.5).long()
                correct = predicted.eq(labels.long().view(-1)).sum().item()

                all_scores.extend(score_prob.detach().cpu().numpy().tolist())
                all_predictions.extend(predicted.detach().cpu().numpy().tolist())
                all_labels.extend(labels.long().view(-1).detach().cpu().numpy().tolist())

                anomaly_map = outputs.get('anomaly_map', None)
                if torch.is_tensor(masks) and torch.is_tensor(anomaly_map):
                    target_masks = masks.float().clamp(0, 1)
                    if target_masks.ndim == 3:
                        target_masks = target_masks[:, None, :, :]
                    anomaly_scores = anomaly_map.float()
                    if anomaly_scores.ndim == 3:
                        anomaly_scores = anomaly_scores[:, None, :, :]
                    if target_masks.shape[-2:] != anomaly_scores.shape[-2:]:
                        target_masks = F.interpolate(
                            target_masks,
                            size=anomaly_scores.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )

                    target_masks = (target_masks >= 0.5).to(torch.long)

                    pixel_ap_metric.update(anomaly_scores.detach(), target_masks.detach())
                    pixel_ap_updates += 1
            else:
                logits = outputs['logits'] if isinstance(outputs, dict) and 'logits' in outputs else outputs
                if active_classes is not None:
                    mask = torch.full_like(logits, float('-inf'))
                    mask[:, active_classes] = 0
                    logits = logits + mask

                loss = self.criterion(logits, labels)
                _, predicted = logits.max(1)
                correct = predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / labels.size(0):.2f}%'
                })

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
        
        avg_loss = total_loss / max(len(test_loader), 1)
        accuracy = 100. * total_correct / total_samples if total_samples > 0 else 0
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        precision_scores, recall_scores, f1_scores, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='binary' if np.unique(all_labels).size <= 2 else 'macro',
            zero_division=0,
        )

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision_scores * 100,
            'recall': recall_scores * 100,
            'f1': f1_scores * 100,
        }

        if len(all_scores) > 1 and np.unique(all_labels).size > 1:
            metrics['image_auroc'] = float(roc_auc_score(all_labels, np.array(all_scores))) * 100
            metrics['image_ap'] = float(average_precision_score(all_labels, np.array(all_scores))) * 100
        else:
            metrics['image_auroc'] = 0.0
            metrics['image_ap'] = 0.0

        if pixel_ap_updates > 0:
            metrics['pixel_ap'] = float(pixel_ap_metric.compute().detach().cpu()) * 100
        else:
            metrics['pixel_ap'] = 0.0

        return metrics
    
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
                print(
                    f"Task {task_id}: "
                    f"Acc={metrics['accuracy']:.2f}% "
                    f"F1={metrics['f1']:.2f}% "
                    f"AUROC={metrics['image_auroc']:.2f}% "
                    f"Pixel-AP={metrics['pixel_ap']:.2f}%"
                )
        
        # Calculate average metrics
        if results:
            avg_metrics = {
                'avg_accuracy': np.mean([m['accuracy'] for m in results.values()]),
                'avg_f1': np.mean([m['f1'] for m in results.values()]),
                'avg_loss': np.mean([m['loss'] for m in results.values()]),
                'avg_image_auroc': np.mean([m['image_auroc'] for m in results.values()]),
                'avg_image_ap': np.mean([m['image_ap'] for m in results.values()]),
                'avg_pixel_ap': np.mean([m['pixel_ap'] for m in results.values()]),
            }
        else:
            avg_metrics = {
                'avg_accuracy': 0,
                'avg_f1': 0,
                'avg_loss': 0,
                'avg_image_auroc': 0,
                'avg_image_ap': 0,
                'avg_pixel_ap': 0,
            }
        
        if verbose:
            print(
                f"\nAverage: "
                f"Acc={avg_metrics['avg_accuracy']:.2f}% "
                f"F1={avg_metrics['avg_f1']:.2f}% "
                f"AUROC={avg_metrics['avg_image_auroc']:.2f}% "
                f"Pixel-AP={avg_metrics['avg_pixel_ap']:.2f}%"
            )
        
        results['average'] = avg_metrics
        
        return results
    
    @torch.no_grad()
    def calculate_forgetting(
        self,
        test_loaders: List[DataLoader],
        baseline_accuracies: Dict[int, float]
    ) -> Dict[str, float]:
        current_results = self.evaluate_all_tasks(test_loaders, verbose=False)
        
        forgetting = {}
        for task_id in baseline_accuracies:
            if task_id in current_results:
                forgetting[task_id] = max(0, baseline_accuracies[task_id] - current_results[task_id]['accuracy'])
        
        avg_forgetting = np.mean(list(forgetting.values())) if forgetting else 0.0
        
        return {
            'per_task_forgetting': forgetting,
            'average_forgetting': avg_forgetting
        }