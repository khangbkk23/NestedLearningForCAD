import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Any
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        optimizer: Optional[optim.Optimizer] = None,
        learning_rate: float = 1e-4,
        use_replay: bool = False,
        replay_batch_size: int = 32,
        task_type: str = 'auto',
        pixel_loss_weight: float = 0.2,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_replay = use_replay
        self.replay_batch_size = replay_batch_size
        self.task_type = task_type
        self.pixel_loss_weight = pixel_loss_weight
        
        self.criterion = nn.CrossEntropyLoss()
        self.image_criterion = nn.BCELoss()
        self.image_logit_criterion = nn.BCEWithLogitsLoss()
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

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
        
    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        self.model.train()
        history = {'loss': [], 'accuracy': [], 'image_loss': [], 'pixel_loss': []}
        
        active_classes = getattr(train_loader.dataset, 'task_classes', None)
        
        if verbose:
            info_str = f"Task {task_id}"
            if active_classes:
                info_str += f" (Classes: {active_classes})"
            print(f"Training {info_str} for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}/{epochs}") if verbose else train_loader
            
            for batch_idx, batch in enumerate(pbar):
                images, labels, masks = self._unpack_batch(batch)
                images, labels = images.to(self.device), labels.to(self.device)
                if torch.is_tensor(masks):
                    masks = masks.to(self.device)

                can_use_replay = (
                    self.use_replay
                    and masks is None
                    and hasattr(self.model, 'sample_from_buffer')
                    and self.model.get_buffer_size() > 0
                )

                if can_use_replay:
                    try:
                        replay_batch = self.model.sample_from_buffer(self.replay_batch_size)
                        if replay_batch is None:
                            raise ValueError("Replay buffer returned no data")

                        if isinstance(replay_batch, (tuple, list)) and len(replay_batch) >= 2:
                            buf_images, buf_labels = replay_batch[0], replay_batch[1]
                        else:
                            raise ValueError("Replay buffer returned unexpected format")

                        buf_images, buf_labels = buf_images.to(self.device), buf_labels.to(self.device)
                        images = torch.cat([images, buf_images])
                        labels = torch.cat([labels, buf_labels])
                    except ValueError:
                        pass 
                
                self.optimizer.zero_grad()
                outputs = self.model(images)

                if self.task_type != 'classification' and self._is_anomaly_output(outputs):
                    labels_float = labels.float().view(-1)
                    image_logit = outputs.get('image_logit', None)
                    image_score = outputs.get('image_score', None)

                    if image_logit is not None:
                        image_logit = image_logit.view(-1)
                        image_loss = self.image_logit_criterion(image_logit, labels_float)
                        score_prob = torch.sigmoid(image_logit)
                    elif image_score is not None:
                        score_prob = image_score.view(-1).clamp(1e-6, 1 - 1e-6)
                        image_loss = self.image_criterion(score_prob, labels_float)
                    else:
                        raise ValueError("Anomaly output must provide image_score or image_logit")

                    pixel_loss_value = 0.0
                    loss = image_loss

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

                        pixel_loss = F.binary_cross_entropy(
                            anomaly_map.clamp(1e-6, 1 - 1e-6),
                            target_masks,
                        )
                        loss = loss + self.pixel_loss_weight * pixel_loss
                        pixel_loss_value = pixel_loss.item()

                    predicted = (score_prob >= 0.5).long()
                    correct += predicted.eq(labels.long().view(-1)).sum().item()
                    history['image_loss'].append(image_loss.item())
                    history['pixel_loss'].append(pixel_loss_value)
                else:
                    logits = outputs['logits'] if isinstance(outputs, dict) and 'logits' in outputs else outputs
                    if not torch.is_tensor(logits):
                        raise TypeError("Model output must be a tensor or contain a 'logits' tensor")

                    if not self.use_replay and active_classes is not None:
                        task_logits = logits[:, active_classes]
                        start_class = active_classes[0]
                        target_mapped = labels - start_class
                        
                        loss = self.criterion(task_logits, target_mapped)
                        _, predicted_local = task_logits.max(1)
                        correct += predicted_local.eq(target_mapped).sum().item()
                    else:
                        loss = self.criterion(logits, labels)
                        _, predicted = logits.max(1)
                        correct += predicted.eq(labels).sum().item()
                    
                    history['image_loss'].append(0.0)
                    history['pixel_loss'].append(0.0)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                if self.use_replay and hasattr(self.model, 'add_to_buffer'):
                    if batch_idx % 5 == 0:
                        self.model.add_to_buffer(images, labels, task_id)
                
                total_loss += loss.item()
                total += labels.size(0)
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100. * correct / total:.2f}%'
                    })
            
            avg_loss = total_loss / max(len(train_loader), 1)
            avg_acc = 100. * correct / max(total, 1)
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.2f}%")
        
        return {
            'loss': sum(history['loss']) / max(len(history['loss']), 1),
            'accuracy': sum(history['accuracy']) / max(len(history['accuracy']), 1),
            'image_loss': sum(history['image_loss']) / max(len(history['image_loss']), 1),
            'pixel_loss': sum(history['pixel_loss']) / max(len(history['pixel_loss']), 1),
            'history': history
        }
    
    def set_learning_rate(self, lr: float):
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr