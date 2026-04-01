import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        optimizer: Optional[optim.Optimizer] = None,
        learning_rate: float = 1e-4,
        use_replay: bool = False,
        replay_batch_size: int = 32
    ):
        self.model = model.to(device)
        self.device = device
        self.use_replay = use_replay
        self.replay_batch_size = replay_batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        self.model.train()
        history = {'loss': [], 'accuracy': []}
        
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
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.use_replay and hasattr(self.model, 'sample_from_buffer') and self.model.get_buffer_size() > 0:
                    try:
                        buf_images, buf_labels = self.model.sample_from_buffer(self.replay_batch_size)
                        buf_images, buf_labels = buf_images.to(self.device), buf_labels.to(self.device)
                        images = torch.cat([images, buf_images])
                        labels = torch.cat([labels, buf_labels])
                    except ValueError:
                        pass 
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                if not self.use_replay and active_classes is not None:
                    task_logits = outputs[:, active_classes]
                    start_class = active_classes[0]
                    target_mapped = labels - start_class
                    
                    loss = self.criterion(task_logits, target_mapped)
                    _, predicted_local = task_logits.max(1)
                    correct += predicted_local.eq(target_mapped).sum().item()
                    
                else:
                    loss = self.criterion(outputs, labels)
                    
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                
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
            
            avg_loss = total_loss / len(train_loader)
            avg_acc = 100. * correct / total
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.2f}%")
        
        return {
            'loss': sum(history['loss']) / len(history['loss']),
            'accuracy': sum(history['accuracy']) / len(history['accuracy']),
            'history': history
        }
    
    def set_learning_rate(self, lr: float):
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr