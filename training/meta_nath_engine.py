import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score

class MetaNATHEngine:
    """
    Engine chuyên biệt dành cho MetaNATHCore.
    Không dùng loss, không dùng optimizer truyền thống.
    Mọi cập nhật diễn ra thông qua Forward Pass (Phase 1 & 2).
    """
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval() # Luôn đóng băng Backbone ở Phase 1-2

    def train_task(self, train_loader, task_id: int, epochs: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Phase 1 & 2: Data Streaming & Consolidation
        """
        self.model.eval()
        total_samples = 0
        approved_count = 0
        total_surprise = 0.0
        total_acc = 0.0
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {task_id} (Phase 1-2) Ep {epoch+1}/{epochs}") if verbose else train_loader
            for batch in pbar:
                images = batch['img'] if isinstance(batch, dict) else batch[0]
                images = images.to(self.device)
                
                with torch.no_grad():
                    out = self.model(images, task_id=task_id, update_coreset=True)
                
                approved_count += int(out['approved']) * images.size(0)
                total_samples += images.size(0)
                total_surprise += out['surprise']
                total_acc += out['acc_score']
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'approved_rate': f"{approved_count/total_samples:.1%}",
                        'coreset_size': f"{len(self.model.coreset)}/{self.model.coreset.max_size}",
                        'acc': f"{out['acc_score']:.3f}"
                    })
                    
        return {
            "approved_rate": approved_count / max(1, total_samples),
            "coreset_size": len(self.model.coreset),
            "avg_surprise": total_surprise / max(1, len(train_loader)),
            "avg_acc": total_acc / max(1, len(train_loader)),
            # Dummy fields to keep run_experiment.py happy
            "loss": 0.0,
            "accuracy": 0.0,
            "image_loss": 0.0,
            "pixel_loss": 0.0,
        }

    def evaluate_task(self, test_loader, task_id: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluation Phase
        """
        self.model.eval()
        all_image_scores = []
        all_image_labels = []
        all_pixel_scores = []
        all_pixel_labels = []
        
        pbar = tqdm(test_loader, desc=f"Eval Task {task_id}") if verbose else test_loader
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['img'] if isinstance(batch, dict) else batch[0]
                labels = batch['anomaly'] if isinstance(batch, dict) else batch[1]
                masks = batch.get('img_mask', None) if isinstance(batch, dict) else (batch[2] if len(batch)>2 else None)
                
                images = images.to(self.device)
                
                out = self.model.score_image(images)
                results = out["batch"] if "batch" in out else [out]
                
                for i, res in enumerate(results):
                    all_image_scores.append(res['s_img'])
                    all_image_labels.append(labels[i].item())
                    
                    if masks is not None:
                        mask_flat = (masks[i].cpu().numpy() > 0).astype(int).flatten()
                        map_flat = res['anomaly_map'].cpu().numpy().flatten()
                        all_pixel_scores.extend(map_flat)
                        all_pixel_labels.extend(mask_flat)

        image_auroc = roc_auc_score(all_image_labels, all_image_scores) if len(np.unique(all_image_labels)) > 1 else 0.0
        
        pixel_auroc = 0.0
        pixel_aupr = 0.0
        if len(all_pixel_labels) > 0 and len(np.unique(all_pixel_labels)) > 1:
            if len(all_pixel_labels) > 2000000:
                idx = np.random.choice(len(all_pixel_labels), 2000000, replace=False)
                sampled_labels = np.array(all_pixel_labels)[idx]
                sampled_scores = np.array(all_pixel_scores)[idx]
                pixel_auroc = roc_auc_score(sampled_labels, sampled_scores)
                pixel_aupr = average_precision_score(sampled_labels, sampled_scores)
            else:
                pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
                pixel_aupr = average_precision_score(all_pixel_labels, all_pixel_scores)

        if verbose:
            print(f"Task {task_id} Eval: Image AUROC: {image_auroc:.4f} | Pixel AUPR: {pixel_aupr:.4f}")
            
        return {
            "image_auroc": image_auroc * 100,
            "pixel_auroc": pixel_auroc * 100,
            "pixel_aupr": pixel_aupr * 100,
            "loss": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "image_ap": 0.0,
            "pixel_f1": 0.0,
            "auroc": image_auroc * 100
        }
