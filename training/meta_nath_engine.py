import torch
import numpy as np
import time
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score

class MetaNATHEngine:
    """
    Engine chuyên biệt dành cho MetaNATHCore.
    Không dùng loss, không dùng optimizer truyền thống.
    Mọi cập nhật diễn ra thông qua Forward Pass (Phase 1 & 2).
    """
    def __init__(
        self,
        model,
        device="cuda",
        nearest_neighbors: int = 2,
        pixel_score_norm: str = "none",
        gaussian_smoothing_sigma: float = 0.0,
    ):
        self.model = model
        self.device = device
        self.nearest_neighbors = max(1, int(nearest_neighbors))
        self.pixel_score_norm = str(pixel_score_norm or "none").lower()
        self.gaussian_smoothing_sigma = max(0.0, float(gaussian_smoothing_sigma))
        if self.pixel_score_norm not in {"none", "minmax", "robust_z"}:
            raise ValueError(
                "pixel_score_norm must be one of: none, minmax, robust_z"
            )
        self.model.to(device)
        self.model.eval() # Luôn đóng băng Backbone ở Phase 1-2

    def train_task(self, train_loader, task_id: int, epochs: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Phase 1 & 2: Data Streaming & Consolidation
        """
        self.model.eval()
        total_samples = 0
        normal_samples = 0
        skipped_anomaly_count = 0
        approved_count = 0
        coreset_update_count = 0
        total_surprise = 0.0
        total_acc = 0.0
        acc_scores = []
        batch_steps = 0
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {task_id} (Phase 1-2) Ep {epoch+1}/{epochs}") if verbose else train_loader
            for batch in pbar:
                images = batch['img'] if isinstance(batch, dict) else batch[0]
                labels = batch['anomaly'] if isinstance(batch, dict) else batch[1]
                total_samples += images.size(0)

                normal_mask = labels == 0
                normal_count = int(normal_mask.sum().item())
                skipped_anomaly_count += images.size(0) - normal_count

                if normal_count == 0:
                    if verbose and isinstance(pbar, tqdm):
                        pbar.set_postfix({
                            'approved_rate': f"{approved_count/max(1, normal_samples):.1%}",
                            'normal': f"{normal_samples}/{total_samples}",
                            'coreset_size': f"{len(self.model.coreset)}/{self.model.coreset.max_size}",
                            'acc': "n/a",
                        })
                    continue

                images = images[normal_mask].to(self.device)
                
                with torch.no_grad():
                    out = self.model(images, task_id=task_id, update_coreset=True)
                
                normal_samples += images.size(0)
                approved_count += int(out['approved']) * images.size(0)
                coreset_update_count += int(out.get('coreset_n_updated', 0))
                total_surprise += out['surprise']
                total_acc += out['acc_score']
                acc_scores.append(out['acc_score'])
                batch_steps += 1
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'approved_rate': f"{approved_count/max(1, normal_samples):.1%}",
                        'normal': f"{normal_samples}/{total_samples}",
                        'coreset_size': f"{len(self.model.coreset)}/{self.model.coreset.max_size}",
                        'acc': f"{out['acc_score']:.3f}"
                    })

        coreset_stats = self.model.coreset.stats()
        acc_array = np.array(acc_scores, dtype=np.float32)
                    
        return {
            "approved_rate": approved_count / max(1, normal_samples),
            "acc_approval_rate": approved_count / max(1, normal_samples),
            "total_samples": total_samples,
            "normal_update_samples": normal_samples,
            "skipped_anomaly_count": skipped_anomaly_count,
            "coreset_update_count": coreset_update_count,
            "coreset_size": len(self.model.coreset),
            "coreset_task_counts": coreset_stats.get("task_counts", {}),
            "avg_surprise": total_surprise / max(1, batch_steps),
            "avg_acc": total_acc / max(1, batch_steps),
            "acc_min": float(acc_array.min()) if acc_array.size else 0.0,
            "acc_p10": float(np.percentile(acc_array, 10)) if acc_array.size else 0.0,
            "acc_p50": float(np.percentile(acc_array, 50)) if acc_array.size else 0.0,
            "acc_p90": float(np.percentile(acc_array, 90)) if acc_array.size else 0.0,
            "acc_max": float(acc_array.max()) if acc_array.size else 0.0,
        }

    def evaluate_task(
        self,
        test_loader,
        task_id: int,
        verbose: bool = True,
        pixel_sample_limit: int = 10000,
    ) -> Dict[str, Any]:
        """
        Evaluation Phase
        """
        self.model.eval()
        all_image_scores = []
        all_image_labels = []
        all_pixel_scores = []
        all_pixel_labels = []
        eval_start = time.time()
        eval_num_images = 0
        
        pbar = tqdm(test_loader, desc=f"Eval Task {task_id}") if verbose else test_loader
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['img'].to(self.device) if isinstance(batch, dict) else batch[0].to(self.device)
                labels = batch['anomaly'] if isinstance(batch, dict) else batch[1]
                masks = batch.get('img_mask', None) if isinstance(batch, dict) else (batch[2] if len(batch)>2 else None)
                
                out = self.model.score_image(images, b=self.nearest_neighbors)
                results = out["batch"] if "batch" in out else [out]
                
                for i, res in enumerate(results):
                    all_image_scores.append(res['s_img'])
                    all_image_labels.append(labels[i].item())
                    eval_num_images += 1
                    
                    if masks is not None:
                        # Lấy mask thực tế (0 hoặc 1)
                        m = masks[i].cpu().numpy()
                        mask_flat = (m > 0.5).astype(np.uint8).flatten()
                        anomaly_map = self._postprocess_anomaly_map(res['anomaly_map'])
                        map_flat = anomaly_map.numpy().flatten()
                        
                        # Sampling ngay tại đây để tránh list khổng lồ
                        # Mỗi ảnh lấy max 10,000 pixel ngẫu nhiên
                        if len(mask_flat) > pixel_sample_limit:
                            indices = np.random.choice(len(mask_flat), pixel_sample_limit, replace=False)
                            all_pixel_scores.extend(map_flat[indices])
                            all_pixel_labels.extend(mask_flat[indices])
                        else:
                            all_pixel_scores.extend(map_flat)
                            all_pixel_labels.extend(mask_flat)

        image_auroc = roc_auc_score(all_image_labels, all_image_scores) if len(np.unique(all_image_labels)) > 1 else 0.0
        
        pixel_auroc = 0.0
        pixel_aupr = 0.0
        if len(all_pixel_labels) > 0 and len(np.unique(all_pixel_labels)) > 1:
            pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
            pixel_aupr = average_precision_score(all_pixel_labels, all_pixel_scores)

        if verbose:
            print(f"Task {task_id} Eval: Image AUROC: {image_auroc:.4f} | Pixel AUPR: {pixel_aupr:.4f}")

        image_ap = average_precision_score(all_image_labels, all_image_scores) if len(np.unique(all_image_labels)) > 1 else 0.0
        eval_seconds = time.time() - eval_start
            
        return {
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "pixel_aupr": pixel_aupr,
            "image_ap": image_ap,
            "auroc": image_auroc,
            "eval_num_images": eval_num_images,
            "eval_seconds": eval_seconds,
        }

    def _postprocess_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Optional pixel-score calibration for pixel AUROC/AUPR experiments."""
        out = anomaly_map.detach().float().cpu()

        sigma = self.gaussian_smoothing_sigma
        if sigma > 0:
            out = self._gaussian_smooth(out, sigma=sigma)

        if self.pixel_score_norm == "minmax":
            min_val = out.min()
            max_val = out.max()
            out = (out - min_val) / (max_val - min_val).clamp_min(1e-6)
        elif self.pixel_score_norm == "robust_z":
            median = out.median()
            mad = (out - median).abs().median().clamp_min(1e-6)
            out = (out - median) / (1.4826 * mad)

        return out

    @staticmethod
    def _gaussian_smooth(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
        radius = max(1, int(round(3 * sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=anomaly_map.dtype)
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, *kernel_2d.shape)

        x = anomaly_map.view(1, 1, *anomaly_map.shape)
        x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        smoothed = F.conv2d(x, kernel_2d)
        return smoothed.squeeze(0).squeeze(0)
