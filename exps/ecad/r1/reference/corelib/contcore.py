import torch
import torch.nn as nn
from typing import Optional
import corelib.utils
from corelib.sampler import BaseSampler

import torch.nn.functional as F
import numpy as np 
from torchvision import transforms
import os
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import metrics


class ContCore(nn.Module):
    # TODO: __init__ constructor argument에 config을 통으로 넣지 마시고, explicit한 type (e.g., torch, str, int) 등의 variable 넣어 주시기 바랍니다
    def __init__(self, 
                 backbone_name: str,
                 feature_extractor: nn.Module,
                 layers_to_extract_from: Optional[str],
                 feature_dim: int,
                 sampling_threshold: int,
                 sampler: BaseSampler,

                 chunk_size: int,
                 image_size: int,
                 save_path: str,
                 device,

        ):
        super(ContCore, self).__init__()
        self.backbone_name = backbone_name
        self.feature_extractor = feature_extractor
        self.sampling_threshold = sampling_threshold

        self.sampler = sampler
        self.layers_to_extract_from = layers_to_extract_from
        self.image_size = image_size
        self.device = device
        self.feature_dim = feature_dim

        self.chunk_size = self._get_max_chunk_size(chunk_size)
        self.features = []
        self.sample_ids = []
        self.coreset_features = None
        self.task_coreset_features = None

        self.save_model_path = os.path.join(save_path, 'model')
        self.save_metrics_path = os.path.join(save_path, 'metrics')
        self.save_seg_path = os.path.join(save_path, 'segmentations')

        os.makedirs(self.save_model_path, exist_ok=True)
        os.makedirs(self.save_metrics_path, exist_ok=True)
        os.makedirs(self.save_seg_path, exist_ok=True)

    def _get_max_chunk_size(self, 
                            chunk_size: int
        ):
        max_chunk_size = corelib.utils.auto_get_max_query_size_pdist(
            fea_dim=self.feature_dim,
            device_type=self.device.type,
            device_index=self.device.index,
            max_usage=0.25,
        )
        if chunk_size > max_chunk_size:
            chunk_size = max_chunk_size

        return chunk_size

    
    def _get_feature_shape(self):
        dummy_img = torch.zeros((1, 3, self.image_size, self.image_size)).to(self.device)
        dummy_sample_id = torch.zeros((1))
        features, _ = self._forward(dummy_img, dummy_sample_id, detach=False, accumulate_features=False)
        return features.squeeze().shape

    @torch.no_grad()
    def _extract_features(self,
                     images: torch.Tensor,
                     sample_ids: torch.Tensor=None,
                     return_patch_shape: bool=False,
        ):
        if isinstance(images, dict):
            images = images["image"]
        images = images.to(torch.float).to(self.device)
        sample_ids = sample_ids.to(self.device)

        if "dinov2" in self.backbone_name:
            features = self.feature_extractor['feature_aggregator'].prepare_tokens(images)
            batch_size = features.shape[0]
            for i, blk in enumerate(self.feature_extractor['feature_aggregator'].blocks): # [9]
                if i <= int(self.layers_to_extract_from[-1]):
                    features = blk(features)
                else:
                    continue
            if 'reg' in self.backbone_name:
                features = features[:, 1 + self.feature_extractor['feature_aggregator'].num_register_tokens:, :]

            fea_shape = int(np.sqrt(features.reshape(batch_size, -1, self.feature_dim).shape[1]))
            patch_shapes = features.reshape(fea_shape, fea_shape, -1).shape[:2]

            sample_ids = torch.broadcast_to(sample_ids[:,None], features.shape[:2]).reshape(-1)
            features = features.reshape(-1, features.shape[-1])
            features = F.normalize(features, dim=-1)
        else:
            features = self.feature_extractor['feature_aggregator'](images)

            features = [features[layer] for layer in self.layers_to_extract_from]

            features = [
                self.feature_extractor['patchifier'].patchify(x, return_spatial_info=True) for x in features
            ]
            patch_shapes = [x[1] for x in features]
            features = [x[0] for x in features]
            ref_num_patches = patch_shapes[0]

            for i in range(1, len(features)):
                _features = features[i]
                patch_dims = patch_shapes[i]

                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                features[i] = _features

            patch_shapes = patch_shapes[0]
            sample_ids = torch.broadcast_to(sample_ids[:,None], features[0].shape[:2]).reshape(-1)

            features = [x.reshape(-1, *x.shape[-3:]) for x in features]
            features = self.feature_extractor["preprocessing"](features)
            features = self.feature_extractor["preadapt_aggregator"](features)

        if return_patch_shape:
            return features, sample_ids, patch_shapes

        return features, sample_ids




    def _forward(self, 
                 images: torch.Tensor, 
                 sample_ids: torch.Tensor, 
                 accumulate_features: bool = True,  ## accumulate when train/sampling and don't when testing
                 return_patch_shape: bool = False):
        """
        images: (n, c, h, w)
        sample_ids: (n)
        output: (n*h//8*w//8, d)
        return patch features from given inputs
        """

        if isinstance(images, dict):
            images = images["image"]

        images = images.to(torch.float).to(self.device)
        sample_ids = sample_ids.to(self.device)

        features, sample_ids = self._extract_features(images, sample_ids, return_patch_shape)

        if accumulate_features:
            self.features.extend(features)
            self.sample_ids.extend(sample_ids)

        return features, sample_ids


    ## sample coresets from patch features
    def _sample_task_coresets(self):
        features = torch.vstack(self.features) ## n, 1024
        sample_ids = torch.vstack(self.sample_ids).squeeze()
        self.task_coreset_features, self.task_coreset_sample_ids = self.sampler.run(features, sample_ids, task_coreset=self.task_coreset_features, base_coreset=self.coreset_features)
        print(f"num. of task coreset features after sampling: {len(self.task_coreset_features)}")
        self.features = []
        self.sample_ids = []
        
    def _meta_greedy_sampling(self):
        print("Meta Greedy Sampling")
        self.coreset_features, self.coreset_sample_ids = self.sampler.meta_greedy_sampling(self.task_coreset_features, 
                                                                                            self.task_coreset_sample_ids,
                                                                                            base_coreset=self.coreset_features,
                                                                                            )
        self.task_coreset_features = None
        self.task_coreset_sample_ids = None
        print(f"num. of coreset features after sampling: {len(self.coreset_features)}")



    def update(
        self,
        images: torch.Tensor,
        sample_ids: torch.Tensor,
    ):
        self._forward(images, sample_ids, accumulate_features=True)
        ## sample if number of features exceed the capable chunk size
        # if len(self.features) > self.sampling_threshold:
        #     print('features are full, starting sampling\n')
        #     self._sample_task_coresets()



    def inference(self, image):
        self.feature_extractor.eval()
        self.anomaly_scorer.fit(detection_features=[self.coreset_features.clone().cpu().numpy()])
        scores, masks = self.predict(image)
        return scores, masks

    def predict(self, images):
        images = images.to(torch.float).to(self.device)
        batch_size = images.shape[0]
        sample_ids = torch.zeros(batch_size).to(torch.float).to(self.device) ## dummy input

        with torch.no_grad():
            features, _, patch_shapes = self._extract_features(images, sample_ids, return_patch_shape=True)
            features = np.asarray(features.cpu().numpy())

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.feature_extractor['patchifier'].unpatch_scores(
                image_scores, batchsize=batch_size
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.feature_extractor['patchifier'].score(image_scores)

            patch_scores = self.feature_extractor['patchifier'].unpatch_scores(
                patch_scores, batchsize=batch_size
            )
            patch_scores = patch_scores.reshape(batch_size, patch_shapes[0], patch_shapes[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        scores = [x for x in image_scores]
        masks = [x for x in masks]
        return scores, masks   
    
    def build_faiss(self, faiss_on_gpu, faiss_num_workers, anomaly_score_num_nn):
        nn_method = corelib.utils.FaissNN(
            faiss_on_gpu, faiss_num_workers
        )
        self.anomaly_scorer = corelib.utils.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.anomaly_segmentor = corelib.utils.RescaleSegmentor(
            device=self.device, target_size=self.image_size
        )



    def save_model(self, step):
        coreset_path = os.path.join(self.save_model_path, f'model_s{step}.pt')
        os.makedirs(os.path.dirname(coreset_path), exist_ok=True)
        ckpt = {"coreset": self.coreset_features.detach().clone(), 
                "sample_ids": self.coreset_sample_ids}
        torch.save(ckpt, coreset_path)
        print(f"Saved the coreset at {coreset_path}!")
    

    def load_model(self):
        ckpt = torch.load(os.path.join(self.save_model_path, 'model.pt'))
        self.coreset_features = ckpt["coreset"].cpu()
        self.coreset_sample_ids = ckpt["sample_ids"]
        print(f"loaded coreset")
        return ckpt

    def load_model_step(self, step):
        ckpt = torch.load(os.path.join(self.save_model_path, f'model_s{step}.pt'))
        self.coreset_features = ckpt["coreset"].cpu()
        self.coreset_sample_ids = ckpt["sample_ids"]
        print(f"loaded coreset")
        return ckpt

    def get_metric_results(self, scores, segmentations, labels_gt, masks_gt, classes):
        result_dict = {}
        for score, seg, lbl, msk_gt, cls in zip(scores, segmentations, labels_gt, masks_gt, classes):
            if not cls in result_dict.keys():
                result_dict[cls] = {'scores':[score],
                                        'segmentations':[seg],
                                        'labels_gt':[lbl],
                                        'masks_gt':[msk_gt]}
            else:
                result_dict[cls]['scores'].append(score)
                result_dict[cls]['segmentations'].append(seg)
                result_dict[cls]['labels_gt'].append(lbl)
                result_dict[cls]['masks_gt'].append(msk_gt)

        result_agg = []
        for k, v in result_dict.items():
            scores = v['scores']
            segmentations = v['segmentations']
            labels_gt = v['labels_gt']
            masks_gt = v['masks_gt']
            auroc_img, auroc_pix, f1_img, f1_pix, ap_img, ap_pix, aupro = self._evaluate(scores, segmentations, labels_gt, masks_gt)
            per_cls_dict = {
                "classname" : k,
                "auroc_img" : auroc_img * 100,
                "auroc_pix" : auroc_pix* 100,
                "f1_img" : f1_img* 100,
                "f1_pix" : f1_pix* 100,
                "ap_img" : ap_img* 100,
                "ap_pix" : ap_pix* 100,
                "aupro" : aupro* 100,
            }
            result_agg.append(per_cls_dict)

        return result_agg

    def _evaluate(self, scores, segmentations, labels_gt, masks_gt):
        
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # scores = np.mean(scores, axis=0)

        auroc_img = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"]

        segmentations = np.array(segmentations)
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        norm_segmentations = np.zeros_like(segmentations)
        for min_score, max_score in zip(min_scores, max_scores):
            norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
        norm_segmentations = norm_segmentations / len(scores)

        auroc_pix = metrics.compute_pixelwise_retrieval_metrics(norm_segmentations, masks_gt)["auroc"]

        sel_idxs = []
        for i in range(len(masks_gt)):
            if np.sum(masks_gt[i]) > 0:
                sel_idxs.append(i)
        aupro = metrics.compute_pixelwise_retrieval_metrics(
            [segmentations[i] for i in sel_idxs],
            [masks_gt[i] for i in sel_idxs],
        )["auroc"]

        norm_segmentations = np.array(norm_segmentations).ravel().tolist()
        masks_gt = np.array(masks_gt).ravel().astype(np.int32).tolist()

        f1_img = metrics.f1_score_max(labels_gt, scores)
        f1_pix = metrics.f1_score_max(masks_gt, norm_segmentations)
        ap_img = average_precision_score(labels_gt, scores)
        ap_pix = average_precision_score(masks_gt, norm_segmentations)

        return auroc_img, auroc_pix, f1_img, f1_pix, ap_img, ap_pix, aupro

    def _save_metrics(self, result_agg, filename):
        result_metric_names = list(result_agg[-1].keys())[1:]
        result_classnames = [results["classname"] for results in result_agg]
        result_scores = [list(results.values())[1:] for results in result_agg]
        corelib.utils.compute_and_store_final_results(
            self.save_metrics_path,
            result_scores,
            column_names=result_metric_names,
            row_names=result_classnames,
            file_name=filename
        )

    def _save_segmentations(self, segmentations, image_paths, mask_paths):
        segmentations = np.array(segmentations)
        min_scores = (
            segmentations.reshape(-1)
            .min(axis=-1)
            .reshape(-1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(-1)
            .max(axis=-1)
            .reshape(-1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        # segmentations = np.mean(segmentations, axis=0)
        
        def image_transform(image):
            in_std = np.array(
                [0.229, 0.224, 0.225]
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                [0.485, 0.456, 0.406]
            ).reshape(-1, 1, 1)

            transform_img = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=in_mean, std=in_std),
            ]
            transform_img = transforms.Compose(transform_img)

            image = transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            transform_mask = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
            transform_mask = transforms.Compose(transform_mask)            
            return transform_mask(mask).numpy()

        corelib.utils.plot_segmentation_images(
            self.save_seg_path,
            image_paths,
            segmentations,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )
        
        
        
        