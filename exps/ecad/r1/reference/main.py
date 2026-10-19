#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
without dist mat (iterative)
build task coreset (tc)
base-task meta-greedy sampling : with dist mat
"""

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import json
from collections import OrderedDict
from datasets.common import *
from utils import *
import copy
import corelib.contcore as contcore
import corelib.utils
import time
import corelib.models as models
import corelib

def main():
    config = get_argparse()
    inc_cls_num = list(map(int, config.incremental_class_list.split('-')))
    config.incremental_class_nums = [inc_cls_num[0]] + [inc_cls_num[1]] * inc_cls_num[2]
    config.device = corelib.utils.set_torch_device(config.gpus)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    trainset_list, testset_list = get_datasets(config)

    device = torch.device(f"cuda:0")

    model = corelib.models.get_contcore_model(
        backbone_name=config.backbone_name,
        image_size=config.image_size,
        layers_to_extract_from=config.layers_to_extract_from, 
        feature_dim=config.feature_dim,
        reduce_feature_dim=config.reduce_feature_dim,
        sampling_threshold=config.sampling_threshold,
        chunk_size=config.chunk_size,        

        sampling_ratio=config.sampling_ratio,
        coreset_size=config.coreset_size,
        project_dim=config.project_dim,
        approximate_ratio=config.approximate_ratio,
        metric=config.metric,

        save_path=config.save_path,
        config=config
    )
    model = model.to(device)

    ## train
    accumulated_testset = []
    train_time = 0
    for step, (trainset, testset) in enumerate(zip(trainset_list, testset_list)):

        ###################
        ###    train    ###
        ###################
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size = config.batch_size,
            shuffle = True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        start = time.time()

        for data in train_loader:
            images = data['image']
            sample_ids = data['sample_id']
            model.update(images, sample_ids)

        model._sample_task_coresets()
        model._meta_greedy_sampling()

        end = time.time()
        train_time += (end - start)
        model.save_model(step=step+1)

        ###################
        ###    test     ###
        ###################

        model.build_faiss(config.faiss_on_gpu, config.faiss_num_workers, config.anomaly_score_num_nn)
        accumulated_testset.append(testset)
        concat_testset = ConcatDataset(accumulated_testset)
        test_loader = torch.utils.data.DataLoader(
            concat_testset,
            batch_size = config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        scores_all = []
        segmentations_all = []
        labels_gt_all = []
        masks_gt_all = []
        class_all = []
        image_paths_all = []
        mask_paths_all = []
        for data in test_loader:
            image = data['image']
            labels_gt = data['is_anomaly'].numpy().tolist()
            masks_gt = data['mask'].numpy().tolist()
            cls = data['classname']
            image_paths = data['image_path']
            mask_paths = data['mask_path']

            scores, segmentations  = model.inference(image)

            scores_all.extend(scores)
            segmentations_all.extend(segmentations)
            labels_gt_all.extend(labels_gt)
            masks_gt_all.extend(masks_gt)
            class_all.extend(cls)
            image_paths_all.extend(image_paths)
            mask_paths_all.extend(mask_paths)
            
        results = model.get_metric_results(scores_all, segmentations_all, labels_gt_all, masks_gt_all, class_all)
        model._save_metrics(results, f'results_s{step+1}.csv')



    with open(os.path.join(config.save_path, f'time.txt'), 'w') as file:
        file.write(f'sampling time: {train_time}')



def get_argparse():
    parser = argparse.ArgumentParser()

    # model arguments
    # feature extractor 
    ## timm based args
    parser.add_argument("--backbone_name", "-b", type=str, default='wideresnet50')
    parser.add_argument('--layers_to_extract_from', "-le", nargs='+', default=['layer2', 'layer3'])
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    parser.add_argument("--feature_dim", type=int, default=1024)

    ## Dinov2 based args
    # parser.add_argument("--backbone_name", "-b", type=str, default='dinov2reg_vit_base_14')
    # parser.add_argument('--layers_to_extract_from', "-le", nargs='+', default=['11'])
    # parser.add_argument("--pretrain_embed_dimension", type=int, default=768)
    # parser.add_argument("--feature_dim", type=int, default=768)


    # non-dino model arguments (from PatchCore)
    # these args are for timm based patchcore sampling, not for DINOv2
    parser.add_argument("--preprocessing", type=str, default="mean")
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--patchstride", type=int, default=1)
    parser.add_argument("--patchscore", type=str, default="max")
    parser.add_argument("--patchoverlap", type=float, default=0.0)
    parser.add_argument("--patchsize_aggregate", "-pa", type=int, default=[])

    # NN Anomaly Scorer parameters.
    parser.add_argument("--anomaly_score_num_nn", type=int, default=1)
    # Faiss
    parser.add_argument("--faiss_on_gpu", type=str2bool, default='True')
    parser.add_argument("--faiss_num_workers", type=int, default=8)

    # Sampler
    parser.add_argument('--metric', type=str, default='L2',choices=['L2', 'cosine'])
    parser.add_argument("--coreset_size", type=int, default=20000) 
    parser.add_argument("--sampling_threshold", type=int, default=20000) ## num features when sampling starts
    parser.add_argument("--sampling_ratio", type=float, default=0.01) ## sampling ratio
    parser.add_argument("--approximate_ratio", type=float, default=0.75) ## approximation ratio for inter-task sampling
    parser.add_argument("--max_chunk_size", type=str2bool, default='True') 
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--project_dim", type=int, default=128) ## project dimension to lower dimension
    parser.add_argument("--reduce_feature_dim", type=str2bool, default=True) ## FLAG: lower dimension projection for inter-task sampling


    ## dataset 
    parser.add_argument('--dataset', type=str, default='mvtec',choices=['mvtec','visa'])
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--augment", type=str2bool, default='True')

    ## misc
    parser.add_argument("--train", type=str2bool, default='True')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--incremental_class_list", type=str, default="1-1-14")

    parser.add_argument("--gpus", type=str, default="0")

    config = parser.parse_args()

    config.save_path = os.path.join(f'./results', config.dataset, config.incremental_class_list, f"Coreset{config.coreset_size}_SampRatio{config.sampling_ratio}_Apprx{config.approximate_ratio}")
    config.model_path = os.path.join(config.save_path, 'model')
    config.metrics_path = os.path.join(config.save_path, 'metrics')
    config.seg_path = os.path.join(config.save_path, 'segmentations')
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.metrics_path, exist_ok=True)
    os.makedirs(config.seg_path, exist_ok=True)


    return config



if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()
