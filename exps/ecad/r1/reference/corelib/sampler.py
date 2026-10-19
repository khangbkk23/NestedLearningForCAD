import abc
from typing import Union, List
from tqdm import tqdm
import gc

import torch
import numpy as np
import faiss
from numpy.linalg import pinv
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
import corelib.automl as automl
from collections import defaultdict
import torch.nn.functional as F
import time


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage
        self._status = defaultdict(float)

    def get_status(self):
        return self._status


    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)



class MetaGreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        device: torch.device,
        sampling_ratio: float = 0.01,
        feature_dim: int = 1024,
        coreset_size: int = 40000,
        reduce_feature_dim: bool= True,
        project_dim: int = 128,
        approximate_ratio: float = 1.0,
        metric: str = 'L2',
    ):
        super().__init__(percentage=sampling_ratio)

        self.feature_dim = feature_dim
        self.reduce_feature_dim = reduce_feature_dim
        self.project_dim = project_dim
        self.metric = metric
        self.device = device
        self.coreset_size = coreset_size
        if coreset_size is None:
            self.coreset_size = self._get_coreset_size(feature_dim=feature_dim)

        self.approximate_ratio = approximate_ratio

        # self._max_coreset_buffer = torch.zeros((2 * coreset_size, feature_dim))    # for v1
        self._max_coreset_buffer = torch.zeros(
            (coreset_size * 2, feature_dim), requires_grad=False, device=device,
        )  
        self._max_coreset_sample_ids = torch.zeros(
            (coreset_size * 2), requires_grad=False, device=device,
        )  
        self._max_task_coreset_buffer = torch.zeros(
            (coreset_size * 2, feature_dim), requires_grad=False, device=device,
        )  
        self._max_task_coreset_sample_ids = torch.zeros(
            (coreset_size * 2), requires_grad=False, device=device,
        )  

        print(f"coreset size: {coreset_size}")


    def _empty_task_coreset_buffer(self,):
        self._max_task_coreset_buffer = torch.zeros(
            (self.coreset_size * 2, self.feature_dim), requires_grad=False, device=self.device,
        )  
        self._max_task_coreset_sample_ids = torch.zeros(
            (self.coreset_size * 2), requires_grad=False, device=self.device,
        )  

    def _get_coreset_size(
        self, 
        feature_dim: int=1024
    ):
        coreset_size = automl.get_max_len_feas(
            feature_dim,
            device=self.device.type,
            max_usage=0.5,
            memory_type="available",
            device_index=self.device.index,
        )
        return coreset_size

    def _adjust_max_coreset_size(self, feature_dim, max_coreset_size):

        max_coreset_size_limit = automl.get_max_num_feas_for_pairwise_dist_matrix(
            fea_dim=feature_dim,
            device_type="cpu",
            max_usage=0.50,
        )

        if max_coreset_size < max_coreset_size_limit:
            return max_coreset_size

        max_coreset_size = max_coreset_size_limit

        return max_coreset_size

    def run(
            self, 
            features, 
            sample_ids, 
            task_coreset=None,
            base_coreset=None,
        ):

        if self.percentage == 1:
            return features, None
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(task_coreset, (torch.Tensor, type(None)))

        if task_coreset is not None:
            print(f"task coreset size: {len(task_coreset)}")
            n_task_coreset = task_coreset.shape[0]
            self._max_task_coreset_buffer[:n_task_coreset] = task_coreset.clone()
        else:
            n_task_coreset = 0
        
        # if task_coreset is not None and base_coreset is not None:
        #     task_coreset = torch.cat((task_coreset, base_coreset))
        
        
        # reduced_features, reduced_task_coreset = self._reduce_features(
        #     features, base_features=task_coreset
        # )

        reduced_features, reduced_task_coreset = self._reduce_features(
            features, base_features=task_coreset
        )


        # sample_indices_with_coreset = self._greedy_sample_indices(reduced_features, reduced_task_coreset)
        # sample_indices_wo_coreset = self._greedy_sample_indices(reduced_features, base_coreset=None)
        # sample_indices = list(set(sample_indices_with_coreset + sample_indices_wo_coreset))
        sample_indices = self._greedy_sample_indices(reduced_features, reduced_task_coreset)
        
        sampled_features = features[sample_indices]
        sampled_ids = sample_ids[sample_indices]

        n_samples = len(sample_indices)
        n_samples_to_delete = n_task_coreset + n_samples - self.coreset_size

        if n_samples_to_delete > 0:
            coreset_features, coreset_sample_ids = self._remove_and_add_to_task_coreset(
                sampled_features, sampled_ids, n_task_coreset
            )
            sample_indices = None
        else:
            coreset_features, coreset_sample_ids = self._add_to_task_coreset(
                sampled_features, sampled_ids, n_task_coreset
            )

        return coreset_features.clone(), coreset_sample_ids.clone()


    def meta_greedy_sampling(
            self, 
            task_coreset,
            task_sample_ids,
            base_coreset=None
        ):
        if base_coreset is not None:
            print(f"base coreset size: {len(base_coreset)}")
            n_base_coreset = base_coreset.shape[0]
            self._max_coreset_buffer[:n_base_coreset] = base_coreset.clone()
        else:
            n_base_coreset = 0
        
        n_samples = len(task_coreset)
        n_samples_to_delete = n_base_coreset + n_samples - self.coreset_size

        if n_samples_to_delete > 0:
            coreset_features, coreset_sample_ids = self._remove_and_add_to_base_coreset(
                task_coreset, task_sample_ids, n_base_coreset
            )
        else:
            coreset_features, coreset_sample_ids = self._add_to_base_coreset(
                task_coreset, task_sample_ids, n_base_coreset
            )
        self._empty_task_coreset_buffer()
        return coreset_features, coreset_sample_ids

    def _add_to_task_coreset(self, sampled_features, sampled_ids, n_base_coreset):
        n_samples = len(sampled_features)
        self._max_task_coreset_buffer[n_base_coreset : n_base_coreset + n_samples] = (
            sampled_features.clone()
        )
        self._max_task_coreset_sample_ids[n_base_coreset : n_base_coreset + n_samples] = (
            sampled_ids.clone()
        )
        coreset_features = self._max_task_coreset_buffer[: n_base_coreset + n_samples].clone()
        coreset_sample_ids = self._max_task_coreset_sample_ids[:n_base_coreset + n_samples].clone()

        print(f"The num. of newly added features to task coreset: {n_samples}")
        return coreset_features, coreset_sample_ids


    def _add_to_base_coreset(self, sampled_features, sampled_ids, n_base_coreset):
        n_samples = len(sampled_features)
        self._max_coreset_buffer[n_base_coreset : n_base_coreset + n_samples] = (
            sampled_features.clone()
        )
        self._max_coreset_sample_ids[n_base_coreset : n_base_coreset + n_samples] = (
            sampled_ids.clone()
        )
        coreset_features = self._max_coreset_buffer[: n_base_coreset + n_samples].clone()
        coreset_sample_ids = self._max_coreset_sample_ids[:n_base_coreset + n_samples].clone()

        print(f"The num. of newly added features to base coreset: {n_samples}")

        return coreset_features, coreset_sample_ids



    def _remove_and_add_to_task_coreset(self, sampled_features, sampled_ids, n_base_coreset):
        print("Meta-greedy sampling task coreset...")
        n_samples = len(sampled_features)

        self._max_task_coreset_buffer[n_base_coreset:n_base_coreset + n_samples] = sampled_features.clone()
        self._max_task_coreset_sample_ids[n_base_coreset:n_base_coreset + n_samples] = sampled_ids.clone()
        
        all_features = self._max_task_coreset_buffer[:n_base_coreset + n_samples].clone()
        all_sample_ids = self._max_task_coreset_sample_ids[:n_base_coreset + n_samples].clone()

        if self.reduce_feature_dim:
            features, _ = self._reduce_features(all_features)
        else:
            features = all_features

        most_greedy_indices = approx_greedy_sample(
            all_features=features,
            n_samples=self.coreset_size,
            device=self.device,
            approximation_rate=self.approximate_ratio,
            metric=self.metric,
        )
        self._max_task_coreset_buffer[: self.coreset_size] = all_features[most_greedy_indices].clone()
        self._max_task_coreset_sample_ids[: self.coreset_size] = all_sample_ids[most_greedy_indices].clone()

        _num_features_added_to_coreset = np.sum(
            np.array(most_greedy_indices) > self.coreset_size
        )
        print(
            f"The num. of newly added features to task coreset is {_num_features_added_to_coreset}"
        )

        return all_features[most_greedy_indices].clone(), all_sample_ids[most_greedy_indices].clone()


    def _remove_and_add_to_base_coreset(self, sampled_features, sampled_ids, n_base_coreset):
        print("Meta-greedy sampling base coreset...")
        n_samples = len(sampled_features)

        self._max_coreset_buffer[n_base_coreset:n_base_coreset + n_samples] = sampled_features.clone()
        self._max_coreset_sample_ids[n_base_coreset:n_base_coreset + n_samples] = sampled_ids.clone()
        
        all_features = self._max_coreset_buffer[:n_base_coreset + n_samples].clone()
        all_sample_ids = self._max_coreset_sample_ids[:n_base_coreset + n_samples].clone()

        if self.reduce_feature_dim:
            features, _ = self._reduce_features(all_features)
        else:
            features = all_features

        most_greedy_indices = approx_greedy_sample(
            all_features=features,
            n_samples=self.coreset_size,
            device=self.device,
            approximation_rate=self.approximate_ratio,
            metric=self.metric,
        )
        self._max_coreset_buffer[: self.coreset_size] = all_features[most_greedy_indices].clone()
        self._max_coreset_sample_ids[: self.coreset_size] = all_sample_ids[most_greedy_indices].clone()

        _num_features_added_to_coreset = np.sum(
            np.array(most_greedy_indices) > self.coreset_size
        )
        print(
            f"The num. of newly added features to base coreset is {_num_features_added_to_coreset}"
        )

        return all_features[most_greedy_indices].clone(), all_sample_ids[most_greedy_indices].clone()

    def _reduce_features(
        self, features: torch.Tensor, base_features: torch.Tensor = None
    ):
        if features.shape[1] == self.project_dim:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.project_dim, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        with torch.no_grad():
            reduced_features = mapper(features)

        if base_features is not None:
            base_features = base_features.to(self.device)
            with torch.no_grad():
                reduced_base_features = mapper(base_features)
        else:
            reduced_base_features = None

        return reduced_features, reduced_base_features

    def _greedy_sample_indices(self, features, base_coreset):

        print("Greedy sampling...")

        n_samples = int(len(features) * self.percentage)
        n_samples = min(n_samples, self.coreset_size)

        sample_indices = iterative_greedy_sampling(
            features=features,
            n_samples=n_samples,
            device=self.device,
            coreset=base_coreset,
        )

        return sample_indices




def greedy_sample_indices(
    features: torch.Tensor,
    n_samples: int,
    device: torch.device,
    f2f_pdist_mat: torch.Tensor = None,
    base_coreset: torch.Tensor = None,
    metric = "L2",
) -> list:
    n_feas = len(features)
    if n_samples == n_feas:
        return torch.arange(n_samples).tolist()
    else:
        assert n_samples < n_feas

    if f2f_pdist_mat is None:
        f2f_pdist_mat = compute_pdist(features, features, device=device, metric=metric)

    assert n_feas == f2f_pdist_mat.shape[0] and n_feas == f2f_pdist_mat.shape[1]

    if base_coreset is None:
        f2c_dists = torch.norm(f2f_pdist_mat, dim=1)
    else:
        f2c_pdist_mat = torch.cdist(features, base_coreset)
        f2c_dists = f2c_pdist_mat.min(dim=-1).values
    f2c_dists = f2c_dists.to(device)
    f2f_pdist_mat = f2f_pdist_mat.to(device)
    sample_indices = []

    for _ in tqdm(
        range(n_samples), leave=False, total=n_samples, desc="Greedy sampling..."
    ):
        selected_idx = torch.argmax(f2c_dists).item()  # this operaiont is extremely light, hence no need to be async
        sample_indices.append(selected_idx)

        dists_feas_to_selected_fea = f2f_pdist_mat[:, selected_idx]
        f2c_dists = torch.minimum(f2c_dists, dists_feas_to_selected_fea)

    torch.cuda.empty_cache()
    return sample_indices


def brute_greedy_sample_indices(
    features: torch.Tensor,
    n_samples: int,
    device: torch.device,
    base_coreset: torch.Tensor = None,
    metric = "L2",
) -> list:

    n_feas = len(features)
    if n_samples == n_feas:
        return torch.arange(n_samples).tolist()
    else:
        assert n_samples < n_feas

    if base_coreset is None:
        sample_indices = [np.random.randint(len(features))]
        min_distances = torch.norm(features - features[sample_indices[0]], dim=1)
    else:
        batch_size = 10000
        offset = 0
        min_val = torch.inf
        feature_chunks = torch.split(features, batch_size)
        for batch in feature_chunks:
            f2c_pdist_mat = compute_pdist(batch, base_coreset, device=device, metric=metric)
            f2c_dist, _ = f2c_pdist_mat.min(dim=-1)
            val, idx = torch.min(f2c_dist, dim=0)
            if val < min_val:
                min_val = val
                min_idx = idx + offset
            offset += batch_size
        sample_indices = [min_idx.item()]
        min_distances = torch.norm(features - features[sample_indices[0]], dim=1)

    for _ in tqdm(range(1, n_samples), desc="brute greedy sampling", leave=False):
        current_distances = torch.norm(features - features[sample_indices[-1]], dim=1)
        min_distances = torch.minimum(min_distances, current_distances)
        sample_indices.append(torch.argmax(min_distances, dim=0).item())
        current_distances = current_distances.cpu()
        del current_distances
        gc.collect()
        torch.cuda.empty_cache()

    return sample_indices


def compute_pdist(
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    batch_size= None,
    metric="L2",
):
    if batch_size is None:
        # auto batch size
        batch_size = automl.auto_get_max_query_size_pdist(  # FIXME:
            fea_dim=X.shape[-1],
            n_support=len(Y),
            max_usage=0.1,
            device_type=device.type,
            device_index=device.index,
        )
        batch_size = max(min(256, batch_size), 128)

    if metric == "L2":
        _compute_pdist = _compute_pdist_l2
    elif metric == 'cosine':
        _compute_pdist = _compute_pdist_cosine
    else:
        raise ValueError("metric incorrectly set")
    return _compute_pdist(X=X, Y=Y, device=device, batch_size=batch_size)


def _compute_pdist_l2(
    X: torch.Tensor, Y: torch.Tensor, device: torch.device, batch_size: int = 128
):
    # X: (n, d)
    # Y: (m, d)
    # n >> m
    # d relatively small
    # Y acts as anchor
    X_pinned = X.pin_memory()

    Yt = Y.to(device, non_blocking=True).T
    Yt_norm = (Yt.pow(2)).sum(dim=0, keepdim=True)

    i0_idxes = range(0, len(X), batch_size)
    Z_slices = [[] for _ in range(len(i0_idxes))]

    for n, i0 in tqdm(
        enumerate(i0_idxes), desc="Computing pdist...", total=len(i0_idxes), leave=False
    ):
        i1 = min(len(X), i0 + batch_size)

        Z_slices[n] = pdist_l2(
            X_pinned[i0:i1].to(device, non_blocking=True), Yt, Yt_norm
        ).to("cpu", non_blocking=True)

    return torch.cat(Z_slices)


def pdist_l2(X: torch.Tensor, Yt: torch.Tensor, Yt_norm: torch.Tensor) -> torch.Tensor:
    # NOTE: this omits sqrt (since we do not need)
    # X: (n, d)
    # Yt: (d, m)
    # Yt_norm: (1, m)

    X_times_X = (X.pow(2)).sum(dim=1, keepdim=True)

    X_times_Y = X.mm(Yt)

    distance_matrix = -2 * X_times_Y + X_times_X + Yt_norm

    return distance_matrix


def _compute_pdist_cosine(
    X: torch.Tensor, Y: torch.Tensor, device: torch.device, batch_size: int = 128
):
    X_pinned = X.pin_memory()
    Y = Y.to(device, non_blocking=True)

    i0_idxes = range(0, len(X), batch_size)
    Z_slices = [[] for _ in range(len(i0_idxes))]

    for n, i0 in tqdm(
        enumerate(i0_idxes), desc="Computing pdist...", total=len(i0_idxes), leave=False
    ):
        i1 = min(len(X), i0 + batch_size)
        Z_slices[n] = pdist_cosine(
            X_pinned[i0:i1].to(device, non_blocking=True), Y
        ).to("cpu", non_blocking=True)

    return torch.cat(Z_slices)

def pdist_cosine(x1, x2):
    x1 = x1 / torch.norm(x1, p=2, dim=1, keepdim=True)
    x2 = x2 / torch.norm(x2, p=2, dim=1, keepdim=True)
    cos_sim = torch.matmul(x1, x2.t()).clip(-1, 1)
    # cos_sim = F.cosine_similarity(x1, x2, dim=1)
    cos_dist = 1 - cos_sim
    return cos_dist


# FIXME: refactor
@torch.no_grad()
def approx_greedy_sample(
    all_features: torch.Tensor,
    n_samples: int,
    device: torch.device,
    approximation_rate=0.75,
    metric = "L2",
):

    assert 0 <= approximation_rate <= 1.0

    n_feas = all_features.shape[0]

    # fea_pdist = compute_pdist(all_features, all_features, device=device, metric=metric)
    fea_pdist = torch.cdist(all_features, all_features)

    ## 2.9~3.0
    if approximation_rate == 0.0:
        sample_indices = greedy_sample_indices(
            features=all_features,
            f2f_pdist_mat=fea_pdist,
            n_samples=n_samples,
            device=device,
            metric=metric,
        )
        sampled_features = all_features[sample_indices]
    else:
        n_approx = int(approximation_rate * n_samples)
        n_approx = max(n_approx, 1)

        assert n_approx <= n_samples

        ## search closest sample
        dist_fea_to_feaset_wo_itself = (
            fea_pdist.fill_diagonal_(float("inf"))
            .topk(k=1, largest=False, dim=-1)
            .values[:, 0]
        )
        fea_pdist.fill_diagonal_(0.0)

        ## select the furthest closest samples as much as approx rate
        approx_pre_sampled_idices = dist_fea_to_feaset_wo_itself.topk(
            k=n_approx, largest=True, sorted=False
        ).indices
        approx_pre_coreset = all_features[approx_pre_sampled_idices]

        # Create a Boolean mask where True indicates the indices to be used for further sampling
        mask = torch.ones(n_feas, dtype=torch.bool)
        mask[approx_pre_sampled_idices] = False
        idxes_to_post_sample = mask.nonzero(as_tuple=False).squeeze()
        feas_to_greedy_sample = all_features[idxes_to_post_sample]


        ## This is bottleneck
        _post_sample_indices = greedy_sample_indices(
            features=feas_to_greedy_sample,
            n_samples=n_samples - len(approx_pre_coreset),
            device=device,
            f2f_pdist_mat=fea_pdist[idxes_to_post_sample][:, idxes_to_post_sample],
            base_coreset=approx_pre_coreset,
            metric=metric,
        )
        ## bottle neck

        post_sample_features = feas_to_greedy_sample[_post_sample_indices]
        pose_sample_feature_indices = idxes_to_post_sample[_post_sample_indices]

        sampled_features = torch.cat([approx_pre_coreset, post_sample_features], dim=0)

        sample_indices = (
            approx_pre_sampled_idices.tolist() + pose_sample_feature_indices.tolist()
        )  # FIXME: incorrect


    return sample_indices



@torch.no_grad()
def iterative_greedy_sampling(
    features, 
    n_samples, 
    device, 
    coreset=None, 
    number_of_starting_points=10
):
    number_of_starting_points = np.clip(
        number_of_starting_points, None, len(features)
    )
    features = features.to(device)
    start_points = np.random.choice(len(features), number_of_starting_points, replace=False).tolist()

    if coreset is None:
        coreset = features.clone()

    ## It doesn't compute the whole N x N distance matrix but just a ew N x n 
    approximate_distance_matrix = torch.cdist(coreset, features[start_points])
    approximate_coreset_anchor_distances = torch.mean(
        approximate_distance_matrix, axis=-1 ## ## num_fea x num_start_fea
    ).reshape(-1, 1)
    sample_indices = []

    for _ in tqdm(range(n_samples), desc="Subsampling..."): 
        # select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
        dist, select_idx = torch.max(approximate_coreset_anchor_distances, dim=0)
        sample_indices.append(select_idx.item())
        coreset_select_distance = torch.cdist(coreset, features[select_idx : select_idx + 1])
        approximate_coreset_anchor_distances = torch.cat(
            [approximate_coreset_anchor_distances, coreset_select_distance],
            dim=-1,
        )
        approximate_coreset_anchor_distances = torch.min(
            approximate_coreset_anchor_distances, dim=1
        ).values.reshape(-1, 1)

    return sample_indices