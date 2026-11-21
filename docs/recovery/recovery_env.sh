# NANM-RECOVERY-01 storage policy: source this before any Python/pip/HF/torch work.
# Keeps caches and temp files off the D:-backed WSL ext4.vhdx. Refuses if /mnt/k is not the K: drvfs mount.
if ! mount | grep -q '^K:\\ on /mnt/k '; then echo "recovery_env: /mnt/k is not K: — refusing" >&2; return 1 2>/dev/null || exit 1; fi
_K=/mnt/k/NLCAD
export XDG_CACHE_HOME=$_K/cache/xdg
export HF_HOME=$_K/cache/hf HF_HUB_CACHE=$_K/cache/hf/hub
export TORCH_HOME=$_K/cache/torch
export PIP_CACHE_DIR=$_K/cache/pip UV_CACHE_DIR=$_K/cache/uv CONDA_PKGS_DIRS=$_K/cache/conda_pkgs
export TMPDIR=$_K/tmp TMP=$_K/tmp TEMP=$_K/tmp
export TRITON_CACHE_DIR=$_K/cache/triton CUDA_CACHE_PATH=$_K/cache/cuda
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""   # recovery is CPU-only
unset _K
