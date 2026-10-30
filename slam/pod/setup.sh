# One-shot GPU pod setup for the SLAM experiments (RunPod Ubuntu image, NVIDIA GPU, no persistent volume needed).
# Installs VGGT-SLAM 2.0 (MIT-SPARK, unchanged) in /root/VGGT-SLAM and downloads NASA's POLAR Traverse views 1-2.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libgomp1 libglib2.0-0 unzip > /dev/null   # open3d needs libEGL
mkdir -p /workspace/polar && cd /workspace/polar
( for v in 1 2; do curl -fsSL -o View$v.zip https://ti.arc.nasa.gov/dataset/PolarTrav/downloads/View$v.zip && unzip -nq View$v.zip; done; echo POLAR_DATA_OK ) > /root/polar_dl.log 2>&1 &
cd /root
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH=$HOME/.local/bin:$PATH
git clone -q https://github.com/MIT-SPARK/VGGT-SLAM && cd VGGT-SLAM
uv venv -q -p 3.11 .venv && source .venv/bin/activate
uv pip install -q -r requirements.txt
mkdir -p third_party && cd third_party
git clone -q https://github.com/Dominic101/salad.git && uv pip install -q -e ./salad
git clone -q https://github.com/MIT-SPARK/VGGT_SPARK.git vggt && uv pip install -q -e ./vggt
cd .. && uv pip install -q -e .
# SAM 3 / Perception Encoder skipped: only used for optional open-set object search (--run_os)
mkdir -p /root/.cache/torch/hub/checkpoints
curl -fsSL -o /root/.cache/torch/hub/checkpoints/dino_salad.ckpt https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
python -c "import torch, open3d; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
wait
cat /root/polar_dl.log
echo SETUP_DONE
