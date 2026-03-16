# -------------------------------- #
# 环境配置
conda create --prefix /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wys_temp_2 python=3.10 pip openssl -y
conda create -n wys_temp_2 python=3.10 -y

conda activate /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wys_temp_2
# 安装torch、torchvision
pip install /c20250502/wangyushen/whl/torch-2.1.1+cu121-cp310-cp310-linux_x86_64.whl
pip install /c20250502/wangyushen/whl/torchvision-0.16.1+cu121-cp310-cp310-linux_x86_64.whl

pip install openmim
# mim install mmcv-full==1.6.0 --no-build-isolation --no-cache-dir # 禁用构建隔离、禁用缓存目录 未试成功

pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html # 可行

# mim install mmdet3d==1.0.0rc6 --no-build-isolation
pip install mmdet3d==1.0.0rc6 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html --no-build-isolation
# 直接下载源码安装
wget https://pypi.tuna.tsinghua.edu.cn/packages/7c/38/5abed519e2c644bd6f447e73128d22ecfe2d55e3672445065ae8ee540b76/mmdet3d-1.0.0rc6.tar.gz
tar -zxvf mmdet3d-1.0.0rc6.tar.gz
cd mmdet3d-1.0.0rc6
sed -i "s/numba==0.53.0/numba>=0.56.0/g" runtime.txt # 避免找不着0.53的numba


mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
pip install setuptools==59.5.0
pip install numpy==1.23.5
pip install wandb

# 修改
from collections import Mapping, Set
from collections.abc import Mapping, Set
# 修改前
from fractions import gcd
# 修改后
from math import gcd

cd models/csrc
python setup.py build_ext --inplace # --inplace表示原地编译

# -------------------------------- #
# 远程仓库
mkdir -p /vepfs-mlp2/c20250502/haoce/wangyushen/.ssh
chmod 700 /vepfs-mlp2/c20250502/haoce/wangyushen/.ssh
ssh-keygen -t rsa -b 4096 -f /vepfs-mlp2/c20250502/haoce/wangyushen/.ssh/id_rsa -C "1394945493@qq.com"
cat /vepfs-mlp2/c20250502/haoce/wangyushen/.ssh/id_rsa.pub


# token 创建token记得勾选repo，不要保存token
git remote set-url origin https://w1394945493:<token>@github.com/w1394945493/SparseOcc.git
git reset --soft origin/main
git restore --staged .
# -------------------------------- #
# SparseOcc 评估
export CUDA_VISIBLE_DEVICES=0
python val.py --config configs/sparseocc_r50_nuimg_704x256_8f.py --weights checkpoints/sparseocc_r50_nuimg_704x256_8f.pth

# 评估(无结果保存)
python /vepfs-mlp2/c20250502/haoce/wangyushen/SparseOcc/val.py \
    --config /vepfs-mlp2/c20250502/haoce/wangyushen/SparseOcc/configs/r50_nuimg_704x256_8f_custom.py \
    --weights /c20250502/wangyushen/Weights/sparseocc/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth

# -------------------------------- #
# SparseOcc 仅推理(可视化(二维图))
python /vepfs-mlp2/c20250502/haoce/wangyushen/SparseOcc/viz_prediction.py \
    --config /vepfs-mlp2/c20250502/haoce/wangyushen/SparseOcc/configs/r50_nuimg_704x256_8f_custom.py \
    --weights /c20250502/wangyushen/Weights/sparseocc/sparseocc_r50_nuimg_704x256_8f_24e_v1.1.pth \
    --viz-dir /c20250502/wangyushen/Outputs/sparseocc/outputs/vis