# okay, start an interactive session on h100 node so we can build environment in the right context
# salloc --account=kempner_alvarez_lab --partition=kempner_h100 --nodes=1 --ntasks-per-node=1 --cpus-per-task=20 --gres=gpu:1 --time=03:00:00 --mem=128G

module load Mambaforge/23.11.0-fasrc01
module load cuda/12.2.0-fasrc01
# module load gcc/13.2.0-fasrc01
module load gcc/9.5.0-fasrc01

# make sure you have your $CONDA_ENV_DIR set to tier1 storage (nano ~/.bashrc), e.g.,
# CONDA_ENV_DIR=/n/alvarez_lab_tier1/Users/alvarez/conda_envs
# base environment with pytorch
conda create -y -n modelvshuman python=3.10.12 mamba ipykernel pip pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge

source activate modelvshuman

# Enforce specific pip install target location

# Determine the Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Construct the target path using the current Conda environment
TARGET_PATH="$CONDA_PREFIX/lib/python$PYTHON_VERSION/site-packages"

# Create or overwrite the pip.conf file in your Conda environment's directory
echo -e "[install]\ntarget=$TARGET_PATH\nscripts=$CONDA_PREFIX/bin" > $CONDA_PREFIX/pip.conf

# The trick/issue is going to be installing ffcv-ssl
mamba install -y pkg-config compilers
mamba install -y opencv=4.6
python -m pip install --upgrade --force-reinstall opencv-python
python -m pip install --no-deps -e git+https://github.com/facebookresearch/FFCV-SSL.git#egg=FFCV-SSL

# install miscellaneous packages
mamba install -y albumentations assertpy attrs boto3 cupy einops nbconvert nbformat submitit terminaltables torchmetrics webdataset
# mamba install -y -c conda-forge texlive-core
python -m pip install --upgrade pip setuptools wheel
# python -m pip install git+https://github.com/loopbio/PyTurboJPEG.git#egg=PyTurboJPEG
python -m pip install --upgrade git+https://github.com/lilohuang/PyTurboJPEG.git#egg=PyTurboJPEG
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install awscli awscli-plugin-endpoint datasets facenet-pytorch fastargs fastprogress huggingface_hub kornia pynvjpeg prettytable timm transformers 'urllib3<2' wandb imgcat pytorch-pfn-extras 

# get pdftex going
# mv $CONDA_PREFIX/pip.conf.bak $CONDA_PREFIX/pip.conf
# mamba install -y -c conda-forge texlive-core texlive-bin texlive-latex-extra
# pdftex -ini -jobname=pdftex -progname=pdftex -translate-file=cp227.tcx *pdftex.ini

# need latest timm
# rm $CONDA_PREFIX/pip.conf
mv $CONDA_PREFIX/pip.conf $CONDA_PREFIX/pip.conf.bak
python -m pip install git+https://github.com/rwightman/pytorch-image-models.git

# install ipykernel so you can choose 'modelvshuman' as a kernel in jupyterlab
mamba install -y fastcore ipykernel numba pandas seaborn ipywidgets -c conda-forge
mamba install -y albumentations cupy einops kornia fastcore fastprogress IPython nbconvert nbformat numba prettytable -c fastai
conda deactivate
conda activate modelvshuman
python -m ipykernel install --user --name=modelvshuman