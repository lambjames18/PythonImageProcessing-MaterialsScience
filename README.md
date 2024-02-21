# UCSB Materials - Machine Learning in Materials Science Seminar Series - Image Processing

This repository contains example python scripts for performing various image processing routines. This is ongoing work and new examples and tutorials are contiually being added. Check back in every so often to see what's new!

### Repo outline

All example data is stored under the `imgs/` directory. For now, this includes...
- an electron backscatter diffraction (EBSD) image stored as a numpy array: `EBSD_data.npy`
- an image of an EBSD pattern: `EBSD-Pattern.tif`
- an infrared thermography (IR) image of a cracked CeO2 pellet (a background is included as well): `CeO2.tiff`, `CeO2_Background.tiff`
- a backscattered electron (BSE) image of a WCu dual-matrix composite: `WCu-Composite.tiff`

### Python environment configuration

This code has been successfully ran using the following conda environment. It is *highly* recommended that one uses a package manager, like miniconda/anaconda/pip, within a virtual environment. Personally, miniconda is my preferred package manager due to it's low footprint and simple user interface. Details on installing and using miniconda can be found [here](https://docs.anaconda.com/free/miniconda/miniconda-install/) and [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment). The following conda commands were used to create the environment used during creating of the scripts in  this repository. Note that `imageproc` is a dummy name for the environment and can (should) be changed to your liking.

conda environment command (windows, and linux I believe):

```
conda create -n imageproc python=3.9

conda activate imageproc

conda install pytorch torchvision torchaudio kornia pytorch-cuda=12.1 numpy matplotlib scikit-learn scikit-image ipympl ipykernel -c pytorch -c conda-forge -c nvidia  # GPU

conda install pytorch torchvision torchaudio kornia numpy matplotlib scikit-learn scikit-image tqdm ipympl ipykernel -c pytorch -c conda-forge  # CPU
```

Note that the `pytorch_cuda=12.1` will need to be modified for your computer. Also note that both the GPU and CPU pytorch versions are shown, so choose based on what is applicable to you.

conda environment command (MacOS):

```
conda create -n imageproc python numpy matplotlib scikit-learn scikit-image ipympl ipykernel pytorch::pytorch torchvision torchaudio -c pytorch -c conda-forge

conda activate imageproc
```


