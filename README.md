# UCSB Materials - Machine Learning in Materials Science Seminar Series - Image Processing

conda environment command (windows, and linux I blieve):

```
conda create -n imageproc python=3.9

conda activate imageproc

conda install pytorch torchvision torchaudio kornia pytorch-cuda=12.1 numpy matplotlib scikit-learn scikit-image ipympl ipykernel -c pytorch -c conda-forge -c nvidia  # GPU

conda install pytorch torchvision torchaudio kornia numpy matplotlib scikit-learn scikit-image tqdm ipympl ipykernel -c pytorch -c conda-forge  # CPU
```

Note that the `pytorch_cuda=12.1` will need to be modified for your computer. If you do not wish to use the GPU, you can remove.

conda environment command (MacOS):

```
conda create -n imageproc python numpy matplotlib scikit-learn scikit-image ipympl ipykernel pytorch::pytorch torchvision torchaudio -c pytorch -c conda-forge

conda activate imageproc
```


