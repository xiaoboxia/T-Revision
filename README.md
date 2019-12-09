# T_Revision  
NeurIPS‘19: Are Anchor Points Really Indispensable in Label-Noise Learning? (PyTorch implementation).  

This is the code for the paper:
[Are Anchor Points Really Indispensable in Label-Noise Learning?](https://papers.nips.cc/paper/8908-are-anchor-points-really-indispensable-in-label-noise-learning)      
Xiaobo Xia, Tongliang Liu, Nannan Wang, Bo Han, Chen Gong, Gang Niu, Masashi Sugiyama.

If you find this code useful in your research, please cite  
```bash
@inproceedings{xia2019t_revision,
  title={Are Anchor Points Really Indispensable in Label-Noise Learning?},
  author={Xia, Xiaobo and Liu, Tongliang and Wang, Nannan and Han, Bo and Gong, Chen and Niu, Gang and Sugiyama, Masashi},
  booktitle={NeurIPS},
  year={2019}
}
```  
## Dependencies
we implement our methods by PyTorch on NVIDIA Tesla V100. The environment is as bellow:
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version >= 0.4.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 9.0
- [Anaconda3](https://www.anaconda.com/)

Install PyTorch and Torchvision (Conda):
```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Install PyTorch and Torchvision (Pip3):
```bash
pip3 install torch torchvision
```
## Experiments
We verify the effectiveness of T_revision on three synthetic noisy datasets (MNIST, CIFAR-10, CIFAR-100), and one real-world noisy dataset (clothing1M). And We provide [datasets](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format).        
Here is an example: 
```bash
python main.py --dataset cifar10 --noise_rate 0.5
```
