# DG4MHealth

# Resolving Requirements
In order to start working on this project, you should install the following packages with anaconda. The following commands works for Win 10 PC + cuda 11.7. 
```
$ """Recommend to do conda clean -all and pip cache purge before installing these packages"""
$ conda create --name dg4mhealth python=3.8
$ """Note, the pytorch-cuda version depends on your cuda version, follow https://pytorch.org/get-started/locally/"""
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip install -U scikit-learn
```

# Data Directory
datasets/
  - dsads
    - data
      - a01
      - ...
      - a19
  - pamap2
    - subject101.dat
    - ...
    - subject108.dat
  - uschad
    - Subject1
    - ...
    - Subject14

# Training & Evaluation
```
$ python dg_supcon.py --dataset dsads --model dsads64 --batch_size 256 --learning_rate 0.08 --temp 0.07 --epochs 500 --cosine 1 --LODO 0 --weighted_supcon 1 --trial 0 --seed 0 > dsads_supcon_LODO_0.txt
```
