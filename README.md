# Real-time Drone Instance Segmentation in Videos Using Optimized Mask Regional-Convolutional Neural Network 

![masked_images](https://github.com/jakariaemon/Custom-object-detection-using-Detectron2/assets/43466665/4fd22d49-ddd1-4170-a654-5b065f211f9c) 

# Installation 

## Required Libraries (Version)  
```
detectron2 == 0.6
torch 2.0.1+cu118
NVCC 11.5
``` 
Install detectron2 (Please use WSL) 
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Install torch 
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
``` 

## Anti-UGV-IS Dataset 

[Google drive](https://drive.google.com/file/d/1LTEWD1X6iWdzTYhWaLLWhnCIZFzvqIgj/view?usp=sharing)


## Training 

- Download the annotated dataset and create split using "04_split_dataset.py".
- To run hyperparameter optimization run "05_hyperparameter_optimization.py". Note that some backbone may took 11GB+ vram. So its better to use google colab (Tesla T4). 
- After that run "06_single_train.py". The model will be saved in the "Output" folder. That can be use for inferencing. 


## Inferencing 
