# DiffuMask
DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models

<p align="center">
<img src="./1684329483514.jpg" width="800px"/>  
<br>
</p>


## :hammer_and_wrench: Getting Started with DiffuMask
### Conda env installation

```
conda create -n DiffuMask python=3.8

conda activate DiffuMask
```

install pydensecrf https://github.com/lucasb-eyer/pydensecrf

```
cd DiffuMask

pip install git+https://github.com/lucasb-eyer/pydensecrf.git

pip install -r requirements.txt
```
```
If there is an error: 

bug for cannot import name 'autocast' from 'torch', 

please refer to the website:  

https://github.com/pesser/stable-diffusion/issues/14
```

### 1. Data and mask generation
```
# generating data and attention map witn stable diffusion (Before generating the data, you need to modify the "hunggingface key" in the "VOC_data_generation.sh" script to your own key. )
sh ./script/DiffusionGeneration/VOC_data_generation.sh
```

or

```
CUDA_VISIBLE_DEVICES=0 python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes aeroplane --thread_num 4 --output ./DiffMask_VOC/ --image_number 8000 --MY_TOKEN '0mzLY4TtGUkaACnbL8geexjfmx7eteZYGVJQZVaqb33qTA4hzkSRZRbgF4qQ'
```

### 2. Refine Mask with AffinityNet (Coarse Mask)
```
# prepare training data for affinity net
sh ./script/prepare_aff_data.sh

# train affinity net
# Before training, you need to download the ResNet-38 ImageNet pre-trained weights and place them in the "./pretrained_model" directory, and rename as 'res38_cls.pth'
sh ./script/train_affinity.sh

# inference affinity net
sh ./script/infer_aff.sh

# generate accurate pseudo label with CRF
sh ./script/curve_threshold.sh
```

### 3. Noise Learning (Cross Validation)

At this stage, it is necessary to train Mask2Former using cross-validation to filter out noisy data. Before training Mask2Former, data augmentation needs to be performed on the dataset.
```
sh ./script/augmentation_VOC.sh
```


To start training the model, please note the following points:

- In this repository, only data for one class (airplanes) is generated. During the training process, it is recommended to include other categories as negative samples. You can generate data for other categories or directly use other categories from the VOC dataset. The corresponding masks are not required, and the labels for other categories can be set to 0.

- During the training process, augmented data (concatenated images) with the original data (original images) all should be used.

### 4. Training Mask2former with clear data

