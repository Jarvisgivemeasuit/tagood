## [ACMMM 2024] TagOOD: A Novel Approach to Out-of-Distribution Detection via Vision-Language Representations and Class Center Learning


## Requirements
- clip==1.0
- colorama==0.4.6
- lightning==2.1.0
- numpy==1.23.3
- opencv_python==4.6.0.66
- Pillow==10.4.0
- ram==0.1
- scikit_learn==1.1.2
- torch==1.12.1
- torchsummary==1.5.1
- torchvision==0.13.1
- tqdm==4.64.1


## Dataset Preparation for Large-scale Experiment

### Install RAM

Please install RAM from [here](https://github.com/xinyu1205/recognize-anything) first.

### In-distribution dataset
Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in ```./data/train``` and ```./data/val```, respectively.

### Out-of-distribution datasets
we follow [MOS](https://github.com/deeplearning-wisc/large_scale_ood) and use Texture, iNaturalist, Places365 and SUN,  and de-duplicated concepts overlapped with ImageNet-1k. To further explore the limitation of our approach, we follow [VIM](https://github.com/haoqiwang/vim) and use ImageNet-O and OpenImage-O. 

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset, which can be download via the following links:
```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```
For Textures, we use the entire dataset, which can be downloaded from their [original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

ImageNet-O and OpenImage-O can be Download from [VIM](https://github.com/haoqiwang/vim).

Please put all downloaded OOD datasets into ```./data/ood_data/```. 

### Pretrained Models

Download our pretrained classifier model from [here](https://drive.google.com/file/d/1TeIg8qgN--1BoOENm8q68umNVpOjDYuF/view?usp=drive_link).

## Demo code for ImageNet Experiment

Run HVCM with ResNet-50 network on a single node with 4 GPUs for 300 epochs with the following command. 

```cd tagood``` and
run ```sh gen_data_tokens.sh``` for generating all the object features.

Run ```sh classifier_training.sh``` for centers training.

Run```sh ood_inference.sh``` for OOD detection. Noticed that for inference, the input from the backbone of RAM is recommended to be the entire image features without any filtering.


## Acknowledgement
[Recognize Anything Model](https://github.com/xinyu1205/recognize-anything)

## Citation
```
@article{li2024tagood,
  title={TagOOD: A Novel Approach to Out-of-Distribution Detection via Vision-Language Representations and Class Center Learning},
  author={Li, Jinglun and Zhou, Xinyu and Jiang, Kaixun and Hong, Lingyi and Guo, Pinxue and Chen, Zhaoyu and Ge, Weifeng and Zhang, Wenqiang},
  journal={arXiv preprint arXiv:2408.15566},
  year={2024}
}
```
