# KFVQA
No-reference video quality assessment with guidance of keyframe extraction

## Intruduction

### Abstract

Video quality assessment (VQA) is a foundational research in computer vision that aims to simulate the human visual system to evaluate video quality and determine its quality level. The most fundamental factor affecting the accuracy of the final score predicted by the video quality assessment model is the extracted frames. Most existing methods prioritize the study of refining feature extraction to improve performance while ignoring improvements in initial frame extraction. To solve these problems, we propose a no-reference video quality assessment with the guidance of keyframe extraction(KFVQA), which aims to improve model performance through keyframe extraction. Specifically, we propose a keyframe extraction module to ensure that the extracted frames present diversity and greater representativeness in content, and avoid excessive similarity and repetition rates. Secondly, we utilize the self-attention for feature focusing module and weighted feature fusion module in KFVQA to better extract spatial features and more accurately focus and emphasize key motion features and spatial feature parts to improve the accuracy and robustness of quality assessment.

## 1. Requirements
```
pytorch
opencv
scipy
pandas
torchvision
torchvideo
```

## 2.Databases
[LSVQ](https://github.com/baidut/PatchVQ)
[KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html)
[Youtube-UGC](https://media.withyoutube.com/)

## 3. Train models
1. Extract video frames
```shell
python extract_frame.py
```
2. Extract motion features
```shell
python extract_SlowFast_features_VQA.py 
```
3. Train the model
```shell
python train.py
```
## 4.Test the model
You can download the trained model via [Google Drive](https://drive.google.com/drive/my-drive).

Test on the public VQA database

```shell
python test_on_pretrained_model.py
```

Test on a single video
```shell
python test_demo.py
```

### Acknowledgements

If you find this code is useful for your research, please cite:
```

```
