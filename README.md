# EETS-Net
The codes for the work "An Edge Enhanced Two-Stage Network for Nuclei Segmentation". 
Thanks to Swin Unet and Swin Transformer for publishing the excellent code。
## 1. Download pre-trained swin transformer model (Swin-B)
* [Get pre-trained model in this link] (https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file): Put pretrained Swin-B into folder "pretrained_ckpt/"

## 2. Prepare data

- The two public datasets we used, CPM-17 and MoNuSeg, are available from the links below.  (https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK) and (https://monuseg.grand-challenge.org/Data/). 

## 3. Environment

- Please prepare an environment with python=3.9, cuda=11.8, numpy=1.22, and install the remaining packages as needed.

## 4. Train/Test

- The batch size we used is 8.  Our test found that batch_size has little effect on the final result. You can modify the value of batch_size according to the size of GPU memory.


## References
* [Swin unet](https://github.com/HuCaoFighting/Swin-Unet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [DE-DCGCN-EE](https://github.com/YangLibuaa/DE-DCGCN-EE)


