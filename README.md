# OSTRA

**OSTRA** is a novel segmentation-then-reconstruction method for segmenting complex open objects in 3D point clouds. This method uses a Segment-Anything Model (SAM) to segment target objects and video object segmentation (VOS) technology to continuously track video frame segmentation targets. Our pipeline enables a complete segmentation process from videos to 3D cloud points and meshes in different level(semantic segmentation, instance segmentation and part segmentation).



**You can check our detailed tutorials [here](./doc/tutorials.md)!**

**SAM-Adapter** Users can utilize SAM-Adapter to generate masks, replacing VOS for mask generation. The adapter can improve SAM's segmentation performance on rice panicles by fine-tuning it with downstream training. We use Tianrun Chen's work as the backbone for SAM-Adapter.

**You can check our detailed tutorials [here](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)!**

## :computer:Getting Started
### Clone OSTRA
This project is tested under python3.9, cuda11.5 and pytorch1.11.0. An equivalent or higher version is recommended.
```shell
#Clone OSTRA
git clone --recursive https://github.com/ganlab/OSTRA.git
cd OSTRA

#Install dependencies:
pip install -r requirements.txt
```

### Install Colmap
Our reconstruction process is based on Colmap. Please follow the instruction and install [Colmap](https://github.com/colmap/colmap) first.

### Prepare Model
All these models are required for OSTRA:
SAM: the default model is [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
DeAOT:the default model is [R50_DeAOTL_PRE_YTB_DAV.pth](https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view)
XMem：the default model is [XMem-s012.pth](https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth)
Grounding-DINO:the default model is [groundingdino_swint_ogc.pth](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth)

### WebUI
We developed WebUI that user can easily access.
```shell
python app.py --device cuda:0
```

## :film_projector: Demo

Two samples of complex object segmentation:
<div align=center class='img_top'>
<img src="./assets/rice.jpg" width="90%"/>
</div>

<div align=center class='img_top'>
<img src="./assets/top.jpg" width="90%"/>
</div>

<div align=center class='img_top'>
<img src="./assets/result.gif" width="90%"/>
</div>

## :book:Citation
Please considering cite our paper if you find this work useful!
```
@misc{xu2023stop,
      title={A One Stop 3D Target Reconstruction and multilevel Segmentation Method}, 
      author={Jiexiong Xu and Weikun Zhao and Zhiyan Tang and Xiangchao Gan},
      year={2023},
      eprint={2308.06974},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
Free for non-profit research purposes. Please contact authors otherwise. The program itself may not be modified in any way and no redistribution is allowed.
No condition is made or to be implied, nor is any warranty given or to be implied, as to the accuracy of OSTRA, or that it will be suitable for any particular purpose or for use under any specific conditions, or that the content or use of OSTRA will not constitute or result in infringement of third-party rights.

## :clap: Acknowledgements
This work is based on [Segment Anything](https://github.com/facebookresearch/segment-anything),  [Track Anything](https://github.com/gaomingqi/Track-Anything/tree/master), [Segment and Track Anything](https://github.com/z-x-yang/Segment-and-Track-Anything), [Colmap](https://github.com/colmap/colmap), [Open3D](https://github.com/isl-org/Open3D) and [SAM-Adapter-backbone](https://github.com/tianrun-chen/SAM-Adapter-PyTorch). The software is developed by following author(s) and supervised by Prof. Xiangchao Gan(gan@njau.edu.cn)

Authors:

Jiexiong Xu 
xujx@stu.njau.edu.cn 
work: framework and reconstruction module

Weikun Zhao 
zhaowk@stu.njau.edu.cn 
work: VOS module
