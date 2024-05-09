# _IS-Fusion:_ Instance-Scene Collaborative Fusion for Multimodal 3D Object Detection
This repository contains the PyTorch implementation of the CVPR'2024 paper (**Highlight**), [*IS-Fusion: Instance-Scene Collaborative Fusion for Multimodal 3D Object Detection*](https://arxiv.org/pdf/2403.15241). This work simultaneously models instance-level and scene-level multimodal contexts to enhance 3D detection performance.

## Updates

* [2024.4.30] Code of IS-Fusion is released. 



## Abstract
Bird’s eye view (BEV) representation has emerged as a dominant solution 
for describing 3D space in autonomous
driving scenarios. However, objects in the BEV representation typically exhibit small sizes, and the associated point cloud context is inherently sparse, which leads to great challenges for reliable 3D perception. In this paper,
we propose IS-FUSION, an innovative multimodal fusion framework that jointly captures the Instance- and Scene-level contextual information. IS-FUSION essentially differs
from existing approaches that only focus on the BEV scenelevel fusion by explicitly incorporating instance-level multimodal information, thus facilitating the instance-centric
tasks like 3D object detection. It comprises a Hierarchical
Scene Fusion (HSF) module and an Instance-Guided Fusion (IGF) module. HSF applies Point-to-Grid and Grid-to-Region transformers to capture the multimodal scene context at different granularities. IGF mines instance candidates, explores their relationships, and aggregates the local
multimodal context for each instance. These instances then
serve as guidance to enhance the scene feature and yield
an instance-aware BEV representation. On the challenging nuScenes benchmark, IS-FUSION outperforms all the
published multimodal works to date.

## Citation
If you find this project is helpful for you, please cite our paper:


    @inproceedings{{yin2024isfusion,
      title={IS-FUSION: Instance-Scene Collaborative Fusion for Multimodal 3D Object Detection},
      author={Yin, Junbo and Shen, Jianbing and Chen, Runnan and Li, Wei and Yang, Ruigang and Frossard, Pascal and Wang, Wenguan},
      booktitle={CVPR},
      year={2024}
    }
    
## Main Results

#### 3D object detection results on nuScenes dataset.



| Method                   | Modality | mAP (val) | NDS (val) | mAP (test) | NDS (test) |
|--------------------------|----------|-----------|-----------|------------|------------|
| TransFusion-L (Baseline) | L        | 65.1      | 70.1      | 65.5       | 70.2       |  
| TransFusion-LC           | L+C      | 67.5      | 71.3      | 68.9       | 71.7       |
| BEVFusion                | L+C      | 68.5      | 71.4      | 70.2       | 72.9       |
| IS-Fusion (Ours)         | L+C      | 72.8      | 74.0      | 73.0       | 75.2       |


## Use IS-Fusion

### Installation

This project is based on torch 1.10.1, mmdet 2.14.0, mmcv 1.4.0 and mmdet3d 0.16.0. Please install mmdet3d following [getting_started.md](docs/getting_started.md). 
In addition, please install TorchEx with `cd mmdet3d/ops/TorchEx` and `pip install -v .`.

### Dataset Preparation
Please refer to [data_preparation.md](docs/data_preparation.md) to prepare the nuScenes dataset.
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

### Training and Evaluation
We provide the multimodal 3D detection config in [isfusion_0075voxel.py](configs/isfusion/isfusion_0075voxel.py). Start the training and evluation by running:
```
bash tools/run-nus.sh extra-tag
```

### Pretrained Models

| Models                    | Link | 
|---------------------------|-----|
| Pretrained Image Backbone | https://drive.google.com/file/d/1k3Eiy5SeeAt36SJVcVwpEUBtal8Uiz9P/view?usp=sharing     |
| Pretrained IS-Fusion      | https://drive.google.com/file/d/1mY2juJ2n0Dw5NWDSraZXrdU1RwkE-40h/view?usp=sharing |


## License

This project is released under MIT license, as seen in [LICENSE](LICENSE).




## Acknowlegement
Our project is partially supported by the following codebase. We would like to thank for their contributions.

* [TransFusion](https://github.com/XuyangBai/TransFusion)
* [AutoAlignV2](https://github.com/zehuichen123/AutoAlignV2)
* [BEVFusion](https://github.com/mit-han-lab/bevfusion)
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
