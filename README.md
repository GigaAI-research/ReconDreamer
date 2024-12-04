<div align="center">   
  
# ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration
</div>

 
## [Project Page](https://recondreamer.github.io/) | [Paper](https://arxiv.org/abs/2411.19548)

# News
- **[2024/12/04]** 🚀 We have released the weights for some scenarios based on [Street Gaussians](https://github.com/zju3dv/street_gaussians), as well as the **NTA-IoU** calculation script.
- **[2024/11/29]** 🚀 Repository Initialization.

# Abstract 
Closed-loop simulation is crucial for end-to-end autonomous driving. Existing sensor simulation methods (e.g., NeRF and 3DGS) reconstruct driving scenes based on conditions that closely mirror training data distributions. However, these methods struggle with rendering novel trajectories, such as lane changes. Recent works have demonstrated that integrating world model knowledge alleviates these issues. Despite their efficiency, these approaches still encounter difficulties in the accurate representation of more complex maneuvers, with multi-lane shifts being a notable example.Therefore, we introduce **ReconDreamer**, which enhances driving scene reconstruction through incremental integration of world model knowledge. Specifically, *DriveRestorer* is proposed to mitigate artifacts via online restoration. This is complemented by a progressive data update strategy designed to ensure high-quality rendering for more complex maneuvers. To the best of our knowledge, **ReconDreamer** is the first method to effectively render in large maneuvers. Experimental results demonstrate that **ReconDreamer** outperforms Street Gaussians in the NTA-IoU, NTL-IoU, and FID, with relative improvements by 24.87%, 6.72%, and 29.97%. Furthermore, **ReconDreamer** surpasses DriveDreamer4D with PVG during large maneuver rendering, as verified by a relative improvement of 195.87% in the NTA-IoU metric and a comprehensive user study.
# ReconDreamer Framework

<img width="1349" alt="method" src="https://github.com/user-attachments/assets/e9d52662-f657-4d56-8b4c-aab8de2549c9">

# Scenario Selection

All selected scenes are sourced from the validation set of the Waymo dataset. The official file names of these scenes, are listed along with their respective starting and ending frames.
| Scene | Start Frame | End Frame |
| :-----| :----: | :----: |
| segment-10359308928573410754_720_000_740_000_with_camera_labels.tfrecord | 120 | 159 |
| segment-11450298750351730790_1431_750_1451_750_with_camera_labels.tfrecord | 0 | 39 |
| segment-12496433400137459534_120_000_140_000_with_camera_labels.tfrecord | 110 | 149 |
| segment-15021599536622641101_556_150_576_150_with_camera_labels.tfrecord | 0 | 39 |
| segment-16767575238225610271_5185_000_5205_000_with_camera_labels.tfrecord | 0 | 39 |
| segment-17860546506509760757_6040_000_6060_000_with_camera_labels.tfrecord | 90 | 129 |
| segment-3015436519694987712_1300_000_1320_000_with_camera_labels.tfrecord | 40 | 79 |
| segment-6637600600814023975_2235_000_2255_000_with_camera_labels.tfrecord | 70 | 109 |

# Install
```
conda create -n recondreamer python=3.8
conda activate recondreamer
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirments.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/simple-waymo-open-dataset-reader
python script/test_gaussian_rasterization.py
```

### Demo

| Scene | Weight                                   | Data                                |
|-------|------------------------------------------|-------------------------------------|
| 005   | [Link to Weight](https://pan.baidu.com/s/1iElO_mg04nqiVsPV7DdTnw?pwd=ub9b) | [Link to Data](https://pan.baidu.com/s/1cLBQK0quZ-ek8Grr7U_EDg?pwd=76yy) |
```
Recondreamer/
├── config/
│   ├── street_gaussians.yaml
│   └── recondreamer.yaml
|── data
├── .....
├── output/
│    └── waymo_full_exp/005
│        ├── recondreamer
│        └── street_gaussians
│── render.py
```
After downloading the data and weights, place the **Data**  contents into the `data/` folder, and move the **Weight**  files into the `output/` directory.
### Rendering
```
python render.py --config configs/recondreamer.yaml
python render.py --config configs/street_gaussians.yaml
```

### Calculate the NTA-IoU metric
| Model | Weight                                   | 
|-------|------------------------------------------|
| Yolo11x.pt   | [Link to Weight](https://pan.baidu.com/s/1-geZP_WvHPNXXNSkGRv9fA?pwd=3piv) | 
```
python script/NTAIou/script/GT.py
python script/NTAIou/script/detect.py
python script/NTAIou/script/calculate.py
```
The detailed metrics will be saved in the file: `script/NTAIou/average_iou_results_005.txt`.

# Rendering Results in Lane Shift @ 3m Novel Trajectory
<div align="center">   

https://github.com/user-attachments/assets/ebcec5e1-af90-4246-a1d7-a1be70157287

</div>
<div align="center">   

https://github.com/user-attachments/assets/746b0592-62cc-4987-b693-e130bef003d5

</div>
<div align="center">   

https://github.com/user-attachments/assets/840a303a-7142-467f-92fc-c4b05c0ed8f8

</div>
<div align="center">   
  
https://github.com/user-attachments/assets/4341d265-fa43-4580-bad8-c287582ee7d4

</div>

# Rendering Results in Lane Shift @ 6m Novel Trajectory

<div align="center">   

https://github.com/user-attachments/assets/8594d451-9702-431f-ada8-88fd1833bd6e

</div>
<div align="center">   

https://github.com/user-attachments/assets/c81dc70b-caed-48e3-a30b-d587a6c0e2fa

</div>
<div align="center">   

https://github.com/user-attachments/assets/0ea19c8b-473a-49a1-900f-95abe95a0371

</div>
<div align="center">   

https://github.com/user-attachments/assets/7f3f4c8b-3d4b-46a5-b0c7-788fd533d9de

</div>


</div>

### Acknowledgements
We would like to thank the following works and projects, for their open research and exploration: [Street Gaussians](https://github.com/zju3dv/street_gaussians), [DriveStudio](https://github.com/ziyc/drivestudio), [DriveDreamer](https://github.com/JeffWang987/DriveDreamer), and [DriveDreamer4D](https://github.com/GigaAI-research/DriveDreamer4D).


### Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{Ni2024ReconDreamerCW,
    title={ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration},
    author={Chaojun Ni and Guosheng Zhao and Xiaofeng Wang and Zheng Zhu and Wenkang Qin and Guan Huang and Chen Liu and Yuyin Chen and Yida Wang and Xueyang Zhang and Yifei Zhan and Kun Zhan and Peng Jia and Xianpeng Lang and Xingang Wang and Wenjun Mei},
    year={2024},
    url={https://arxiv.org/abs/2411.19548}
}
```
