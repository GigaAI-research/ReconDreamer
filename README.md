<div align="center">   
  
# ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration
</div>

 
## [Project Page](https://recondreamer.github.io/) | [Paper]()

# News
- **[2024/11/30]** Repository Initialization.

# Abstract 
Closed-loop Simulation is Crucial for End-to-end Autonomous Driving. Existing sensor simulation methods (e.g., NeRF and 3DGS) reconstruct driving scenes based on conditions that closely mirror training data distributions. However, these methods struggle with rendering novel trajectories, such as lane changes. Recent work, DriveDreamer4D, has demonstrated that integrating world model knowledge alleviates these issues. Although the training-free integration approach is efficient, it still struggles to render larger maneuvers, such as multi-lane shifts.Therefore, we introduce **ReconDreamer**, which enhances driving scene reconstruction through incremental integration of world model knowledge. Specifically, based on the world model, *DriveRestorer* is proposed to mitigate ghosting artifacts via online restoration. Additionally, we propose the progressive data update strategy to ensure high-quality rendering for larger maneuvers. Notably, **ReconDreamer** is the first method to effectively render in large maneuvers (e.g., across multiple lanes, spanning up to 6 meters).Additionally, experimental results demonstrate that **ReconDreamer** outperforms Street Gaussians in the NTA-IoU, NTL-IoU, and FID, with a relative improvement by 24.87%, 6.72%, and 29.97%. Furthermore, **ReconDreamer** surpasses DriveDreamer4D with PVG during large maneuver rendering, as verified by a relative improvement of 195.87% in the NTA-IoU metric and a comprehensive user study.
# DriveDreamer4D Framework

<img width="1349" alt="method" src="[https://github.com/user-attachments/assets/3bb9f09f-2743-4c5b-bb4f-b3d6ea675f56](https://github.com/user-attachments/assets/e0499805-8610-44f6-b4a6-3ec14056df20)">

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

# Rendering Results in Lane Shift @ 3m Novel Trajectory


<div align="center">   
  
https://github.com/user-attachments/assets/41043245-b813-4378-a43c-684bf0334d4b

</div>
<div align="center">   

  https://github.com/user-attachments/assets/0133c96b-61e8-41f4-a94b-85316c1c98d4

</div>
<div align="center">   

  https://github.com/user-attachments/assets/a381d30b-bc78-4a63-8f7c-613bc8bee0c6

</div>
<div align="center">   

  https://github.com/user-attachments/assets/5556e61d-cf7b-4539-8c2f-3acefb256c1d

</div>


# Rendering Results in Lane Shift @ 6m Novel Trajectory

<div align="center">   

https://github.com/user-attachments/assets/4cb4e8c5-37d3-4d55-9495-71f7e054db77

</div>
<div align="center">   

https://github.com/user-attachments/assets/fb7b3a64-926c-4aab-96f7-f8f6e641e322

</div>
<div align="center">   

https://github.com/user-attachments/assets/75119b8b-ba72-40a5-b13e-467791d36907

</div>
<div align="center">   

https://github.com/user-attachments/assets/07da54cf-7cd8-4159-8723-6bf708d25c87


</div>


