<div align="center">   
  
# ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration
</div>

 
## [Project Page](https://recondreamer.github.io/) | [Paper]()

# News
- **[2024/11/29]** Repository Initialization.

# Abstract 
Closed-loop Simulation is Crucial for End-to-end Autonomous Driving. Existing sensor simulation methods (e.g., NeRF and 3DGS) reconstruct driving scenes based on conditions that closely mirror training data distributions. However, these methods struggle with rendering novel trajectories, such as lane changes. Recent work, DriveDreamer4D, has demonstrated that integrating world model knowledge alleviates these issues. Although the training-free integration approach is efficient, it still struggles to render larger maneuvers, such as multi-lane shifts.Therefore, we introduce **ReconDreamer**, which enhances driving scene reconstruction through incremental integration of world model knowledge. Specifically, based on the world model, *DriveRestorer* is proposed to mitigate ghosting artifacts via online restoration. Additionally, we propose the progressive data update strategy to ensure high-quality rendering for larger maneuvers. Notably, **ReconDreamer** is the first method to effectively render in large maneuvers (e.g., across multiple lanes, spanning up to 6 meters).Additionally, experimental results demonstrate that **ReconDreamer** outperforms Street Gaussians in the NTA-IoU, NTL-IoU, and FID, with a relative improvement by 24.87%, 6.72%, and 29.97%. Furthermore, **ReconDreamer** surpasses DriveDreamer4D with PVG during large maneuver rendering, as verified by a relative improvement of 195.87% in the NTA-IoU metric and a comprehensive user study.
# DriveDreamer4D Framework

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


