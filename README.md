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
  
https://github.com/user-attachments/assets/f5247777-d2be-4d14-adf3-9389a3d3f58c

</div>
<div align="center">   

https://github.com/user-attachments/assets/bf224943-e43f-414b-b498-928833938c18

</div>
<div align="center">   

 https://github.com/user-attachments/assets/49c7fc1a-73c0-4cb0-914a-484c0b7bffbc

</div>
<div align="center">   

  https://github.com/user-attachments/assets/2b98710c-a5ff-4005-ae50-a09586275960

</div>




# Rendering Results in Lane Shift @ 6m Novel Trajectory

<div align="center">   

https://github.com/user-attachments/assets/d0df1043-066f-46e5-9ce5-4ce680762a06

</div>
<div align="center">   

https://github.com/user-attachments/assets/6239614e-232a-47f9-8dd2-012508cdb2b2

</div>
<div align="center">   

https://github.com/user-attachments/assets/9d1e7af2-34a5-483f-b123-3f65cc1ad00d

</div>
<div align="center">   

https://github.com/user-attachments/assets/f50cdf5d-5b0d-4cba-a0ce-df44bd8cc84b


</div>


