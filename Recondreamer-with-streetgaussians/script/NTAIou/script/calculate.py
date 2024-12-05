import os
import os
from shapely.geometry import box
from shapely.ops import unary_union
import numpy as np
import argparse

def calculate_iou(boxA, boxB):   
    x_inter_min = max(boxA[0], boxB[0])  
    y_inter_min = max(boxA[1], boxB[1])  
    x_inter_max = min(boxA[2], boxB[2])  
    y_inter_max = min(boxA[3], boxB[3])  
    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)  
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])  
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  
    union_area = boxA_area + boxB_area - inter_area  
    iou = inter_area / union_area  
    return iou


def find_closest_box(gt_box, ge_boxes):
    gt_center = [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]
    min_distance = 10
    closest_box = None
    for ge_box in ge_boxes:
        ge_center = [(ge_box[0] + ge_box[2]) / 2, (ge_box[1] + ge_box[3]) / 2]
        distance = np.linalg.norm(np.array(gt_center) - np.array(ge_center))
        if distance < min_distance:
            min_distance = distance
            closest_box = ge_box
    return closest_box if min_distance <= 10 else None


def iou(args):
    shift_values = [-1, -2, -3, 1, 2, 3, 4, 5]
    avg_ious = {}

    for scene_id in args.scene_ids:
        with open(args.output_file.replace('.txt',f'_{scene_id}.txt'), 'w') as outfile:
            for data in args.datasets:
                shift_value_iou = {sv: [] for sv in shift_values}
                detect_base_dir = os.path.join(args.output_dir, args.exp,scene_id,data, 'detect_txt')
                gt_base_dir = args.gt_base_dir

                detect_folders = [f for f in os.listdir(detect_base_dir)]

                for detect_folder in detect_folders:
                    try:
                        shift_value = int(detect_folder.split('shifting_')[1].split('_')[0])
                    except (IndexError, ValueError):
                        print(f"Could not determine shift value from folder name: {detect_folder}")
                        continue

                    if shift_value not in shift_values:
                        continue

                    gt_folder = f'{shift_value}'
                    gt_path = os.path.join(gt_base_dir, gt_folder)

                    if not os.path.exists(gt_path):
                        continue

                    detect_files = [f for f in os.listdir(os.path.join(detect_base_dir, detect_folder)) if f.endswith('.txt')]

                    for detect_file in detect_files:
                        detect_where = os.path.join(detect_base_dir, detect_folder, detect_file)
                        with open(detect_where, 'r') as df:
                            detect_data = [list(map(float, line.strip().split(','))) for line in df.readlines()]

                        gt_file = os.path.join(gt_path, detect_file).replace('_rgb.', '.')

                        if not os.path.exists(gt_file):
                            print(f"Warning: Corresponding GT file {gt_file} does not exist.")
                            continue
                        
                        with open(gt_file, 'r') as gtf:
                            gt_data = [list(map(float, line.strip().split(','))) for line in gtf.readlines()]
                    
                        for gt_box in gt_data:
                            closest_detect_box = find_closest_box(gt_box, detect_data)
                            iou = calculate_iou(gt_box, closest_detect_box) if closest_detect_box is not None else 0
                            shift_value_iou[shift_value].append(iou)

                outfile.write(f"Results for dataset: {data}\n")
                avg_ious[data] = {}
                for shift_val in shift_values:
                    avg_iou = sum(shift_value_iou[shift_val]) / len(shift_value_iou[shift_val]) if shift_value_iou[shift_val] else 0
                    outfile.write(f"Average IoU for shift value {shift_val}: {avg_iou:.4f}\n")
                    avg_ious[data][shift_val] = avg_iou
                outfile.write("\n")  

            outfile.write("Improvement of recondreamer over street_gaussians:\n")
            for shift_val in shift_values:
                street_avg_iou = avg_ious['street_gaussians'][shift_val]
                recondreamer_avg_iou = avg_ious['recondreamer'][shift_val]
                if street_avg_iou == 0:
                    improvement_percentage = 'Inf'  
                else:
                    improvement_percentage = ((recondreamer_avg_iou - street_avg_iou) / street_avg_iou) * 100
                improvement = recondreamer_avg_iou - street_avg_iou
                outfile.write(f"Shift value {shift_val}: recondreamer improved by {improvement_percentage:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and compare average IoU for different datasets.")
    parser.add_argument('--exp', type=str, default='waymo_full_exp', help='Experiment name')
    parser.add_argument('--output_file', type=str, default='script/NTAIou/average_iou_results.txt', help='Output file to write results')
    parser.add_argument('--output_dir', type=str, default='output', help='Base directory for output files')
    parser.add_argument('--gt_base_dir', type=str, default='data/005/shift_gt', help='Base directory for ground truth files')
    parser.add_argument('--datasets', nargs='+', default=['street_gaussians', 'recondreamer'], help='Datasets to process')
    parser.add_argument('--scene_ids', nargs='+', default=['005'], help='Datasets to process')

    args = parser.parse_args()
    iou(args)