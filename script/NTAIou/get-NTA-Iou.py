import os
import os
from shapely.geometry import box
from shapely.ops import unary_union
import numpy as np
import argparse
import sys
sys.path.append('/mnt/pfs/users/chaojun.ni/1-code/release-code/')
from script.NTAIou.script.detect import *
from script.NTAIou.script.calculate import *
from script.NTAIou.script.GT import *

def parse_args():
    parser = argparse.ArgumentParser(description="Process images with YOLO model and save detection results.")
    parser.add_argument('--exp', type=str, default='waymo_full_exp', help='Experiment name')
    parser.add_argument('--model_path', type=str, default='pt/yolo11x.pt', help='Path to the YOLO model file')
    parser.add_argument('--data_types', nargs='+', default=['street_gaussians', 'recondreamer'], help='Data types to process')
    parser.add_argument('--target_size', type=int, nargs=2, default=[480, 320], help='Target size for resizing images (width, height)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--class_ids', type=int, nargs='+', default=[2, 5], help='Class IDs to filter detections by')
    parser.add_argument('--scene_ids', type=int, nargs='+', default=['005'])
    parser.add_argument('--output_file', type=str, default='script/NTAIou/average_iou_results.txt', help='Output file to write results')
    parser.add_argument('--output_dir', type=str, default='output', help='Base directory for output files')
    parser.add_argument('--gt_base_dir', type=str, default='data/005/shift_gt', help='Base directory for ground truth files')
    parser.add_argument('--datasets', nargs='+', default=['street_gaussians', 'recondreamer'], help='Datasets to process')
    return parser.parse_args()

def main(args):
    GT(args)
    detect(args)
    iou(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)