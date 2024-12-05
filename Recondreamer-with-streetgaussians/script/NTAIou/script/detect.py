import argparse
from ultralytics import YOLO
import cv2
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Process images with YOLO model and save detection results.")
    parser.add_argument('--exp', type=str, default='waymo_full_exp', help='Experiment name')
    parser.add_argument('--model_path', type=str, default='pt/yolo11x.pt', help='Path to the YOLO model file')
    parser.add_argument('--data_types', nargs='+', default=['street_gaussians', 'recondreamer'], help='Data types to process')
    parser.add_argument('--target_size', type=int, nargs=2, default=[480, 320], help='Target size for resizing images (width, height)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--class_ids', type=int, nargs='+', default=[2, 5], help='Class IDs to filter detections by')
    parser.add_argument('--scene_ids', type=int, nargs='+', default=['005'])
    return parser.parse_args()

def detect(args):
    model = YOLO(args.model_path)


    for data in args.data_types:
        for scene_id in args.scene_ids: 
            inputs = []
            base_dir = f'output/{args.exp}/{scene_id}/{data}/trajectory'
            for action in os.listdir(base_dir):
                path = os.path.join(base_dir, action)
                if os.path.exists(path) and os.path.isdir(path):
                    inputs.append(path)
            
            output_root = f"output/{args.exp}/{scene_id}/{data}/detect_txt"
            os.makedirs(output_root, exist_ok=True)  # Ensure the root output directory exists
            
            for input_path in inputs:
                print(input_path)
                input_files = sorted(glob.glob(os.path.join(input_path, '*')))
                input_files = [file for file in input_files if file.endswith('rgb.png')]
                for file in input_files:
                    if file.lower().endswith((".jpg", ".png", ".jpeg")):
                        image = cv2.imread(file)
                        resized_image = cv2.resize(image, tuple(args.target_size))
                        output_dir = os.path.join(output_root, file.split('/')[-2])
                        os.makedirs(output_dir, exist_ok=True)  
                        image_name = file.split('/')[-1].replace('_train.', '.').replace('.png', '')
                        output_file = os.path.join(output_dir, f"{image_name}.txt") 
                        results = model(resized_image)
                        with open(output_file, 'w') as f:
                            for r in results:
                                for box in r.boxes:
                                    if int(box.cls[0].cpu().numpy()) not in args.class_ids:
                                        continue
                                    conf = box.conf.cpu().numpy()
                                    if conf > args.confidence_threshold:  
                                        xyxy = box.xyxy[0].cpu().numpy() 
                                        x1, y1, x2, y2 = map(int, xyxy) 
                                        f.write(f"{x1},{y1},{x2},{y2}\n")
                                        cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2
                                        
        print("Processing completed for all images.")

if __name__ == "__main__":
    args = parse_args()
    detect(args)