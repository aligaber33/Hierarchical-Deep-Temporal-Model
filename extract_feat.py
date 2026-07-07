import os
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from Annotation_loader import load_tracking_annot 

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
videos_root = Path('videos-splitted')
annot_root = Path('volleyball_tracking_annotation')
# Shifted target to person-level to cleanly match your models configuration layout
output_root = Path('features/person-level/resnet')

def check():
    print('Torch version:', torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available.")
        num_devices = torch.cuda.device_count()
        print(f"Number of GPU devices: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")


def prepare_model(image_level=False):
    if image_level:
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)
    model.eval()

    return model, preprocess, device


def find_video_dir_for_clip(video_id, clip_id):
    for group_folder in videos_root.iterdir():
        if group_folder.is_dir() and group_folder.name.startswith("videos_g"):
            potential_path = group_folder / video_id / clip_id
            if potential_path.is_dir():
                return potential_path
    return None


def extract_features(clip_dir_path, annot_file, output_file, model, preprocess, device, image_level=False):
    frame_boxes = load_tracking_annot(str(annot_file))
    
    # ENHANCED: Enforce fixed dimensions during extraction to keep matrices uniform
    target_frames = 9
    target_people = 12
    feature_dim = 2048
    
    clip_matrix_list = []
    sorted_frames = sorted(frame_boxes.keys())[:target_frames] # Ensure we take up to 9 frames max

    with torch.no_grad():
        for frame_id in sorted_frames:
            boxes_info = frame_boxes[frame_id]
            try:
                img_path = Path(clip_dir_path) / f'{frame_id}.jpg'
                if not img_path.exists():
                    continue
                    
                image = Image.open(img_path).convert('RGB')

                if image_level:
                    preprocessed_image = preprocess(image).unsqueeze(0).to(device)
                    dnn_repr = model(preprocessed_image)
                    dnn_repr = dnn_repr.view(1, -1).cpu().numpy() # Shape: [1, 2048]
                    clip_matrix_list.append(dnn_repr)
                else:
                    preprocessed_images = []
                    for box_info in boxes_info[:target_people]: # Limit to max 12 people
                        x1, y1, x2, y2 = map(int, box_info.box)
                        cropped_image = image.crop((x1, y1, x2, y2))
                        preprocessed_images.append(preprocess(cropped_image).unsqueeze(0))

                    if not preprocessed_images:
                        # Append a dummy frame matrix of zeros if empty
                        clip_matrix_list.append(np.zeros((target_people, feature_dim), dtype=np.float32))
                        continue

                    # Batch process all players in this frame at once
                    batch_tensor = torch.cat(preprocessed_images).to(device)
                    dnn_repr = model(batch_tensor)
                    dnn_repr = dnn_repr.view(len(preprocessed_images), -1).cpu().numpy()  # Shape: [Num_players, 2048]
                    
                    # ENHANCED: Pad people dimension with zeros up to exactly 12 right here
                    current_people = dnn_repr.shape[0]
                    if current_people < target_people:
                        padding = np.zeros((target_people - current_people, feature_dim), dtype=np.float32)
                        dnn_repr = np.vstack([dnn_repr, padding])
                        
                    clip_matrix_list.append(dnn_repr) # Perfect Shape: [12, 2048]

            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")

    # ENHANCED: Stack into a pure, clean 3D numeric array instead of a dictionary object
    if len(clip_matrix_list) > 0:
        final_clip_array = np.stack(clip_matrix_list, axis=0) # Shape: [Num_Frames, 12, 2048] or [Num_Frames, 1, 2048]
        
        # Final temporal padding block if the clip had fewer than 9 frames total
        if final_clip_array.shape[0] < target_frames:
            pad_frames = target_frames - final_clip_array.shape[0]
            spatial_dim = 1 if image_level else target_people
            temporal_padding = np.zeros((pad_frames, spatial_dim, feature_dim), dtype=np.float32)
            final_clip_array = np.vstack([final_clip_array, temporal_padding])
            
        # Save as a pure numeric array matrix file
        np.save(output_file, final_clip_array)


if __name__ == '__main__':
    check()

    # ENHANCED: Swapped to False for person-level multi-person activity tracking tasks
    image_level = True 
    model, preprocess, device = prepare_model(image_level)

    if not annot_root.exists():
        print(f"Error: {annot_root} folder missing.")
        exit()

    video_dirs = sorted(os.listdir(str(annot_root)))

    for idx, video_dir in enumerate(video_dirs):
        video_annot_path = annot_root / video_dir
        if not video_annot_path.is_dir():
            continue

        print(f'[{idx + 1}/{len(video_dirs)}] Processing Video ID: {video_dir}')

        clips_dir = sorted(os.listdir(str(video_annot_path)))
        video_output_dir = output_root / video_dir
        video_output_dir.mkdir(parents=True, exist_ok=True)

        for clip_dir in clips_dir:
            clip_annot_path = video_annot_path / clip_dir
            if not clip_annot_path.is_dir():
                continue

            annot_file = clip_annot_path / f'{clip_dir}.txt'
            if not annot_file.exists():
                continue

            clip_dir_path = find_video_dir_for_clip(video_dir, clip_dir)
            if clip_dir_path is None:
                print(f"\t⚠️ Warning: Visual frames missing for {video_dir}/{clip_dir}, skipping feature extraction.")
                continue

            output_file = video_output_dir / f'{clip_dir}.npy'
            
            print(f"\tExtracting features for Clip: {clip_dir}")
            extract_features(
                clip_dir_path=str(clip_dir_path),
                annot_file=annot_file,
                output_file=str(output_file),
                model=model,
                preprocess=preprocess,
                device=device,
                image_level=image_level
            )