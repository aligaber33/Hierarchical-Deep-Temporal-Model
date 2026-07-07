import cv2
import os
import pickle
from pathlib import Path
from typing import List
from Box_info import BoxInfo

videos_root = Path('videos-splitted')
annot_root = Path('volleyball_tracking_annotation')


def load_tracking_annot(path):
    with open(path, 'r') as file:
        # Initialize dictionary to hold box info for all potential players
        player_boxes = {idx: [] for idx in range(100)}  # Expanded range to prevent missing high-value tracking IDs
        frame_boxes_dct = {}

        for line in file:
            if not line.strip():
                continue
            box_info = BoxInfo(line)
            
            # REMOVED: The rigid "if box_info.player_ID > 11" filter to ensure all players pass through
            player_boxes[box_info.player_ID].append(box_info)

        for player_ID, boxes_info in player_boxes.items():
            if not boxes_info:
                continue
                
            # Keep track data intact, or adjust window clipping safely
            if len(boxes_info) > 11:
                # If you want to keep ALL frames for a baseline, change this to: boxes_info = boxes_info
                boxes_info = boxes_info[5:-6]
            else:
                mid = len(boxes_info) // 2
                boxes_info = boxes_info[max(0, mid - 4) : min(len(boxes_info), mid + 5)]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []
                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct


def vis_clip(annot_path, video_dir):
    frame_boxes_dct = load_tracking_annot(annot_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    video_writer = None
    output_video_path = Path("output_clip.mp4")

    for frame_id in sorted(frame_boxes_dct.keys()):
        boxes_info = frame_boxes_dct[frame_id]
        img_path = Path(video_dir) / f'{frame_id}.jpg'
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Frame missing at path {img_path}")
            continue

        for box_info in boxes_info:
            x1, y1, x2, y2 = map(int, box_info.box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(box_info.category), (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

        if video_writer is None:
            height, width, _ = image.shape
            video_writer = cv2.VideoWriter(str(output_video_path), 0x7634706d, 5.0, (width, height))

        video_writer.write(image)

    if video_writer is not None:
        video_writer.release()
        print(f"\n🚀 Success! Video saved safely to workspace root: {output_video_path}")
    else:
        print("Error: No frames were compiled into a video.")


def load_video_annot(video_annot):
    if not os.path.exists(video_annot):
        return {}
    with open(video_annot, 'r') as file:
        clip_category_dct = {}
        for line in file:
            items = line.strip().split(' ')[:2]
            if len(items) < 2:
                continue
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]
        return clip_category_dct


# 1. FIXED DEFINITION: Takes both parameters
def find_video_dir_for_clip(video_id, clip_id):
    """
    Looks inside the separate videos_root ('videos-splitted') 
    across all 'videos_gX' folders, accounting for the inner video_id folder.
    """
    for group_folder in videos_root.iterdir():
        if group_folder.is_dir() and group_folder.name.startswith("videos_g"):
            potential_path = group_folder / video_id / clip_id
            if potential_path.is_dir():
                return potential_path
    return None


def load_volleyball_dataset():
    if not annot_root.exists():
        print(f"Directory Error: {annot_root} not found.")
        return {}
        
    video_dirs = sorted(os.listdir(str(annot_root)))
    videos_annot = {}

    for idx, video_dir in enumerate(video_dirs):
        video_annot_path = annot_root / video_dir
        if not video_annot_path.is_dir():
            continue

        print(f'[{idx + 1}/{len(video_dirs)}] Processing Annotation Video ID: {video_dir}')

        clips_dir = sorted(os.listdir(str(video_annot_path)))
        clip_annot = {}

        for clip_dir in clips_dir:
            clip_annot_path = video_annot_path / clip_dir
            if not clip_annot_path.is_dir():
                continue

            actual_video_dir = find_video_dir_for_clip(video_dir, clip_dir)
            if actual_video_dir is None:
                continue

            video_annot_file = actual_video_dir / 'annotations.txt'
            clip_category_dct = load_video_annot(str(video_annot_file))
            
            if clip_dir not in clip_category_dct:
                continue

            annot_file = clip_annot_path / f'{clip_dir}.txt'
            if not annot_file.exists():
                continue
                
            frame_boxes_dct = load_tracking_annot(str(annot_file))

            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot


def create_pkl_version():
    videos_annot = load_volleyball_dataset()
    out_pkl = Path('annot_all.pkl')
    with open(out_pkl, 'wb') as file:
        pickle.dump(videos_annot, file)
    print(f"Dataset successfully compiled into pickle format at {out_pkl}")


if __name__ == '__main__':
    # TARGET CONFIGURATION FOR DEBUGGING A SINGLE CLIP
    target_video_id = "4"
    target_clip_id = "24745"
    
    annot_file = annot_root / target_video_id / target_clip_id / f"{target_clip_id}.txt"
    
    # 3. FIXED CALL: Passing both target variables down here as well!
    clip_dir_path = find_video_dir_for_clip(target_video_id, target_clip_id)

    print(f"Looking for tracking files in: {annot_file}")
    print(f"Located image frames in: {clip_dir_path}")
    
    if annot_file.exists() and clip_dir_path is not None:
        vis_clip(str(annot_file), str(clip_dir_path))
    else:
        print("🚨 File Configuration Error: Could not link annotation text file to its video images.")