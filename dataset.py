import os
import json
import random
import traceback
import numpy as np
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import repeat
from scipy.spatial.transform import Rotation as R


def get_local_rotation_matrix(x_angle, y_angle, z_angle):
    """
    Generate a local 4x4 rotation matrix from the given Euler angles in degrees.
    """
    r_matrix = R.from_euler('xyz', [x_angle, y_angle, z_angle], degrees=True).as_matrix()

    local_c2w = np.eye(4)
    local_c2w[:3, :3] = r_matrix
    return local_c2w

def crop_and_resize(target_size, fxfycxcy, square_crop):
    target_width, target_height = target_size
    fx, fy, cx, cy, h, w = fxfycxcy

    # squre crop
    if square_crop:
        min_size = min(w, h)
        start_h = (h - min_size) // 2
        start_w = (w - min_size) // 2
        cx -= start_w
        cy -= start_h


    if square_crop:
        min_size = min(w, h)
        new_fx = fx * (target_width / min_size)
        new_fy = fy * (target_height / min_size)
        new_cx = cx * (target_width / min_size)
        new_cy = cy * (target_height / min_size)
    else:
        new_fx = fx * (target_width / w)
        new_fy = fy * (target_height / h)
        new_cx = cx * (target_width / w)
        new_cy = cy * (target_height / h)
        
    return [new_fx, new_fy, new_cx, new_cy]

def resize_pano(image, depth, mask, target_size):
    target_width, target_height = target_size
    
    resized_image = cv2.resize(image, (target_width, target_height))
    if depth is not None:
        resized_depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        resized_depth = resized_depth[:, :, np.newaxis]
    else:
        resized_depth = None
    
    if mask is not None:
        resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask[:, :, np.newaxis]
    else:
        resized_mask = None

    return resized_image, resized_depth, resized_mask

class Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.evaluation = config.get("evaluation", False)
        self.viewpoint_max_view = config.data.get("viewpoint_max_view", 12)
        self.view_select_dict = config.data.get("view_select_dict", {})

        if self.evaluation and "data_eval" in config:
            self.config.data.update(config.data_eval)
        
        data_path_text = config.data.data_path
        data_root_dir = config.data.root_data_dir

        with open(data_path_text, 'r') as f:
            self.data_path = f.readlines()
        self.data_path = [x.strip() for x in self.data_path]
        self.data_path = [os.path.join(data_root_dir, x) for x in self.data_path if len(x) > 0]
        total_count = len(self.data_path)
        print(f"Finish load data from str, total: {total_count}\n")
        
        if not config.get("inference", False):
            np.random.shuffle(self.data_path)
        
        
    def __len__(self):
        return len(self.data_path)

    def process_frames(self, frames):
        fxfycxcy_list = []

        resize_h = self.config.data.get("resize_h", -1)
        patch_size = self.config.model.patch_size * self.config.model.get("patch_factor", 2)
        square_crop = self.config.data.square_crop

        resize_w = resize_h
        resize_h = int(round(resize_h / patch_size)) * patch_size # 
        resize_w = int(round(resize_w / patch_size)) * patch_size # 
        for frame in frames:
            fxfycxcyhw = [frame["fx"], frame["fy"], frame["cx"], frame["cy"], frame["h"], frame["w"]]
            fxfycxcy = crop_and_resize((resize_w, resize_h), fxfycxcyhw, square_crop)
            fxfycxcy_list.append(fxfycxcy)
        intrinsics = torch.tensor(fxfycxcy_list, dtype=torch.float32)  # (num_frames, 4)
        c2ws = np.stack([np.array(frame["c2w"]) for frame in frames])
        c2ws = torch.from_numpy(c2ws).float()
        c2w_bucket = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=c2ws.shape[0]).clone()
        c2w_bucket[:, :3] = c2ws[:, :3]  # (num_frames, 4, 4)

        return intrinsics, c2w_bucket

    def process_pano_frames(self, frames):
        image_list = []
        depth_list = []
        mask_list = []
        resize_h_pano = self.config.data.get("resize_h_pano", -1)
        patch_size = self.config.model.patch_size * self.config.model.get("patch_factor", 2)
        filter_top_down = self.config.data.get("filter_top_down", False)

        resize_w_pano = int(resize_h_pano * 2) # 512
        resize_h_pano = int(round(resize_h_pano / patch_size)) * patch_size # 256
        resize_w_pano = int(round(resize_w_pano / patch_size)) * patch_size # 512
        for frame in frames:
            image = np.array(Image.open(frame["image_path"]))
            h, w = image.shape[:2]
            if w/h != 2:
                return None, None, None, None, False

            if "depth_path" in frame and os.path.exists(frame["depth_path"]):
                depth = np.array(Image.open(frame["depth_path"])) # [h, w]
                depth = depth[:, :, np.newaxis] # [h, w, 1]
            else:
                depth = np.zeros((h, w, 1), dtype=image.dtype)
            
            if "mask_path" in frame and os.path.exists(frame["mask_path"]):
                mask = np.array(Image.open(frame["mask_path"]))/255
                if len(mask.shape) == 2:
                    mask = mask[:, :, np.newaxis]
            else:
                mask = np.ones((h, w, 1), dtype=image.dtype)
                if filter_top_down:
                    mask[:int(h//5), :, :] = 0
                    mask[int(4*h//5):, :, :] = 0
            
            depth_scale = 1.0
            if "depth_scale" in frame:
                depth_scale = frame["depth_scale"]
            image, depth, mask = resize_pano(image, depth, mask, (resize_w_pano, resize_h_pano))
            depth = depth * 1.0 / depth_scale # Convert back to meters

            image_list.append(torch.from_numpy(image / 255.0).permute(2, 0, 1).float())  # (3, resize_h, resize_w)
            depth_list.append(torch.from_numpy(depth).permute(2, 0, 1).float())  # (1, resize_h, resize_w)
            mask_list.append(torch.from_numpy(mask).permute(2, 0, 1).float()) # (1, resize_h, resize_w)

        images = torch.stack(image_list, dim=0)
        depths = torch.stack(depth_list, dim=0)
        masks = torch.stack(mask_list, dim=0) # (v, 1, resize_h, resize_w)
        c2ws = np.stack([np.array(frame["c2w"]) for frame in frames])
        c2ws = torch.from_numpy(c2ws).float()
        
        c2w_bucket = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=c2ws.shape[0]).clone()
        c2w_bucket[:, :3] = c2ws[:, :3]  # (num_frames, 4, 4)

        return images, depths, masks, c2w_bucket, True

    def __getitem__(self, idx):
        try:
            # Load the perspective-view metadata for the current scene
            data_path = self.data_path[idx] 
            data_path_class = data_path.split("/")[-1]

            viewpoints_path = None
            view_name_list = []
            room_id_list = []
            if data_path_class == "map.json" or data_path_class == "map_eval_12.json" or data_path_class == "map_eval.json":
                map_json = json.load(open(data_path, 'r'))
                room_id = 0
                for map_key in map_json.keys():
                    view_name_list.append(map_key)
                    room_id_list.append(room_id)
                    for map_value in map_json[map_key]:
                        view_name_list.append(map_value)
                        room_id_list.append(room_id)
                    room_id += 1
                viewpoints_path = os.path.dirname(data_path)
            else:
                print(f"error loading data_path_class: {data_path_class}")
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            # Load all panorama frames
            frames_pano = []
            for view_name in view_name_list:
                frame_pano = {}
                frame_pano["c2w"] = np.loadtxt(os.path.join(viewpoints_path, "viewpoints", view_name, "extrinsics.txt"))
                frame_pano["image_path"] = os.path.join(viewpoints_path, "viewpoints", view_name, "panoImage_1600.jpg")
                frame_pano["mask_path"] = os.path.join(viewpoints_path, "viewpoints", view_name, "pano_mask.png")
                frame_pano["view_name"] = view_name
                frames_pano.append(frame_pano)
            
            num_input_frames = len(frames_pano)
            if num_input_frames > self.viewpoint_max_view:
                num_input_frames = self.viewpoint_max_view
     
            # get input frames_pano range
            input_frames_pano_idx = list(range(0, len(frames_pano))) # Panorama views
            random_indices = np.random.choice(len(input_frames_pano_idx), num_input_frames, replace=False)
            input_frame_idx = [input_frames_pano_idx[i] for i in random_indices]
            input_frame_room_id = [room_id_list[i] for i in random_indices]

            input_frames = [frames_pano[i] for i in input_frame_idx]
            input_frames_view_name = ""
            for input_idx in range(len(input_frames)):
                if input_idx == len(input_frames) - 1:
                    input_frames_view_name += input_frames[input_idx]["view_name"]
                else:
                    input_frames_view_name += (input_frames[input_idx]["view_name"] + "-")

            input_images, input_depths, input_masks, input_c2ws, succ_status = self.process_pano_frames(input_frames)
            if succ_status == False:
                print(f"error succ_status: {succ_status}, data_path: {data_path}")
                return self.__getitem__(random.randint(0, len(self) - 1))

            pose_variations = [(0, 0, 0), (0, -270, 0), (0, -180, 0), (0, -90, 0), (90, 0, 0), (-90, 0, 0)]
            # Load all perspective-view frames
            frames = []
            for input_pano_frame in input_frames:
                view_name = input_pano_frame["view_name"]
                for angles in pose_variations:
                    local_rot_mat = get_local_rotation_matrix(*angles)
                    base_c2w = np.loadtxt(os.path.join(viewpoints_path, "viewpoints", view_name, "extrinsics.txt"))
                    new_c2w = base_c2w @ local_rot_mat
                    frame_data = {}
                    frame_data["c2w"] = new_c2w
                    frame_data["fx"] = self.config.data.resize_h / 2
                    frame_data["fy"] = self.config.data.resize_h / 2
                    frame_data["cx"] = self.config.data.resize_h / 2
                    frame_data["cy"] = self.config.data.resize_h / 2
                    frame_data["h"] = self.config.data.resize_h
                    frame_data["w"] = self.config.data.resize_h
                    frames.append(frame_data)
            num_target_frames = len(frames)

            target_intr, target_c2ws = self.process_frames(frames)
            
            # Reject scenes with excessively large translations
            if (target_c2ws[:, :3, 3] > 1e3).any():
                print(f"encounter large translation in target poses: {target_c2ws[:, :3, 3].max()}")
                assert False
            if (input_c2ws[:, :3, 3] > 1e3).any():
                print(f"encounter large translation in input poses: {input_c2ws[:, :3, 3].max()}")
                assert False

            # Camera poses must not contain NaNs
            if any(torch.isnan(torch.det(target_c2ws[:, :3, :3]))):
                print(f"encounter nan in target poses: {target_c2ws[:, :3, :3]}")
                assert False
            if any(torch.isnan(torch.det(input_c2ws[:, :3, :3]))):
                print(f"encounter nan in input poses: {input_c2ws[:, :3, :3]}")
                assert False
            
            # Verify that each rotation matrix has determinant 1
            if not torch.allclose(torch.det(target_c2ws[:, :3, :3]), torch.det(target_c2ws[:, :3, :3]).new_tensor(1.0)):
                print(f"det of target poses not equal to 1")
                assert False
            if not torch.allclose(torch.det(input_c2ws[:, :3, :3]), torch.det(input_c2ws[:, :3, :3]).new_tensor(1.0)):
                print(f"det of input poses not equal to 1")
                assert False

            # normalize input camera poses
            position_avg = input_c2ws[:, :3, 3].mean(0) # (3,)
            forward_avg = input_c2ws[:, :3, 2].mean(0) # (3,)
            down_avg = input_c2ws[:, :3, 1].mean(0) # (3,)

            # --- Safeguard 1: check whether forward_avg is too small ---
            if torch.norm(forward_avg) < 1e-6:
                # If camera directions cancel out completely, fall back to the default Z axis
                forward_avg = torch.tensor([0.0, 0.0, 1.0], device=input_c2ws.device).float()
            else:
                forward_avg = F.normalize(forward_avg, dim=0)

            # --- Safeguard 2: check down_avg and apply Gram-Schmidt orthogonalization ---
            # First, try to compute an orthogonalized down vector
            down_avg_ortho = down_avg - down_avg.dot(forward_avg) * forward_avg
            
            if torch.norm(down_avg_ortho) < 1e-6:
                # This means either:
                # 1. the original down_avg is zero, or
                # 2. the original down_avg is parallel to forward_avg
                # In either case, create a fallback vector that is not parallel to forward_avg.
                
                # Try the Y axis first
                fallback_down = torch.tensor([0.0, 1.0, 0.0], device=input_c2ws.device).float()
                # If forward is nearly aligned with Y, switch to the X axis instead
                if torch.abs(torch.dot(forward_avg, fallback_down)) > 0.99:
                    fallback_down = torch.tensor([1.0, 0.0, 0.0], device=input_c2ws.device).float()
                
                # Orthogonalize again using the fallback direction
                down_avg_ortho = fallback_down - fallback_down.dot(forward_avg) * forward_avg
                down_avg = F.normalize(down_avg_ortho, dim=0)
            else:
                # Standard normalization path
                down_avg = F.normalize(down_avg_ortho, dim=0)

            # Compute the right vector; the safeguards above ensure this cross product is safe
            right_avg = torch.cross(down_avg, forward_avg, dim=0)

            # Build the normalization transform
            pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1) # (3, 4)
            pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0) # (4, 4)
            
            # Invert the transform; the matrix should be orthogonal here, so inversion is stable
            pos_avg_inv = torch.inverse(pos_avg)

            input_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), input_c2ws)
            target_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), target_c2ws)
            
            if torch.isnan(input_c2ws).any() or torch.isinf(input_c2ws).any():
                print("encounter nan or inf in input poses")
                assert False

            if torch.isnan(target_c2ws).any() or torch.isinf(target_c2ws).any():
                print("encounter nan or inf in target poses")
                assert False
            
            input_depths_mask = (input_depths > 0) * (input_masks > 0)
            input_room_ids = torch.tensor(input_frame_room_id).long()

            ret_dict = {
                "input_images": input_images,  # (num_input, 3, resize_pano_h, resize_pano_w)
                "input_depths": input_depths, # (num_input, 1, resize_pano_h, resize_pano_w)
                "input_depths_mask": input_depths_mask, # (num_input, 1, resize_h, resize_w)
                "input_masks": (input_masks > 0), # (num_input, 1, resize_pano_h, resize_pano_w)
                "input_c2ws": input_c2ws,  # (num_input, 4, 4)
                "target_fxfycxcy": target_intr,  # (num_target, 4)
                "target_c2ws": target_c2ws, # (num_target, 4, 4)
                "input_room_ids": input_room_ids,
                "input_target_scene_name": viewpoints_path.split("/")[-1],
                "input_view_names": input_frames_view_name,
            }

        except:
            traceback.print_exc()
            print(f"error loading data: {self.data_path[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        return ret_dict
