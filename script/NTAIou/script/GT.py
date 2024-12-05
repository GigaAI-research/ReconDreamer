import os
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
import numpy as np
import cv2
from PIL import Image
import mmengine
import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))
from tqdm import tqdm
import imageio 
import copy
MAX_MAP = 75

UNKNOWN = 0
FRONT = 1
FRONT_LEFT = 2
FRONT_RIGHT = 3
SIDE_LEFT = 4
SIDE_RIGHT = 5
H = 1280
W = 1920
scale = 1920/480

OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

calib_view_map = {
            'FRONT':0,
            'FRONT_LEFT':1,
            'FRONT_RIGHT':2,
            'SIDE_LEFT':3,
            'SIDE_RIGHT':4
        }
name_val_map = {
            'FRONT':FRONT,
            'FRONT_LEFT':FRONT_LEFT,
            'FRONT_RIGHT':FRONT_RIGHT,
            'SIDE_LEFT':SIDE_LEFT,
            'SIDE_RIGHT':SIDE_RIGHT
        }
cls_map = {
            1:0,
            2:1,
            4:2
            }
img_view_map = {
            'FRONT':0,
            'FRONT_LEFT':1,
            'FRONT_RIGHT':3,
            'SIDE_LEFT':2,
            'SIDE_RIGHT':4
        }

line_cls_map = {
            'road_edge':0,
            'crosswalk':1,
            'road_line':2
        }

box_skeleton=[[0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]]

def get_intrinsic(calib):
    intrinsic = calib.intrinsic
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic

def get_extrinsic(calib):
    extrinsic = np.array(calib.extrinsic.transform).reshape(4,4)
    extrinsic = extrinsic@OPENCV2DATASET
    return extrinsic

def _get_cam_calib(frame, cam_type):
    
    ego2world = np.array(frame.pose.transform).reshape(4,4)
    cam_calib = frame.context.camera_calibrations[calib_view_map[cam_type]]
    assert cam_calib.name==name_val_map[cam_type]
    cam_intrinsic = get_intrinsic(cam_calib)
    cam2ego = get_extrinsic(cam_calib)
    height = cam_calib.height
    width = cam_calib.width
    calib ={
        'intrinsic': cam_intrinsic,
        'ego2world': ego2world,
        'cam2ego': cam2ego,
        'height': height,
        'width': width,
    }
    
    # if self.ref_ego_poses is not None or self.inter_novel:
    #     c2w = ego2world@cam2ego
    #     self.labels['c2w'].append(c2w)
        
    return calib

def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy(),world_points

def _get_label(frame, cam_type):
    boxes_3d = []
    corners = []
    types = []
    world_corners = []
    calib = frame.context.camera_calibrations[calib_view_map[cam_type]]
    
    for label in frame.laser_labels:
        '''
        enum Type {
          TYPE_UNKNOWN = 0;
          TYPE_VEHICLE = 1;
          TYPE_PEDESTRIAN = 2;
          TYPE_SIGN = 3;
          TYPE_CYCLIST = 4;
        }
        '''
        _type = label.type
        if _type not in cls_map:
            continue
        
        box = label.box
        if not box.ByteSize():
            continue  # Filter out labels that do not have a camera_synced_box.
        if  label.num_top_lidar_points_in_box< 10:
            continue  # Filter out likely occluded objects.
        
        box_coords = np.array(
                [
                    [
                        box.center_x,
                        box.center_y,
                        box.center_z,
                        box.length,
                        box.width,
                        box.height,
                        box.heading,
                    ]
                ]
            )
        # get ego box
        corner_3d = box_utils.get_upright_3d_box_corners(box_coords)[0].numpy()  # [8, 3]
        
        # Project box corners from vehicle coordinates onto the image.
        corner,world_corner = project_vehicle_to_image(
            frame.pose, calib, corner_3d
        )

        ok = corner[:,2]
        if not ok.any():
            continue
        
        
        
        # # waymo box format to nuScenes format
        corner = corner[[1,2,6,5,0,3,7,4],:2]
        world_corners.append(world_corner)
        corners.append(corner)
        boxes_3d.append(box_coords[0])
        types.append(_type)
        
    
    if len(boxes_3d) > 0:
        boxes_3d = np.stack(boxes_3d, axis=0).astype(np.float32)
        corners = np.stack(corners, axis=0).astype(np.float32)
        world_corners = np.stack(world_corners, axis=0).astype(np.float32)
        
    else:
        boxes_3d = np.zeros((0,7), dtype=np.float32)
        corners = np.zeros((0,8,2), dtype=np.float32)
        world_corners = np.zeros((0,8,3), dtype=np.float32)
        

    return boxes_3d,corners,world_corners,types

def get_box_canvas(corners,types,calib):
    '''
    heading (0,3,5,3)                      heading (4,5,6,7)
    waymo box format                       nuScenes box format
                  
        4 --------- 7                      7 --------- 6
        /|         / |                     /|         / |
      / |        /  |                    / |        /  |
      /  |       /   |          --->     /  |       /   |
    5 -------- 6    3                  3 -------- 2    5
    |   0      |   /                   |   4      |   /
    |          |  /                    |          |  /
    |          | /                     |          | /
    1 -------- 2                       0 -------- 1
              
    '''

    def pt_out_img(pt):
        return (pt[0]==-1 and pt[1]==-1)
    
    corners = [corner for index, corner in enumerate(corners) if types[index] == 1]
    types = [1] * len(corners)


    box_canvas=np.zeros((3,calib['height'],calib['width']),dtype=np.uint8)
    if len(types)==0:
        box_canvas = Image.fromarray(np.transpose(box_canvas,(1,2,0)))
        return box_canvas
    
    thickness = 12
    corners = np.array(corners, dtype=np.int32)
    color_line = 255
    
    for i, corner in enumerate(corners):
        w_1 = abs(corner[0,0]-corner[2,0])
        h_1 = abs(corner[0,1]-corner[2,1])
        w_2 = abs(corner[4,0]-corner[6,0])
        h_2 = abs(corner[4,1]-corner[6,1])
        if w_1*h_1<w_2*h_2:
            for i_st,i_end in box_skeleton:
                if (pt_out_img(corner[i_st]) or pt_out_img(corner[i_end])) :
                    continue
                cv2.line(
                    box_canvas[cls_map[types[i]]],
                    (corner[i_st, 0], corner[i_st, 1]),
                    (corner[i_end, 0], corner[i_end, 1]),
                    color =color_line,
                    thickness=thickness,
                )
            if not (pt_out_img(corner[4]) or pt_out_img(corner[6])):
                cv2.line(
                    box_canvas[cls_map[types[i]]],
                    (corner[4, 0], corner[4, 1]),
                    (corner[6, 0], corner[6, 1]),
                    color =color_line,
                    thickness=thickness,
                )
            if not (pt_out_img(corner[5]) or pt_out_img(corner[7])):
                cv2.line(
                    box_canvas[cls_map[types[i]]],
                    (corner[5, 0], corner[5, 1]),
                    (corner[7, 0], corner[7, 1]),
                    color =color_line,
                    thickness=thickness,
                ) 
            
        else:
            if not (pt_out_img(corner[4]) or pt_out_img(corner[6])):
                cv2.line(
                    box_canvas[cls_map[types[i]]],
                    (corner[4, 0], corner[4, 1]),
                    (corner[6, 0], corner[6, 1]),
                    color =color_line,
                    thickness=thickness,
                )
            if not (pt_out_img(corner[5]) or pt_out_img(corner[7])):
                cv2.line(
                    box_canvas[cls_map[types[i]]],
                    (corner[5, 0], corner[5, 1]),
                    (corner[7, 0], corner[7, 1]),
                    color =color_line,
                    thickness=thickness,
                ) 
            
            
            for i_st,i_end in box_skeleton:
                if pt_out_img(corner[i_st]) or pt_out_img(corner[i_end]):
                      continue
                cv2.line(
                    box_canvas[cls_map[types[i]]],
                    (corner[i_st, 0], corner[i_st, 1]),
                    (corner[i_end, 0], corner[i_end, 1]),
                    color =color_line,
                    thickness=thickness,
                )

    box_canvas = Image.fromarray(np.transpose(box_canvas,(1,2,0)))

    x_final = []
    for i, corner in enumerate(corners):
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        for point in corner:
            if not pt_out_img(point):
                x_min = min(x_min, point[0]) 
                y_min = min(y_min, point[1]) 
                x_max = max(x_max, point[0])
                y_max = max(y_max, point[1])
        x_final.append([x_min,y_min,x_max,y_max])

    return box_canvas,x_final

def _get_hdmap( frame,bev_hdmap):
    

    # if len(frame.map_features)>0:
    #     bev_hdmap = frame.map_feature
    #     assert 
    offset  = frame.map_pose_offset
    offset = np.array([offset.x,offset.y,offset.z,0])
    vectors = []
    
    for line in bev_hdmap:
        # get lines
        if line.HasField('road_edge'):
            vector = list(line.road_edge.polyline)
            _type = 'road_edge'
        elif line.HasField('road_line'):
            vector = list(line.road_line.polyline)
            _type = 'road_line'
        elif line.HasField('crosswalk'):
            vector = list(line.crosswalk.polygon)
            _type = 'crosswalk'
        else:
            continue
        
        # get points 
        pts = []
        for _pt in vector:
            pt = np.array([_pt.x,_pt.y,_pt.z,1])
            pt -= offset
            pts.append(pt)
        pts = np.stack(pts)
        
        # crosswalk only save the long side
        if _type == 'crosswalk':
            if pts.shape[0]== 4:
                dist = np.square(pts[1:]-pts[:-1]).sum(1)
                idx = np.argsort(dist)[-1]
                # pts = pts[idx:idx+2]
                if idx ==2:
                    idx = 0
                vectors.append((pts[idx:idx+2],_type))
                vectors.append((pts[[idx+2,(idx+3)%4]],_type))
            # assert idx ==1
        else:
            vectors.append((pts,_type))
    hdmap = vectors
    
    return hdmap

def view_points_depth(points, view, normalize):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view
    nbr_points = points.shape[1]
    points = np.dot(viewpad, points)
    points = points[:3, :]
    depth = points[2, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    return points, depth

def draw_hdmap(vectors,w2c,calib):
    map_canvas = np.zeros((3,calib['height'],calib['width']), dtype=np.uint8)
    for pts, _type in vectors:
        pts = w2c@pts.T

        # filter points out of map
        keep_index = []
        dist = np.sqrt(np.square(pts[:-1]).sum(0))
        for i in range(pts.shape[1]):
            if dist[i]<MAX_MAP and abs(pts[1,i])<10:
                keep_index.append(i)
        if len(keep_index)<2:
            continue
        
        # split pts if index not continuous
        index = np.array(keep_index)
        flag = (index[1:]-index[:-1])>1
        split_idx = [0]
        if flag.any():
            split_idx += list(np.argwhere(flag).reshape(-1)+1)
            pass
        split_idx+=[len(keep_index)]

        for idx_start,idx_end in zip(split_idx[:-1],split_idx[1:]):
            idx = keep_index[idx_start:idx_end]
            if len(idx)<2:
                continue
            _pts = pts[:,idx]
            
            _pts,_depth = view_points_depth(_pts,calib['intrinsic'],normalize=True)
            _pts = _pts[:, _depth > 1e-3]
            _pts = _pts[:2, :]
            _pts=_pts.T
            cv2.polylines(map_canvas[line_cls_map[_type]], [_pts.astype(np.int32)], False, color=255, thickness=12)

    filter_mask = np.ones_like(map_canvas)
    filter_mask[1] = map_canvas[0]==0
    map_canvas[1] = map_canvas[1]*filter_mask[1]
    filter_mask[2] = map_canvas[1]==0
    map_canvas[2] = map_canvas[2]*filter_mask[1]   

    map_canvas = Image.fromarray(np.transpose(map_canvas,(1,2,0)))
    return map_canvas

def prepare_data():
    split = 'validation'

    data_root = '/mnt/data-2/users/zhaoguosheng/1-code/2-drivedreamer/datasets/waymo/source_data/'+split
    save_root = '/mnt/data-2/users/zhaoguosheng/1-code/2-drivedreamer/datasets/waymo/pkl_data/'+split
    os.makedirs(save_root,exist_ok=True)
    scenes = os.listdir(data_root)
    scenes.sort()
    scene_ids = [17,18,29,56,65,81,108,113,164]
    cam_type = 'FRONT'
    for scene_id in tqdm(scene_ids):
        save_dir = os.path.join(save_root,str(scene_id).zfill(3))
        os.makedirs(save_root,exist_ok=True)
        scene = scenes[scene_id]
        data_dict = {
            'scene_id':scene_id
        }
        offsets = []
        calibs = []
        corners = []
        names = []
        hdmaps = []
        world_corners = []
        scene_data =  tf.data.TFRecordDataset(os.path.join(data_root,scene))
        for i,data in enumerate(scene_data):
            frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
            if i == 0:
                bev_hdmap = frame.map_features
                imgs = frame.images
                image = tf.image.decode_png(imgs[img_view_map[cam_type]].image).numpy()
                imageio.imwrite(os.path.join('/mnt/data-2/users/zhaoguosheng/1-code/1-drivestudio/cache',split+'_'+str(scene_id).zfill(3)+'.png'),image)
            
            hdmap = _get_hdmap(frame,bev_hdmap)
            
            calib = _get_cam_calib(frame,cam_type)
            boxes_3d,corner,world_corner,name = _get_label(frame,cam_type)
            offset  = frame.map_pose_offset
            offset = np.array([offset.x,offset.y,offset.z,0])
            
            offsets.append(offset)
            calibs.append(calib)
            corners.append(corner)
            names.append(name)
            hdmaps.append(hdmap)
            world_corners.append(world_corner)
        data_dict.update(
            {
                'offsets':offsets,
                'calibs':calibs,
                'corners':corners,
                'names':names,
                'hdmap':hdmaps,
                'world_corner':world_corners
            }
        )    
        mmengine.dump(data_dict,os.path.join(save_dir,'label.pkl'))


def draw_2d_box_on_image(xyxy, image_path='/mnt/pfs/users/chaojun.ni/3-data/waymo_stgs/validation/164/images/000000_0.png', target_size=(480, 320)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    if image is None:
        raise ValueError(f"load error: {image_path}")
    
    color = (0, 0, 255)  
    scale_x = 1
    scale_y = scale_x
    for data in xyxy:
        x_min, y_min, x_max, y_max = data
        x_min = int(x_min * scale_x)
        y_min = int(y_min * scale_y)
        x_max = int(x_max * scale_x)
        y_max = int(y_max * scale_y)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    return resized_image



data_root = 'data/'
for scene in  ['005']:        
    data_dict = mmengine.load(os.path.join(data_root,scene,'label.pkl'))
    world_corners = data_dict['world_corner']
    calibs = data_dict['calibs']
    hdmaps = data_dict['hdmap']
    labels = data_dict['names']
    num_frame = len(hdmaps)
    for shifting in [-3,-2,-1,0,1,2,3,4,5]:
        base_dir = os.path.join(data_root, scene, 'shift_gt', f'{shifting}')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        for i in range(num_frame):
            calib = calibs[i]
            world_corner = world_corners[i]
            vectors = hdmaps[i]
            label = labels[i]
            cam2ego = copy.deepcopy(calib['cam2ego'])
            ego2world = copy.deepcopy(calib['ego2world'])
            intrinsic = calib['intrinsic']

            cam2ego[1,3] += shifting

            c2w = ego2world@cam2ego
            w2c = np.linalg.inv(c2w)
            w2c = w2c.astype(np.float128)
            corners=[]
            for corner in world_corner:
                corner = np.concatenate([corner,np.ones((corner.shape[0],1))],axis=-1)
                corner = w2c@corner.T
                corner,depth=view_points_depth(corner,intrinsic,normalize=True)
                corner = corner / scale
                corners.append(corner.T[:,:2])
            
            filename = f'{i:06d}_0.txt'
            filename = os.path.join(base_dir,filename)
            try:
                corners = np.stack(corners)
                box_map,xyxy = get_box_canvas(corners,label,calib)
                hdmap = draw_hdmap(vectors,w2c,calib)
                new_3d_image = draw_2d_box_on_image(xyxy,target_size=(480, 320))
                max_x = 480
                max_y = 320
                with open(filename, 'w') as file:
                    for sublist in xyxy:
                        x_min, y_min, x_max, y_max = map(int, sublist)
                        
                        if x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0:
                            continue  
                        
                        if x_max > max_x or y_max > max_y:
                            continue  
                    
                        file.write(','.join(map(str, sublist)) + '\n')
                print(f"数据已成功写入 {filename}，每行一个子列表")
            except:
                print('')
            




    

