import json
import numpy as np
import cv2

def load_intrinsic(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    # 获取第一个键名
    id = list(data.keys())[0]
    #print(id)

    # 读取相机内参矩阵
    intrinsic_data = data[id]['param']['cam_K']['data']
    intrinsic = np.array(intrinsic_data).reshape(3, 3)
    
    # 读取相机畸变参数
    distortion_data = data[id]['param']['cam_dist']['data']
    distortion_size = data[id]['param']['cam_dist']['cols']
    distortion = np.array(distortion_data[0][:distortion_size], dtype=np.float32)
    
    return intrinsic, distortion

def load_extrinsic(filename):
    try:
        with open(filename, 'r') as f:
            root = json.load(f)
    except IOError:
        print(f"Error opening {filename}")
        exit(1)
    
    id = next(iter(root.keys()))
    print(id)
    data = root[id]["param"]["sensor_calib"]["data"]

    extrinsic = np.array([
        [data[0][0], data[0][1], data[0][2], data[0][3]],
        [data[1][0], data[1][1], data[1][2], data[1][3]],
        [data[2][0], data[2][1], data[2][2], data[2][3]],
        [data[3][0], data[3][1], data[3][2], data[3][3]]
    ], dtype=float)

    return extrinsic