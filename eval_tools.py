import numpy as np
import cv2
import open3d as o3d
import time
from scipy.spatial.transform import Rotation as R
from lidar_processing import parse_pcd, POINT_COLOR_MAP

def rotation_matrix_to_rpy(R):
    """
    将旋转矩阵转换为 Roll, Pitch, Yaw 角
    :param R: 3x3 旋转矩阵
    :return: (roll, pitch, yaw)
    """
    assert R.shape == (3, 3), "输入矩阵必须是 3x3 矩阵"

    # 计算 Pitch (theta)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw

def deg2rad(degrees):
    return degrees * np.pi / 180

def get_delta_t(R_mat, t0, t1, t2):

    # 创建变换矩阵
    deltaT = np.eye(4)
    deltaT[:3, :3] = R_mat
    deltaT[0, 3] = t0
    deltaT[1, 3] = t1
    deltaT[2, 3] = t2
    
    return deltaT



def mask_registration_loss(R_mat, t0, t1, t2, distance_img, register_cloud, K):
    PointCnt = 0
    deltaT = get_delta_t(R_mat, t0, t1, t2)
    #print(self.curr_optim_extrinsic_)
    T = deltaT
    print(T)

    valid_points = []
    
    for src_pt_index in range(register_cloud.shape[0]):
        src_pt = register_cloud[src_pt_index]
        
        if not np.isfinite(src_pt[0]) or not np.isfinite(src_pt[1]) or not np.isfinite(src_pt[2]):
            continue
        vec = np.array([src_pt[0], src_pt[1], src_pt[2], 1.0])
        cam_point = T @ vec
        cam_vec = cam_point[:3]
        vec_2d = K @ cam_vec
        
        if vec_2d[2] > 0:
            x = int(np.round(vec_2d[0] / vec_2d[2]))
            y = int(np.round(vec_2d[1] / vec_2d[2]))

            if 0 <= x < distance_img.shape[1] and 0 <= y < distance_img.shape[0]:
                pixel = distance_img[y, x]
                pixel_gray = np.mean(pixel)

                if pixel_gray > 55:
                    PointCnt += 1

    return float(PointCnt) / register_cloud.shape[0]

def cost_function(R_mat, t0, t1, t2, distance_img, register_cloud, K):
    registration_cost = mask_registration_loss(R_mat, t0, t1, t2, distance_img, register_cloud, K)
    return registration_cost

def eval_function(R_mat, t0, t1, t2, distance_img, register_cloud, K):
        eval_result = cost_function(R_mat, t0, t1, t2, distance_img, register_cloud, K)

        print("final loss: ", eval_result)