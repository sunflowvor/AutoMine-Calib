import open3d
import numpy as np
import os
import sys

from utils import deg2rad, get_delta_t, binarize_image, generate_distance_weight_map, extract_3d_center, extract_2d_center
from utils import show_and_save_proj, generate_weight_map
from load_tools import load_intrinsic, load_extrinsic
from img_processing import undistortion, undistortion_color, extract_pc_cate
from lidar_processing import extract_feature, post_lidarseg
from calculate_stereo import calculate_extrinsic_matrix
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import distance_transform_edt
from eval_tools import eval_function, rotation_matrix_to_rpy
from optimizer import Optimizer
from kalman import KalmanFilterForExtrinsics
from calculate_error import calculate_error

import numpy as np
import cv2
import open3d as o3d
import time
from scipy.spatial.transform import Rotation as R
from lidar_processing import parse_pcd, POINT_COLOR_MAP

img_mask_set = {"void": 0, "sky": 1, "road": 2, "massif": 3, "rut": 4,
                "road_edge": 5, "brushwood": 6, "mining_truck": 7, "soft_soil": 8, "gravel": 9,
                "puddle": 10, "electric_shovel": 11, "road_testing_equipment": 12, "engineering_vehicle": 13,
                "pushdozer": 14, "truck": 15, "car": 16, "tussock": 17, "vegetation": 18,
                "blocking_stone": 19, "excavator": 20, "construction": 21, "ice_cream_cone": 22, "fence": 23,
                "road_sign": 24, "cabel": 25, "conmunication_pole": 26, "telegraph_pole": 27, "watering_car": 28,
                "oil_truck": 29, "human": 30, "wide_body_truck": 31, "bus":32}
front_segments = [2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32]
init_segments = [7, 11, 12, 13, 14, 15, 16, 20, 21, 24, 28, 29, 30, 31, 32]

img_id_mask_set = {v: k for k, v in img_mask_set.items()}

POINT_LABLES = {0: "void", 1: "mining_truck", 2: "wide_body_truck", 3: "truck", 4: "bus", \
                6: "engineering_vehicle", 7: "excavator", 8: "pushdozer", \
                9: "watering_car", 10: "oil_truck", 11: "electric_shovel", \
                12: "human", 13: "road", 14: "rut", 15: "gravel", 16: "soft_soil", \
                17: "road_edge", 18: "puddle", 19: "taft", 20: "construction", \
                21: "vegetation", 22: "tussock", 23: "sky", 24: "massif", 25: "fence", \
                26: "road_sign", 27: "blocking_stone", 28: "car", 29: "ice_cream_cone", \
                30: "telegraph_pole", 31: "communication_pole", 32: "road_testing_equipment"}

def compress_to_range(data, min_target=0.9, max_target=1.0):
    """
    将一个列表中的数值线性压缩到[min_target, max_target]区间。
    
    :param data: 输入的数值列表
    :param min_target: 目标区间的最小值（默认为0.9）
    :param max_target: 目标区间的最大值（默认为1.0）
    :return: 压缩后的数值列表
    """
    data = np.array(data)  # 转为numpy数组便于操作
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val == min_val:
        # 如果最大值和最小值相等，直接设置为目标区间的中值
        return [min_target + (max_target - min_target) / 2] * len(data)
    
    # 线性压缩公式
    scaled_data = min_target + (data - min_val) * (max_target - min_target) / (max_val - min_val)
    return scaled_data.tolist()  # 转为列表返回


def mask_weight(image_mask_np, common_category):
    return generate_weight_map(image_mask_np, init_segments, common_category)

class Optimizer_all:
    def __init__(self, curr_optim_intrinsic, curr_optim_right_intrinsic, curr_optim_extrinsic, stereo_R, stereo_t, decent_step):
        self.curr_optim_extrinsic_ = curr_optim_extrinsic
        self.curr_optim_intrinsic = curr_optim_intrinsic
        self.curr_optim_right_intrinsic = curr_optim_right_intrinsic
        self.stereo_R_ = stereo_R
        self.stereo_t_ = stereo_t
        self.decent_step = decent_step


    def mask_registration_loss(self, var, left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask):


        PointCnt = 0
        deltaT = get_delta_t(var)
        T_lidar_camera = self.curr_optim_extrinsic_ @ deltaT
        T_stereo = np.eye(4)
        T_stereo[:3, :3] = self.stereo_R_
        T_stereo[:3, 3:] = self.stereo_t_

        T_stereo_inv = np.linalg.inv(T_stereo)

        T_lidar_right = np.matmul(T_stereo, T_lidar_camera)

        valid_points = []

        pcd_mask_to_img_cate = np.zeros_like(left_image_mask_np)

        pcd_xyz_entend = np.hstack((pcd_xyz, np.ones((pcd_xyz.shape[0], 1))))
        cam_point_left = T_lidar_camera @ pcd_xyz_entend.T
        cam_vec_left = cam_point_left[:3]
        vec_2d_left = self.curr_optim_intrinsic @ cam_vec_left
        proj_points_left = vec_2d_left.T

        # 转换为像素坐标 (x, y, z)
        u_left = proj_points_left[:, 0] / proj_points_left[:, 2]  # x 坐标 (N,)
        v_left = proj_points_left[:, 1] / proj_points_left[:, 2]  # y 坐标 (N,)
        z_left = proj_points_left[:, 2]                   # 深度值 (N,)

        # 筛选有效投影点
        valid_mask_left = (z_left > 0) & \
                    (u_left >= 0) & (u_left < left_image_mask_np.shape[1]) & \
                    (v_left >= 0) & (v_left < left_image_mask_np.shape[0])

        # 仅保留有效点
        u_valid_left = np.floor(u_left[valid_mask_left]).astype(int)
        v_valid_left = np.floor(v_left[valid_mask_left]).astype(int)
        indices_valid_left = np.where(valid_mask_left)[0]  # 对应的点云索引

        # 从掩码图像中获取像素值
        pixels_left = left_image_mask_np[v_valid_left, u_valid_left]

        # 从权重中获取像素值
        weights_left = left_mask[v_valid_left, u_valid_left]

        # 映射点云类别
        point_categories_left = pcd_mask_map_to_camera[indices_valid_left]

        result_left = np.where(pixels_left == point_categories_left, 1, 0)
        result_left = np.multiply(result_left, weights_left)

        #----------------right-----------------
        cam_point_right = T_lidar_right @ pcd_xyz_entend.T
        cam_vec_right = cam_point_right[:3]
        vec_2d_right = self.curr_optim_right_intrinsic @ cam_vec_right
        proj_points_right = vec_2d_right.T

        # 转换为像素坐标 (x, y, z)
        u_right = proj_points_right[:, 0] / proj_points_right[:, 2]  # x 坐标 (N,)
        v_right = proj_points_right[:, 1] / proj_points_right[:, 2]  # y 坐标 (N,)
        z_right = proj_points_right[:, 2]                   # 深度值 (N,)

        # 筛选有效投影点
        valid_mask_right = (z_right > 0) & \
                    (u_right >= 0) & (u_right < right_image_mask_np.shape[1]) & \
                    (v_right >= 0) & (v_right < right_image_mask_np.shape[0])

        # 仅保留有效点
        u_valid_right = np.floor(u_right[valid_mask_right]).astype(int)
        v_valid_right = np.floor(v_right[valid_mask_right]).astype(int)
        indices_valid_right = np.where(valid_mask_right)[0]  # 对应的点云索引

        # 从掩码图像中获取像素值
        pixels_right = right_image_mask_np[v_valid_right, u_valid_right]

        weights_right = right_mask[v_valid_right, u_valid_right]

        # 映射点云类别
        point_categories_right = pcd_mask_map_to_camera[indices_valid_right]

        result_right = np.where(pixels_right == point_categories_right, 1, 0)
        result_right = np.multiply(result_right, weights_right)

        score_left = np.sum(result_left)
        score_right = np.sum(result_right)
        return (score_left + score_right) / (pcd_xyz.shape[0] * 2)

    def cost_function(self, var, left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask):
        registration_cost = self.mask_registration_loss(var, left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)
        return registration_cost

    def random_search_params(self, search_count, delta_roll, delta_pitch, delta_yaw, delta_x, delta_y, delta_z, \
                left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask):
        better_cnt = 0.0
        var = np.zeros(6)
        bestVal = np.zeros(6)

        cpu_time = time.process_time()

        # 获取当前时间
        current_time = time.time()

        # 计算种子
        seed = int((current_time - cpu_time) * 1e6)

        # 初始化随机数生成器
        generator = np.random.default_rng(seed)

        distribution_x = generator.uniform(-delta_x, delta_x, search_count)
        distribution_y = generator.uniform(-delta_y, delta_y, search_count)
        distribution_z = generator.uniform(-delta_z, delta_z, search_count)
        distribution_roll = generator.uniform(-delta_roll, delta_roll, search_count)
        distribution_pitch = generator.uniform(-delta_pitch, delta_pitch, search_count)
        distribution_yaw = generator.uniform(-delta_yaw, delta_yaw, search_count)

        maxPointCnt = self.cost_function(var, left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)

        for i in range(search_count):
            var[0] = distribution_roll[i]
            var[1] = distribution_pitch[i]
            var[2] = distribution_yaw[i]
            var[3] = distribution_x[i]
            var[4] = distribution_y[i]
            var[5] = distribution_z[i]

            cnt = self.cost_function(var, left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)

            #print(var)
            if cnt > maxPointCnt:
                better_cnt += 1
                maxPointCnt = cnt
                bestVal = var.copy()

        self.curr_optim_extrinsic_ = self.curr_optim_extrinsic_ @ get_delta_t(bestVal)
        self.current_cost_fun_ = maxPointCnt
        self.current_fc_score_ = 1 - better_cnt / search_count

    def calibrate(self, left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask):

        var = np.zeros(6)
        bestVal = np.zeros(6)
        best_extrinsic_vec = np.zeros(6)
        varName = ["roll", "pitch", "yaw", "tx", "ty", "tz"]
        direction = [1, 0, -1]

        stereo_T = np.eye(4)
        stereo_T[:3, :3] = self.stereo_R_
        stereo_T[:3, 3:] = self.stereo_t_
        stere_T_inv = np.linalg.inv(stereo_T)
        self.stere_R_inv = stere_T_inv[:3, :3]
        self.stere_t_inv = stere_T_inv[:3, 3:]

        maxPointCnt = self.cost_function(var, left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)

        #print("before score: ", maxPointCnt)

        iteration_num = 0
        rotation_matrix = self.curr_optim_extrinsic_[:3, :3]
        ea = cv2.Rodrigues(rotation_matrix)[0].ravel()
        target_extrinsic_vec = np.zeros(6)
        curr_optim_extrinsic_vec = np.zeros(6)

        is_violence_search = False

        if is_violence_search:
            rpy_resolution = 1
            xyz_resolution = 0.1
            for i1 in range(-5, 5):
                for i2 in range(-5, 5):
                    for i3 in range(-5, 5):
                        for i4 in range(-5, 5):
                            for i5 in range(-5, 5):
                                for i6 in range(-5, 5):
                                    var[0] = i1 * rpy_resolution
                                    var[1] = i2 * rpy_resolution
                                    var[2] = i3 * rpy_resolution
                                    var[3] = i4 * xyz_resolution
                                    var[4] = i5 * xyz_resolution
                                    var[5] = i6 * xyz_resolution
                                    #print(var)
                                    cnt = self.cost_function(var, left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)
                                    #print(cnt)
                                    if cnt > maxPointCnt:
                                        maxPointCnt = cnt
                                        bestVal = var.copy()
                                        #print("match point increase to: ", maxPointCnt)
            self.curr_optim_extrinsic_ = self.curr_optim_extrinsic_ @ get_delta_t(bestVal)
            self.current_cost_fun_ = maxPointCnt
            return self.curr_optim_extrinsic_, self.current_cost_fun_

        else:
            f1, f2 = False, False
            rotation_step = 0.1 * self.decent_step 
            translation_step = 0.1 * self.decent_step
            for k in range(10):
                if k % 2 == 0:
                    self.random_search_params(10, rotation_step, rotation_step, rotation_step,\
                        translation_step, translation_step, translation_step, \
                        left_image_mask_np, right_image_mask_np,\
                    pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)
                    f1 = self.current_fc_score_ == 1
                else:
                    self.random_search_params(10, rotation_step, rotation_step, rotation_step,\
                    translation_step, translation_step, translation_step,\
                    left_image_mask_np, right_image_mask_np,\
                    pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)
                    f2 = self.current_fc_score_ == 1
                if f1 and f2:
                    break
            #print("checking level 1")
            for k in range(10):
                self.random_search_params(10, rotation_step, rotation_step, rotation_step,\
                    translation_step, translation_step, translation_step,\
                    left_image_mask_np, right_image_mask_np,\
                    pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)
                if self.current_fc_score_ == 1:
                    break
            #print("checking level 2")
            for k in range(10):
                self.random_search_params(10, rotation_step, rotation_step, rotation_step, \
                    translation_step, translation_step, translation_step,\
                    left_image_mask_np, right_image_mask_np,\
                    pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)
                if self.current_fc_score_ == 1:
                    break
            #print("checking level 3")
            for k in range(10):
                self.random_search_params(10, rotation_step, rotation_step, rotation_step, \
                    translation_step, translation_step, translation_step,\
                    left_image_mask_np, right_image_mask_np,\
                    pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_mask, right_mask)
                if self.current_fc_score_ == 1:
                    break
            #img_points, lidar_points = self.save_project_result(left_image_mask_np, image_ori, pcd_xyz, idx_name)
            return self.curr_optim_extrinsic_, self.current_cost_fun_

    def rotation2eul(self, rotation_matrix):
        r = R.from_matrix(rotation_matrix)
        return r.as_euler('zyx', degrees=False)

    def save_project_result(self, filename, distance_img, register_cloud, idx_name):
        # 保存结果的方法
        rotation_matrix = self.curr_optim_extrinsic_[:3, :3]
        translation = self.curr_optim_extrinsic_[:3, 3]
        eulerAngle = self.rotation2eul(rotation_matrix)

        img_points = [] 
        lidar_points = [] 

        image = distance_img.copy()
        for src_pt_index in range(register_cloud.shape[0]):
            src_pt = register_cloud[src_pt_index]
            if not np.isfinite(src_pt[0]) or not np.isfinite(src_pt[1]) or not np.isfinite(src_pt[2]):
                continue
            vec = np.array([src_pt[0], src_pt[1], src_pt[2], 1.0])
            cam_point = self.curr_optim_extrinsic_ @ vec
            cam_vec = cam_point[:3]
            vec_2d = self.curr_optim_intrinsic @ cam_vec

            if vec_2d[2] > 0:
                x = int(np.round(vec_2d[0] / vec_2d[2]))
                y = int(np.round(vec_2d[1] / vec_2d[2]))

                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    centerCircle2 = (x, y)
                    color = (0, 255, 0)
                    cv2.circle(image, centerCircle2, 3, color, -1, 0)
                    img_points.append([x, y])
                    lidar_points.append([src_pt[0], src_pt[1], src_pt[2]])

        cv2.imwrite("./" + filename.split(".")[0] + "_" + idx_name + "_test.png", image)
        return np.array(img_points), np.array(lidar_points)


def lidar_camera_registration(lidar_points, img_points, K):
    _, rvec, tvec, inliers = cv2.solvePnPRansac(lidar_points, img_points, K, None)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

def project(points, camera_params, K):
    #print(camera_params)
    R_vec = camera_params[0, :3]
    t = camera_params[0, 3:6]
    R, _ = cv2.Rodrigues(R_vec)
    P = K @ np.hstack((R, t.reshape(3, 1)))
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_proj = P @ points_hom.T
    points_proj = points_proj[:2] / points_proj[2]
    return points_proj.T

def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))

    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    return (points_proj - points_2d).ravel()

def bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d, K):
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    res = least_squares(reprojection_error, x0, args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))
    return res.x

def draw_matches(img1, img2, keypoints1, keypoints2, matches):
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_img

# 数值梯度计算函数
def numerical_gradient(f, x, epsilon=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = np.copy(x)
        x2 = np.copy(x)
        x1[i] += epsilon
        x2[i] -= epsilon
        grad[i] = (f(x1) - f(x2)) / (2 * epsilon)
    return grad

# 定义损失函数
# 定义phi函数
def loss_function(r_vec, t, points, mask, K):

    R_mat = R.from_rotvec(r_vec).as_matrix()

    width = mask.shape[1]
    height = mask.shape[0]
    
    Image_plane = np.zeros((height, width))
    Image_edges = np.matmul(K, np.matmul(R_mat, points.transpose(1, 0)) + t.reshape(-1, 1)).transpose(1, 0)

    Image_edges[:, 0] /= Image_edges[:, 2]
    Image_edges[:, 1] /= Image_edges[:, 2]

    # 舍弃掉小数部分，并将坐标限制在图像尺寸内
    x_coords = np.clip(Image_edges[:, 0].astype(np.int32), 0, width - 1)
    y_coords = np.clip(Image_edges[:, 1].astype(np.int32), 0, height - 1)

    # 使用 NumPy 的高级索引将点云映射到图像平面
    Image_plane[y_coords, x_coords] = 1

    result_phi = np.sum(mask * Image_plane)

    return result_phi

def extrinsic_calibration_optimization_solver(r0, t0, H, K, P_G, omega, tau, delta, phi_0):
    # 初始化
    E = {}
    T_index = {}
    rotation_0 = R.from_matrix(r0)
    rot_vec_0 = rotation_0.as_rotvec()
    E[0] = phi_0
    T_index[0] = [rot_vec_0, t0]
    delay = 0.5

    r_vec = rot_vec_0
    t_vec = t0

    for n in range(tau + 1):

        if len(E.keys()) > omega:
            min_key = min(E, key=E.get)
            # 删除具有最小值的键值对
            del E[min_key]
            del T_index[min_key]

        #    delete the min E
        eta = 1.0

        #grad_r = numerical_gradient(lambda r: loss_function(r, t_vec, P_G, H, K), r_vec)
        grad_t = numerical_gradient(lambda t: loss_function(r_vec, t, P_G, H, K), t_vec)

        iter_count = 0
        back_tracking = True

        while back_tracking and iter_count <= delta:
            #r_vec_new = r_vec - eta * grad_r
            r_vec_new = r_vec
            t_vec_new = t_vec - eta * grad_t
            new_loss = loss_function(r_vec_new, t_vec_new, P_G, H, K)
            #print(new_loss)
            if new_loss > min(E.values()):
                r_vec = r_vec_new
                t_vec = t_vec_new
                back_tracking = False
            else:
                eta = delay * eta

            iter_count += 1

        E[n + 1] = new_loss
        T_index[n + 1] = [r_vec_new, t_vec_new]
        if min(E.values()) == max(E.values()):
            break

    max_key = max(E, key=E.get)
    result_r_ve, result_t_vec = T_index[max_key][0], T_index[max_key][1]
    return result_r_ve, result_t_vec

    # 假设我们有一个函数来确定区域O和B
def get_regions_from_image(image):
        # 此处添加你的图像处理代码来获取区域O和B
        # 例如，假设我们使用阈值分割来确定区域
        O = set()
        B = set()
        height, width = image.shape[:2]
        
        for y in range(height):
            for x in range(width):
                pixel_value = image[y, x]
                if np.all(pixel_value == 1):  # 假设白色区域为O
                    O.add((x, y))
                else:  # 其他颜色区域为B
                    B.add((x, y))
        
        return O, B

# 定义H_DT函数
def H_DT(O, B, alpha_1):

    # 计算距离变换
    distance_to_zero = distance_transform_edt(O)
    # 归一化距离矩阵，使其值在0到1之间
    min_val = np.min(distance_to_zero)
    max_val = np.max(distance_to_zero)
    normalized_distance_matrix = (distance_to_zero - min_val) / (max_val - min_val)
    normalized_distance_matrix = 1 - normalized_distance_matrix
    normalized_distance_matrix = O * normalized_distance_matrix

    return normalized_distance_matrix

# 定义H_DT函数
def H_IDT(O, B, alpha_1):
    threshold = 100
    distance_to_zero = distance_transform_edt(B)
    distance_to_zero[distance_to_zero > threshold] = threshold

    min_val = np.min(distance_to_zero)
    max_val = np.max(distance_to_zero)
    normalized_distance_matrix = (distance_to_zero - min_val) / (max_val - min_val)
    normalized_distance_matrix = 1 - normalized_distance_matrix
    normalized_distance_matrix = B * normalized_distance_matrix
    
    return normalized_distance_matrix

# 定义H函数，假设它是一个掩膜图层
def H(point, mask):
    x, y, _ = point
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[int(y), int(x)] / 255.0  # 归一化掩膜值到[0, 1]
    return 0

# 定义K函数，使用相机内参矩阵K将3D点投影到2D平面
def project_to_image_plane(point, K):
    point_homogeneous = np.append(np.array(point), 1)  # 转换为齐次坐标
    projected_point = np.dot(K, point_homogeneous)
    projected_point /= projected_point[2]  # 归一化
    return projected_point[:2]  # 返回2D坐标

# 定义phi函数
def phi(R_mat, t, points, mask, K):

    width = mask.shape[1]
    height = mask.shape[0]
    
    Image_plane = np.zeros((height, width))
    Image_edges = np.matmul(K, np.matmul(R_mat, points.transpose(1, 0)) + t.reshape(-1, 1)).transpose(1, 0)

    Image_edges[:, 0] /= Image_edges[:, 2]
    Image_edges[:, 1] /= Image_edges[:, 2]

    # 舍弃掉小数部分，并将坐标限制在图像尺寸内
    x_coords = np.clip(Image_edges[:, 0].astype(np.int32), 0, width - 1)
    y_coords = np.clip(Image_edges[:, 1].astype(np.int32), 0, height - 1)

    # 使用 NumPy 的高级索引将点云映射到图像平面
    Image_plane[y_coords, x_coords] = 1

    result_phi = np.sum(mask * Image_plane)

    return result_phi

def project_points(points, rotation, translation, K):
    projected_points = K @ (rotation @ points.T + translation.reshape(-1, 1))
    projected_points /= projected_points[2, :]
    return projected_points[:2, :].T

# 定义重投影误差函数
def reprojection_error(params, points_3d, mask, K, R_fixed):
    rotation_vector = params[:3]
    translation = params[3:]
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    T_predict = np.eye(4)
    T_predict[:3, :3] = rotation_matrix
    T_predict[:3, 3] = translation
    
    # 计算投影点
    projected_points = project_points(points_3d, rotation_matrix, translation, K)
    
    # 重投影误差：确保投影点落在mask的1区域内
    reprojection_residuals = []
    for point in projected_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            #print(mask.shape)
            if  np.mean(mask[y, x]) < 55:
                reprojection_residuals.append(1)  # 落在0区域，增加误差
            else:
                reprojection_residuals.append(0)  # 落在1区域，没有误差
        else:
            reprojection_residuals.append(1)  # 超出图像范围，增加误差
    
    # 旋转矩阵相等的约束误差
    rotation_matrix_residuals = (T_predict - R_fixed).ravel()
    
    # 合并误差
    return np.hstack((reprojection_residuals, rotation_matrix_residuals))


def main(args):
    
    squence_image_path = "../folder_01/left"
    squence_right_image_path = "../folder_01/right"
    squence_pcd_path = "../folder_01/pointseg"
    intrinsic_json_path = "../folder_01/center_camera-intrinsic.json"
    right_intrinsic_json_path = "../folder_01/right_camera-intrinsic.json"
    extrinsic_json_path = "../folder_01/top_center_lidar-to-center_camera-extrinsic.json"
    
    intrinsic, distortion = load_intrinsic(intrinsic_json_path)
    right_intrinsic, right_distortion = load_intrinsic(right_intrinsic_json_path)
    extrinsic = load_extrinsic(extrinsic_json_path)

    squence_image_files = os.listdir(squence_image_path)
    squence_right_image_files = os.listdir(squence_right_image_path)
    squence_pcd_files = os.listdir(squence_pcd_path)

    squence_image_files = sorted(squence_image_files)
    squence_right_image_files = sorted(squence_right_image_files)
    squence_pcd_files = sorted(squence_pcd_files)

    R_01_kalman = None
    R_01_kalman_list = []
    scores_list = []
    frame_index = 0
    init_process = False

    decent_step = 1

    for each_image_file in squence_image_files:
        each_image_path = os.path.join(squence_image_path, each_image_file)
        each_right_image_path = os.path.join(squence_right_image_path, each_image_file)
        each_pcd_path = os.path.join(squence_pcd_path, each_image_file.replace("png", "pcd"))

        left_image = undistortion(each_image_path, intrinsic, distortion)
        left_image_color = undistortion_color(each_image_path, intrinsic, distortion)
        right_image = undistortion(each_right_image_path, right_intrinsic, right_distortion)
        right_image_color = undistortion_color(each_right_image_path, intrinsic, distortion)

        left_image_mask_path = each_image_path.replace("left", "left_label")
        left_image_mask = undistortion(left_image_mask_path, intrinsic, distortion)
        left_image_mask_np = np.array(left_image_mask)
        left_image_category_ids = np.unique(left_image_mask_np)
        left_image_category = [img_id_mask_set[key] for key in img_id_mask_set if key in left_image_category_ids]

        right_image_mask_path = each_right_image_path.replace("right", "right_label")
        right_image_mask = undistortion(right_image_mask_path, right_intrinsic, right_distortion)
        right_image_mask_np = np.array(right_image_mask)
        right_image_category_ids = np.unique(right_image_mask_np)
        right_image_category = [img_id_mask_set[key] for key in img_id_mask_set if key in right_image_category_ids]

        pcd_mask, pcd_xyz = extract_pc_cate(each_pcd_path)
        lidar_category = [POINT_LABLES[key] for key in POINT_LABLES if key in pcd_mask]
        intersection = set(left_image_category) & set(right_image_category) & set(lidar_category)
        # 转换为列表（如果需要）
        common_category_str = list(intersection)
        pcd_mask_map_to_camera_str = np.vectorize(POINT_LABLES.get)(pcd_mask)
        pcd_mask_map_to_camera = np.vectorize(img_mask_set.get)(pcd_mask_map_to_camera_str)
        common_category = [img_mask_set[key] for key in img_mask_set if key in common_category_str]

        intersection_front_segments = set(common_category) & set(front_segments)
        intersection_init_segments = set(common_category) & set(init_segments)

        left_weight_map = mask_weight(left_image_mask_np, common_category)
        right_weight_map = mask_weight(right_image_mask_np, common_category)

        if init_process is False:
            init_process = True
            if len(intersection_front_segments) > 3 and len(intersection_init_segments) > 0:
                print("supporting initialization")
                ddd_center = extract_3d_center(pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments)
                dd_center = extract_2d_center(left_image_mask_np, intersection_front_segments)
                assert len(ddd_center) == len(dd_center)
                ddd_center = np.vstack(ddd_center).reshape(-1, 3)
                dd_center = np.vstack(dd_center).reshape(-1, 2)

                if ddd_center.shape[0] > 4:
                    ddd_center = ddd_center[:4, :]
                    dd_center = dd_center[:4, :]

                success, init_rvec, init_tvec = cv2.solvePnP(ddd_center, dd_center, intrinsic, distortion, flags=cv2.SOLVEPNP_P3P)
                init_R, _ = cv2.Rodrigues(init_rvec)
                init_extrinsic_matrix = np.hstack((init_R, init_tvec))
                init_extrinsic_matrix = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.], [1., 0., 0., 0.]])
                print("Initial：[R|T] = \n", init_extrinsic_matrix)
                print("计算误差-Initial：")
                calculate_error(init_extrinsic_matrix, extrinsic)

                register_cloud = extract_feature(left_image, each_pcd_path, init_extrinsic_matrix, intrinsic)
                show_and_save_proj(left_image, register_cloud, intrinsic, init_extrinsic_matrix, "init")
                init_extrinsic_matrix = np.concatenate([init_extrinsic_matrix, np.array([[0, 0, 0, 1]])], axis = 0)
                R_01_kalman = init_extrinsic_matrix

            else:
                print("not supporting initialization")
                continue

        R12, t12, kp1_12, kp2_12, matches12, match_points1, match_points2 = \
                calculate_extrinsic_matrix(left_image, right_image, intrinsic, right_intrinsic)

        matched_img = draw_matches(left_image, right_image, kp1_12, kp2_12, matches12)
        cv2.imwrite('img_matches12.png', matched_img)

        optimizer_best = Optimizer_all(intrinsic, right_intrinsic, extrinsic, R12, t12, decent_step)
        best_score = optimizer_best.cost_function(np.zeros(6), left_image_mask_np, right_image_mask_np,\
                     pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_weight_map, right_weight_map)
        print("best_score:", best_score)

        optimizer_all = Optimizer_all(intrinsic, right_intrinsic, R_01_kalman, R12, t12, decent_step)

        # if frame_index == 0:

        #     init_score = optimizer_all.cost_function(np.zeros(6), left_image_mask_np, right_image_mask_np,\
        #             pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_weight_map, right_weight_map)
        #     R_01_kalman_list.append(R_01_kalman)
        #     scores_list.append(init_score)

        R_01_kalman, score_left_1 = optimizer_all.calibrate(left_image_mask_np, right_image_mask_np,\
                pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments, left_weight_map, right_weight_map)

        print("计算误差-mutli")
        calculate_error(R_01_kalman, extrinsic)
        print("score:", score_left_1)

        kf = KalmanFilterForExtrinsics()

        # 假设输入外参矩阵和打分
        R_01_kalman_list.append(R_01_kalman)
        scores_list.append(score_left_1)

        scores_list_range = compress_to_range(scores_list, 0.1, 0.98)
        #print(scores_list_range)

        #初始化 Kalman 滤波器
        for i, (ext, score) in enumerate(zip(R_01_kalman_list, scores_list_range)):
            #print(f"Frame {i + 1}")
            kf.update(ext.flatten(), score)  # 传入展平的外参矩阵和打分
            current_state = kf.get_state()
        R_01_kalman = current_state
        
        print("sequential-计算误差")
        calculate_error(R_01_kalman, extrinsic)
        #print(R_01_kalman)
        show_and_save_proj(left_image_color, pcd_xyz, intrinsic, R_01_kalman, str(frame_index))
        #assert 0 
        frame_index += 1
        #decent_step *= 0.9

if __name__ == "__main__":
    main(sys.argv)