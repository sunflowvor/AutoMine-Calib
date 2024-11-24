import numpy as np
import cv2
import open3d as o3d
import time
from scipy.spatial.transform import Rotation as R
from lidar_processing import parse_pcd, POINT_COLOR_MAP

def deg2rad(degrees):
    return degrees * np.pi / 180

def get_delta_t(var):
    # 计算旋转矩阵
    rotation = R.from_euler('zyx', [deg2rad(var[2]), deg2rad(var[1]), deg2rad(var[0])])
    deltaR = rotation.as_matrix()
    
    # 创建变换矩阵
    deltaT = np.eye(4)
    deltaT[:3, :3] = deltaR
    deltaT[0, 3] = var[3]
    deltaT[1, 3] = var[4]
    deltaT[2, 3] = var[5]
    
    return deltaT

def binarize_image(image):
    # 将大于0的值置为1，其余保持为0
    binary_image = np.where(image > 0, 1, 0)
    return binary_image

def generate_distance_weight_map(mask):
    # 确保mask是二值化的（0和1）
    mask = mask.astype(np.uint8)
    
    # 计算距离变换，计算每个1像素到最近0像素的距离
    mask = mask[:, :, 0]
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # 将原mask中为0的部分直接设置为0
    dist_transform[mask == 0] = 0
    
    # 归一化，将距离值缩放到 [0, 1] 之间
    max_val = dist_transform.max()
    if max_val > 0:
        dist_transform = dist_transform / max_val
    
    return dist_transform

class Optimizer:
    def __init__(self, curr_optim_intrinsic, curr_optim_extrinsic):
        self.curr_optim_extrinsic_ = curr_optim_extrinsic
        self.curr_optim_intrinsic = curr_optim_intrinsic

    def mask_registration_loss(self, var, distance_img, register_cloud, weight_map):
        PointCnt = 0
        deltaT = get_delta_t(var)
        #print(self.curr_optim_extrinsic_)
        T = self.curr_optim_extrinsic_ @ deltaT

        valid_points = []

        #bin_left_distance_img = binarize_image(distance_img)
        #weight_left_distance_img = generate_distance_weight_map(bin_left_distance_img)
        
        for src_pt_index in range(register_cloud.shape[0]):
            src_pt = register_cloud[src_pt_index]
            
            if not np.isfinite(src_pt[0]) or not np.isfinite(src_pt[1]) or not np.isfinite(src_pt[2]):
                continue
            vec = np.array([src_pt[0], src_pt[1], src_pt[2], 1.0])
            cam_point = T @ vec
            cam_vec = cam_point[:3]
            vec_2d = self.curr_optim_intrinsic @ cam_vec
            
            if vec_2d[2] > 0:
                x = int(np.round(vec_2d[0] / vec_2d[2]))
                y = int(np.round(vec_2d[1] / vec_2d[2]))

                if 0 <= x < distance_img.shape[1] and 0 <= y < distance_img.shape[0]:
                    pixel = distance_img[y, x]
                    pixel_gray = np.mean(pixel)
                    weight_value = weight_map[y, x]

                    PointCnt += pixel_gray * weight_value

        return float(PointCnt) / register_cloud.shape[0]

    def cost_function(self, var, distance_img, register_cloud, weight_map):
        registration_cost = self.mask_registration_loss(var, distance_img, register_cloud, weight_map)
        return registration_cost

    def random_search_params(self, search_count, delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, weight_map):
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
        maxPointCnt = self.cost_function(var, self.register_img_, self.register_cloud_, weight_map)

        for i in range(search_count):
            var[0] = distribution_x[i]
            var[1] = distribution_y[i]
            var[2] = distribution_z[i]
            var[3] = distribution_roll[i]
            var[4] = distribution_pitch[i]
            var[5] = distribution_yaw[i]
            cnt = self.cost_function(var, self.register_img_, self.register_cloud_, weight_map)
            #print(var)
            if cnt > maxPointCnt:
                better_cnt += 1
                maxPointCnt = cnt
                bestVal = var.copy()

        self.curr_optim_extrinsic_ = self.curr_optim_extrinsic_ @ get_delta_t(bestVal)
        self.current_cost_fun_ = maxPointCnt
        self.current_fc_score_ = 1 - better_cnt / search_count

    def calibrate(self, distance_img, register_cloud, idx_name, delay_scale, weight_map):
        self.register_img_ = distance_img
        self.register_cloud_ = register_cloud

        # SaveProjectResult("before.png", distance_img, original_cloud)
        var = np.zeros(6)
        bestVal = np.zeros(6)
        best_extrinsic_vec = np.zeros(6)
        varName = ["roll", "pitch", "yaw", "tx", "ty", "tz"]
        direction = [1, 0, -1]
        maxPointCnt = self.cost_function(var, distance_img, register_cloud, weight_map)

        print("before score: ", maxPointCnt)

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
                                    cnt = self.cost_function(var, distance_img, register_cloud, weight_map)
                                    #print(cnt)
                                    if cnt > maxPointCnt:
                                        maxPointCnt = cnt
                                        bestVal = var.copy()
                                        #print("match point increase to: ", maxPointCnt)
            self.curr_optim_extrinsic_ = self.curr_optim_extrinsic_ @ get_delta_t(bestVal)
            self.current_cost_fun_ = maxPointCnt
        else:
            f1, f2 = False, False
            rotation_step = 0.1
            translation_step = 0.1
            for k in range(10):
                if k % 2 == 0:
                    self.random_search_params(10, translation_step, translation_step, translation_step, \
                        rotation_step, rotation_step, rotation_step, weight_map)
                    f1 = self.current_fc_score_ == 1
                else:
                    self.random_search_params(10, translation_step / 10, translation_step / 10, translation_step / 10, \
                        rotation_step / 10, rotation_step / 10, rotation_step / 10, weight_map)
                    f2 = self.current_fc_score_ == 1
                if f1 and f2:
                    break
            #print("checking level 1")
            for k in range(10):
                self.random_search_params(10, translation_step, translation_step / 10, translation_step / 10,\
                 0, 0, 0, weight_map)
                if self.current_fc_score_ == 1:
                    break
            #print("checking level 2")
            for k in range(10):
                self.random_search_params(10, translation_step / 10, translation_step, translation_step / 10,\
                 0, 0, 0, weight_map)
                if self.current_fc_score_ == 1:
                    break
            #print("checking level 3")
            for k in range(10):
                self.random_search_params(10, translation_step / 10, translation_step / 10, translation_step,\
                 0, 0, 0, weight_map)
                if self.current_fc_score_ == 1:
                    break

        #print("after loss: ", self.current_cost_fun_)
        #print("curr ex:", self.curr_optim_extrinsic_)
        img_points, lidar_points = self.save_project_result("feature_projection.png", distance_img, register_cloud, idx_name)
        return img_points, lidar_points, self.curr_optim_extrinsic_, self.current_cost_fun_

    def calibrate_t(self, distance_img, register_cloud, idx_name):
        self.register_img_ = distance_img
        self.register_cloud_ = register_cloud

        # SaveProjectResult("before.png", distance_img, original_cloud)
        var = np.zeros(6)
        bestVal = np.zeros(6)
        best_extrinsic_vec = np.zeros(6)
        varName = ["roll", "pitch", "yaw", "tx", "ty", "tz"]
        direction = [1, 0, -1]
        maxPointCnt = self.cost_function(var, distance_img, register_cloud)

        #print("before loss: ", maxPointCnt)

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
                                    cnt = self.cost_function(var, distance_img, register_cloud)
                                    #print(cnt)
                                    if cnt > maxPointCnt:
                                        maxPointCnt = cnt
                                        bestVal = var.copy()
                                        #print("match point increase to: ", maxPointCnt)
            self.curr_optim_extrinsic_ = self.curr_optim_extrinsic_ @ get_delta_t(bestVal)
            self.current_cost_fun_ = maxPointCnt
        else:
            f1, f2 = False, False
            for k in range(10):
                if k % 2 == 0:
                    self.random_search_params(100, 0, 0, 0, 0.5, 0.5, 0.5)
                    f1 = self.current_fc_score_ == 1
                else:
                    self.random_search_params(100, 0, 0, 0, 0.25, 0.25, 0.25)
                    f2 = self.current_fc_score_ == 1
                if f1 and f2:
                    break
            #print("checking level 1")
            for k in range(10):
                self.random_search_params(10, 0, 0, 0, 0.25, 0.25, 0.25)
                if self.current_fc_score_ == 1:
                    break
            #print("checking level 2")
            for k in range(10):
                self.random_search_params(10, 0, 0, 0, 0.05, 0.05, 0.05)
                if self.current_fc_score_ == 1:
                    break
            #print("checking level 3")
            for k in range(10):
                self.random_search_params(10, 0, 0, 0, 0.025, 0.025, 0.025)
                if self.current_fc_score_ == 1:
                    break

        #print("after loss: ", self.current_cost_fun_)
        #print("curr ex:", self.curr_optim_extrinsic_)
        img_points, lidar_points = self.save_project_result("feature_projection.png", distance_img, register_cloud, idx_name)
        return img_points, lidar_points, self.curr_optim_extrinsic_, self.current_cost_fun_

    def rotation2eul(self, rotation_matrix):
        r = R.from_matrix(rotation_matrix)
        return r.as_euler('zyx', degrees=False)

    def save_project_result(self, filename, distance_img, register_cloud, idx_name):
        # 保存结果的方法
        
        rotation_matrix = self.curr_optim_extrinsic_[:3, :3]
        translation = self.curr_optim_extrinsic_[:3, 3]
        eulerAngle = self.rotation2eul(rotation_matrix)

        #print(rotation_matrix)
        #print("refine extrinsic:")
        #print("rotation:", eulerAngle)
        #print("translation:", translation)

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

        cv2.imwrite("./" + idx_name + "_test.png", image)
        return np.array(img_points), np.array(lidar_points)

    def show_all_points(self, distance_imgs, register_cloud):
        point_cloud_data = parse_pcd(register_cloud)
        x = point_cloud_data['x']
        y = point_cloud_data['y']
        z = point_cloud_data['z']
        label = point_cloud_data['label']
        object_id = point_cloud_data['object']

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
        xyz = np.concatenate([x, y, z], axis = 1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        colors = np.zeros((xyz.shape[0], 3))

        for each_label in np.unique(label):
            color_255 = POINT_COLOR_MAP[each_label]
            array_data = np.array(color_255)
            result_array = array_data / 255
            result_tuple = tuple(result_array)
            colors[label == each_label] = result_tuple

        pcd.colors = o3d.utility.Vector3dVector(colors)

        label = label[xyz[:, 0] > 5]
        xyz = xyz[xyz[:, 0] > 5]

        ones_vector = np.ones((xyz.shape[0], 1))
        extended_selected_points = np.concatenate([xyz, ones_vector], axis = 1)

        cam_point = np.matmul(self.curr_optim_extrinsic_, extended_selected_points.transpose(1, 0))
        cam_vec = cam_point[:3]
        vec_2d = np.matmul(self.curr_optim_intrinsic, cam_vec).transpose(1, 0)

        register_cloud = []
        register_edge = []
        
        for each_index in range(vec_2d.shape[0]):
            each_point = vec_2d[each_index]

            if each_point[2] > 0:
                x = int(np.round(each_point[0] / each_point[2]))
                y = int(np.round(each_point[1] / each_point[2]))
    
                if 0 <= x < distance_imgs.shape[1] and 0 <= y < distance_imgs.shape[0]:
                        pc = np.array([extended_selected_points[each_index, 0], \
                                        extended_selected_points[each_index, 1], \
                                        extended_selected_points[each_index, 2]])
                        register_cloud.append(pc)
                        register_edge.append(np.array([x, y]))
        return np.array(register_edge), np.array(register_cloud)

    def ideal_img(self, register_cloud):
        point_cloud_data = parse_pcd(register_cloud)
        x = point_cloud_data['x']
        y = point_cloud_data['y']
        z = point_cloud_data['z']
        label = point_cloud_data['label']
        object_id = point_cloud_data['object']

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
        xyz = np.concatenate([x, y, z], axis = 1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        colors = np.zeros((xyz.shape[0], 3))

        for each_label in np.unique(label):
            color_255 = POINT_COLOR_MAP[each_label]
            array_data = np.array(color_255)
            result_array = array_data / 255
            result_tuple = tuple(result_array)
            colors[label == each_label] = result_tuple

        pcd.colors = o3d.utility.Vector3dVector(colors)

        label = label[xyz[:, 0] > 5]
        xyz = xyz[xyz[:, 0] > 5]

        ones_vector = np.ones((xyz.shape[0], 1))
        extended_selected_points = np.concatenate([xyz, ones_vector], axis = 1)

        cam_point = np.matmul(np.array([[0, -1, 0, 0], [0, 0, -1, 0],[1, 0, 0, 0],[0, 0, 0, 1]]), extended_selected_points.transpose(1, 0))
        cam_vec = cam_point[:3]

        vec_2d = np.matmul(np.array([[1200, 0, 2048/2], [0, 1200, 1536/2], [0, 0, 1]]), cam_vec).transpose(1, 0)

        register_cloud = []
        register_edge = []
        size = 1536, 2048
        m = np.zeros(size, dtype=np.uint8) # ?
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        
        for each_index in range(vec_2d.shape[0]):
            each_point = vec_2d[each_index]

            if each_point[2] > 0:
                x = int(np.round(each_point[0] / each_point[2]))
                y = int(np.round(each_point[1] / each_point[2]))
    
                if 0 <= x < 2048 and 0 <= y < 1536:
                        pc = np.array([extended_selected_points[each_index, 0], \
                                        extended_selected_points[each_index, 1], \
                                        extended_selected_points[each_index, 2]])
                        register_cloud.append(pc)
                        register_edge.append(np.array([x, y]))
                        color = (0, 255, 0)
                        cv2.circle(m, (x, y), 3, color, -1, 0)
        return np.array(register_edge), np.array(register_cloud)