from scipy.ndimage import label, find_objects
import numpy as np
from lidar_processing import post_lidarseg, extract_center, parse_pcd
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt

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
    #print(mask.shape)
    #mask = mask[:, :, 0]
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # 将原mask中为0的部分直接设置为0
    dist_transform[mask == 0] = 0
    
    # 归一化，将距离值缩放到 [0, 1] 之间
    max_val = dist_transform.max()
    if max_val > 0:
        dist_transform = dist_transform / max_val
    
    return dist_transform

def extract_3d_center(pcd_mask_map_to_camera, pcd_xyz, intersection_front_segments):
    extract_3d_center_list = []
    intersection_front_segments = sorted(intersection_front_segments)
    for each_segment in intersection_front_segments:

        mask = np.isin(pcd_mask_map_to_camera, each_segment)

        selected_points = pcd_xyz[mask]

        # 计算每个轴的中位数，得到中心点坐标
        selected_points_center_x = np.median(selected_points[:, 0])
        selected_points_center_y = np.median(selected_points[:, 1])
        selected_points_center_z = np.median(selected_points[:, 2])
        selected_points_center_point = np.array([selected_points_center_x, selected_points_center_y, selected_points_center_z])

        # 计算每个点到中心点的欧氏距离
        selected_points_distances = np.linalg.norm(selected_points - selected_points_center_point, axis=1)

        # 找到距离最小的点索引
        selected_points_closest_index = np.argmin(selected_points_distances)
        center_point = selected_points[selected_points_closest_index]
        extract_3d_center_list.append(center_point)
    return extract_3d_center_list


def extract_2d_center(left_image_mask_np, intersection_front_segments):
    extract_2d_center_list = []
    intersection_front_segments = sorted(intersection_front_segments)
    for each_segment in intersection_front_segments:
        binary_array = np.zeros_like(left_image_mask_np)
        binary_array[left_image_mask_np == each_segment] = 1

        labeled_array, num_features = label(binary_array)
        
        # Step 2: Find slices for each connected component
        regions = find_objects(labeled_array)

        # Step 3: Calculate the size of each region
        region_sizes = [(i + 1, np.sum(labeled_array[regions[i]] == i + 1)) for i in range(num_features)]

        # Step 4: Find the largest region
        largest_region_label, largest_region_size = max(region_sizes, key=lambda x: x[1])

        # Step 5: Create a mask for the largest region
        largest_region_mask = (labeled_array == largest_region_label)
        
        # 找到值为 1 的所有坐标
        y_coords, x_coords = np.where(largest_region_mask == 1)

        # 找到 x 和 y 的极值
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        top_left = (x_min, y_min)   # 左上角
        top_right = (x_max, y_min)  # 右上角
        bottom_left = (x_min, y_max) # 左下角
        bottom_right = (x_max, y_max) # 右下角
        center_img = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))

        egdes_points_2d = np.array([[x_min, (y_min + y_max)/2], [x_max, (y_min + y_max)/2], \
            [(x_min + x_max)/2, y_min], [(x_min + x_max)/2, y_max]])
        center_2d = np.array([[(x_min + x_max)/2, (y_min + y_max)/2]])
        extract_2d_center_list.append(center_2d)

    return extract_2d_center_list

def show_and_save_proj(image, register_cloud, intrinsic, extrinsic, flag):
    image_save = image.copy()
    for src_pt_index in range(register_cloud.shape[0]):
        src_pt = register_cloud[src_pt_index]
        if not np.isfinite(src_pt[0]) or not np.isfinite(src_pt[1]) or not np.isfinite(src_pt[2]):
            continue
        vec = np.array([src_pt[0], src_pt[1], src_pt[2], 1.0])
        cam_point = extrinsic @ vec
        cam_vec = cam_point[:3]
        vec_2d = intrinsic @ cam_vec

        if vec_2d[2] > 0:
            x = int(np.round(vec_2d[0] / vec_2d[2]))
            y = int(np.round(vec_2d[1] / vec_2d[2]))

            if 0 <= x < image_save.shape[1] and 0 <= y < image_save.shape[0]:
                centerCircle2 = (x, y)
                color = (0, 255, 0)
                cv2.circle(image_save, centerCircle2, 3, color, -1, 0)

    cv2.imwrite("./" + flag + "_test.png", image_save)

def generate_weight_map(image_mask_np, init_segments, common_category):
    #print(image_mask_np.shape)
    #mask_front = (image_mask_np != 0).astype(int)
    common_init_segments = list(set(common_category) & set(init_segments))
    calculate_mask = np.isin(image_mask_np, common_category).astype(np.uint8)
    front_mask = np.isin(image_mask_np, common_init_segments).astype(np.uint8)

    back_segments = [x for x in common_category if x not in init_segments]
    back_mask = np.isin(image_mask_np, back_segments).astype(np.uint8)
    back_mask[:300, :] = 0

    # Step 1: 标记独立区域
    labeled_array, num_features = label(front_mask)

    # Step 2: 计算每个区域的中心和最远点距离
    region_centers = []
    region_radii = []

    for i in range(1, num_features + 1):
        # 获取当前区域的坐标
        region_coords = np.argwhere(labeled_array == i)
        
        # 计算中心点
        center = region_coords.mean(axis=0)
        region_centers.append(center)
        
        # 计算最远点距离
        distances = np.linalg.norm(region_coords - center, axis=1)
        max_distance = distances.max()
        region_radii.append(max_distance)

    # # Step 3: 绘图
    # plt.figure(figsize=(8, 8))
    # plt.imshow(vehicle_mask, cmap="gray")
    # plt.title("Regions with Centers and Circles")

    # # 绘制中心点和圆
    # for center, radius in zip(region_centers, region_radii):
    #     center_y, center_x = center  # 注意坐标顺序
    #     plt.plot(center_x, center_y, "ro")  # 绘制中心点
    #     circle = plt.Circle((center_x, center_y), radius, color="b", fill=False, linewidth=2)
    #     plt.gca().add_artist(circle)

    # plt.axis("equal")
    # plt.show()

    # 输出每个区域的中心和半径
    # for i, (center, radius) in enumerate(zip(region_centers, region_radii), 1):
    #     print(f"区域 {i}: 中心 = {center}, 最大半径 R = {radius}")
    
    # 创建权重图
    vehicle_weight_map = np.zeros_like(image_mask_np)

    # 创建坐标网格
    x = np.arange(vehicle_weight_map.shape[1])
    y = np.arange(vehicle_weight_map.shape[0])
    xx, yy = np.meshgrid(x, y)

    # 对每个圆处理
    for index in range(len(region_centers)):
        center_x, center_y = region_centers[index][1], region_centers[index][0]
        radius = region_radii[index]
        
        # 计算到圆心的距离
        distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        
        # 线性插值权重：从中心 (0) 到边缘 (1)
        weights = np.clip(distances / radius, 0, 1)
        weights[distances > radius] = 0  # 圆外部分设置为 0
        
        # 将权重合并到整体图中
        vehicle_weight_map = np.maximum(vehicle_weight_map, weights)

    # Step 3: 可视化权重图
    vehicle_weight_map = vehicle_weight_map / 255

    # 获取所有值为 1 的位置的 y 坐标
    y_indices, x_indices = np.where(back_mask == 1)

    # 计算 y 的最小值和最大值，最小值对应最上方，最大值对应最下方
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    # # 创建一个与原图像大小相同的权重图
    # back_weights = np.zeros_like(back_mask, dtype=float)
    # # 根据 y 坐标计算线性权重
    # for y, x in zip(y_indices, x_indices):
    #     # 计算 y 对应的权重，最上面的 y 对应权重为 0，最下面的 y 对应权重为 1
    #     weight = (y - y_min) / (y_max - y_min)  # 线性映射
    #     back_weights[y, x] = weight
    # back_weights = back_weights / 255

    height, width = image_mask_np.shape[0], image_mask_np.shape[1]
    # 生成坐标网格
    y_back, x_back = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    
    # 距离计算：到边缘的最小距离
    distance_to_edge = np.minimum.reduce([x_back, y_back, width - 1 - x_back, height - 1 - y_back])
    
    # 将距离归一化到 [0, 1]，边缘为1，中心为0
    max_distance = np.max(distance_to_edge)
    back_weights = 1 - (distance_to_edge / max_distance)
    back_weights = back_weights / 255

    # cv2.imshow("image", vehicle_weight_map * front_mask * 255)
    # cv2.waitKey(0)
    # assert 0 
    front_weight = vehicle_weight_map * front_mask + 0.5 * back_weights * back_mask
    weight_mask = front_weight * calculate_mask

    # cv2.imshow("image", weight_mask * 255)
    # cv2.waitKey(0)
    # assert 0 

    return weight_mask