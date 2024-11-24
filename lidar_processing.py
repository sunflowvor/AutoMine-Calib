import cv2
import numpy as np
import open3d as o3d
# 28: "cable" 5 CAR
POINT_LABLES = {0: "VOID", 1: "mining_truck", 2: "wide_body_truck", 3: "truck", 4: "bus", \
                6: "engineering_vehicle", 7: "excavator", 8: "pushdozer", \
                9: "watering_car", 10: "fuel_tank_car", 11: "electric_shovel", \
                12: "human", 13: "road", 14: "rut", 15: "gravel", 16: "soft_soil", \
                17: "road_edge", 18: "puddle", 19: "taft", 20: "construction", \
                21: "vegetation", 22: "tussock", 23: "sky", 24: "massif", 25: "fence", \
                26: "road_sign", 27: "blocking_stone", 28: "car", 29: "ice_cream_cone", \
                30: "telegraph_pole", 31: "communication_pole", 32: "road_testing_equipment"}

POINT_COLOR_MAP = {0: (0, 0, 0),
               23: (70, 130, 180),#sky
               13: (128, 64, 128),#road
               24: (0, 80, 100),#massif
               14: (111, 74, 0),#rut
               17: (244, 35, 232),#curb
               22: (0, 255, 0),#brushwood
               1: (180, 165, 180),#mining_truck
               16: (81, 0, 81),#mollisoil
               15: (102, 102, 156),#gravel
               18: (250, 170, 160),#puddle
               11: (0, 0, 110),#electric_shovel
               32: (220, 0, 0),#road test equipment
               6: (100, 20, 60),#engineering vehicle
               8: (152, 251, 152),#pushdozer
               3: (0, 0, 70),#truck
               5: (0, 0, 142),#car
               19: (230, 150, 140),#mound
               21: (107, 142, 35),#vegetation
               27: (0, 0, 230),#blocking_stone
               7: (150, 120, 90),#excavator
               20: (70, 70, 70),#constrcution
               29: (119, 11, 32),#traffic cone
               25: (190, 153, 153),#fence
               26: (220, 220, 0),#road_sign
               28: (0, 142, 142),#cable
               31: (250, 170, 30),#conmunication pole
               30: (153, 153, 153),#telegraph pole
               9: (255, 0, 0),#watering car
               10: (0, 0, 90),#oil truck
               12: (220, 20, 60),#human
               2: (150, 100, 100),#wide_body_truck
               4: (0,60,100)#bus
           }

def parse_pcd(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = []
    data = []
    is_data = False

    for line in lines:
        if line.startswith('DATA'):
            is_data = True
        elif is_data:
            data.append(line.strip())
        else:
            header.append(line.strip())

    # Extract fields information from header
    fields_line = next(line for line in header if line.startswith('FIELDS'))
    fields = fields_line.split()[1:]

    # Read the data into a structured numpy array
    dtype_list = []
    for field in fields:
        if field in ['x', 'y', 'z']:
            dtype_list.append((field, 'f4'))
        elif field in ['label', 'object']:
            dtype_list.append((field, 'i4'))
        elif field == 'rgb':
            dtype_list.append((field, 'u4'))
    #print(fields)
    structured_array = np.genfromtxt(data, dtype=dtype_list)
    return structured_array

def extract_feature(distance_images, register_cloud, extrinsic, intrinsic):

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


    label = label[(xyz[:, 0] > 5) & (xyz[:, 0] < 80)]
    xyz = xyz[(xyz[:, 0] > 5) & (xyz[:, 0] < 80)]

    unique_values = np.unique(label)
    #assert 0 
    unique_names = []
    for i in unique_values:
        name_index = POINT_LABLES[i]
        unique_names.append(name_index)

    common_elements = list(set(unique_names))
    available_elsements = ['road_edge', "truck", "car"]
    selected_elements = list(set(common_elements) & set(available_elsements))

    point_keys_with_value = [key for key, value in POINT_LABLES.items() if value in selected_elements]
    #print(point_keys_with_value)
    #print(img_keys_with_value)

    #print(point_keys_with_value)

    #selected_points = xyz[label == point_keys_with_value[0]]
    mask = np.isin(label, point_keys_with_value)

    # 使用布尔掩码从 xyz 中选择对应的点
    selected_points = xyz[mask]

    ones_vector = np.ones((selected_points.shape[0], 1))
    extended_selected_points = np.concatenate([selected_points, ones_vector], axis = 1)

    cam_point = np.matmul(extrinsic, extended_selected_points.transpose(1, 0))
    cam_vec = cam_point[:3]
    vec_2d = np.matmul(intrinsic, cam_vec).transpose(1, 0)

    register_cloud = []
    
    for each_index in range(vec_2d.shape[0]):
        each_point = vec_2d[each_index]

        if each_point[2] > 0:
            x = int(np.round(each_point[0] / each_point[2]))
            y = int(np.round(each_point[1] / each_point[2]))
 
            if 0 <= x < distance_images.shape[1] and 0 <= y < distance_images.shape[0]:
                    pc = np.array([extended_selected_points[each_index, 0], \
                                    extended_selected_points[each_index, 1], \
                                    extended_selected_points[each_index, 2], 1])
                    register_cloud.append(pc)
                    #cloud_size += 1

    return np.array(register_cloud)

def extract_center(register_cloud, intrinsic, flag):
    if flag == "road_edge":
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

        # 根据 y 坐标的条件进行划分
        point_cloud_data_right = [item for item in point_cloud_data if item[1] <= 0]
        point_cloud_data_right = np.array(point_cloud_data_right)

        #processing right edge
        x_right = point_cloud_data_right['x']
        y_right = point_cloud_data_right['y']
        z_right = point_cloud_data_right['z']
        label_right = point_cloud_data_right['label']
        object_id_right = point_cloud_data_right['object']

        x_right = x_right.reshape(-1, 1)
        y_right = y_right.reshape(-1, 1)
        z_right = z_right.reshape(-1, 1)
        xyz_right = np.concatenate([x_right, y_right, z_right], axis = 1)
        pcd_right = o3d.geometry.PointCloud()
        pcd_right.points = o3d.utility.Vector3dVector(xyz_right)

        label_right = label_right[(xyz_right[:, 0] > 5) & (xyz_right[:, 0] < 80)]
        xyz_right = xyz_right[(xyz_right[:, 0] > 5) & (xyz_right[:, 0] < 80)]

        unique_values_right = np.unique(label_right)
        #print("Unique values:")
        #print(unique_values)
        unique_names_right = []
        for i in unique_values_right:
            name_index = POINT_LABLES[i]
            unique_names_right.append(name_index)

        common_elements_right = list(set(unique_names_right))
        available_elsements_right = [flag]
        selected_elements_right = list(set(common_elements_right) & set(available_elsements_right))

        point_keys_with_value_right = [key for key, value in POINT_LABLES.items() if value in selected_elements_right]

        #selected_points = xyz[label == point_keys_with_value[0]]
        mask_right = np.isin(label_right, point_keys_with_value_right)

        # 使用布尔掩码从 xyz 中选择对应的点
        selected_points_right = xyz_right[mask_right]

        # 计算每个轴的中位数，得到中心点坐标
        selected_points_right_center_x = np.median(selected_points_right[:, 0])
        selected_points_right_center_y = np.median(selected_points_right[:, 1])
        selected_points_right_center_z = np.median(selected_points_right[:, 2])
        selected_points_right_center_point = np.array([selected_points_right_center_x, \
                selected_points_right_center_y, selected_points_right_center_z])

        # 计算每个点到中心点的欧氏距离
        selected_points_right_distances = np.linalg.norm(selected_points_right - selected_points_right_center_point, axis=1)

        # 找到距离最小的点索引
        selected_points_right_closest_index = np.argmin(selected_points_right_distances)
        right_center_point = selected_points_right[selected_points_right_closest_index]

        return right_center_point

    else:
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


        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
        xyz = np.concatenate([x, y, z], axis = 1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        label = label[(xyz[:, 0] > 5) & (xyz[:, 0] < 80)]
        xyz = xyz[(xyz[:, 0] > 5) & (xyz[:, 0] < 80)]

        unique_values = np.unique(label)
        #print("Unique values:")
        #print(unique_values)
        unique_names = []
        for i in unique_values:
            name_index = POINT_LABLES[i]
            unique_names.append(name_index)

        common_elements = list(set(unique_names))
        available_elsements = [flag]
        selected_elements = list(set(common_elements) & set(available_elsements))

        point_keys_with_value = [key for key, value in POINT_LABLES.items() if value in selected_elements]

        #selected_points = xyz[label == point_keys_with_value[0]]
        mask = np.isin(label, point_keys_with_value)

        # 使用布尔掩码从 xyz 中选择对应的点
        selected_points = xyz[mask]

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
        return center_point


def post_lidarseg(register_cloud, intrinsic, flag):

    if flag == "road_edge":
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

        # 根据 y 坐标的条件进行划分
        point_cloud_data_right = [item for item in point_cloud_data if item[1] <= 0]
        point_cloud_data_left = [item for item in point_cloud_data if item[1] > 0]
        point_cloud_data_right = np.array(point_cloud_data_right)
        point_cloud_data_left = np.array(point_cloud_data_left)
        #assert 0 

        # # 打印结果
        # print("四个角点的 3D 坐标分别为：")
        # print("最小 X 的 3D 点：", corner_points_3d[0])
        # print("最大 X 的 3D 点：", corner_points_3d[1])
        # print("最小 Y 的 3D 点：", corner_points_3d[2])
        # print("最大 Y 的 3D 点：", corner_points_3d[3])

        # 使用 Open3D 可视化点云
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(xyz_left)

        # 可视化点云
        # o3d.visualization.draw_geometries([pcd_left])
        # o3d.visualization.draw_geometries([pcd]])
        # o3d.visualization.draw_geometries([pcd])
        # assert 0 

        #edge_right
    
        x_right = point_cloud_data_right['x']
        y_right = point_cloud_data_right['y']
        z_right = point_cloud_data_right['z']
        label_right = point_cloud_data_right['label']
        object_id_right = point_cloud_data_right['object']

        x_right = x_right.reshape(-1, 1)
        y_right = y_right.reshape(-1, 1)
        z_right = z_right.reshape(-1, 1)
        xyz_right = np.concatenate([x_right, y_right, z_right], axis = 1)
        pcd_right = o3d.geometry.PointCloud()
        pcd_right.points = o3d.utility.Vector3dVector(xyz_right)
        colors_right = np.zeros((xyz_right.shape[0], 3))

        for each_label_right in np.unique(label_right):
            color_255_right = POINT_COLOR_MAP[each_label_right]
            array_data_right = np.array(color_255_right)
            result_array_right = array_data_right / 255
            result_tuple_right = tuple(result_array_right)
            colors_right[label_right == each_label_right] = result_tuple_right

        pcd_right.colors = o3d.utility.Vector3dVector(colors_right)

        label_right = label_right[(xyz_right[:, 0] > 5) & (xyz_right[:, 0] < 80)]
        xyz_right = xyz_right[(xyz_right[:, 0] > 5) & (xyz_right[:, 0] < 80)]

        unique_values_right = np.unique(label_right)
        #print("Unique values:")
        #print(unique_values)
        unique_names_right = []
        for i in unique_values_right:
            name_index = POINT_LABLES[i]
            unique_names_right.append(name_index)

        common_elements_right = list(set(unique_names_right))
        available_elsements_right = [flag]
        selected_elements_right = list(set(common_elements_right) & set(available_elsements_right))

        point_keys_with_value_right = [key for key, value in POINT_LABLES.items() if value in selected_elements_right]

        #selected_points = xyz[label == point_keys_with_value[0]]
        mask_right = np.isin(label_right, point_keys_with_value_right)

        # 使用布尔掩码从 xyz 中选择对应的点
        selected_points_right = xyz_right[mask_right]

        ones_vector_right = np.ones((selected_points_right.shape[0], 1))
        extended_selected_points_right = np.concatenate([selected_points_right, ones_vector_right], axis = 1)
        extrinsic = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        cam_point_right = np.matmul(extrinsic, extended_selected_points_right.transpose(1, 0))
        cam_vec_right = cam_point_right[:3]
        vec_2d_right = np.matmul(intrinsic, cam_vec_right).transpose(1, 0)
        # 3. 找到未归一化 2D 投影的边界
        x_coords_right = vec_2d_right[:, 0]
        y_coords_right = vec_2d_right[:, 1]

        # 获取 2D 投影边界的最小最大值索引
        min_x_index_right = np.argmin(x_coords_right)
        max_x_index_right = np.argmax(x_coords_right)
        min_y_index_right = np.argmin(y_coords_right)
        max_y_index_right = np.argmax(y_coords_right)

        # 4. 获取四个角点对应的 3D 点
        egdes_points_3d_right = selected_points_right[[min_x_index_right, max_x_index_right, min_y_index_right, max_y_index_right]]

        # # 打印结果
        # print("四个角点的 3D 坐标分别为：")
        # print("最小 X 的 3D 点：", corner_points_3d[0])
        # print("最大 X 的 3D 点：", corner_points_3d[1])
        # print("最小 Y 的 3D 点：", corner_points_3d[2])
        # print("最大 Y 的 3D 点：", corner_points_3d[3])

        # 使用 Open3D 可视化点云
        #o3d.visualization.draw_geometries([pcd])



        return egdes_points_3d_right

    else:

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

        label = label[(xyz[:, 0] > 5) & (xyz[:, 0] < 80)]
        xyz = xyz[(xyz[:, 0] > 5) & (xyz[:, 0] < 80)]


        unique_values = np.unique(label)
        #print("Unique values:")
        #print(unique_values)
        unique_names = []
        for i in unique_values:
            name_index = POINT_LABLES[i]
            unique_names.append(name_index)

        common_elements = list(set(unique_names))
        available_elsements = [flag]
        selected_elements = list(set(common_elements) & set(available_elsements))

        point_keys_with_value = [key for key, value in POINT_LABLES.items() if value in selected_elements]

        #selected_points = xyz[label == point_keys_with_value[0]]
        mask = np.isin(label, point_keys_with_value)

        # 使用布尔掩码从 xyz 中选择对应的点
        selected_points = xyz[mask]

        ones_vector = np.ones((selected_points.shape[0], 1))
        extended_selected_points = np.concatenate([selected_points, ones_vector], axis = 1)
        extrinsic = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        cam_point = np.matmul(extrinsic, extended_selected_points.transpose(1, 0))
        cam_vec = cam_point[:3]
        vec_2d = np.matmul(intrinsic, cam_vec).transpose(1, 0)
        # 3. 找到未归一化 2D 投影的边界
        x_coords = vec_2d[:, 0]
        y_coords = vec_2d[:, 1]

        # 获取 2D 投影边界的最小最大值索引
        min_x_index = np.argmin(x_coords)
        max_x_index = np.argmax(x_coords)
        min_y_index = np.argmin(y_coords)
        max_y_index = np.argmax(y_coords)

        # 4. 获取四个角点对应的 3D 点
        egdes_points_3d = selected_points[[min_x_index, max_x_index, min_y_index, max_y_index]]

        # # 打印结果
        # print("四个角点的 3D 坐标分别为：")
        # print("最小 X 的 3D 点：", corner_points_3d[0])
        # print("最大 X 的 3D 点：", corner_points_3d[1])
        # print("最小 Y 的 3D 点：", corner_points_3d[2])
        # print("最大 Y 的 3D 点：", corner_points_3d[3])

        # 使用 Open3D 可视化点云
        #o3d.visualization.draw_geometries([pcd])



        return egdes_points_3d