import cv2
import numpy as np

def undistortion(image_path, intrinsic, distortion):
    # 读取图像
    image = cv2.imread(image_path, 0)
    
    # 将Eigen::Matrix3d转换为cv::Mat
    K = intrinsic.astype(np.float32)
    
    # 初始化映射矩阵
    map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3, dtype=np.float32), K, (image.shape[1], image.shape[0]), cv2.CV_32FC1)
    
    # 应用重映射来校正图像
    threshold = cv2.remap(image, map1, map2, interpolation=cv2.INTER_NEAREST)
    
    return threshold

def undistortion_color(image_path, intrinsic, distortion):
    # 读取图像
    image = cv2.imread(image_path, 1)
    
    # 将Eigen::Matrix3d转换为cv::Mat
    K = intrinsic.astype(np.float32)
    
    # 初始化映射矩阵
    map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3, dtype=np.float32), K, (image.shape[1], image.shape[0]), cv2.CV_32FC1)
    
    # 应用重映射来校正图像
    threshold = cv2.remap(image, map1, map2, interpolation=cv2.INTER_NEAREST)
    
    return threshold

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

def extract_pc_cate(pcd_path):
    point_cloud_data = parse_pcd(pcd_path)
    x = point_cloud_data['x']
    y = point_cloud_data['y']
    z = point_cloud_data['z']
    label = point_cloud_data['label']

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    xyz = np.concatenate([x, y, z], axis = 1)
    label = label[(xyz[:, 0] > 5) & (xyz[:, 0] < 80)]
    xyz = xyz[(xyz[:, 0] > 5) & (xyz[:, 0] < 80)]
    return label, xyz
