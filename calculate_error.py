import numpy as np

# label = np.array([[-0.00533616, -0.999986, 0.00064036, 0.504211],
#                 [0.0342852, -0.000822947, -0.999412, -0.118845],
#                 [0.999398, -0.00531107, 0.0342891, -0.105546],
#                 [0, 0, 0, 1]])

#kitti
#label = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
#                [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
#                [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
#                [0, 0, 0, 1]])

def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角（roll, pitch, yaw）
    R: 3x3 旋转矩阵
    返回：roll, pitch, yaw
    """
    assert R.shape == (3, 3), "旋转矩阵必须是3x3的"

    # 从旋转矩阵中提取pitch
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6  # 如果sy接近零，我们认为这是一个奇异情况

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw

def calculate_error(T_matrix, label):

    R_matrix = T_matrix[:3, :3]
    t_matrix = T_matrix[:3, 3:]

    r_vector = rotation_matrix_to_euler_angles(R_matrix)

    label_R_matrix = label[:3, :3]
    label_t_matrix = label[:3, 3:]

    l_vector = rotation_matrix_to_euler_angles(label_R_matrix)


    r_error = [np.abs(r_vector[0] - l_vector[0]), np.abs(r_vector[1] - l_vector[1]), np.abs(r_vector[2] - l_vector[2])]
    t_error = [np.abs(t_matrix[0,0] - label_t_matrix[0,0]), np.abs(t_matrix[1,0] - label_t_matrix[1,0]), np.abs(t_matrix[2,0] - label_t_matrix[2,0])]

    print("t_error:", r_error)
    print("t_error:", t_error)
