import numpy as np

class KalmanFilterForExtrinsics:
    def __init__(self, state_dim=16):
        """
        初始化 Kalman 滤波器
        state_dim: 状态维度，16 对应 4x4 外参矩阵展平
        """
        self.state_dim = state_dim
        
        # 状态向量 (展平的 4x4 外参矩阵)
        self.state = np.zeros(state_dim)
        
        # 状态转移矩阵 (单位矩阵, 表示外参矩阵基本平稳)
        self.F = np.eye(state_dim)
        
        # 测量矩阵 (单位矩阵, 表示直接观察所有状态)
        self.H = np.eye(state_dim)
        
        # 状态协方差 (初始的不确定性)
        self.P = np.eye(state_dim) * 100
        
        # 过程噪声协方差 (假设系统变化很小)
        self.Q = np.eye(state_dim) * 0.01
        
        # 测量噪声协方差 (根据打分动态调整)
        self.R = np.eye(state_dim) * 10

    def update(self, measurement, score):
        """
        更新状态 (基于观测值和打分)
        measurement: 新的外参矩阵 (展平成向量)
        score: 打分 (0-1), 表示观测可信度
        """
        # 动态调整测量噪声协方差 (分数越高, 噪声越小)
        self.R = np.eye(self.state_dim) * (1.0 / max(score, 1e-6))
        
        # 预测阶段
        pred_state = self.F @ self.state
        pred_P = self.F @ self.P @ self.F.T + self.Q
        
        # 更新阶段
        innovation = measurement - self.H @ pred_state
        innovation_cov = self.H @ pred_P @ self.H.T + self.R
        K = pred_P @ self.H.T @ np.linalg.inv(innovation_cov)  # 卡尔曼增益
        
        self.state = pred_state + K @ innovation
        self.P = (np.eye(self.state_dim) - K @ self.H) @ pred_P

    def get_state(self):
        """
        返回当前状态 (4x4 外参矩阵形式)
        """
        return self.state.reshape(4, 4)
