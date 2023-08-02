import numpy as np
from pykalman import KalmanFilter

import matplotlib.pyplot as plt

def Kalman_traj_smooth(data, process_noise_std, measurement_noise_std):
    '''
    使用卡尔曼滤波器对轨迹数据进行平滑处理
    
    参数
    ----
    data : DataFrame
        轨迹数据，包含time、lon、lat三列
    process_noise_std : float or list
        过程噪声标准差，如果是list，则认为是过程噪声协方差矩阵的对角线元素
    measurement_noise_std : float or list
        观测噪声标准差，如果是list，则认为是观测噪声协方差矩阵的对角线元素

    返回
    ----
    data : DataFrame
        平滑后的轨迹数据
    '''
    # 拷贝数据，避免修改原始数据
    data = data.copy()
    # 轨迹数据转换为numpy数组
    observations = data[['X', 'Y']].values
    timestamps = data['TIMESTAMP']
    # F-状态转移矩阵
    transition_matrix = np.array([[1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    # H-观测矩阵
    observation_matrix = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0]])
    # R-观测噪声协方差矩阵
    # 如果measurement_noise_std是list，则认为是观测噪声协方差矩阵的对角线元素
    if isinstance(measurement_noise_std, list):
        observation_covariance = np.diag(measurement_noise_std)**2
    else:
        observation_covariance = np.eye(2) * measurement_noise_std**2
    # Q-过程噪声协方差矩阵
    # 如果process_noise_std是list，则认为是过程噪声协方差矩阵的对角线元素
    if isinstance(process_noise_std, list):
        transition_covariance = np.diag(process_noise_std)**2
    else:
        transition_covariance = np.eye(4) * process_noise_std**2
    # 初始状态
    initial_state_mean = [observations[0, 0], observations[0, 1], 0, 0]
    # 初始状态协方差矩阵
    initial_state_covariance = np.eye(4) * 1
    # 初始化卡尔曼滤波器
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    # 使用卡尔曼滤波器进行平滑处理
    # 先创建变量存储平滑后的状态
    smoothed_states = np.zeros((len(observations), 4))
    # 将初始状态存储到平滑后的状态中
    smoothed_states[0, :] = initial_state_mean
    # 从第二个状态开始，进行循环迭代
    current_state = initial_state_mean
    current_covariance = initial_state_covariance
    if len(observations) <= 2:
        return data
    for i in range(1, len(observations)):
        # 计算时间间隔
        # dt = (timestamps.iloc[i] - timestamps.iloc[i - 1]).total_seconds()  
        dt = (timestamps.iloc[i] - timestamps.iloc[i - 1])
        # 更新状态转移矩阵
        kf.transition_matrices = np.array([[1, 0, dt, 0],
                                           [0, 1, 0, dt],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        # 根据当前状态的预测情况与观测结果进行状态估计
        current_state, current_covariance = kf.filter_update(
            current_state, current_covariance, observations[i]
        )
        # 将平滑后的状态存储到变量中
        smoothed_states[i, :] = current_state 
    # 将平滑后的数据结果添加到原始数据中
    
    
    # plt.figure(figsize=(10,10), dpi=400)
    # source_x = data['X'].values
    # source_y = data['Y'].values
    # plt.plot(source_x, source_y, color='b', label="source")
    # smooth_x = smoothed_states[:, 0]
    # smooth_y = smoothed_states[:, 1]
    # plt.plot(smooth_x, smooth_y, color='g', label="smooth")
    # plt.legend()
    # plt.savefig("./smooth.png")
    
    
    data['X'] = smoothed_states[:, 0]
    data['Y'] = smoothed_states[:, 1]
    
    
    return data
