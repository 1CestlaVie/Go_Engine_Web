"""
配置文件
包含模型训练和对弈的各种参数
"""

# 棋盘设置
BOARD_SIZE = 19  # 棋盘大小 (9, 13, 或 19)

# 模型架构参数
MODEL_CONFIG = {
    'num_channels': 128,  # 卷积通道数
    'num_residual_blocks': 5,  # 残差块数量（增加可提升性能但训练更慢）
}

# 训练参数
TRAIN_CONFIG = {
    'num_games': 100,  # 生成的训练游戏数量
    'num_epochs': 20,  # 训练轮数
    'batch_size': 32,  # 批次大小
    'learning_rate': 0.001,  # 学习率
    'use_gpu': True,  # 是否使用GPU（如果可用）
}

# 自我对弈参数
SELFPLAY_CONFIG = {
    'temperature': 1.0,  # 温度参数（控制探索程度，越大越随机）
    'num_games': 100,  # 自我对弈局数
}

# GUI设置
GUI_CONFIG = {
    'cell_size': 30,  # 每个格子的像素大小
    'move_delay': 500,  # 自动落子延迟（毫秒）
    'show_probabilities': False,  # 是否显示落子概率
}

# 对弈设置
BATTLE_CONFIG = {
    'num_games': 5,  # 对弈局数
    'max_moves': 400,  # 单局最大手数
}

# 文件路径
PATHS = {
    'models_dir': 'models',
    'default_model': 'models/go_model.pth',
    'default_onnx': 'models/go_model.onnx',
}
