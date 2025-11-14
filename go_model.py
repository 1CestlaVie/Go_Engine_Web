"""
围棋神经网络模型（基于PyTorch）
(v1.2 - 修复ONNX导出 + 添加Dropout)

采用类似AlphaGo Zero的架构：卷积神经网络 + 残差块
输入: (batch, 3, 19, 19) - 棋盘状态
输出:
  - policy: (batch, 361) - 落子概率（19x19个位置）
  - value: (batch, 1) - 局面评估值 [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GoModel(nn.Module):
    """围棋神经网络模型"""

    def __init__(self, board_size: int = 19, num_channels: int = 128, num_residual_blocks: int = 10):
        super(GoModel, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels

        # 初始卷积层
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])

        # --- [ 新增 ] Dropout 层 ---
        # p=0.5 表示在训练时随机丢弃 50% 的神经元
        self.dropout = nn.Dropout(p=0.5)
        # --- [ 结束 ] ---

        # 策略头 (Policy Head)
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # 价值头 (Value Head)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch, 3, board_size, board_size) - 棋盘状态
        Returns:
            policy: (batch, board_size * board_size) - 落子概率logits
            value: (batch, 1) - 局面评估值
        """
        # 初始卷积
        out = F.relu(self.bn_input(self.conv_input(x)))

        # 残差块
        for block in self.residual_blocks:
            out = block(out)

        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        # --- [ 新增 ] 在全连接层前应用 Dropout ---
        # 注意: model.train() 时激活, model.eval() 时自动关闭
        policy = self.dropout(policy)
        policy = self.policy_fc(policy)  # 输出logits

        # 价值头
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        # --- [ 新增 ] 在全连接层前应用 Dropout ---
        value = self.dropout(value)
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def predict(self, board_state):
        """
        预测单个棋盘状态
        (已修复设备 bug)
        """
        self.eval() # self.eval() 会自动关闭 Dropout
        with torch.no_grad():
            if isinstance(board_state, np.ndarray):
                board_state = torch.FloatTensor(board_state)

            board_state = board_state.unsqueeze(0)  # 添加batch维度

            # 自动检测设备
            device = next(self.parameters()).device
            board_state = board_state.to(device)

            policy_logits, value = self.forward(board_state)

            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()

        return policy, value


import numpy as np


def create_model(board_size: int = 19, num_channels: int = 128, num_residual_blocks: int = 10):
    """创建模型"""
    return GoModel(board_size, num_channels, num_residual_blocks)


def export_to_onnx(model: GoModel, output_path: str, board_size: int = 19):
    """
    导出模型到ONNX格式
    (已修复设备 bug)
    """
    model.eval() # self.eval() 会自动关闭 Dropout

    # --- [ Bug 修复 ] ---
    # 1. 自动检测模型所在的设备 (是 'cpu' 还是 'cuda:0')
    device = next(model.parameters()).device

    # 2. 在同一个设备上创建示例输入
    dummy_input = torch.randn(1, 3, board_size, board_size, device=device)
    # --- [ 修复结束 ] ---

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")


def load_model(checkpoint_path: str, board_size: int = 19, num_channels: int = 128, num_residual_blocks: int = 10):
    """
    加载模型
    Args:
        checkpoint_path: 检查点路径
        board_size: 棋盘大小
        num_channels: 通道数
        num_residual_blocks: 残差块数量
    Returns:
        model: 加载的模型
    """
    # [ 修复 ] 确保加载模型时使用正确的参数
    # 尝试从检查点读取参数
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # 从检查点获取参数(如果存在)，否则使用默认值
    board_size = checkpoint.get('board_size', board_size)
    num_channels = checkpoint.get('num_channels', num_channels)
    num_residual_blocks = checkpoint.get('num_residual_blocks', num_residual_blocks)

    model = create_model(board_size, num_channels, num_residual_blocks)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model