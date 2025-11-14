"""
ONNX推理引擎
用于加载和运行ONNX格式的围棋模型
"""

import onnxruntime as ort
import numpy as np
from typing import Tuple
from go_engine import GoEngine


class ONNXGoPlayer:
    """ONNX围棋AI玩家"""

    def __init__(self, model_path: str, board_size: int = 19):
        """
        初始化ONNX推理引擎
        Args:
            model_path: ONNX模型路径
            board_size: 棋盘大小
        """
        self.board_size = board_size
        self.model_path = model_path

        # 检查模型文件是否存在
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 检查文件大小
        file_size = os.path.getsize(model_path)
        print(f"Model file size: {file_size} bytes")

        # 检查相关的.data文件是否存在
        data_file_path = model_path + ".data"
        if os.path.exists(data_file_path):
            data_file_size = os.path.getsize(data_file_path)
            print(f"Found associated data file: {data_file_path} ({data_file_size} bytes)")
        else:
            print(f"No associated data file found for: {model_path}")

        # 加载ONNX模型
        print(f"Loading ONNX model from {model_path}...")
        try:
            # 显式指定使用CPU执行提供者，避免可能的编码问题
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"Failed to load model: {e}")
            # 尝试提供更多调试信息
            import traceback
            traceback.print_exc()
            raise

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.policy_output_name = self.session.get_outputs()[0].name
        self.value_output_name = self.session.get_outputs()[1].name

        print(f"Model loaded successfully!")
        print(f"Input: {self.input_name}")
        print(f"Outputs: {self.policy_output_name}, {self.value_output_name}")

    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预测落子策略和局面价值
        Args:
            board_state: 棋盘状态 (3, board_size, board_size)
        Returns:
            policy: 落子概率分布 (board_size * board_size,)
            value: 局面价值 [-1, 1]
        """
        # 确保输入形状正确
        if board_state.ndim == 3:
            board_state = np.expand_dims(board_state, 0)  # 添加batch维度

        # 转换为float32
        board_state = board_state.astype(np.float32)

        # 运行推理
        outputs = self.session.run(
            [self.policy_output_name, self.value_output_name],
            {self.input_name: board_state}
        )

        policy_logits = outputs[0][0]  # (board_size * board_size,)
        value = outputs[1][0][0]  # scalar

        # 将logits转换为概率
        policy = self.softmax(policy_logits)

        return policy, value

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def select_move(self, engine: GoEngine, temperature: float = 0.1) -> Tuple[int, int]:
        """
        选择落子位置
        Args:
            engine: 围棋引擎
            temperature: 温度参数（越小越确定性，越大越随机）
        Returns:
            move: (row, col) 或 (-1, -1) 表示pass
        """
        # 获取棋盘状态
        board_state = engine.get_board_state()

        # 预测
        policy, value = self.predict(board_state)

        # 获取合法落子
        valid_moves = engine.get_valid_moves(engine.current_player)

        # 屏蔽非法落子
        legal_mask = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for move in valid_moves:
            if move != (-1, -1):
                move_idx = move[0] * self.board_size + move[1]
                legal_mask[move_idx] = 1.0

        # 应用合法性屏蔽
        policy = policy * legal_mask
        policy_sum = policy.sum()

        if policy_sum > 0:
            policy = policy / policy_sum

            # 应用温度
            if temperature != 1.0:
                policy = np.power(policy, 1.0 / temperature)
                policy = policy / policy.sum()

            # 选择概率最高的落子（贪婪策略）
            if temperature < 0.01:
                move_idx = np.argmax(policy)
            else:
                # 按概率采样
                move_idx = np.random.choice(self.board_size * self.board_size, p=policy)

            move = (move_idx // self.board_size, move_idx % self.board_size)
        else:
            # 没有合法落子，pass
            move = (-1, -1)

        return move

    def get_move_probabilities(self, engine: GoEngine) -> np.ndarray:
        """
        获取所有位置的落子概率（用于可视化）
        Args:
            engine: 围棋引擎
        Returns:
            probabilities: (board_size, board_size) 每个位置的概率
        """
        board_state = engine.get_board_state()
        policy, _ = self.predict(board_state)

        # 获取合法落子
        valid_moves = engine.get_valid_moves(engine.current_player)

        # 屏蔽非法落子
        legal_mask = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        for move in valid_moves:
            if move != (-1, -1):
                move_idx = move[0] * self.board_size + move[1]
                legal_mask[move_idx] = 1.0

        policy = policy * legal_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum

        # 重塑为棋盘形状
        probabilities = policy.reshape(self.board_size, self.board_size)
        return probabilities


def test_onnx_model(model_path: str):
    """
    测试ONNX模型
    Args:
        model_path: ONNX模型路径
    """
    print("Testing ONNX model...")

    # 创建玩家
    player = ONNXGoPlayer(model_path)

    # 创建围棋引擎
    engine = GoEngine()

    # 测试几步落子
    for i in range(5):
        print(f"\n--- Move {i+1} ---")
        print(f"Current player: {'Black' if engine.current_player == 1 else 'White'}")

        # 选择落子
        move = player.select_move(engine, temperature=0.5)
        print(f"Selected move: {move}")

        # 执行落子
        if not engine.make_move(move[0], move[1]):
            print("Invalid move!")
            break

        # 显示棋盘
        print("\nBoard:")
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in range(engine.board_size):
            print(' '.join(symbols[engine.board[row, col]] for col in range(engine.board_size)))

    print("\nONNX model test completed!")


if __name__ == '__main__':
    # 测试模型
    test_onnx_model('models/go_model.onnx')
