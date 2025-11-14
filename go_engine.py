"""
围棋规则引擎
实现围棋的基本规则，包括落子、提子、打劫等
"""

import numpy as np
from typing import Tuple, Set, Optional, List

class GoEngine:
    """围棋规则引擎"""

    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)  # 0: 空, 1: 黑, 2: 白
        self.current_player = 1  # 1: 黑, 2: 白
        self.ko_point = None  # 打劫点
        self.history = []  # 历史状态
        self.captured_stones = {1: 0, 2: 0}  # 提子数
        self.pass_count = 0  # 连续pass次数

    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.ko_point = None
        self.history = []
        self.captured_stones = {1: 0, 2: 0}
        self.pass_count = 0

    def copy(self):
        """复制当前状态"""
        new_engine = GoEngine(self.board_size)
        new_engine.board = self.board.copy()
        new_engine.current_player = self.current_player
        new_engine.ko_point = self.ko_point
        new_engine.history = self.history.copy()
        new_engine.captured_stones = self.captured_stones.copy()
        new_engine.pass_count = self.pass_count
        return new_engine

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """获取邻居位置"""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                neighbors.append((r, c))
        return neighbors

    def get_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """获取连通的棋子组"""
        if self.board[row, col] == 0:
            return set()

        color = self.board[row, col]
        group = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))

            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr, nc] == color and (nr, nc) not in group:
                    stack.append((nr, nc))

        return group

    def get_liberties(self, group: Set[Tuple[int, int]]) -> int:
        """计算棋子组的气"""
        liberties = set()
        for r, c in group:
            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr, nc] == 0:
                    liberties.add((nr, nc))
        return len(liberties)

    def remove_group(self, group: Set[Tuple[int, int]]):
        """提子"""
        for r, c in group:
            self.board[r, c] = 0

    def is_valid_move(self, row: int, col: int, player: int) -> bool:
        """检查落子是否合法"""
        # 检查是否在棋盘内
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False

        # 检查位置是否为空
        if self.board[row, col] != 0:
            return False

        # 检查是否是打劫点
        if self.ko_point == (row, col):
            return False

        # 模拟落子，检查是否自杀
        temp_engine = self.copy()
        temp_engine.board[row, col] = player

        # 先提掉对方没气的棋子
        opponent = 3 - player
        captured_any = False
        for nr, nc in temp_engine.get_neighbors(row, col):
            if temp_engine.board[nr, nc] == opponent:
                group = temp_engine.get_group(nr, nc)
                if temp_engine.get_liberties(group) == 0:
                    temp_engine.remove_group(group)
                    captured_any = True

        # 检查自己的棋子组是否有气
        my_group = temp_engine.get_group(row, col)
        if temp_engine.get_liberties(my_group) == 0:
            return False  # 自杀，不合法

        return True

    def make_move(self, row: int, col: int) -> bool:
        """落子"""
        if row == -1 and col == -1:  # Pass
            self.pass_count += 1
            self.current_player = 3 - self.current_player
            self.ko_point = None
            self.history.append(self.board.copy())
            return True

        self.pass_count = 0

        if not self.is_valid_move(row, col, self.current_player):
            return False

        # 落子
        self.board[row, col] = self.current_player

        # 提掉对方没气的棋子
        opponent = 3 - self.current_player
        captured_groups = []
        for nr, nc in self.get_neighbors(row, col):
            if self.board[nr, nc] == opponent:
                group = self.get_group(nr, nc)
                if self.get_liberties(group) == 0:
                    captured_groups.append(group)

        # --- [ 修复BUG ] ---
        # 1. 提子 (必须在打劫检测之前)
        for group in captured_groups:
            self.captured_stones[self.current_player] += len(group)
            self.remove_group(group)

        # 2. 更新打劫点 (现在 get_liberties 是正确的)
        self.ko_point = None
        if len(captured_groups) == 1 and len(captured_groups[0]) == 1:
            # 可能是打劫
            my_group = self.get_group(row, col)
            if len(my_group) == 1 and self.get_liberties(my_group) == 1:
                # 确实是打劫
                ko_r, ko_c = list(captured_groups[0])[0]
                self.ko_point = (ko_r, ko_c)
        # --- [ 修复结束 ] ---

        # 切换玩家
        self.current_player = 3 - self.current_player
        self.history.append(self.board.copy())

        return True

    def is_game_over(self) -> bool:
        """判断游戏是否结束（双方连续pass）"""
        return self.pass_count >= 2

    def get_valid_moves(self, player: int) -> List[Tuple[int, int]]:
        """获取所有合法落子位置"""
        valid_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.is_valid_move(r, c, player):
                    valid_moves.append((r, c))
        valid_moves.append((-1, -1))  # Pass
        return valid_moves

    def get_board_state(self) -> np.ndarray:
        """
        获取棋盘状态，用于神经网络输入
        返回形状: (3, board_size, board_size)
        通道0: 当前玩家的棋子
        通道1: 对手的棋子
        通道2: 全1矩阵（表示当前玩家）
        """
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == (3 - self.current_player)).astype(np.float32)
        state[2] = np.ones((self.board_size, self.board_size), dtype=np.float32)
        return state

    def simple_score(self) -> Tuple[float, float]:
        """
        简单计分（中国规则）
        返回: (黑方得分, 白方得分)
        """
        black_score = 0
        white_score = 7.5  # 贴目

        # 计算棋子数和地
        visited = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in visited:
                    continue

                if self.board[r, c] == 1:
                    black_score += 1
                    visited.add((r, c))
                elif self.board[r, c] == 2:
                    white_score += 1
                    visited.add((r, c))
                else:
                    # 计算空点属于谁
                    territory = self.get_territory(r, c)
                    visited.update(territory)
                    owner = self.get_territory_owner(territory)
                    if owner == 1:
                        black_score += len(territory)
                    elif owner == 2:
                        white_score += len(territory)

        return black_score, white_score

    def get_territory(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """获取连通的空白区域"""
        if self.board[row, col] != 0:
            return set()

        territory = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in territory or self.board[r, c] != 0:
                continue
            territory.add((r, c))

            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr, nc] == 0 and (nr, nc) not in territory:
                    stack.append((nr, nc))

        return territory

    def get_territory_owner(self, territory: Set[Tuple[int, int]]) -> int:
        """判断领地属于谁 (1: 黑, 2: 白, 0: 未定)"""
        adjacent_colors = set()
        for r, c in territory:
            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr, nc] != 0:
                    adjacent_colors.add(self.board[nr, nc])

        if len(adjacent_colors) == 1:
            return list(adjacent_colors)[0]
        return 0  # 未定
