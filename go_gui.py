"""
围棋对弈可视化GUI平台
使用pygame实现，支持两个ONNX模型对弈
"""

import pygame
import sys
import time
from typing import Optional, List, Tuple
from go_engine import GoEngine
from onnx_inference import ONNXGoPlayer


class GoGUI:
    """围棋GUI界面"""

    def __init__(self, board_size: int = 19, cell_size: int = 30):
        """
        初始化GUI
        Args:
            board_size: 棋盘大小
            cell_size: 每个格子的像素大小
        """
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 50
        self.info_panel_width = 300

        # 计算窗口大小
        self.board_width = cell_size * (board_size - 1)
        self.window_width = self.board_width + 2 * self.margin + self.info_panel_width
        self.window_height = self.board_width + 2 * self.margin

        # 颜色定义
        self.COLOR_BG = (220, 179, 92)  # 棋盘背景色
        self.COLOR_LINE = (0, 0, 0)  # 线条颜色
        self.COLOR_BLACK = (0, 0, 0)  # 黑棋
        self.COLOR_WHITE = (255, 255, 255)  # 白棋
        self.COLOR_STAR = (0, 0, 0)  # 星位
        self.COLOR_LAST_MOVE = (255, 0, 0)  # 最后落子标记
        self.COLOR_INFO_BG = (240, 240, 240)  # 信息面板背景
        self.COLOR_TEXT = (0, 0, 0)  # 文字颜色

        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Go AI Battle Platform")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 32)

        # 游戏状态
        self.engine = GoEngine(board_size)
        self.last_move = None
        self.game_paused = False
        self.auto_play = True
        self.move_delay = 500  # 自动对弈延迟(ms)

    def board_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        """将棋盘坐标转换为屏幕坐标"""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        return x, y

    def screen_to_board(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """将屏幕坐标转换为棋盘坐标"""
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row, col
        return None

    def draw_board(self):
        """绘制棋盘"""
        # 背景
        self.screen.fill(self.COLOR_BG)

        # 信息面板背景
        pygame.draw.rect(self.screen, self.COLOR_INFO_BG,
                        (self.board_width + 2 * self.margin, 0,
                         self.info_panel_width, self.window_height))

        # 网格线
        for i in range(self.board_size):
            # 横线
            start_x, start_y = self.board_to_screen(i, 0)
            end_x, end_y = self.board_to_screen(i, self.board_size - 1)
            pygame.draw.line(self.screen, self.COLOR_LINE, (start_x, start_y), (end_x, end_y), 1)

            # 竖线
            start_x, start_y = self.board_to_screen(0, i)
            end_x, end_y = self.board_to_screen(self.board_size - 1, i)
            pygame.draw.line(self.screen, self.COLOR_LINE, (start_x, start_y), (end_x, end_y), 1)

        # 星位
        star_positions = []
        if self.board_size == 19:
            star_positions = [(3, 3), (3, 9), (3, 15),
                            (9, 3), (9, 9), (9, 15),
                            (15, 3), (15, 9), (15, 15)]
        elif self.board_size == 13:
            star_positions = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif self.board_size == 9:
            star_positions = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]

        for row, col in star_positions:
            x, y = self.board_to_screen(row, col)
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), 4)

    def draw_stones(self):
        """绘制棋子"""
        stone_radius = int(self.cell_size * 0.45)

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.engine.board[row, col] != 0:
                    x, y = self.board_to_screen(row, col)
                    color = self.COLOR_BLACK if self.engine.board[row, col] == 1 else self.COLOR_WHITE

                    # 绘制棋子
                    pygame.draw.circle(self.screen, color, (x, y), stone_radius)
                    pygame.draw.circle(self.screen, self.COLOR_LINE, (x, y), stone_radius, 1)

                    # 标记最后落子
                    if self.last_move == (row, col):
                        marker_radius = int(stone_radius * 0.3)
                        marker_color = self.COLOR_WHITE if self.engine.board[row, col] == 1 else self.COLOR_BLACK
                        pygame.draw.circle(self.screen, marker_color, (x, y), marker_radius)

    def draw_info(self, player1_name: str = "Model 1", player2_name: str = "Model 2",
                  game_number: int = 1, total_games: int = 5, scores: Tuple[int, int] = (0, 0)):
        """
        绘制信息面板
        Args:
            player1_name: 玩家1名称
            player2_name: 玩家2名称
            game_number: 当前对局数
            total_games: 总对局数
            scores: (玩家1得分, 玩家2得分)
        """
        info_x = self.board_width + 2 * self.margin + 10
        y_offset = 20

        # 标题
        title = self.font_large.render("AI Battle", True, self.COLOR_TEXT)
        self.screen.blit(title, (info_x, y_offset))
        y_offset += 50

        # 对局信息
        game_info = self.font.render(f"Game {game_number}/{total_games}", True, self.COLOR_TEXT)
        self.screen.blit(game_info, (info_x, y_offset))
        y_offset += 40

        # 当前玩家
        current_player = "Black" if self.engine.current_player == 1 else "White"
        player_text = self.font.render(f"Turn: {current_player}", True, self.COLOR_TEXT)
        self.screen.blit(player_text, (info_x, y_offset))
        y_offset += 40

        # 玩家1信息（黑方）
        p1_text = self.font.render(f"{player1_name} (Black)", True, self.COLOR_TEXT)
        self.screen.blit(p1_text, (info_x, y_offset))
        y_offset += 25
        p1_score = self.font_small.render(f"Wins: {scores[0]}", True, self.COLOR_TEXT)
        self.screen.blit(p1_score, (info_x, y_offset))
        y_offset += 35

        # 玩家2信息（白方）
        p2_text = self.font.render(f"{player2_name} (White)", True, self.COLOR_TEXT)
        self.screen.blit(p2_text, (info_x, y_offset))
        y_offset += 25
        p2_score = self.font_small.render(f"Wins: {scores[1]}", True, self.COLOR_TEXT)
        self.screen.blit(p2_score, (info_x, y_offset))
        y_offset += 40

        # 手数
        move_count = len(self.engine.history)
        move_text = self.font_small.render(f"Moves: {move_count}", True, self.COLOR_TEXT)
        self.screen.blit(move_text, (info_x, y_offset))
        y_offset += 30

        # 提子数
        captured_text = self.font_small.render("Captured:", True, self.COLOR_TEXT)
        self.screen.blit(captured_text, (info_x, y_offset))
        y_offset += 20
        cap_black = self.font_small.render(f"  Black: {self.engine.captured_stones[1]}", True, self.COLOR_TEXT)
        self.screen.blit(cap_black, (info_x, y_offset))
        y_offset += 20
        cap_white = self.font_small.render(f"  White: {self.engine.captured_stones[2]}", True, self.COLOR_TEXT)
        self.screen.blit(cap_white, (info_x, y_offset))
        y_offset += 40

        # 控制说明
        controls_title = self.font.render("Controls:", True, self.COLOR_TEXT)
        self.screen.blit(controls_title, (info_x, y_offset))
        y_offset += 25

        controls = [
            "SPACE: Pause/Resume",
            "N: Next Move",
            "R: Reset Game",
            "Q: Quit"
        ]
        for control in controls:
            text = self.font_small.render(control, True, self.COLOR_TEXT)
            self.screen.blit(text, (info_x, y_offset))
            y_offset += 20

    def update(self):
        """更新显示"""
        pygame.display.flip()

    def render(self, player1_name: str = "Model 1", player2_name: str = "Model 2",
              game_number: int = 1, total_games: int = 5, scores: Tuple[int, int] = (0, 0)):
        """渲染整个界面"""
        self.draw_board()
        self.draw_stones()
        self.draw_info(player1_name, player2_name, game_number, total_games, scores)
        self.update()

    def show_game_result(self, winner: str, black_score: float, white_score: float):
        """显示对局结果"""
        # 半透明遮罩
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # 结果文字
        result_text = self.font_large.render(f"Winner: {winner}", True, (255, 255, 255))
        score_text = self.font.render(f"Black: {black_score:.1f}  White: {white_score:.1f}",
                                     True, (255, 255, 255))

        # 居中显示
        result_rect = result_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 20))
        score_rect = score_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 20))

        self.screen.blit(result_text, result_rect)
        self.screen.blit(score_text, score_rect)

        self.update()


def run_ai_battle(model1_path: str, model2_path: str, num_games: int = 5,
                  board_size: int = 19, move_delay: int = 500):
    """
    运行AI对弈
    Args:
        model1_path: 模型1的ONNX路径
        model2_path: 模型2的ONNX路径
        num_games: 对弈局数
        board_size: 棋盘大小
        move_delay: 落子延迟(ms)
    """
    # 创建GUI
    gui = GoGUI(board_size)

    # 加载两个模型
    print(f"Loading Model 1 from {model1_path}...")
    player1 = ONNXGoPlayer(model1_path, board_size)

    print(f"Loading Model 2 from {model2_path}...")
    player2 = ONNXGoPlayer(model2_path, board_size)

    # 得分统计
    scores = [0, 0]  # [player1_wins, player2_wins]

    # 进行多局对弈
    for game_num in range(1, num_games + 1):
        print(f"\n=== Game {game_num}/{num_games} ===")

        # 重置棋盘
        gui.engine.reset()
        gui.last_move = None

        # 对局循环
        running = True
        paused = False
        last_move_time = pygame.time.get_ticks()
        max_moves = 400  # 最大手数

        while running:
            current_time = pygame.time.get_ticks()

            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_r:
                        gui.engine.reset()
                        gui.last_move = None

            # 渲染界面
            gui.render(
                player1_name="Model 1",
                player2_name="Model 2",
                game_number=game_num,
                total_games=num_games,
                scores=tuple(scores)
            )

            # 检查游戏是否结束
            if gui.engine.is_game_over() or len(gui.engine.history) >= max_moves:
                # 计算得分
                black_score, white_score = gui.engine.simple_score()
                winner = "Black (Model 1)" if black_score > white_score else "White (Model 2)"

                if black_score > white_score:
                    scores[0] += 1
                else:
                    scores[1] += 1

                print(f"Game Over! Winner: {winner}")
                print(f"Score - Black: {black_score:.1f}, White: {white_score:.1f}")

                # 显示结果
                gui.show_game_result(winner, black_score, white_score)
                pygame.time.wait(3000)  # 等待3秒
                break

            # 自动落子
            if not paused and current_time - last_move_time >= move_delay:
                # 选择当前玩家
                current_player = player1 if gui.engine.current_player == 1 else player2
                player_name = "Model 1" if gui.engine.current_player == 1 else "Model 2"

                # 选择落子
                move = current_player.select_move(gui.engine, temperature=0.3)
                print(f"{player_name} plays: {move}")

                # 执行落子
                if gui.engine.make_move(move[0], move[1]):
                    gui.last_move = move if move != (-1, -1) else None
                    last_move_time = current_time
                else:
                    print(f"Invalid move: {move}")

            gui.clock.tick(60)  # 60 FPS

    # 显示最终结果
    print(f"\n=== Final Results ===")
    print(f"Model 1 wins: {scores[0]}")
    print(f"Model 2 wins: {scores[1]}")

    # 等待关闭
    print("\nPress Q to quit...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                waiting = False
        gui.clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    # 测试GUI（需要先训练模型）
    run_ai_battle(
        model1_path='models/go_model.onnx',
        model2_path='models/go_model.onnx',
        num_games=5,
        board_size=19,
        move_delay=500
    )
