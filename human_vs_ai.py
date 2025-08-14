import sys
import os
import time
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QFrame, QGroupBox, QFormLayout,
    QLineEdit, QComboBox, QCheckBox, QTextEdit, QSplitter
)
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtCore import Qt, QSize, QTimer
import numpy as np
import torch as th

# 【修复】导入游戏环境的路径
from .environment import (GameEnvironment, PieceType, SQ_TO_POS, POS_TO_SQ,
                              ACTION_SPACE_SIZE, REVEAL_ACTIONS_COUNT, REGULAR_MOVE_ACTIONS_COUNT,
                              MAX_CONSECUTIVE_MOVES_FOR_DRAW)

# 导入新的AI模型
try:
    from .net_model import Model, CustomNetwork
    AI_AVAILABLE = True
    print("AI模型支持已加载")
except ImportError:
    AI_AVAILABLE = False
    print("警告: 未找到AI模型库(model.py)，只能进行人人对战")

class BitboardGridWidget(QWidget):
    """一个专门用于可视化单个bitboard的4x4网格小部件。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(1)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.labels = []

        # 创建16个小方块来代表bitboard的16个位
        for i in range(16):
            label = QLabel()
            label.setFixedSize(15, 15)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: #DDDDDD; border: 1px solid #BBBBBB;")
            self.labels.append(label)
            row, col = SQ_TO_POS[i]
            self.grid_layout.addWidget(label, row, col)

        self.setLayout(self.grid_layout)

    def update_bitboard(self, bb_value: int):
        """根据传入的bitboard整数值更新网格颜色。"""
        for i in range(16):
            if (bb_value >> i) & 1:
                self.labels[i].setStyleSheet("background-color: #4CAF50; border: 1px solid #388E3C;")
            else:
                self.labels[i].setStyleSheet("background-color: #DDDDDD; border: 1px solid #BBBBBB;")

class MainWindow(QMainWindow):
    """主窗口，支持人机对战的暗棋游戏。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.game = GameEnvironment()

        # GUI 内部状态
        self.selected_from_sq = None
        self.valid_action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=int)

        # AI 相关状态
        self.ai_model = None
        self.ai_player = None  # None表示人人对战，1或-1表示AI控制的玩家
        self.ai_thinking = False
        self.game_over = False

        # AI思考计时器
        self.ai_timer = QTimer()
        self.ai_timer.setSingleShot(True)
        self.ai_timer.timeout.connect(self.make_ai_move)

        self.setWindowTitle("暗棋 - 人机对战")
        self.setGeometry(100, 100, 1400, 900)

        self.setup_ui()
        self.reset_game()

    def setup_ui(self):
        """设置用户界面。"""
        # 主分割器
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # --- 左侧面板：游戏控制 ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # AI设置组
        ai_group = QGroupBox("AI 设置")
        ai_layout = QFormLayout()

        # 游戏模式选择
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["人 vs 人", "人 vs AI (你是红方)", "人 vs AI (你是黑方)", "AI vs AI"])
        if not AI_AVAILABLE:
            self.mode_combo.setEnabled(False)
            self.mode_combo.setToolTip("需要安装 sb3-contrib 来启用AI功能")
        ai_layout.addRow("游戏模式:", self.mode_combo)

        # AI模型路径
        self.model_path_edit = QLineEdit("./models/self_play_final/model.pt")
        self.model_path_edit.setPlaceholderText("输入AI模型文件路径")
        ai_layout.addRow("AI模型路径:", self.model_path_edit)

        # 加载AI按钮
        self.load_ai_button = QPushButton("加载 AI 模型")
        self.load_ai_button.clicked.connect(self.load_ai_model)
        if not AI_AVAILABLE:
            self.load_ai_button.setEnabled(False)
        ai_layout.addRow(self.load_ai_button)

        # AI状态显示
        self.ai_status_label = QLabel("AI状态: 未加载")
        ai_layout.addRow(self.ai_status_label)

        # AI思考延迟
        self.ai_delay_edit = QLineEdit("500")
        self.ai_delay_edit.setPlaceholderText("毫秒")
        ai_layout.addRow("AI思考延迟:", self.ai_delay_edit)

        ai_group.setLayout(ai_layout)

        # 游戏控制组
        control_group = QGroupBox("游戏控制")
        control_layout = QVBoxLayout()

        self.new_game_button = QPushButton("开始新游戏")
        self.new_game_button.clicked.connect(self.new_game)
        control_layout.addWidget(self.new_game_button)

        self.reset_button = QPushButton("重置当前游戏")
        self.reset_button.clicked.connect(self.reset_game)
        control_layout.addWidget(self.reset_button)

        control_group.setLayout(control_layout)

        # 游戏日志
        log_group = QGroupBox("游戏日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        left_layout.addWidget(ai_group)
        left_layout.addWidget(control_group)
        left_layout.addWidget(log_group)
        left_layout.addStretch()

        # --- 中间面板：棋盘 ---
        board_widget = QWidget()
        board_layout = QVBoxLayout(board_widget)

        # 棋盘
        self.board_buttons = []
        board_group = QGroupBox("游戏棋盘")
        board_grid_layout = QGridLayout()
        board_grid_layout.setSpacing(5)

        font = QFont("SimSun", 18, QFont.Weight.Bold)
        for r in range(4):
            row_buttons = []
            for c in range(4):
                button = QPushButton()
                button.setFixedSize(80, 80)
                button.setFont(font)
                button.clicked.connect(lambda _, pos=(r, c): self.on_board_click(pos))
                board_grid_layout.addWidget(button, r, c)
                row_buttons.append(button)
            self.board_buttons.append(row_buttons)

        board_group.setLayout(board_grid_layout)

        # 游戏状态显示
        status_group = QGroupBox("游戏状态")
        status_layout = QFormLayout()

        self.current_player_label = QLineEdit()
        self.current_player_label.setReadOnly(True)
        self.scores_label = QLineEdit()
        self.scores_label.setReadOnly(True)
        self.move_counter_label = QLineEdit()
        self.move_counter_label.setReadOnly(True)
        self.game_status_label = QLineEdit()
        self.game_status_label.setReadOnly(True)

        status_layout.addRow("当前玩家:", self.current_player_label)
        status_layout.addRow("得分 (红-黑):", self.scores_label)
        status_layout.addRow("连续未吃子:", self.move_counter_label)
        status_layout.addRow("游戏状态:", self.game_status_label)

        status_group.setLayout(status_layout)

        board_layout.addWidget(board_group)
        board_layout.addWidget(status_group)
        board_layout.addStretch()

        # --- 右侧面板：详细状态 ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 阵亡棋子
        dead_group = QGroupBox("阵亡棋子")
        dead_layout = QFormLayout()
        self.dead_pieces_red_label = QLineEdit()
        self.dead_pieces_red_label.setReadOnly(True)
        self.dead_pieces_black_label = QLineEdit()
        self.dead_pieces_black_label.setReadOnly(True)
        dead_layout.addRow("红方阵亡:", self.dead_pieces_red_label)
        dead_layout.addRow("黑方阵亡:", self.dead_pieces_black_label)
        dead_group.setLayout(dead_layout)

        # === 新增: 详细的Bitboards可视化 ===
        bitboard_group = QGroupBox("Bitboards 可视化")
        bitboard_main_layout = QVBoxLayout()

        # 通用 Bitboards
        self.hidden_bb_widget = BitboardGridWidget()
        self.empty_bb_widget = BitboardGridWidget()
        bb_common_layout = QFormLayout()
        bb_common_layout.addRow("Hidden Bitboard:", self.hidden_bb_widget)
        bb_common_layout.addRow("Empty Bitboard:", self.empty_bb_widget)
        bitboard_main_layout.addLayout(bb_common_layout)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        bitboard_main_layout.addWidget(line)

        # 玩家相关的 Bitboards
        player_bb_layout = QHBoxLayout()
        
        # 红方
        red_bb_group = QGroupBox("红方 (Player 1) Bitboards")
        self.red_bb_layout = QFormLayout()
        red_bb_group.setLayout(self.red_bb_layout)
        player_bb_layout.addWidget(red_bb_group)
        
        # 黑方
        black_bb_group = QGroupBox("黑方 (Player -1) Bitboards")
        self.black_bb_layout = QFormLayout()
        black_bb_group.setLayout(self.black_bb_layout)
        player_bb_layout.addWidget(black_bb_group)
        
        self.player_bb_widgets = {1: {}, -1: {}}
        for p, layout in [(1, self.red_bb_layout), (-1, self.black_bb_layout)]:
            revealed_widget = BitboardGridWidget()
            layout.addRow("Revealed BB:", revealed_widget)
            self.player_bb_widgets[p]['revealed'] = revealed_widget
            for pt in PieceType:
                pt_widget = BitboardGridWidget()
                layout.addRow(f"{pt.name} BB:", pt_widget)
                self.player_bb_widgets[p][pt.value] = pt_widget

        bitboard_main_layout.addLayout(player_bb_layout)
        bitboard_group.setLayout(bitboard_main_layout)

        right_layout.addWidget(dead_group)
        right_layout.addWidget(bitboard_group)
        right_layout.addStretch()

        # 添加到主分割器
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(board_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([350, 500, 550])

    def load_ai_model(self):
        """加载AI模型。"""
        if not AI_AVAILABLE:
            self.log_message("错误: 未安装AI模型库 (model.py)")
            return

        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self.log_message("错误: 请输入模型路径")
            return

        try:
            device = th.device("cuda" if th.cuda.is_available() else "cpu")
            self.ai_model = Model(self.game.observation_space, device)
            self.ai_model.network.load_state_dict(th.load(model_path, map_location=device))
            self.ai_status_label.setText("AI状态: 已加载")
            self.log_message(f"成功加载AI模型: {model_path}")
        except Exception as e:
            self.ai_status_label.setText("AI状态: 加载失败")
            self.log_message(f"加载AI模型失败: {str(e)}")

    def new_game(self):
        """开始新游戏，根据选择的模式设置AI。"""
        mode = self.mode_combo.currentText()

        if "AI" in mode and self.ai_model is None:
            self.log_message("错误: 请先加载AI模型")
            return

        # 设置AI玩家
        if mode == "人 vs 人":
            self.ai_player = None
        elif mode == "人 vs AI (你是红方)":
            self.ai_player = -1  # AI控制黑方
        elif mode == "人 vs AI (你是黑方)":
            self.ai_player = 1   # AI控制红方
        elif mode == "AI vs AI":
            self.ai_player = "both"  # AI控制双方

        self.log_message(f"--- 开始新游戏: {mode} ---")
        self.reset_game()


    def reset_game(self):
        """重置游戏状态。"""
        self.selected_from_sq = None
        self.ai_thinking = False
        self.game_over = False
        # 【修正】reset方法返回的info中包含初始动作掩码
        _, info = self.game.reset()
        self.valid_action_mask = info.get('action_mask', np.zeros(ACTION_SPACE_SIZE, dtype=int))
        self.update_gui()

        # 如果AI是红方或AI对战模式，让AI先行
        if not self.game_over and (self.ai_player == self.game.current_player or self.ai_player == "both"):
            self.schedule_ai_move()

    def on_board_click(self, pos):
        """处理棋盘点击事件 (已优化选择逻辑)。"""
        if self.game_over or self.ai_thinking:
            return

        # 检查是否轮到人类玩家
        if self.ai_player is not None and (self.ai_player == self.game.current_player or self.ai_player == "both"):
            return

        clicked_sq = POS_TO_SQ[pos]
        piece_at_click = self.game.board[clicked_sq]

        # --- 情况1: 未选择任何棋子 ---
        if self.selected_from_sq is None:
            # 尝试执行翻棋动作
            action_index = self.game.coords_to_action.get(pos)
            if action_index is not None and action_index < REVEAL_ACTIONS_COUNT:
                if self.valid_action_mask[action_index]:
                    self.make_move(action_index)
                    return

            # 如果点击的不是翻棋位置，则尝试选择一个己方棋子
            if piece_at_click and piece_at_click.revealed and piece_at_click.player == self.game.current_player:
                self.selected_from_sq = clicked_sq
                self.update_gui()
        
        # --- 情况2: 已选择一个棋子 ---
        else:
            from_pos = tuple(SQ_TO_POS[self.selected_from_sq])
            to_pos = tuple(pos)
            
            # 尝试执行移动/攻击
            action_index = self.game.coords_to_action.get((from_pos, to_pos))
            if action_index is not None and self.valid_action_mask[action_index]:
                self.make_move(action_index)
            # [UX优化] 如果点击的是另一个自己的棋子，则切换选择
            elif piece_at_click and piece_at_click.revealed and piece_at_click.player == self.game.current_player:
                self.selected_from_sq = clicked_sq
                self.update_gui()
            # 否则，取消选择
            else:
                self.selected_from_sq = None
                self.update_gui()

    def make_move(self, action_index):
        """
        执行一步棋并更新状态。
        """
        if self.game_over:
            return
        
        # 记录移动日志
        coords = self.game.action_to_coords.get(action_index)
        player_name = "红方" if self.game.current_player == 1 else "黑方"
        move_desc = ""
        if action_index < REVEAL_ACTIONS_COUNT:
            move_desc = f"翻开了 ({coords[0]}, {coords[1]}) 位置的棋子"
        else:
            from_sq = POS_TO_SQ[coords[0]]
            to_sq = POS_TO_SQ[coords[1]]
            attacker = self.game.board[from_sq]
            defender = self.game.board[to_sq]
            attacker_name = attacker.piece_type.name if attacker else "未知"
            if defender is None:
                move_desc = f"将 {attacker_name} 从 {coords[0]} 移动到 {coords[1]}"
            else:
                defender_name = defender.piece_type.name if defender else "未知"
                move_desc = f"用 {attacker_name} 从 {coords[0]} 吃掉了 {coords[1]} 的 {defender_name}"
        self.log_message(f"{player_name}: {move_desc}")
        
        # 执行动作，获取新状态
        _, _, terminated, truncated, info = self.game.step(action_index)
        self.selected_from_sq = None
        self.valid_action_mask = info.get('action_mask')
        winner = info.get('winner')

        if terminated or truncated:
            self.game_over = True
            if winner == 1:
                self.log_message("--- 游戏结束: 红方获胜! ---")
            elif winner == -1:
                self.log_message("--- 游戏结束: 黑方获胜! ---")
            else:
                self.log_message("--- 游戏结束: 平局! ---")

        self.update_gui()

        # 如果游戏没结束且轮到AI，安排AI移动
        if not self.game_over and self.ai_player is not None and \
           (self.ai_player == self.game.current_player or self.ai_player == "both"):
            self.schedule_ai_move()

    def schedule_ai_move(self):
        """安排AI移动。"""
        if self.game_over or self.ai_model is None:
            return

        delay = int(self.ai_delay_edit.text() or "500")
        self.ai_thinking = True
        self.update_gui() # 立即更新UI以显示“思考中”
        
        player_name = "红方" if self.game.current_player == 1 else "黑方"
        self.log_message(f"{player_name} (AI) 正在思考...")
        
        self.ai_timer.start(delay)

    def make_ai_move(self):
        """执行AI移动。"""
        if self.ai_model is None or self.game_over:
            self.ai_thinking = False
            return

        try:
            state = self.game.get_state()
            action_mask = self.valid_action_mask
            legal_actions = np.where(action_mask)[0]

            if len(legal_actions) == 0:
                self.log_message("AI发现无棋可走，游戏逻辑应已处理此情况。")
                self.ai_thinking = False
                return

            action = self.ai_model.predict(state, legal_actions)
            
            # AI移动后，立即清除思考状态
            self.ai_thinking = False
            self.make_move(int(action))

        except Exception as e:
            self.log_message(f"AI移动出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.ai_thinking = False
        finally:
            # 确保UI在任何情况下都能刷新
            self.update_gui()


    def log_message(self, message):
        """添加日志消息。"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def update_gui(self):
        """更新GUI显示。"""
        self.update_board_display()
        self.update_status_display()
        self.update_bitboard_display()

    def update_board_display(self):
        """更新棋盘显示 (已优化高亮逻辑)。"""
        red_map = {
            PieceType.GENERAL: "帥", PieceType.ADVISOR: "仕", PieceType.ELEPHANT: "相",
            PieceType.CHARIOT: "俥", PieceType.HORSE: "傌", PieceType.CANNON: "炮", PieceType.SOLDIER: "兵"
        }
        black_map = {
            PieceType.GENERAL: "將", PieceType.ADVISOR: "士", PieceType.ELEPHANT: "象",
            PieceType.CHARIOT: "車", PieceType.HORSE: "馬", PieceType.CANNON: "炮", PieceType.SOLDIER: "卒"
        }

        # 计算高亮目标
        reveal_targets = set()
        normal_move_targets = set()
        cannon_attack_targets = set()
        movable_pieces_pos = set()

        is_human_turn = (not self.ai_thinking and not self.game_over and
                         (self.ai_player is None or (self.ai_player != "both" and self.ai_player != self.game.current_player)))

        if is_human_turn:
            if self.selected_from_sq is not None:
                # --- 已选择棋子：高亮目标位置 ---
                from_pos_selected = tuple(SQ_TO_POS[self.selected_from_sq])
                selected_piece = self.game.board[self.selected_from_sq]
                is_cannon = selected_piece and selected_piece.piece_type == PieceType.CANNON

                for action_index, is_valid in enumerate(self.valid_action_mask):
                    if not is_valid: continue
                    coords = self.game.action_to_coords.get(action_index)
                    if coords and isinstance(coords, tuple) and len(coords) == 2 and isinstance(coords[0], tuple):
                        if tuple(coords[0]) == from_pos_selected:
                            target_pos = tuple(coords[1])
                            if is_cannon and action_index >= (REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT):
                                cannon_attack_targets.add(target_pos)
                            else:
                                normal_move_targets.add(target_pos)
            else:
                # --- 未选择棋子：高亮可翻开的棋子和可移动的棋子 ---
                for action_index, is_valid in enumerate(self.valid_action_mask):
                    if not is_valid: continue
                    coords = self.game.action_to_coords.get(action_index)
                    if not coords: continue

                    if action_index < REVEAL_ACTIONS_COUNT:
                        reveal_targets.add(coords)
                    else: # 移动或攻击动作
                        movable_pieces_pos.add(tuple(coords[0]))


        # 更新按钮显示
        for r in range(4):
            for c in range(4):
                sq = POS_TO_SQ[(r, c)]
                pos = (r, c)
                button = self.board_buttons[r][c]
                piece = self.game.board[sq]

                # 基础样式
                stylesheet = "QPushButton { border: 2px solid #AAAAAA; }"
                
                button.setEnabled(is_human_turn)
                if not is_human_turn and not self.game_over:
                     stylesheet += "QPushButton { background-color: #F0F0F0; }"
                
                # --- 应用高亮样式 (顺序很重要) ---
                # 1. 目标位置高亮
                if pos in cannon_attack_targets:
                    stylesheet += "QPushButton { background-color: #FFC0CB; }" # 粉色 (炮攻击)
                elif pos in normal_move_targets:
                    stylesheet += "QPushButton { background-color: #90EE90; }" # 浅绿 (普通移动)
                elif pos in reveal_targets:
                    stylesheet += "QPushButton { background-color: #ADD8E6; }" # 浅蓝 (翻棋)
                # 2. 可选棋子高亮 (如果未选择棋子)
                elif pos in movable_pieces_pos:
                     stylesheet += "QPushButton { background-color: #FFFFE0; }" # 淡黄 (可选)

                # 3. 选中的棋子边框 (最高优先级)
                if self.selected_from_sq == sq:
                    stylesheet += "QPushButton { border-color: #0078D7; border-width: 4px; }"

                button.setStyleSheet(stylesheet)

                # 设置文本和颜色
                if piece is None:
                    button.setText("")
                elif not piece.revealed:
                    button.setText("暗")
                    button.setStyleSheet(button.styleSheet() + "QPushButton { color: black; }")
                elif piece.player == 1:
                    button.setText(red_map[piece.piece_type])
                    button.setStyleSheet(button.styleSheet() + "QPushButton { color: red; }")
                else:
                    button.setText(black_map[piece.piece_type])
                    button.setStyleSheet(button.styleSheet() + "QPushButton { color: blue; }")

    def update_status_display(self):
        """更新状态显示。"""
        # 当前玩家
        player_name = "红方" if self.game.current_player == 1 else "黑方"
        player_role = ""
        if self.ai_player is not None and not self.game_over:
             if self.ai_player == self.game.current_player or self.ai_player == "both":
                 player_role = " (AI)"
             else:
                 player_role = " (你)"
        
        self.current_player_label.setText(f"{player_name}{player_role}")
        
        # 分数
        scores = self.game.scores
        self.scores_label.setText(f"{scores[1]} - {scores[-1]}")
        
        # 连续步数
        self.move_counter_label.setText(f"{self.game.move_counter} / {MAX_CONSECUTIVE_MOVES_FOR_DRAW}")
        
        # 游戏状态
        if self.game_over:
            self.game_status_label.setText("游戏结束")
        elif self.ai_thinking:
            self.game_status_label.setText("AI思考中...")
        else:
            self.game_status_label.setText("进行中")
        
        # 阵亡棋子
        dead_red_str = ', '.join([p.piece_type.name for p in self.game.dead_pieces[1]])
        dead_black_str = ', '.join([p.piece_type.name for p in self.game.dead_pieces[-1]])
        self.dead_pieces_red_label.setText(dead_red_str if dead_red_str else "无")
        self.dead_pieces_black_label.setText(dead_black_str if dead_black_str else "无")

    def update_bitboard_display(self):
        """更新所有Bitboard的可视化网格。"""
        def vector_to_bitboard(vector):
            # 使用 numpy 的方式来加速转换
            powers_of_2 = 1 << np.arange(len(vector), dtype=np.uint64)
            result = np.sum(powers_of_2[np.where(vector)])
            return int(result)

        # 更新基础状态向量
        self.hidden_bb_widget.update_bitboard(vector_to_bitboard(self.game.hidden_vector))
        self.empty_bb_widget.update_bitboard(vector_to_bitboard(self.game.empty_vector))

        # 更新玩家相关的向量
        for p in [1, -1]:
            self.player_bb_widgets[p]['revealed'].update_bitboard(
                vector_to_bitboard(self.game.revealed_vectors[p])
            )
            for pt in PieceType:
                bb_val = vector_to_bitboard(self.game.piece_vectors[p][pt.value])
                self.player_bb_widgets[p][pt.value].update_bitboard(bb_val)

if __name__ == '__main__':
    # 确保项目根目录在 sys.path 中，以便能正确找到 game 和 utils 模块
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())