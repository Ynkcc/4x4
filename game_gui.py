# game_gui.py
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QFrame, QGroupBox, QFormLayout,
    QLineEdit
)
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtCore import Qt, QSize
import numpy as np

# 导入您提供的游戏环境
from Game import GameEnvironment, PieceType, SQ_TO_POS, POS_TO_SQ, ACTION_TYPE_REVEAL

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
                # 如果该位为1，高亮显示
                self.labels[i].setStyleSheet("background-color: #4CAF50; border: 1px solid #388E3C;")
            else:
                self.labels[i].setStyleSheet("background-color: #DDDDDD; border: 1px solid #BBBBBB;")

class MainWindow(QMainWindow):
    """主窗口，整合了游戏棋盘、状态显示和控制按钮。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.game = GameEnvironment()
        
        # GUI 内部状态
        self.selected_from_sq = None
        self.valid_action_mask = np.zeros(self.game.ACTION_SPACE_SIZE, dtype=int)

        self.setWindowTitle("暗棋游戏逻辑验证GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # --- 主布局 ---
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- 左侧：棋盘和控制 ---
        left_panel = QVBoxLayout()
        self.board_buttons = []
        board_group = QGroupBox("游戏棋盘")
        board_layout = QGridLayout()
        board_layout.setSpacing(5)

        font = QFont("SimSun", 18, QFont.Bold)
        for r in range(4):
            row_buttons = []
            for c in range(4):
                button = QPushButton()
                button.setFixedSize(80, 80)
                button.setFont(font)
                button.clicked.connect(lambda _, pos=(r, c): self.on_board_click(pos))
                board_layout.addWidget(button, r, c)
                row_buttons.append(button)
            self.board_buttons.append(row_buttons)
        
        board_group.setLayout(board_layout)
        
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()
        self.reset_button = QPushButton("重置游戏 (Reset)")
        self.reset_button.clicked.connect(self.reset_game)
        control_layout.addWidget(self.reset_button)
        control_group.setLayout(control_layout)

        left_panel.addWidget(board_group)
        left_panel.addWidget(control_group)

        # --- 右侧：内部状态显示 ---
        right_panel = QVBoxLayout()
        state_group = QGroupBox("内部状态变量")
        self.state_layout = QFormLayout()

        # 基本信息
        self.current_player_label = QLineEdit()
        self.scores_label = QLineEdit()
        self.move_counter_label = QLineEdit()
        self.dead_pieces_red_label = QLineEdit()
        self.dead_pieces_black_label = QLineEdit()
        for label in [self.current_player_label, self.scores_label, self.move_counter_label, self.dead_pieces_red_label, self.dead_pieces_black_label]:
            label.setReadOnly(True)

        self.state_layout.addRow(QLabel("当前玩家:"), self.current_player_label)
        self.state_layout.addRow(QLabel("得分 (红 - 黑):"), self.scores_label)
        self.state_layout.addRow(QLabel("连续未吃子:"), self.move_counter_label)
        self.state_layout.addRow(QLabel("红方阵亡棋子:"), self.dead_pieces_red_label)
        self.state_layout.addRow(QLabel("黑方阵亡棋子:"), self.dead_pieces_black_label)
        
        state_group.setLayout(self.state_layout)
        
        # Bitboards 可视化
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
            # Revealed BB
            revealed_widget = BitboardGridWidget()
            layout.addRow("Revealed BB:", revealed_widget)
            self.player_bb_widgets[p]['revealed'] = revealed_widget
            # Piece Type BBs
            for pt in PieceType:
                pt_widget = BitboardGridWidget()
                layout.addRow(f"{pt.name} BB:", pt_widget)
                self.player_bb_widgets[p][pt.value] = pt_widget

        bitboard_main_layout.addLayout(player_bb_layout)
        bitboard_group.setLayout(bitboard_main_layout)

        right_panel.addWidget(state_group)
        right_panel.addWidget(bitboard_group)
        
        # --- 整合主布局 ---
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)
        
        self.reset_game()

    def reset_game(self):
        """重置游戏状态并刷新整个GUI。"""
        self.selected_from_sq = None
        _, info = self.game.reset()
        self.valid_action_mask = info['action_mask']
        self.update_gui()

    def on_board_click(self, pos):
        """处理棋盘按钮的点击事件。"""
        clicked_sq = POS_TO_SQ[pos]
        
        # 检查是否是翻棋动作
        if self.selected_from_sq is None:
            action_index = self.game.coords_to_action.get(pos)
            if action_index is not None and action_index < self.game.REVEAL_ACTIONS_COUNT:
                if self.valid_action_mask[action_index]:
                    self.game.step(action_index)
                    self.valid_action_mask = self.game.action_masks()
                    self.update_gui()
                    return

        # 检查是否是移动/攻击动作
        # 情况1：选择起始棋子
        if self.selected_from_sq is None:
            piece = self.game.board[clicked_sq]
            if piece and piece.revealed and piece.player == self.game.current_player:
                self.selected_from_sq = clicked_sq
                self.update_gui() # 高亮可选目标
        # 情况2：已选择起始棋子，现在选择目标
        else:
            from_pos = SQ_TO_POS[self.selected_from_sq]
            to_pos = pos
            action_index = self.game.coords_to_action.get((from_pos, to_pos))
            
            # 如果点击有效目标
            if action_index is not None and self.valid_action_mask[action_index]:
                self.game.step(action_index)
                self.selected_from_sq = None
                self.valid_action_mask = self.game.action_masks()
                self.update_gui()
            # 如果点击其他地方（包括点自己），则取消选择
            else:
                self.selected_from_sq = None
                self.update_gui() # 取消高亮

    def update_gui(self):
        """根据当前游戏状态，刷新所有UI组件。"""
        self.update_board_display()
        self.update_state_display()
        self.update_bitboard_display()

    def update_board_display(self):
        """更新4x4棋盘的显示。"""
        red_map = {
            PieceType.GENERAL: "帥", PieceType.ADVISOR: "仕", PieceType.ELEPHANT: "相",
            PieceType.CHARIOT: "俥", PieceType.HORSE: "傌", PieceType.CANNON: "炮", PieceType.SOLDIER: "兵"
        }
        black_map = {
            PieceType.GENERAL: "將", PieceType.ADVISOR: "士", PieceType.ELEPHANT: "象",
            PieceType.CHARIOT: "車", PieceType.HORSE: "馬", PieceType.CANNON: "炮", PieceType.SOLDIER: "卒"
        }

        # 预先计算所有合法目标位置
        valid_targets = set()
        if self.selected_from_sq is not None:
            from_pos = SQ_TO_POS[self.selected_from_sq]
            for action_id, is_valid in enumerate(self.valid_action_mask):
                if is_valid:
                    coords = self.game.action_to_coords.get(action_id)
                    if isinstance(coords, tuple) and len(coords) == 2 and isinstance(coords[0], tuple):
                        if coords[0] == from_pos:
                             valid_targets.add(coords[1])
        
        for r in range(4):
            for c in range(4):
                sq = POS_TO_SQ[r, c]
                button = self.board_buttons[r][c]
                piece = self.game.board[sq]
                
                # --- 设置样式 ---
                stylesheet = "QPushButton { border: 2px solid #AAAAAA; }"
                
                # 高亮合法动作
                is_valid_action = False
                # 检查翻棋
                action_idx_reveal = self.game.coords_to_action.get((r,c))
                if action_idx_reveal is not None and self.valid_action_mask[action_idx_reveal]:
                    is_valid_action = True
                
                # 检查移动目标
                if (r,c) in valid_targets:
                    is_valid_action = True
                
                if self.selected_from_sq == sq:
                    stylesheet += "QPushButton { border-color: #0078D7; border-width: 4px; }" # 选中的棋子
                elif is_valid_action:
                    stylesheet += "QPushButton { background-color: #90EE90; }" # 合法目标
                
                button.setStyleSheet(stylesheet)

                # --- 设置文本和颜色 ---
                if piece is None:
                    button.setText("")
                elif not piece.revealed:
                    button.setText("暗")
                    button.setForegroundRole(QPalette.ColorRole.ButtonText)
                elif piece.player == 1:
                    button.setText(red_map[piece.piece_type])
                    button.setStyleSheet(button.styleSheet() + "QPushButton { color: red; }")
                else: # player == -1
                    button.setText(black_map[piece.piece_type])
                    button.setStyleSheet(button.styleSheet() + "QPushButton { color: blue; }")

    def update_state_display(self):
        """更新右侧的状态信息文本。"""
        # 当前玩家
        player_str = "红方 (Player 1)" if self.game.current_player == 1 else "黑方 (Player -1)"
        self.current_player_label.setText(player_str)
        
        # 分数
        scores = self.game.scores
        self.scores_label.setText(f"{scores[1]} - {scores[-1]}")
        
        # 连续步数
        self.move_counter_label.setText(f"{self.game.move_counter} / {self.game.MAX_CONSECUTIVE_MOVES}")

        # 阵亡棋子
        dead_red_str = ', '.join([p.piece_type.name for p in self.game.dead_pieces[1]])
        dead_black_str = ', '.join([p.piece_type.name for p in self.game.dead_pieces[-1]])
        self.dead_pieces_red_label.setText(dead_red_str if dead_red_str else "无")
        self.dead_pieces_black_label.setText(dead_black_str if dead_black_str else "无")

    def update_bitboard_display(self):
        """更新所有Bitboard的可视化网格。"""
        self.hidden_bb_widget.update_bitboard(self.game.hidden_bitboard)
        self.empty_bb_widget.update_bitboard(self.game.empty_bitboard)

        for p in [1, -1]:
            self.player_bb_widgets[p]['revealed'].update_bitboard(self.game.revealed_bitboards[p])
            for pt in PieceType:
                bb_val = self.game.piece_bitboards[p][pt.value]
                self.player_bb_widgets[p][pt.value].update_bitboard(bb_val)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())