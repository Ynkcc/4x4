# game_gui_cython.py - 使用 Cython 优化版本的 GUI

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QFrame, QGroupBox, QFormLayout,
    QLineEdit
)
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtCore import Qt, QSize
import numpy as np

# 导入 Cython 优化的游戏环境
from Game_cython_simple import GameEnvironment, PieceType
from Game import SQ_TO_POS, POS_TO_SQ  # 这些常量仍然可以从原版导入

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
        self.game = GameEnvironment()  # 使用 Cython 优化版本
        
        self.setWindowTitle("暗棋游戏 GUI (Cython 优化版)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：游戏棋盘
        board_frame = QFrame()
        board_frame.setFrameStyle(QFrame.Box)
        board_frame.setFixedSize(400, 400)
        board_layout = QGridLayout(board_frame)
        
        self.board_buttons = []
        for i in range(16):
            button = QPushButton()
            button.setFixedSize(80, 80)
            button.setFont(QFont("Arial", 16, QFont.Bold))
            button.clicked.connect(lambda checked, idx=i: self.on_square_clicked(idx))
            self.board_buttons.append(button)
            row, col = SQ_TO_POS[i]
            board_layout.addWidget(button, row, col)
        
        # 右侧：状态显示和控制
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 游戏信息
        info_group = QGroupBox("游戏信息 (Cython 版本)")
        info_layout = QFormLayout(info_group)
        
        self.current_player_label = QLabel()
        self.score_label = QLabel()
        self.move_counter_label = QLabel()
        self.game_status_label = QLabel()
        
        info_layout.addRow("当前玩家:", self.current_player_label)
        info_layout.addRow("得分:", self.score_label)
        info_layout.addRow("连续移动:", self.move_counter_label)
        info_layout.addRow("游戏状态:", self.game_status_label)
        
        # 控制按钮
        controls_group = QGroupBox("控制")
        controls_layout = QVBoxLayout(controls_group)
        
        reset_button = QPushButton("重置游戏")
        reset_button.clicked.connect(self.reset_game)
        
        random_move_button = QPushButton("随机移动")
        random_move_button.clicked.connect(self.make_random_move)
        
        auto_play_button = QPushButton("自动游戏")
        auto_play_button.clicked.connect(self.auto_play)
        
        controls_layout.addWidget(reset_button)
        controls_layout.addWidget(random_move_button)
        controls_layout.addWidget(auto_play_button)
        
        # Bitboard 可视化
        bitboard_group = QGroupBox("Bitboard 可视化")
        bitboard_layout = QVBoxLayout(bitboard_group)
        
        # 创建bitboard显示网格
        bitboard_grids_layout = QGridLayout()
        
        self.bitboard_widgets = {}
        labels = ["隐藏", "空格", "红兵", "红炮", "红马", "红车", "红象", "红士", "红帅"]
        
        for i, label in enumerate(labels):
            grid_widget = BitboardGridWidget()
            self.bitboard_widgets[label] = grid_widget
            
            # 创建标签和网格的组合
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.addWidget(QLabel(label))
            container_layout.addWidget(grid_widget)
            container_layout.setContentsMargins(2, 2, 2, 2)
            
            row, col = i // 3, i % 3
            bitboard_grids_layout.addWidget(container, row, col)
        
        bitboard_layout.addLayout(bitboard_grids_layout)
        
        # 组装右侧面板
        control_layout.addWidget(info_group)
        control_layout.addWidget(controls_group)
        control_layout.addWidget(bitboard_group)
        control_layout.addStretch()
        
        # 组装主布局
        main_layout.addWidget(board_frame)
        main_layout.addWidget(control_panel)
        
        # 初始化游戏
        self.reset_game()
    
    def reset_game(self):
        """重置游戏状态"""
        state, info = self.game.reset()
        self.update_display()
    
    def on_square_clicked(self, square_index):
        """处理棋盘方格点击事件"""
        action_mask = self.game.action_masks()
        
        # 检查翻棋动作
        if square_index < 16 and action_mask[square_index]:
            self.make_move(square_index)
            return
        
        # 检查移动动作 (简化处理，实际需要根据游戏状态判断)
        print(f"点击了方格 {square_index}")
    
    def make_move(self, action_index):
        """执行移动"""
        try:
            state, reward, terminated, truncated, info = self.game.step(action_index)
            self.update_display()
            
            if terminated or truncated:
                winner = info.get('winner')
                if winner == 1:
                    self.game_status_label.setText("红方获胜!")
                elif winner == -1:
                    self.game_status_label.setText("黑方获胜!")
                else:
                    self.game_status_label.setText("平局!")
        except Exception as e:
            print(f"移动错误: {e}")
    
    def make_random_move(self):
        """进行随机移动"""
        action_mask = self.game.action_masks()
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            self.make_move(action)
    
    def auto_play(self):
        """自动游戏直到结束"""
        max_moves = 200
        moves = 0
        
        while moves < max_moves:
            action_mask = self.game.action_masks()
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) == 0:
                break
            
            action = np.random.choice(valid_actions)
            state, reward, terminated, truncated, info = self.game.step(action)
            
            moves += 1
            if terminated or truncated:
                break
        
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        # 更新棋盘显示
        for i in range(16):
            button = self.board_buttons[i]
            piece = self.game.board[i]
            
            if piece is None:
                button.setText("")
                button.setStyleSheet("background-color: #F5F5DC;")
            elif not piece.revealed:
                button.setText("暗")
                button.setStyleSheet("background-color: #8B4513; color: white;")
            else:
                # 显示棋子
                piece_chars = {
                    PieceType.SOLDIER: "兵" if piece.player == 1 else "卒",
                    PieceType.CANNON: "炮",
                    PieceType.HORSE: "傌" if piece.player == 1 else "馬", 
                    PieceType.CHARIOT: "俥" if piece.player == 1 else "車",
                    PieceType.ELEPHANT: "相" if piece.player == 1 else "象",
                    PieceType.ADVISOR: "仕" if piece.player == 1 else "士",
                    PieceType.GENERAL: "帥" if piece.player == 1 else "將"
                }
                
                button.setText(piece_chars[piece.piece_type])
                if piece.player == 1:
                    button.setStyleSheet("background-color: #FFB6C1; color: red;")
                else:
                    button.setStyleSheet("background-color: #ADD8E6; color: blue;")
        
        # 更新游戏信息
        player_name = "红方" if self.game.current_player == 1 else "黑方"
        self.current_player_label.setText(player_name)
        self.score_label.setText(f"红: {self.game.scores[1]} - 黑: {self.game.scores[-1]}")
        self.move_counter_label.setText(f"{self.game.move_counter}")
        self.game_status_label.setText("进行中")
        
        # 更新 bitboard 可视化
        self.bitboard_widgets["隐藏"].update_bitboard(self.game.hidden_bitboard)
        self.bitboard_widgets["空格"].update_bitboard(self.game.empty_bitboard)
        
        # 更新棋子 bitboards (红方)
        piece_names = ["兵", "炮", "马", "车", "象", "士", "帅"]
        for i, name in enumerate(piece_names):
            if f"红{name}" in self.bitboard_widgets:
                self.bitboard_widgets[f"红{name}"].update_bitboard(
                    int(self.game.piece_bitboards[1, i])
                )

def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    print("暗棋游戏 GUI (Cython 优化版) 已启动!")
    print("性能已优化，游戏运行更流畅!")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
