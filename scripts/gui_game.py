#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于PySide6的4x4暗棋GUI对弈系统
"""

import sys
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                               QComboBox, QMessageBox, QFrame, QButtonGroup,
                               QProgressBar, QTextEdit, QSplitter, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QIcon

from sb3_contrib import MaskablePPO
from Game import GameEnvironment, PieceType

class AIWorker(QThread):
    """AI计算线程，避免GUI阻塞"""
    action_ready = Signal(int)
    
    def __init__(self, model, obs, action_mask):
        super().__init__()
        self.model = model
        self.obs = obs
        self.action_mask = action_mask
        self.action = None
    
    def run(self):
        action, _ = self.model.predict(self.obs, action_masks=self.action_mask, deterministic=False)
        self.action_ready.emit(int(action))

class BoardButton(QPushButton):
    """棋盘按钮类"""
    
    def __init__(self, row, col):
        super().__init__()
        self.row = row
        self.col = col
        self.piece = None
        self.is_highlighted = False
        
        # 设置按钮大小和样式
        self.setFixedSize(80, 80)
        self.setFont(QFont("Arial", 16, QFont.Bold))
        
        # 默认样式
        self.setStyleSheet("""
            QPushButton {
                background-color: #F5DEB3;
                border: 2px solid #8B4513;
                border-radius: 8px;
                color: #2F4F4F;
            }
            QPushButton:hover {
                background-color: #FFE4B5;
            }
            QPushButton:pressed {
                background-color: #DEB887;
            }
        """)
        
        self.update_display()
    
    def set_piece(self, piece):
        """设置棋子"""
        self.piece = piece
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.piece is None:
            self.setText("")
            return
        
        # 棋子符号映射
        red_symbols = {
            PieceType.G: "帥", PieceType.F: "仕", PieceType.E: "相", 
            PieceType.D: "俥", PieceType.C: "傌", PieceType.B: "炮", PieceType.A: "兵"
        }
        black_symbols = {
            PieceType.G: "將", PieceType.F: "士", PieceType.E: "象", 
            PieceType.D: "車", PieceType.C: "馬", PieceType.B: "炮", PieceType.A: "卒"
        }
        
        if not self.piece.revealed:
            self.setText("暗")
            self.setStyleSheet("""
                QPushButton {
                    background-color: #696969;
                    border: 2px solid #2F2F2F;
                    border-radius: 8px;
                    color: #F0F0F0;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #808080;
                }
            """)
        else:
            if self.piece.player == 1:  # 红方
                symbol = red_symbols[self.piece.piece_type]
                self.setText(symbol)
                self.setStyleSheet("""
                    QPushButton {
                        background-color: #FFB6C1;
                        border: 2px solid #DC143C;
                        border-radius: 8px;
                        color: #8B0000;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #FFC0CB;
                    }
                """)
            else:  # 黑方
                symbol = black_symbols[self.piece.piece_type]
                self.setText(symbol)
                self.setStyleSheet("""
                    QPushButton {
                        background-color: #ADD8E6;
                        border: 2px solid #4169E1;
                        border-radius: 8px;
                        color: #000080;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #B0E0E6;
                    }
                """)
    
    def highlight(self, highlight=True):
        """高亮显示"""
        self.is_highlighted = highlight
        if highlight:
            current_style = self.styleSheet()
            self.setStyleSheet(current_style + """
                QPushButton {
                    border: 3px solid #FFD700;
                    background-color: #FFFFE0;
                }
            """)
        else:
            self.update_display()

class GameGUI(QMainWindow):
    """游戏主界面"""
    
    def __init__(self):
        super().__init__()
        
        self.env = None
        self.ai_model = None
        self.ai_worker = None
        self.human_player = 1  # 1为红方，-1为黑方
        self.ai_player = -1
        self.selected_button = None
        self.valid_actions = []
        self.action_buttons = []
        self.game_over = False
        
        self.setup_ui()
        self.setup_game()
    
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("4x4暗棋 AI对弈系统")
        self.setFixedSize(1000, 700)
        
        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧游戏区域
        game_area = self.create_game_area()
        
        # 右侧控制区域
        control_area = self.create_control_area()
        
        # 添加到主布局
        main_layout.addWidget(game_area, 2)
        main_layout.addWidget(control_area, 1)
    
    def create_game_area(self):
        """创建游戏区域"""
        game_widget = QWidget()
        game_layout = QVBoxLayout(game_widget)
        
        # 游戏标题
        title_label = QLabel("4x4暗棋对弈")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        game_layout.addWidget(title_label)
        
        # 当前玩家显示
        self.current_player_label = QLabel("当前玩家: 红方")
        self.current_player_label.setAlignment(Qt.AlignCenter)
        self.current_player_label.setFont(QFont("Arial", 14))
        game_layout.addWidget(self.current_player_label)
        
        # 分数显示
        self.score_label = QLabel("分数 - 红方: 0  黑方: 0")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont("Arial", 12))
        game_layout.addWidget(self.score_label)
        
        # 棋盘
        self.board_widget = QWidget()
        self.board_layout = QGridLayout(self.board_widget)
        self.board_layout.setSpacing(2)
        
        # 创建棋盘按钮
        self.board_buttons = []
        for row in range(4):
            button_row = []
            for col in range(4):
                button = BoardButton(row, col)
                button.clicked.connect(lambda checked, r=row, c=col: self.on_board_click(r, c))
                self.board_layout.addWidget(button, row, col)
                button_row.append(button)
            self.board_buttons.append(button_row)
        
        game_layout.addWidget(self.board_widget)
        
        # 游戏状态
        self.status_label = QLabel("点击'开始游戏'开始")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #666666;")
        game_layout.addWidget(self.status_label)
        
        return game_widget
    
    def create_control_area(self):
        """创建控制区域"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # 模型选择组
        model_group = QGroupBox("AI模型选择")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.load_available_models()
        model_layout.addWidget(self.model_combo)
        
        control_layout.addWidget(model_group)
        
        # 玩家选择组
        player_group = QGroupBox("选择您的一方")
        player_layout = QVBoxLayout(player_group)
        
        self.player_combo = QComboBox()
        self.player_combo.addItem("红方 (先手)", 1)
        self.player_combo.addItem("黑方 (后手)", -1)
        player_layout.addWidget(self.player_combo)
        
        control_layout.addWidget(player_group)
        
        # 游戏控制按钮
        button_group = QGroupBox("游戏控制")
        button_layout = QVBoxLayout(button_group)
        
        self.start_button = QPushButton("开始游戏")
        self.start_button.clicked.connect(self.start_game)
        self.start_button.setFont(QFont("Arial", 12))
        button_layout.addWidget(self.start_button)
        
        self.restart_button = QPushButton("重新开始")
        self.restart_button.clicked.connect(self.restart_game)
        self.restart_button.setEnabled(False)
        button_layout.addWidget(self.restart_button)
        
        control_layout.addWidget(button_group)
        
        # 动作提示区域
        action_group = QGroupBox("可用动作")
        action_layout = QVBoxLayout(action_group)
        
        self.action_list = QTextEdit()
        self.action_list.setFixedHeight(150)
        self.action_list.setReadOnly(True)
        action_layout.addWidget(self.action_list)
        
        control_layout.addWidget(action_group)
        
        # 游戏日志
        log_group = QGroupBox("游戏日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        control_layout.addWidget(log_group)
        
        return control_widget
    
    def load_available_models(self):
        """加载可用的AI模型"""
        self.model_combo.clear()
        log_dir = "./banqi_ppo_logs/"
        
        model_files = []
        
        # 检查最佳模型
        best_model = os.path.join(log_dir, "best_model.zip")
        if os.path.exists(best_model):
            model_files.append(("最佳模型", best_model))
        
        # 检查最终模型
        final_model = os.path.join(log_dir, "banqi_ppo_model.zip")
        if os.path.exists(final_model):
            model_files.append(("最终模型", final_model))
        
        # 检查检查点模型
        if os.path.exists(log_dir):
            checkpoint_files = []
            for file in os.listdir(log_dir):
                if file.startswith("rl_model_") and file.endswith("_steps.zip"):
                    steps = int(file.replace("rl_model_", "").replace("_steps.zip", ""))
                    checkpoint_files.append((steps, file))
            
            checkpoint_files.sort(key=lambda x: x[0], reverse=True)
            for steps, file in checkpoint_files[:10]:  # 只显示最新的10个
                model_files.append((f"检查点模型 ({steps} steps)", os.path.join(log_dir, file)))
        
        if not model_files:
            self.model_combo.addItem("未找到模型文件", "")
        else:
            for name, path in model_files:
                self.model_combo.addItem(name, path)
    
    def setup_game(self):
        """设置游戏环境"""
        self.env = GameEnvironment(render_mode=None)
    
    def start_game(self):
        """开始游戏"""
        model_path = self.model_combo.currentData()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "请选择有效的AI模型！")
            return
        
        # 加载AI模型
        try:
            # 创建临时环境用于获取环境规格
            temp_env = GameEnvironment()
            self.ai_model = MaskablePPO.load(model_path, env=temp_env)
            self.log_message(f"AI模型加载成功: {self.model_combo.currentText()}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"AI模型加载失败: {str(e)}")
            return
        
        # 设置玩家
        self.human_player = self.player_combo.currentData()
        self.ai_player = -self.human_player
        
        # 重置游戏
        self.game_over = False
        obs, info = self.env.reset()
        self.update_board()
        self.update_status()
        
        # 启用重新开始按钮
        self.restart_button.setEnabled(True)
        self.start_button.setEnabled(False)
        
        self.log_message(f"游戏开始！人类玩家: {'红方' if self.human_player == 1 else '黑方'}")
        self.log_message(f"AI玩家: {'红方' if self.ai_player == 1 else '黑方'}")
        
        # 如果AI先手，让AI下棋
        if self.env.current_player == self.ai_player:
            self.ai_move()
        else:
            self.update_human_actions()
    
    def restart_game(self):
        """重新开始游戏"""
        self.game_over = False
        if self.env:
            obs, info = self.env.reset()
            self.update_board()
            self.update_status()
            self.clear_highlights()
            self.log_message("游戏重新开始")
            
            if self.env.current_player == self.ai_player:
                self.ai_move()
            else:
                self.update_human_actions()
    
    def on_board_click(self, row, col):
        """处理棋盘点击事件"""
        if self.game_over or not self.ai_model:
            return
        
        if self.env.current_player != self.human_player:
            return
        
        # 计算动作索引
        pos_idx = row * 4 + col
        
        # 检查所有可能的动作
        valid_actions = self.env.action_masks()
        
        available_actions = []
        for action_sub_idx in range(5):
            action_idx = pos_idx * 5 + action_sub_idx
            if valid_actions[action_idx] == 1:
                available_actions.append((action_idx, action_sub_idx))
        
        if not available_actions:
            self.status_label.setText("这个位置没有可用动作")
            return
        
        if len(available_actions) == 1:
            # 只有一个动作，直接执行
            action_idx, _ = available_actions[0]
            self.execute_human_action(action_idx)
        else:
            # 多个动作，让用户选择
            self.show_action_menu(row, col, available_actions)
    
    def show_action_menu(self, row, col, available_actions):
        """显示动作选择菜单"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"选择动作 ({row},{col})")
        dialog.setFixedSize(300, 200)
        
        layout = QVBoxLayout(dialog)
        
        label = QLabel(f"位置 ({row},{col}) 的可用动作:")
        layout.addWidget(label)
        
        for action_idx, action_sub_idx in available_actions:
            if action_sub_idx == 4:
                button_text = "翻开棋子"
            else:
                directions = ["向上", "向下", "向左", "向右"]
                button_text = directions[action_sub_idx]
            
            button = QPushButton(button_text)
            button.clicked.connect(lambda checked, idx=action_idx: self.select_action(dialog, idx))
            layout.addWidget(button)
        
        # 取消按钮
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(dialog.reject)
        layout.addWidget(cancel_button)
        
        dialog.exec()
    
    def select_action(self, dialog, action_idx):
        """选择动作"""
        dialog.accept()
        self.execute_human_action(action_idx)
    
    def execute_human_action(self, action_idx):
        """执行人类玩家的动作"""
        if self.game_over:
            return
        
        # 描述动作
        pos_idx = action_idx // 5
        action_sub_idx = action_idx % 5
        row = pos_idx // 4
        col = pos_idx % 4
        
        if action_sub_idx == 4:
            action_desc = f"翻开 ({row},{col})"
        else:
            directions = ["向上", "向下", "向左", "向右"]
            action_desc = f"从 ({row},{col}) {directions[action_sub_idx]}"
        
        self.log_message(f"人类玩家: {action_desc}")
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        
        # 更新界面
        self.update_board()
        self.update_status()
        
        # 检查游戏结束
        if terminated or truncated:
            self.handle_game_over(info)
        else:
            # AI回合
            QTimer.singleShot(500, self.ai_move)  # 延迟500ms让用户看到结果
    
    def ai_move(self):
        """AI下棋"""
        if self.game_over or not self.ai_model:
            return
        
        if self.env.current_player != self.ai_player:
            return
        
        self.status_label.setText("AI正在思考...")
        
        # 获取状态和动作掩码
        obs = self.env.get_state()
        action_mask = self.env.action_masks()
        
        # 在后台线程中计算AI动作
        self.ai_worker = AIWorker(self.ai_model, obs, action_mask)
        self.ai_worker.action_ready.connect(self.on_ai_action_ready)
        self.ai_worker.start()
    
    def on_ai_action_ready(self, action):
        """AI动作准备完成"""
        if self.game_over:
            return
        
        # 描述AI动作
        pos_idx = action // 5
        action_sub_idx = action % 5
        row = pos_idx // 4
        col = pos_idx % 4
        
        if action_sub_idx == 4:
            action_desc = f"翻开 ({row},{col})"
        else:
            directions = ["向上", "向下", "向左", "向右"]
            action_desc = f"从 ({row},{col}) {directions[action_sub_idx]}"
        
        self.log_message(f"AI玩家: {action_desc}")
        
        # 执行AI动作
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 更新界面
        self.update_board()
        self.update_status()
        
        # 检查游戏结束
        if terminated or truncated:
            self.handle_game_over(info)
        else:
            # 人类回合
            self.update_human_actions()
    
    def update_board(self):
        """更新棋盘显示"""
        for row in range(4):
            for col in range(4):
                piece = self.env.board[row, col]
                self.board_buttons[row][col].set_piece(piece)
    
    def update_status(self):
        """更新状态显示"""
        current_player_text = "红方" if self.env.current_player == 1 else "黑方"
        self.current_player_label.setText(f"当前玩家: {current_player_text}")
        
        red_score = self.env.scores[1]
        black_score = self.env.scores[-1]
        self.score_label.setText(f"分数 - 红方: {red_score}  黑方: {black_score}")
        
        if self.env.current_player == self.human_player:
            self.status_label.setText("轮到您下棋")
        else:
            self.status_label.setText("AI正在思考...")
    
    def update_human_actions(self):
        """更新人类玩家的可用动作"""
        if self.env.current_player != self.human_player:
            return
        
        valid_actions = self.env.action_masks()
        valid_action_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_action_indices) == 0:
            self.action_list.setText("无可用动作")
            return
        
        action_text = "可用动作：\n"
        for action_idx in valid_action_indices:
            pos_idx = action_idx // 5
            action_sub_idx = action_idx % 5
            row = pos_idx // 4
            col = pos_idx % 4
            
            if action_sub_idx == 4:
                desc = f"翻开 ({row},{col})"
            else:
                directions = ["向上", "向下", "向左", "向右"]
                desc = f"从 ({row},{col}) {directions[action_sub_idx]}"
            
            action_text += f"• {desc}\n"
        
        self.action_list.setText(action_text)
    
    def handle_game_over(self, info):
        """处理游戏结束"""
        self.game_over = True
        winner = info.get('winner')
        
        if winner == 1:
            result_text = "红方获胜！"
            if self.human_player == 1:
                result_text += " 🎉 恭喜您获胜！"
            else:
                result_text += " AI获胜！"
        elif winner == -1:
            result_text = "黑方获胜！"
            if self.human_player == -1:
                result_text += " 🎉 恭喜您获胜！"
            else:
                result_text += " AI获胜！"
        else:
            result_text = "平局！"
        
        self.status_label.setText(result_text)
        self.log_message(f"游戏结束: {result_text}")
        
        # 显示结果对话框
        QMessageBox.information(self, "游戏结束", result_text)
        
        # 重新启用开始按钮
        self.start_button.setEnabled(True)
    
    def clear_highlights(self):
        """清除高亮显示"""
        for row in range(4):
            for col in range(4):
                self.board_buttons[row][col].highlight(False)
    
    def log_message(self, message):
        """记录消息到日志"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = GameGUI()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
