#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于PySide6的AI vs AI对弈观察系统
"""

import sys
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                               QComboBox, QMessageBox, QFrame, QSpinBox,
                               QProgressBar, QTextEdit, QSplitter, QGroupBox,
                               QCheckBox, QSlider)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QIcon

from sb3_contrib import MaskablePPO
from Game import GameEnvironment, PieceType
from gui_game import BoardButton, AIWorker

class GameStatistics:
    """游戏统计类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.games_played = 0
        self.red_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_moves = 0
        self.red_total_score = 0
        self.black_total_score = 0
    
    def add_game(self, winner, moves, red_score, black_score):
        self.games_played += 1
        self.total_moves += moves
        self.red_total_score += red_score
        self.black_total_score += black_score
        
        if winner == 1:
            self.red_wins += 1
        elif winner == -1:
            self.black_wins += 1
        else:
            self.draws += 1
    
    def get_stats(self):
        if self.games_played == 0:
            return {
                'games': 0, 'red_wins': 0, 'black_wins': 0, 'draws': 0,
                'red_win_rate': 0, 'black_win_rate': 0, 'draw_rate': 0,
                'avg_moves': 0, 'avg_red_score': 0, 'avg_black_score': 0
            }
        
        return {
            'games': self.games_played,
            'red_wins': self.red_wins,
            'black_wins': self.black_wins,
            'draws': self.draws,
            'red_win_rate': self.red_wins / self.games_played * 100,
            'black_win_rate': self.black_wins / self.games_played * 100,
            'draw_rate': self.draws / self.games_played * 100,
            'avg_moves': self.total_moves / self.games_played,
            'avg_red_score': self.red_total_score / self.games_played,
            'avg_black_score': self.black_total_score / self.games_played
        }

class AIvsAIGUI(QMainWindow):
    """AI vs AI对弈观察界面"""
    
    def __init__(self):
        super().__init__()
        
        self.env = None
        self.red_ai_model = None
        self.black_ai_model = None
        self.ai_worker = None
        self.game_over = False
        self.is_running = False
        self.current_game = 0
        self.total_games = 1
        self.auto_mode = False
        self.move_count = 0
        
        self.statistics = GameStatistics()
        
        self.setup_ui()
        self.setup_game()
    
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("AI vs AI 对弈观察系统")
        self.setFixedSize(1200, 800)
        
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
        title_label = QLabel("AI vs AI 对弈观察")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        game_layout.addWidget(title_label)
        
        # 游戏进度
        self.progress_label = QLabel("准备开始...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setFont(QFont("Arial", 12))
        game_layout.addWidget(self.progress_label)
        
        # 当前玩家和分数
        info_layout = QHBoxLayout()
        
        self.current_player_label = QLabel("当前玩家: 红方")
        self.current_player_label.setFont(QFont("Arial", 14))
        info_layout.addWidget(self.current_player_label)
        
        self.score_label = QLabel("分数 - 红方: 0  黑方: 0")
        self.score_label.setFont(QFont("Arial", 14))
        info_layout.addWidget(self.score_label)
        
        game_layout.addLayout(info_layout)
        
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
                button.setEnabled(False)  # AI对弈时禁用点击
                self.board_layout.addWidget(button, row, col)
                button_row.append(button)
            self.board_buttons.append(button_row)
        
        game_layout.addWidget(self.board_widget)
        
        # 游戏状态
        self.status_label = QLabel("选择AI模型并开始对弈")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #666666;")
        game_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        game_layout.addWidget(self.progress_bar)
        
        return game_widget
    
    def create_control_area(self):
        """创建控制区域"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # AI模型选择组
        model_group = QGroupBox("AI模型选择")
        model_layout = QVBoxLayout(model_group)
        
        # 红方AI
        red_label = QLabel("红方AI:")
        red_label.setStyleSheet("color: #DC143C; font-weight: bold;")
        model_layout.addWidget(red_label)
        
        self.red_model_combo = QComboBox()
        self.load_available_models(self.red_model_combo)
        model_layout.addWidget(self.red_model_combo)
        
        # 黑方AI
        black_label = QLabel("黑方AI:")
        black_label.setStyleSheet("color: #4169E1; font-weight: bold;")
        model_layout.addWidget(black_label)
        
        self.black_model_combo = QComboBox()
        self.load_available_models(self.black_model_combo)
        model_layout.addWidget(self.black_model_combo)
        
        control_layout.addWidget(model_group)
        
        # 对弈设置组
        settings_group = QGroupBox("对弈设置")
        settings_layout = QVBoxLayout(settings_group)
        
        # 对弈局数
        games_layout = QHBoxLayout()
        games_layout.addWidget(QLabel("对弈局数:"))
        self.games_spinbox = QSpinBox()
        self.games_spinbox.setRange(1, 1000)
        self.games_spinbox.setValue(10)
        games_layout.addWidget(self.games_spinbox)
        settings_layout.addLayout(games_layout)
        
        # 延迟设置
        delay_layout = QHBoxLayout()
        delay_layout.addWidget(QLabel("动作延迟(ms):"))
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setRange(0, 2000)
        self.delay_slider.setValue(500)
        self.delay_label = QLabel("500")
        self.delay_slider.valueChanged.connect(lambda v: self.delay_label.setText(str(v)))
        delay_layout.addWidget(self.delay_slider)
        delay_layout.addWidget(self.delay_label)
        settings_layout.addLayout(delay_layout)
        
        # 自动模式
        self.auto_checkbox = QCheckBox("自动连续对弈")
        settings_layout.addWidget(self.auto_checkbox)
        
        control_layout.addWidget(settings_group)
        
        # 控制按钮组
        button_group = QGroupBox("控制")
        button_layout = QVBoxLayout(button_group)
        
        self.start_button = QPushButton("开始对弈")
        self.start_button.clicked.connect(self.start_games)
        self.start_button.setFont(QFont("Arial", 12))
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_games)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.reset_button = QPushButton("重置统计")
        self.reset_button.clicked.connect(self.reset_statistics)
        button_layout.addWidget(self.reset_button)
        
        control_layout.addWidget(button_group)
        
        # 统计信息组
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFixedHeight(150)
        stats_layout.addWidget(self.stats_text)
        
        control_layout.addWidget(stats_group)
        
        # 游戏日志
        log_group = QGroupBox("游戏日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        control_layout.addWidget(log_group)
        
        # 更新统计显示
        self.update_statistics_display()
        
        return control_widget
    
    def load_available_models(self, combo):
        """加载可用的AI模型"""
        combo.clear()
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
            for steps, file in checkpoint_files[:15]:  # 显示最新的15个
                model_files.append((f"检查点模型 ({steps} steps)", os.path.join(log_dir, file)))
        
        if not model_files:
            combo.addItem("未找到模型文件", "")
        else:
            for name, path in model_files:
                combo.addItem(name, path)
    
    def setup_game(self):
        """设置游戏环境"""
        self.env = GameEnvironment(render_mode=None)
    
    def start_games(self):
        """开始对弈"""
        red_model_path = self.red_model_combo.currentData()
        black_model_path = self.black_model_combo.currentData()
        
        if not red_model_path or not os.path.exists(red_model_path):
            QMessageBox.warning(self, "错误", "请选择有效的红方AI模型！")
            return
        
        if not black_model_path or not os.path.exists(black_model_path):
            QMessageBox.warning(self, "错误", "请选择有效的黑方AI模型！")
            return
        
        # 加载AI模型
        try:
            # 创建临时环境用于获取环境规格
            temp_env = GameEnvironment()
            
            # 加载模型时提供环境信息
            self.red_ai_model = MaskablePPO.load(red_model_path, env=temp_env)
            self.black_ai_model = MaskablePPO.load(black_model_path, env=temp_env)
            
            self.log_message(f"红方AI模型加载成功: {self.red_model_combo.currentText()}")
            self.log_message(f"黑方AI模型加载成功: {self.black_model_combo.currentText()}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"AI模型加载失败: {str(e)}")
            return
        
        # 设置对弈参数
        self.total_games = self.games_spinbox.value()
        self.current_game = 0
        self.auto_mode = self.auto_checkbox.isChecked()
        self.is_running = True
        
        # 重置统计
        if not self.auto_mode or self.current_game == 0:
            self.statistics.reset()
        
        # 更新UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(self.total_games)
        self.progress_bar.setValue(0)
        
        # 开始第一局
        self.start_next_game()
    
    def start_next_game(self):
        """开始下一局游戏"""
        if not self.is_running or self.current_game >= self.total_games:
            self.finish_games()
            return
        
        self.current_game += 1
        self.game_over = False
        self.move_count = 0
        
        # 重置环境
        obs, info = self.env.reset()
        self.update_board()
        self.update_status()
        
        # 更新进度
        self.progress_label.setText(f"第 {self.current_game} / {self.total_games} 局")
        self.progress_bar.setValue(self.current_game - 1)
        
        self.log_message(f"第 {self.current_game} 局开始")
        
        # 开始AI对弈
        self.ai_move()
    
    def ai_move(self):
        """AI下棋"""
        if self.game_over or not self.is_running:
            return
        
        # 获取当前AI模型
        if self.env.current_player == 1:
            current_model = self.red_ai_model
            player_name = "红方AI"
        else:
            current_model = self.black_ai_model
            player_name = "黑方AI"
        
        self.status_label.setText(f"{player_name}正在思考...")
        
        # 获取状态和动作掩码
        obs = self.env.get_state()
        action_mask = self.env.action_masks()
        
        # 在后台线程中计算AI动作
        self.ai_worker = AIWorker(current_model, obs, action_mask)
        self.ai_worker.action_ready.connect(self.on_ai_action_ready)
        self.ai_worker.start()
    
    def on_ai_action_ready(self, action):
        """AI动作准备完成"""
        if self.game_over or not self.is_running:
            return
        
        self.move_count += 1
        
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
        
        player_name = "红方AI" if self.env.current_player == 1 else "黑方AI"
        self.log_message(f"回合 {self.move_count}: {player_name} - {action_desc}")
        
        # 执行AI动作
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 更新界面
        self.update_board()
        self.update_status()
        
        # 检查游戏结束
        if terminated or truncated:
            self.handle_game_over(info)
        else:
            # 继续下一步
            delay = self.delay_slider.value()
            if delay > 0:
                QTimer.singleShot(delay, self.ai_move)
            else:
                self.ai_move()
    
    def handle_game_over(self, info):
        """处理游戏结束"""
        self.game_over = True
        winner = info.get('winner')
        
        # 记录统计
        self.statistics.add_game(
            winner, self.move_count, 
            self.env.scores[1], self.env.scores[-1]
        )
        
        # 显示结果
        if winner == 1:
            result_text = "红方AI获胜！"
        elif winner == -1:
            result_text = "黑方AI获胜！"
        else:
            result_text = "平局！"
        
        self.log_message(f"第 {self.current_game} 局结束: {result_text}")
        self.log_message(f"总回合数: {self.move_count}, 最终分数: 红方 {self.env.scores[1]}, 黑方 {self.env.scores[-1]}")
        
        # 更新统计显示
        self.update_statistics_display()
        
        # 继续下一局或结束
        if self.is_running:
            delay = max(1000, self.delay_slider.value())  # 至少1秒间隔
            QTimer.singleShot(delay, self.start_next_game)
    
    def finish_games(self):
        """完成所有对弈"""
        self.is_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        stats = self.statistics.get_stats()
        self.log_message(f"所有对弈完成！总局数: {stats['games']}")
        self.log_message(f"红方胜率: {stats['red_win_rate']:.1f}%, 黑方胜率: {stats['black_win_rate']:.1f}%")
        
        self.status_label.setText("对弈完成")
        self.progress_label.setText("对弈完成")
    
    def stop_games(self):
        """停止对弈"""
        self.is_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self.log_message("对弈被手动停止")
        self.status_label.setText("对弈已停止")
    
    def reset_statistics(self):
        """重置统计"""
        self.statistics.reset()
        self.update_statistics_display()
        self.log_message("统计信息已重置")
    
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
    
    def update_statistics_display(self):
        """更新统计显示"""
        stats = self.statistics.get_stats()
        
        stats_text = f"""对弈统计 ({stats['games']} 局)
━━━━━━━━━━━━━━━━━━━━━━━━
胜负统计:
  红方胜利: {stats['red_wins']} 局 ({stats['red_win_rate']:.1f}%)
  黑方胜利: {stats['black_wins']} 局 ({stats['black_win_rate']:.1f}%)
  平局: {stats['draws']} 局 ({stats['draw_rate']:.1f}%)

游戏数据:
  平均回合数: {stats['avg_moves']:.1f}
  平均分数: 红方 {stats['avg_red_score']:.1f}, 黑方 {stats['avg_black_score']:.1f}
━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        self.stats_text.setText(stats_text)
    
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
    window = AIvsAIGUI()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
