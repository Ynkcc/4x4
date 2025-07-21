#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI版本启动脚本
"""

import sys
import os
import subprocess
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap, QIcon

class LauncherDialog(QDialog):
    """启动器对话框"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4x4暗棋 GUI 启动器")
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("4x4暗棋 AI对弈系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel("选择您要启动的模式")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setStyleSheet("color: #666666; margin-bottom: 20px;")
        layout.addWidget(subtitle_label)
        
        # 人机对弈按钮
        human_button = QPushButton("🎮 人机对弈")
        human_button.setFont(QFont("Arial", 14))
        human_button.setFixedHeight(60)
        human_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        human_button.clicked.connect(self.launch_human_vs_ai)
        layout.addWidget(human_button)
        
        # AI对弈观察按钮
        ai_button = QPushButton("🤖 AI对弈观察")
        ai_button.setFont(QFont("Arial", 14))
        ai_button.setFixedHeight(60)
        ai_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        ai_button.clicked.connect(self.launch_ai_vs_ai)
        layout.addWidget(ai_button)
        
        # 分隔线
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #E0E0E0; margin: 20px 0;")
        layout.addWidget(separator)
        
        # 控制台版本按钮
        console_label = QLabel("控制台版本:")
        console_label.setFont(QFont("Arial", 10))
        console_label.setStyleSheet("color: #666666;")
        layout.addWidget(console_label)
        
        console_layout = QHBoxLayout()
        
        console_human_button = QPushButton("控制台人机对弈")
        console_human_button.setFont(QFont("Arial", 10))
        console_human_button.clicked.connect(self.launch_console_human)
        console_layout.addWidget(console_human_button)
        
        console_ai_button = QPushButton("控制台AI对弈")
        console_ai_button.setFont(QFont("Arial", 10))
        console_ai_button.clicked.connect(self.launch_console_ai)
        console_layout.addWidget(console_ai_button)
        
        layout.addLayout(console_layout)
        
        # 退出按钮
        exit_button = QPushButton("退出")
        exit_button.clicked.connect(self.close)
        layout.addWidget(exit_button)
        
        # 检查依赖
        self.check_dependencies()
    
    def check_dependencies(self):
        """检查依赖包"""
        try:
            import PySide6
            import stable_baselines3
            import sb3_contrib
            import gymnasium
        except ImportError as e:
            QMessageBox.warning(self, "依赖检查", f"缺少依赖包: {str(e)}\n\n请运行:\npip install PySide6 stable-baselines3[extra] sb3-contrib gymnasium")
    
    def launch_human_vs_ai(self):
        """启动人机对弈GUI"""
        try:
            self.close()
            subprocess.run([sys.executable, "gui_game.py"])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动失败: {str(e)}")
    
    def launch_ai_vs_ai(self):
        """启动AI对弈观察GUI"""
        try:
            self.close()
            subprocess.run([sys.executable, "gui_ai_vs_ai.py"])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动失败: {str(e)}")
    
    def launch_console_human(self):
        """启动控制台人机对弈"""
        try:
            self.close()
            subprocess.run([sys.executable, "play_with_human.py"])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动失败: {str(e)}")
    
    def launch_console_ai(self):
        """启动控制台AI对弈"""
        try:
            self.close()
            subprocess.run([sys.executable, "ai_vs_ai.py"])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动失败: {str(e)}")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 检查是否有训练好的模型
    log_dir = "./banqi_ppo_logs/"
    model_exists = False
    
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            if file.endswith(".zip"):
                model_exists = True
                break
    
    if not model_exists:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("提示")
        msg.setText("未找到训练好的AI模型！\n\n请先运行训练脚本:\npython train.py")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.button(QMessageBox.Ok).setText("继续")
        msg.button(QMessageBox.Cancel).setText("退出")
        
        if msg.exec() == QMessageBox.Cancel:
            sys.exit(0)
    
    # 显示启动器
    dialog = LauncherDialog()
    dialog.exec()

if __name__ == "__main__":
    main()
