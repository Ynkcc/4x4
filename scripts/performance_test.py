#!/usr/bin/env python3
"""
性能测试脚本 - 测试暗棋游戏环境的性能
使用固定随机种子完成1000局游戏，并统计各项性能指标
支持对比新旧版本的性能差异
新增 'profile' 模式，用于深度性能剖析
"""

import time
import numpy as np
import random
import statistics
from collections import defaultdict
import sys
import os
import cProfile
import pstats
from io import StringIO

# 添加路径以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bitboard_version'))

# 导入原版游戏环境
try:
    from original_version.Game import GameEnvironment as OriginalGameEnvironment
    ORIGINAL_VERSION_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 original_version/Game.py，将跳过原版测试")
    ORIGINAL_VERSION_AVAILABLE = False

# 导入 Bitboard 版本游戏环境
try:
    from Game_bitboard import GameEnvironment as BitboardGameEnvironment
    BITBOARD_VERSION_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 bitboard_version/Game_bitboard.py，将跳过 Bitboard 测试")
    BITBOARD_VERSION_AVAILABLE = False

# 导入 Cython 优化版本
try:
    from Game_cython import GameEnvironment as CythonGameEnvironment
    CYTHON_VERSION_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 Game_cython，将跳过 Cython 测试")
    CYTHON_VERSION_AVAILABLE = False


class PerformanceTester:
    """游戏环境性能测试器"""
    
    def __init__(self, random_seed=42):
        """
        初始化性能测试器
        
        Args:
            random_seed: 固定的随机种子，确保测试结果可重现
        """
        self.random_seed = random_seed
        self.reset_random_seeds()
        
    def reset_random_seeds(self):
        """重置所有随机种子"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def run_single_game(self, env, max_steps=1000):
        """
        运行单局游戏并收集统计信息
        
        Args:
            env: 游戏环境
            max_steps: 最大步数限制
            
        Returns:
            dict: 包含游戏统计信息的字典
        """
        obs, info = env.reset()
        total_steps = 0
        total_reward = 0
        reveal_actions = 0
        move_actions = 0
        cannon_actions = 0
        
        action_times = []
        step_times = []
        
        # 获取动作空间大小（适应新旧版本）
        action_space_size = env.action_space.n
        
        for step in range(max_steps):
            # 测量动作选择时间
            action_start = time.perf_counter()
            
            # 获取有效动作并随机选择
            action_mask = info.get('action_mask', np.ones(action_space_size))
            valid_actions = np.where(action_mask)[0]
            
            if len(valid_actions) == 0:
                break
                
            action = np.random.choice(valid_actions)
            action_time = time.perf_counter() - action_start
            action_times.append(action_time)
            
            # 统计动作类型（根据版本调整）
            if hasattr(env, 'ACTION_SPACE_SIZE') and env.ACTION_SPACE_SIZE == 112:
                # 新版本 (bitboard)
                if action < 16:  # 翻棋动作
                    reveal_actions += 1
                elif action < 64:  # 普通移动
                    move_actions += 1
                else:  # 炮击动作
                    cannon_actions += 1
            else:
                # 旧版本 (每个位置5个动作: 上,下,左,右,翻)
                action_type = action % 5
                if action_type == 4:  # 翻棋
                    reveal_actions += 1
                elif action_type in [0, 1, 2, 3]:  # 移动/攻击
                    move_actions += 1
                # 旧版本没有专门的炮击动作类型统计
            
            # 测量环境步进时间
            step_start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
            
            total_reward += reward
            total_steps += 1
            
            if terminated or truncated:
                break
        
        return {
            'steps': total_steps,
            'total_reward': total_reward,
            'reveal_actions': reveal_actions,
            'move_actions': move_actions,
            'cannon_actions': cannon_actions,
            'avg_action_time': np.mean(action_times) if action_times else 0,
            'avg_step_time': np.mean(step_times) if step_times else 0,
            'max_step_time': np.max(step_times) if step_times else 0,
            'min_step_time': np.min(step_times) if step_times else 0
        }
    
    def run_performance_test(self, num_games=1000, max_steps_per_game=1000, version='new'):
        """
        运行完整的性能测试
        
        Args:
            num_games: 要运行的游戏局数
            max_steps_per_game: 每局游戏的最大步数
            version: 'new' 或 'old' 或 'both' 选择测试版本
            
        Returns:
            dict: 性能测试结果统计
        """
        if version == 'both':
            return self.run_comparison_test(num_games, max_steps_per_game)
        
        print(f"开始性能测试 - {num_games}局游戏 ({'新版本' if version == 'new' else '旧版本'})")
        print(f"使用随机种子: {self.random_seed}")
        print("=" * 50)
        
        # 重置随机种子确保一致性
        self.reset_random_seeds()
        
        # 选择环境版本
        if version == 'original':
            if not ORIGINAL_VERSION_AVAILABLE:
                raise ValueError("原版环境不可用")
            env = OriginalGameEnvironment()
        elif version == 'bitboard':
            if not BITBOARD_VERSION_AVAILABLE:
                raise ValueError("Bitboard版本环境不可用")
            env = BitboardGameEnvironment()
        elif version == 'cython' and CYTHON_VERSION_AVAILABLE:
            env = CythonGameEnvironment()
        else:
            raise ValueError(f"不支持的版本: {version} 或对应版本不可用")
        
        env.reset(seed=self.random_seed)
        
        # 统计变量
        game_stats = []
        total_start_time = time.perf_counter()
        
        # 运行所有游戏
        for game_idx in range(num_games):
            if (game_idx + 1) % 100 == 0:
                print(f"已完成 {game_idx + 1}/{num_games} 局游戏...")
            
            game_start = time.perf_counter()
            stats = self.run_single_game(env, max_steps_per_game)
            game_time = time.perf_counter() - game_start
            stats['game_time'] = game_time
            stats['game_index'] = game_idx + 1
            
            game_stats.append(stats)
        
        total_time = time.perf_counter() - total_start_time
        
        # 计算汇总统计
        summary_stats = self._calculate_summary_stats(game_stats, total_time)
        summary_stats['version'] = version
        
        # 输出结果
        self._print_results(summary_stats, game_stats)
        
        return {
            'summary': summary_stats,
            'detailed_stats': game_stats
        }
    
    def run_comparison_test(self, num_games=1000, max_steps_per_game=1000):
        """
        运行三个版本的对比测试
        
        Args:
            num_games: 要运行的游戏局数
            max_steps_per_game: 每局游戏的最大步数
            
        Returns:
            dict: 包含三个版本对比的测试结果
        """
        available_versions = []
        if ORIGINAL_VERSION_AVAILABLE:
            available_versions.append(('original', '原版'))
        if BITBOARD_VERSION_AVAILABLE:
            available_versions.append(('bitboard', 'Bitboard版本'))
        if CYTHON_VERSION_AVAILABLE:
            available_versions.append(('cython', 'Cython优化版本'))
        
        if len(available_versions) < 2:
            print("错误: 至少需要两个版本才能进行对比测试")
            return None
        
        print(f"开始版本对比测试 - 每个版本{num_games}局游戏")
        print(f"使用随机种子: {self.random_seed}")
        print("=" * 60)
        
        results = {}
        
        # 测试每个可用版本
        for version_key, version_name in available_versions:
            print(f"\n正在测试{version_name}...")
            results[version_key] = self.run_performance_test(num_games, max_steps_per_game, version_key)
            print("\n" + "=" * 60)
        
        # 生成对比报告
        print("\n版本对比报告")
        self._print_multi_version_comparison(results)
        
        return results
    
    def run_comprehensive_test(self, num_games=1000, max_steps_per_game=1000):
        """
        运行全面的性能测试，包括所有可用版本
        """
        results = {}
        
        print(f"开始全面性能测试 - 每个版本{num_games}局游戏")
        print(f"使用随机种子: {self.random_seed}")
        print("=" * 60)
        
        # 测试原版
        if ORIGINAL_VERSION_AVAILABLE:
            print("\n正在测试原版...")
            results['original'] = self.run_performance_test(num_games, max_steps_per_game, 'original')
        
        # 测试Bitboard版本
        if BITBOARD_VERSION_AVAILABLE:
            print("\n" + "=" * 60)
            print("\n正在测试Bitboard版本...")
            results['bitboard'] = self.run_performance_test(num_games, max_steps_per_game, 'bitboard')
        
        # 测试 Cython 版本
        if CYTHON_VERSION_AVAILABLE:
            print("\n" + "=" * 60)
            print("\n正在测试Cython优化版本...")
            results['cython'] = self.run_performance_test(num_games, max_steps_per_game, 'cython')
        
        # 生成综合对比报告
        print("\n" + "=" * 60)
        self._print_comprehensive_report(results)
        
        return results
    
    def run_single_version_test(self, env_class, version_name, num_games=1000, max_steps_per_game=1000):
        """
        运行单个版本的性能测试
        """
        print(f"开始 {version_name} 性能测试 - {num_games}局游戏")
        print(f"使用随机种子: {self.random_seed}")
        print("=" * 60)
        
        # 确定版本标识
        if env_class == OriginalGameEnvironment:
            version = 'original'
        elif BITBOARD_VERSION_AVAILABLE and env_class == BitboardGameEnvironment:
            version = 'bitboard'
        elif CYTHON_VERSION_AVAILABLE and env_class == CythonGameEnvironment:
            version = 'cython'
        else:
            version = 'unknown'
        
        results = self.run_performance_test(num_games, max_steps_per_game, version)
        
        print("\n" + "=" * 60)
        print(f"{version_name} 性能测试结果:")
        print("=" * 60)
        self._print_single_version_report(results['summary'], version_name)
        
        return results
    
    def _calculate_summary_stats(self, game_stats, total_time):
        """计算汇总统计信息"""
        if not game_stats:
            return {}
        
        # 提取各项指标
        steps = [g['steps'] for g in game_stats]
        game_times = [g['game_time'] for g in game_stats]
        rewards = [g['total_reward'] for g in game_stats]
        reveal_actions = [g['reveal_actions'] for g in game_stats]
        move_actions = [g['move_actions'] for g in game_stats]
        cannon_actions = [g['cannon_actions'] for g in game_stats]
        avg_action_times = [g['avg_action_time'] for g in game_stats]
        avg_step_times = [g['avg_step_time'] for g in game_stats]
        
        return {
            'total_games': len(game_stats),
            'total_time': total_time,
            'avg_time_per_game': total_time / len(game_stats),
            'games_per_second': len(game_stats) / total_time,
            
            # 步数统计
            'avg_steps_per_game': statistics.mean(steps),
            'median_steps_per_game': statistics.median(steps),
            'min_steps_per_game': min(steps),
            'max_steps_per_game': max(steps),
            'std_steps_per_game': statistics.stdev(steps) if len(steps) > 1 else 0,
            
            # 时间统计
            'avg_game_time': statistics.mean(game_times),
            'median_game_time': statistics.median(game_times),
            'min_game_time': min(game_times),
            'max_game_time': max(game_times),
            
            # 奖励统计
            'avg_total_reward': statistics.mean(rewards),
            'median_total_reward': statistics.median(rewards),
            'min_total_reward': min(rewards),
            'max_total_reward': max(rewards),
            
            # 动作类型统计
            'avg_reveal_actions': statistics.mean(reveal_actions),
            'avg_move_actions': statistics.mean(move_actions),
            'avg_cannon_actions': statistics.mean(cannon_actions),
            
            # 性能统计
            'avg_action_selection_time': statistics.mean(avg_action_times),
            'avg_env_step_time': statistics.mean(avg_step_times),
            'total_steps': sum(steps),
            'steps_per_second': sum(steps) / total_time
        }
    
    def _print_results(self, summary_stats, game_stats):
        """打印测试结果"""
        print("\n" + "=" * 50)
        print("性能测试结果")
        print("=" * 50)
        
        print(f"总游戏局数: {summary_stats['total_games']}")
        print(f"总测试时间: {summary_stats['total_time']:.2f} 秒")
        print(f"平均每局游戏时间: {summary_stats['avg_time_per_game']:.4f} 秒")
        print(f"游戏执行速度: {summary_stats['games_per_second']:.2f} 局/秒")
        
        print("\n--- 游戏步数统计 ---")
        print(f"平均步数: {summary_stats['avg_steps_per_game']:.2f}")
        print(f"中位数步数: {summary_stats['median_steps_per_game']:.2f}")
        print(f"最少步数: {summary_stats['min_steps_per_game']}")
        print(f"最多步数: {summary_stats['max_steps_per_game']}")
        print(f"步数标准差: {summary_stats['std_steps_per_game']:.2f}")
        
        print("\n--- 游戏时间统计 ---")
        print(f"平均游戏时间: {summary_stats['avg_game_time']:.4f} 秒")
        print(f"中位数游戏时间: {summary_stats['median_game_time']:.4f} 秒")
        print(f"最短游戏时间: {summary_stats['min_game_time']:.4f} 秒")
        print(f"最长游戏时间: {summary_stats['max_game_time']:.4f} 秒")
        
        print("\n--- 奖励统计 ---")
        print(f"平均总奖励: {summary_stats['avg_total_reward']:.3f}")
        print(f"中位数总奖励: {summary_stats['median_total_reward']:.3f}")
        print(f"最小总奖励: {summary_stats['min_total_reward']:.3f}")
        print(f"最大总奖励: {summary_stats['max_total_reward']:.3f}")
        
        print("\n--- 动作类型统计 ---")
        print(f"平均翻棋动作数: {summary_stats['avg_reveal_actions']:.2f}")
        print(f"平均移动动作数: {summary_stats['avg_move_actions']:.2f}")
        print(f"平均炮击动作数: {summary_stats['avg_cannon_actions']:.2f}")
        
        print("\n--- 性能指标 ---")
        print(f"总步数: {summary_stats['total_steps']}")
        print(f"步数执行速度: {summary_stats['steps_per_second']:.2f} 步/秒")
        print(f"平均动作选择时间: {summary_stats['avg_action_selection_time']*1000:.3f} 毫秒")
        print(f"平均环境步进时间: {summary_stats['avg_env_step_time']*1000:.3f} 毫秒")
        
        # 找出最快和最慢的游戏
        fastest_game = min(game_stats, key=lambda x: x['game_time'])
        slowest_game = max(game_stats, key=lambda x: x['game_time'])
        
        print("\n--- 极值分析 ---")
        print(f"最快游戏: 第{fastest_game['game_index']}局，用时{fastest_game['game_time']:.4f}秒，{fastest_game['steps']}步")
        print(f"最慢游戏: 第{slowest_game['game_index']}局，用时{slowest_game['game_time']:.4f}秒，{slowest_game['steps']}步")

    def _calculate_comparison_metrics(self, new_stats, old_stats):
        """计算新旧版本的对比指标"""
        comparison = {}
        
        # 计算性能提升比例
        metrics_to_compare = [
            'games_per_second', 'steps_per_second', 'avg_time_per_game',
            'avg_game_time', 'avg_env_step_time', 'avg_action_selection_time'
        ]
        
        for metric in metrics_to_compare:
            old_val = old_stats.get(metric, 0)
            new_val = new_stats.get(metric, 0)
            
            if old_val != 0:
                if metric in ['avg_time_per_game', 'avg_game_time', 'avg_env_step_time', 'avg_action_selection_time']:
                    # 对于时间指标，值越小越好
                    improvement = (old_val - new_val) / old_val * 100
                else:
                    # 对于速度指标，值越大越好
                    improvement = (new_val - old_val) / old_val * 100
                
                comparison[f'{metric}_improvement_percent'] = improvement
            else:
                comparison[f'{metric}_improvement_percent'] = 0
        
        return comparison
    
    def _print_multi_version_comparison(self, results):
        """打印多版本对比报告"""
        print("=" * 80)
        
        # 收集所有版本的统计信息
        versions = []
        version_names = {
            'original': '原版',
            'bitboard': 'Bitboard版本', 
            'cython': 'Cython优化版本'
        }
        
        for version_key, result in results.items():
            version_name = version_names.get(version_key, version_key)
            versions.append((version_name, result['summary']))
        
        if len(versions) < 2:
            print("需要至少两个版本才能进行对比")
            return
        
        # 打印性能对比表
        print(f"{'版本':<20} {'游戏/秒':<12} {'步/秒':<12} {'平均游戏时间(ms)':<18} {'平均步时间(ms)':<16}")
        print("-" * 80)
        
        for version_name, stats in versions:
            games_per_sec = stats.get('games_per_second', 0)
            steps_per_sec = stats.get('steps_per_second', 0)
            avg_game_time = stats.get('avg_game_time', 0) * 1000  # 转换为毫秒
            avg_step_time = stats.get('avg_env_step_time', 0) * 1000  # 转换为毫秒
            
            print(f"{version_name:<20} {games_per_sec:<12.1f} {steps_per_sec:<12.1f} {avg_game_time:<18.3f} {avg_step_time:<16.5f}")
        
        # 计算相对于第一个版本的加速比
        if len(versions) > 1:
            baseline_name, baseline = versions[0]
            print(f"\n加速比 (相对于{baseline_name}):")
            print("-" * 50)
            
            for version_name, stats in versions[1:]:
                speedup_games = stats.get('games_per_second', 0) / baseline.get('games_per_second', 1)
                speedup_steps = stats.get('steps_per_second', 0) / baseline.get('steps_per_second', 1)
                
                print(f"{version_name}: 游戏执行 {speedup_games:.2f}x, 步执行 {speedup_steps:.2f}x")

    def _print_comparison_report(self, new_stats, old_stats):
        """打印详细的对比报告 (保留向后兼容性)"""
        print("版本性能对比报告")
        print("=" * 60)
        
        # 主要性能指标对比
        print("\n--- 主要性能指标对比 ---")
        print(f"{'指标':<25} {'版本1':<15} {'版本2':<15} {'提升':<10}")
        print("-" * 65)
        
        metrics = [
            ('游戏执行速度(局/秒)', 'games_per_second', '{:.2f}'),
            ('步数执行速度(步/秒)', 'steps_per_second', '{:.2f}'),
            ('平均每局时间(秒)', 'avg_time_per_game', '{:.4f}'),
            ('平均游戏时间(秒)', 'avg_game_time', '{:.4f}'),
            ('环境步进时间(毫秒)', 'avg_env_step_time', '{:.3f}'),
            ('动作选择时间(毫秒)', 'avg_action_selection_time', '{:.3f}'),
        ]
        
        comparison = self._calculate_comparison_metrics(new_stats, old_stats)
        
        for name, key, fmt in metrics:
            new_val = new_stats.get(key, 0)
            old_val = old_stats.get(key, 0)
            
            # 对时间指标转换为毫秒显示
            if 'time' in key and key != 'avg_time_per_game' and key != 'avg_game_time':
                new_val *= 1000
                old_val *= 1000
            
            improvement = comparison.get(f'{key}_improvement_percent', 0)
            improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "N/A"
            
            print(f"{name:<25} {fmt.format(new_val):<15} {fmt.format(old_val):<15} {improvement_str:<10}")
        
        # 总结
        print("\n--- 性能提升总结 ---")
        game_speed_improvement = comparison.get('games_per_second_improvement_percent', 0)
        step_speed_improvement = comparison.get('steps_per_second_improvement_percent', 0)
        game_time_improvement = comparison.get('avg_game_time_improvement_percent', 0)
        
        print(f"游戏执行速度提升: {game_speed_improvement:+.1f}%")
        print(f"步数执行速度提升: {step_speed_improvement:+.1f}%")
        print(f"游戏时间减少: {game_time_improvement:+.1f}%")
        
        if game_speed_improvement > 0:
            print(f"\n✓ 版本1在游戏执行速度上有 {game_speed_improvement:.1f}% 的提升")
        elif game_speed_improvement < 0:
            print(f"\n✗ 版本1在游戏执行速度上有 {abs(game_speed_improvement):.1f}% 的下降")
        else:
            print(f"\n- 两个版本在游戏执行速度上基本相同")
        
        # 内存使用和其他指标对比
        print("\n--- 游戏特征对比 ---")
        print(f"{'特征':<20} {'版本1':<15} {'版本2':<15}")
        print("-" * 50)
        
        feature_metrics = [
            ('平均步数', 'avg_steps_per_game'),
            ('平均翻棋动作', 'avg_reveal_actions'),
            ('平均移动动作', 'avg_move_actions'),
            ('平均炮击动作', 'avg_cannon_actions'),
            ('平均总奖励', 'avg_total_reward'),
        ]
        
        for name, key in feature_metrics:
            new_val = new_stats.get(key, 0)
            old_val = old_stats.get(key, 0)
            print(f"{name:<20} {new_val:<15.2f} {old_val:<15.2f}")

    def _print_comprehensive_report(self, results):
        """打印综合性能对比报告"""
        print("\n综合性能对比报告")
        print("=" * 80)
        
        # 收集所有版本的统计信息
        versions = []
        version_names = {
            'original': '原版',
            'bitboard': 'Bitboard版本',
            'cython': 'Cython优化版本'
        }
        
        for version_key, result in results.items():
            version_name = version_names.get(version_key, version_key)
            versions.append((version_name, result['summary']))
        
        if len(versions) < 2:
            print("需要至少两个版本才能进行对比")
            return
        
        # 打印性能对比表
        print(f"{'版本':<20} {'游戏/秒':<12} {'步/秒':<12} {'平均游戏时间(ms)':<18} {'平均步时间(ms)':<16}")
        print("-" * 80)
        
        for version_name, stats in versions:
            games_per_sec = stats.get('games_per_second', 0)
            steps_per_sec = stats.get('steps_per_second', 0)
            avg_game_time = stats.get('avg_game_time', 0) * 1000  # 转换为毫秒
            avg_step_time = stats.get('avg_env_step_time', 0) * 1000  # 转换为毫秒
            
            print(f"{version_name:<20} {games_per_sec:<12.1f} {steps_per_sec:<12.1f} {avg_game_time:<18.3f} {avg_step_time:<16.5f}")
        
        # 计算相对于第一个版本的加速比
        if len(versions) > 1:
            baseline_name, baseline = versions[0]
            print(f"\n加速比 (相对于{baseline_name}):")
            print("-" * 50)
            
            for version_name, stats in versions[1:]:
                speedup_games = stats.get('games_per_second', 0) / baseline.get('games_per_second', 1)
                speedup_steps = stats.get('steps_per_second', 0) / baseline.get('steps_per_second', 1)
                
                print(f"{version_name}: 游戏执行 {speedup_games:.2f}x, 步执行 {speedup_steps:.2f}x")

    def _print_single_version_report(self, stats, version_name):
        """打印单个版本的性能报告"""
        print(f"{version_name} 性能统计:")
        print("-" * 40)
        print(f"总游戏局数: {stats.get('total_games', 0)}")
        print(f"总执行时间: {stats.get('total_time', 0):.3f} 秒")
        print(f"平均每局时间: {stats.get('avg_game_time', 0)*1000:.3f} 毫秒")
        print(f"游戏执行速度: {stats.get('games_per_second', 0):.1f} 局/秒")
        print(f"步执行速度: {stats.get('steps_per_second', 0):.1f} 步/秒")
        print(f"平均步时间: {stats.get('avg_step_time', 0)*1000:.5f} 毫秒")
        print(f"平均每局步数: {stats.get('avg_steps_per_game', 0):.1f}")
        print(f"平均总奖励: {stats.get('avg_total_reward', 0):.2f}")
        
        # 动作类型统计
        print(f"\n动作类型分布:")
        print(f"平均翻棋动作: {stats.get('avg_reveal_actions', 0):.1f}")
        print(f"平均移动动作: {stats.get('avg_move_actions', 0):.1f}")
        print(f"平均炮击动作: {stats.get('avg_cannon_actions', 0):.1f}")

# ==============================================================================
# --- 新增功能：深度性能剖析 ---
# ==============================================================================
def profile_environment(env_class, version_name, num_games=10, max_steps=1000, random_seed=42):
    """
    使用 cProfile 对指定的游戏环境运行多局游戏进行性能剖析

    Args:
        env_class: 要测试的环境类 (e.g., NewGameEnvironment)
        version_name: 版本名称，用于显示 (e.g., "新版本")
        num_games: 用于剖析的游戏局数
        max_steps: 每局最大步数
        random_seed: 随机种子
    """
    print("\n" + "=" * 60)
    print(f"开始对 {version_name} 进行深度性能剖析 ({num_games} 局游戏)")
    print("=" * 60)

    # 初始化环境和profiler
    env = env_class()
    env.reset(seed=random_seed)
    profiler = cProfile.Profile()
    
    # 将游戏循环置于 profiler 的监控下
    profiler.enable()
    
    for _ in range(num_games):
        obs, info = env.reset()
        for step in range(max_steps):
            action_mask = info.get('action_mask', np.ones(env.action_space.n))
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    
    profiler.disable()
    
    # 打印剖析结果
    s = StringIO()
    # sort_stats('cumulative') 会将时间花在最深层函数调用栈上的函数排在最前面
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30) # 打印前30个最耗时的函数
    
    print(s.getvalue())
    print(f"--- {version_name} 剖析报告结束 ---\n")
    print("报告解读提示:")
    print("  - ncalls: 函数被调用的总次数")
    print("  - tottime: 函数本身执行所花费的总时间 (不包括其调用的子函数)")
    print("  - percall (tottime/ncalls): 函数单次执行的平均时间")
    print("  - cumtime: 函数及其所有子函数执行所花费的累计时间")
    print("  - percall (cumtime/ncalls): 函数单次调用（包括子函数）的平均时间")
    print("  - filename:lineno(function): 函数位置")
    print("\n请重点关注 'cumtime' 和 'tottime' 最高的函数，它们是性能瓶颈所在。")


def main():
    """主函数 - 运行性能测试"""
    # 可配置参数
    RANDOM_SEED = 42
    NUM_GAMES = 1000
    MAX_STEPS_PER_GAME = 1000
    PROFILE_GAMES = 10 # 用于profile的局数，不宜过多，否则报告太长
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        test_mode = sys.argv[1].lower()
        if test_mode not in ['original', 'bitboard', 'cython', 'both', 'all', 'profile']:
            print("用法: python performance_test.py [original|bitboard|cython|both|all|profile]")
            print("  original - 只测试原版")
            print("  bitboard - 只测试Bitboard版本")
            print("  cython   - 只测试Cython优化版本")
            print("  both     - 对比测试可用版本")
            print("  all      - 测试所有可用版本")
            print("  profile  - 对所有版本进行深度性能剖析")
            sys.exit(1)
    else:
        test_mode = 'all'  # 默认测试所有版本
    
    # 检查版本可用性
    if test_mode in ['original', 'both', 'all', 'profile'] and not ORIGINAL_VERSION_AVAILABLE:
        print("警告: 原版不可用")
        
    if test_mode in ['bitboard', 'both', 'all', 'profile'] and not BITBOARD_VERSION_AVAILABLE:
        print("警告: Bitboard版本不可用")
        
    if test_mode in ['cython', 'both', 'all', 'profile'] and not CYTHON_VERSION_AVAILABLE:
        print("警告: Cython优化版本不可用")
            
    tester = PerformanceTester(random_seed=RANDOM_SEED)

    # --- 根据模式选择执行的逻辑 ---
    
    if test_mode == 'profile':
        print("开始深度性能剖析...")
        if BITBOARD_VERSION_AVAILABLE:
            profile_environment(BitboardGameEnvironment, "Bitboard版本", num_games=PROFILE_GAMES, random_seed=RANDOM_SEED)
        if ORIGINAL_VERSION_AVAILABLE:
            profile_environment(OriginalGameEnvironment, "原版", num_games=PROFILE_GAMES, random_seed=RANDOM_SEED)
        if CYTHON_VERSION_AVAILABLE:
            profile_environment(CythonGameEnvironment, "Cython优化版本", num_games=PROFILE_GAMES, random_seed=RANDOM_SEED)

    elif test_mode == 'all':
        print("开始全面性能对比测试...")
        tester.run_comprehensive_test(
            num_games=NUM_GAMES,
            max_steps_per_game=MAX_STEPS_PER_GAME
        )
    
    elif test_mode == 'both':
        print("开始版本对比测试...")
        tester.run_comparison_test(
            num_games=NUM_GAMES,
            max_steps_per_game=MAX_STEPS_PER_GAME
        )
    
    elif test_mode == 'original':
        if ORIGINAL_VERSION_AVAILABLE:
            print("开始原版性能测试...")
            tester.run_single_version_test(
                OriginalGameEnvironment, "原版",
                num_games=NUM_GAMES,
                max_steps_per_game=MAX_STEPS_PER_GAME
            )
        else:
            print("错误: 原版不可用")
            sys.exit(1)
    
    elif test_mode == 'bitboard':
        if BITBOARD_VERSION_AVAILABLE:
            print("开始Bitboard版本性能测试...")
            tester.run_single_version_test(
                BitboardGameEnvironment, "Bitboard版本",
                num_games=NUM_GAMES,
                max_steps_per_game=MAX_STEPS_PER_GAME
            )
        else:
            print("错误: Bitboard版本不可用")
            sys.exit(1)
            
    elif test_mode == 'cython':
        if CYTHON_VERSION_AVAILABLE:
            print("开始Cython版本性能测试...")
            tester.run_single_version_test(
                CythonGameEnvironment, "Cython优化版本",
                num_games=NUM_GAMES,
                max_steps_per_game=MAX_STEPS_PER_GAME
            )
        else:
            print("错误: Cython版本不可用")
            sys.exit(1)

if __name__ == "__main__":
    main()