# test_game.py (Merged)

import pytest
import numpy as np
from Game import GameEnvironment, Piece, PieceType, ULL, SQ_TO_POS, POS_TO_SQ
from gymnasium import spaces

# ==============================================================================
# --- 测试环境的主要功能 ---
# ==============================================================================

class TestGameEnvironment:
    """测试GameEnvironment类的各项功能"""
    
    def setup_method(self):
        """在每个测试方法执行前，创建一个新的、干净的环境实例。"""
        self.env = GameEnvironment()
        
    def test_initialization_and_constants(self):
        """测试1.1: 环境初始化及核心常量"""
        assert self.env.ACTION_SPACE_SIZE == 80
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)
        assert self.env.action_space.n == 80
        assert self.env.MAX_CONSECUTIVE_MOVES == 40
        assert self.env.WINNING_SCORE == 60

        # 测试棋子价值
        expected_values = {
            PieceType.A: 4, PieceType.B: 10, PieceType.C: 10, PieceType.D: 10,
            PieceType.E: 10, PieceType.F: 20, PieceType.G: 30
        }
        assert self.env.PIECE_VALUES == expected_values
        
        # 测试棋子数量
        expected_counts = {
            PieceType.A: 2, PieceType.B: 1, PieceType.C: 1, PieceType.D: 1,
            PieceType.E: 1, PieceType.F: 1, PieceType.G: 1
        }
        assert self.env.PIECE_MAX_COUNTS == expected_counts
        
        # 测试预计算表
        assert 'king' in self.env.attack_tables
        assert 'rays' in self.env.attack_tables
        assert 'move_to_action' in self.env.attack_tables

    def test_reset(self):
        """测试1.2 & 1.3: 环境重置功能"""
        state, info = self.env.reset(seed=42)
        
        assert isinstance(state, np.ndarray) and state.shape == (self.env.state_size,)
        assert isinstance(info, dict) and 'action_mask' in info
        assert len(info['action_mask']) == self.env.ACTION_SPACE_SIZE
        assert self.env.current_player == 1
        assert self.env.move_counter == 0
        assert self.env.scores == {-1: 0, 1: 0}
        assert len([p for p in self.env.board if p is not None]) == 16
        assert all(not p.revealed for p in self.env.board if p is not None)
        assert np.sum(info['action_mask']) == 16  # 初始时只有16个翻棋动作

    def test_can_attack_rules(self):
        """测试2.4: 验证 'can_attack' 方法严格遵守新的棋子等级规则"""
        p1 = 1
        p2 = -1
        
        # 创建所有棋子类型
        soldier_p1 = Piece(PieceType.A, p1); soldier_p1.revealed = True  # 兵 value=0
        cannon_p1 = Piece(PieceType.B, p1); cannon_p1.revealed = True    # 炮 value=1
        horse_p1 = Piece(PieceType.C, p1); horse_p1.revealed = True      # 马 value=2
        chariot_p1 = Piece(PieceType.D, p1); chariot_p1.revealed = True  # 车 value=3
        elephant_p1 = Piece(PieceType.E, p1); elephant_p1.revealed = True # 象 value=4
        guard_p1 = Piece(PieceType.F, p1); guard_p1.revealed = True      # 仕 value=5
        general_p1 = Piece(PieceType.G, p1); general_p1.revealed = True  # 将 value=6
        
        # 对方棋子
        soldier_p2 = Piece(PieceType.A, p2); soldier_p2.revealed = True  # 兵 value=0
        cannon_p2 = Piece(PieceType.B, p2); cannon_p2.revealed = True    # 炮 value=1
        horse_p2 = Piece(PieceType.C, p2); horse_p2.revealed = True      # 马 value=2
        chariot_p2 = Piece(PieceType.D, p2); chariot_p2.revealed = True  # 车 value=3
        elephant_p2 = Piece(PieceType.E, p2); elephant_p2.revealed = True # 象 value=4
        guard_p2 = Piece(PieceType.F, p2); guard_p2.revealed = True      # 仕 value=5
        general_p2 = Piece(PieceType.G, p2); general_p2.revealed = True  # 将 value=6

        # 特殊规则: 兵 > 将
        assert self.env.can_attack(soldier_p1, general_p2) == True
        assert self.env.can_attack(general_p2, soldier_p1) == False

        # 同类型棋子可以互吃
        assert self.env.can_attack(general_p2, general_p1) == True
        assert self.env.can_attack(general_p1, general_p2) == True

        # 只能吃比自己等级低的棋子
        assert self.env.can_attack(cannon_p1, soldier_p2) == True
        assert self.env.can_attack(soldier_p2, cannon_p1) == False
        
        # 不允许吃同阵营已翻开棋子
        assert self.env.can_attack(general_p1, guard_p1) == False
        assert self.env.can_attack(guard_p1, elephant_p1) == False
        assert self.env.can_attack(elephant_p1, chariot_p1) == False
        assert self.env.can_attack(chariot_p1, horse_p1) == False
        assert self.env.can_attack(horse_p1, cannon_p1) == False
        assert self.env.can_attack(cannon_p1, soldier_p1) == False
        
        # 测试未翻开的棋子不能攻击
        hidden_soldier = Piece(PieceType.A, p1)  # 未翻开
        assert self.env.can_attack(hidden_soldier, general_p2) == False
        assert self.env.can_attack(horse_p2, hidden_soldier) == False
        
        # 测试非炮棋子只能攻击已翻开的棋子
        hidden_enemy = Piece(PieceType.A, p2)  # 未翻开的敌方棋子
        assert self.env.can_attack(guard_p1, hidden_enemy) == False
        assert self.env.can_attack(horse_p1, hidden_enemy) == False
        
        # 测试炮的特殊规则 - 炮可以攻击任何对方棋子,未翻开的棋子
        assert self.env.can_attack(cannon_p1, hidden_enemy) == True
        assert self.env.can_attack(cannon_p1, hidden_soldier) == True
        assert self.env.can_attack(cannon_p1, general_p2) == True  # 炮也可以攻击将

        # 测试炮不能攻击已翻开的己方棋子
        hidden_friend = Piece(PieceType.A, p1)  # 未翻开的己方棋子
        assert self.env.can_attack(cannon_p1, hidden_friend) == True
        assert self.env.can_attack(cannon_p1, soldier_p1) == False  # 炮不能攻击己方已翻开的棋子

    def test_action_reveal(self):
        """测试2.1: 翻棋动作的正确性 (确定性测试)"""
        self.env.reset(seed=42)
        action_index = 5 * 5 + 4  # 翻开位置在 sq=5 的棋子
        
        piece_to_reveal = self.env.board[5]
        player_of_piece = piece_to_reveal.player
        piecetype_of_piece = piece_to_reveal.piece_type

        self.env.step(action_index)

        assert piece_to_reveal.revealed == True
        assert not (self.env.hidden_pieces_bitboard & ULL(5))
        assert (self.env.piece_bitboards[player_of_piece][piecetype_of_piece.value] & ULL(5))
        assert self.env.move_counter == 0
        assert self.env.current_player == -1

    def test_cannon_logic(self):
        """测试2.5: 炮的攻击、移动限制和友军伤害限制规则 (确定性测试)"""
        # --- 手动设置棋盘状态 ---
        self.env.reset()
        self.env.board[0] = Piece(PieceType.B, 1);  self.env.board[0].revealed = True
        self.env.board[1] = Piece(PieceType.A, 1);  self.env.board[1].revealed = True
        self.env.board[2] = Piece(PieceType.D, -1); self.env.board[2].revealed = True
        self.env.board[3] = Piece(PieceType.C, 1);  self.env.board[3].revealed = True
        self.env.board[4] = None
        
        # --- 重建Bitboards以匹配棋盘 ---
        for p in [1, -1]:
            self.env.piece_bitboards[p] = [0] * self.env.NUM_PIECE_TYPES
            self.env.occupied_bitboards[p] = 0
        self.env.hidden_pieces_bitboard = 0
        self.env.occupied_bitboards[0] = 0
        for sq, piece in enumerate(self.env.board):
            if piece:
                self.env.piece_bitboards[piece.player][piece.piece_type.value] |= ULL(sq)
                self.env.occupied_bitboards[piece.player] |= ULL(sq)
        self.env.occupied_bitboards[0] = self.env.occupied_bitboards[1] | self.env.occupied_bitboards[-1]
        self.env.current_player = 1
        
        # --- 开始验证 ---
        mask = self.env.action_masks()
        attack_action_index = self.env.attack_tables['move_to_action'][0][1]
        
        # 1. 验证炮可以攻击敌方棋子 (P1炮在0, 隔着1, 攻击P2车在2)
        assert mask[attack_action_index] == 1, "炮应该能攻击敌方棋子"

        # 2. 验证炮不能移动到空格 (规则: 炮不能移动)
        # 尝试生成移动到 sq=4 的动作
        assert mask[self.env.attack_tables['move_to_action'][0].get(4, -1)] == 0, "炮不应该能移动到空格"

        # 3. 验证炮不能攻击己方棋子 (action_masks内部逻辑已阻止)
        # 炮在0, 炮架在1, 目标是己方马在3。该方向只有一个合法攻击目标(sq=2)，所以不会生成攻击己方棋子的动作。

    def test_termination_by_score(self):
        """测试4.1: 分数达到上限时游戏结束"""
        self.env.reset()
        self.env.board[0] = Piece(PieceType.G, 1); self.env.board[0].revealed = True
        self.env.board[1] = Piece(PieceType.A, -1); self.env.board[1].revealed = True
        
        self.env.scores[1] = self.env.WINNING_SCORE - 1
        self.env.current_player = 1
        
        action_index = self.env.attack_tables['move_to_action'][0][1]
        _, _, terminated, _, info = self.env.step(action_index)
        
        assert terminated == True, "达到胜利分数后，游戏应该终止"
        assert info['winner'] == 1, "获胜方应该是玩家1"

    def test_truncation_by_move_limit(self):
        """测试4.3: 连续移动达到上限时游戏平局"""
        self.env.reset()
        self.env.board[0] = Piece(PieceType.A, 1); self.env.board[0].revealed = True
        self.env.board[1] = None
        self.env.occupied_bitboards.update({1: ULL(0), 0: ULL(0)})

        self.env.move_counter = self.env.MAX_CONSECUTIVE_MOVES - 1
        self.env.current_player = 1

        action_index = self.env.attack_tables['move_to_action'][0][1]
        _, _, _, truncated, info = self.env.step(action_index)

        assert truncated == True, "达到移动次数上限后，游戏应该截断"
        assert info['winner'] == 0, "达到移动次数上限后，应该是平局"
        
    def test_player_switching(self):
        """测试玩家切换"""
        self.env.reset()
        initial_player = self.env.current_player
        self.env.step(4) # 执行任意翻棋动作
        assert self.env.current_player == -initial_player

    def test_bitboard_consistency(self):
        """测试Bitboard的一致性"""
        self.env.reset()
        total_pieces = bin(self.env.occupied_bitboards[0]).count('1')
        assert total_pieces == 16
        
        player1_pieces = bin(self.env.occupied_bitboards[1]).count('1')
        player_neg1_pieces = bin(self.env.occupied_bitboards[-1]).count('1')
        hidden_pieces = bin(self.env.hidden_pieces_bitboard).count('1')
        assert player1_pieces + player_neg1_pieces + hidden_pieces == 16

    def test_render(self):
        """测试渲染功能"""
        env_render = GameEnvironment(render_mode='human')
        env_render.reset()
        try:
            env_render.render()
        except Exception as e:
            pytest.fail(f"渲染失败: {e}")

# ==============================================================================
# --- 测试独立的Bitboard工具函数 ---
# ==============================================================================
class TestBitboardUtils:
    """测试Bitboard操作的单独测试类"""
    
    def test_ull_function(self):
        """测试ULL函数"""
        assert ULL(0) == 1
        assert ULL(2) == 4
        assert ULL(15) == 32768
        
    def test_position_conversion(self):
        """测试位置转换"""
        assert POS_TO_SQ[0, 0] == 0
        assert POS_TO_SQ[3, 3] == 15
        assert SQ_TO_POS[0] == (0, 0)
        assert SQ_TO_POS[15] == (3, 3)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])