# rllib_version_complete/utils/elo.py

import json
import os
from typing import Dict

from .constants import SELF_PLAY_OUTPUT_DIR, ELO_DEFAULT, ELO_K_FACTOR

ELO_RATINGS_FILE = os.path.join(SELF_PLAY_OUTPUT_DIR, "elo_ratings.json")

def load_elo_ratings() -> Dict[str, float]:
    """从文件加载Elo评分。如果文件不存在，返回空字典。"""
    if os.path.exists(ELO_RATINGS_FILE):
        try:
            with open(ELO_RATINGS_FILE, 'r') as f:
                return json.load().get("elo", {})
        except (json.JSONDecodeError, IOError):
            print(f"警告: 无法读取或解析Elo文件: {ELO_RATINGS_FILE}。将使用默认值。")
            return {}
    return {}

def save_elo_ratings(elo_ratings: Dict[str, float]):
    """将Elo评分保存到文件。"""
    try:
        with open(ELO_RATINGS_FILE, 'w') as f:
            json.dump({"elo": elo_ratings}, f, indent=4)
    except IOError:
        print(f"错误: 无法写入Elo文件: {ELO_RATINGS_FILE}")

def update_elo(
    elo_ratings: Dict[str, float],
    player_a_id: str,
    player_b_id: str,
    player_a_win_rate: float
) -> Dict[str, float]:
    """
    根据单场比赛的胜率更新两个玩家的Elo评分。

    Args:
        elo_ratings: 当前所有玩家的Elo字典。
        player_a_id: 玩家A的ID。
        player_b_id: 玩家B的ID。
        player_a_win_rate: 玩家A对B的胜率 (0.0 to 1.0)。

    Returns:
        更新后的Elo字典。
    """
    player_a_elo = elo_ratings.get(player_a_id, ELO_DEFAULT)
    player_b_elo = elo_ratings.get(player_b_id, ELO_DEFAULT)

    # Elo公式：计算期望胜率
    expected_win_a = 1 / (1 + 10 ** ((player_b_elo - player_a_elo) / 400))

    # 更新Elo
    new_player_a_elo = player_a_elo + ELO_K_FACTOR * (player_a_win_rate - expected_win_a)
    # B的得分变化与A相反
    new_player_b_elo = player_b_elo - ELO_K_FACTOR * (player_a_win_rate - expected_win_a)

    elo_ratings[player_a_id] = new_player_a_elo
    elo_ratings[player_b_id] = new_player_b_elo

    return elo_ratings
