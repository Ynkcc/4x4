// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/anqi_4x4/anqi_4x4.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace anqi_4x4 {

namespace {
// --- 游戏元数据 ---
// 定义了游戏的基本属性，如游戏类型、信息结构等。
const GameType kGameType{
    /*short_name=*/"anqi_4x4",
    /*long_name=*/"Chinese Dark Chess (Anqi) 4x4",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,  // 初始棋盘布局是随机的
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false, // 不提供
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<const AnqiGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// --- 动作编码/解码 ---
// 对应 Python 中的 _precompute_action_mappings，但采用动态计算方式
// 而非预计算查找表。
struct ActionInfo {
  enum Type { Reveal, Move, CannonAttack };
  Type type;
  int idx;  // 棋盘位置索引 (0-15)
  int dir;  // 0:上, 1:下, 2:左, 3:右
};

ActionInfo DecodeAction(Action action) {
  if (action < kMoveActionBase) {
    return {ActionInfo::Reveal, action - kRevealActionBase, -1};
  } else if (action < kCannonActionBase) {
    int rel_action = action - kMoveActionBase;
    return {ActionInfo::Move, rel_action / 4, rel_action % 4};
  } else {
    int rel_action = action - kCannonActionBase;
    return {ActionInfo::CannonAttack, rel_action / 4, rel_action % 4};
  }
}

Action EncodeAction(const ActionInfo& info) {
  if (info.type == ActionInfo::Reveal) {
    return kRevealActionBase + info.idx;
  } else if (info.type == ActionInfo::Move) {
    return kMoveActionBase + info.idx * 4 + info.dir;
  } else {  // CannonAttack
    return kCannonActionBase + info.idx * 4 + info.dir;
  }
}

// 棋子类型到名称的映射
std::string PieceTypeToString(PieceType piece_type) {
  switch (piece_type) {
    case PieceType::kSoldier:  return "S"; // 兵
    case PieceType::kCannon:   return "C"; // 炮
    case PieceType::kHorse:    return "H"; // 马
    case PieceType::kChariot:  return "R"; // 车
    case PieceType::kElephant: return "E"; // 象
    case PieceType::kAdvisor:  return "A"; // 士
    case PieceType::kGeneral:  return "G"; // 将
    default: SpielFatalError("Unknown piece type"); return "X";
  }
}

}  // namespace

// --- 坐标转换 ---
int PosToIdx(int r, int c) { return r * kBoardCols + c; }
int PosToIdx(const Pos& pos) { return pos.r * kBoardCols + pos.c; }
Pos IdxToPos(int idx) { return {idx / kBoardCols, idx % kBoardCols}; }

// --- Piece ---
std::string Piece::ToString() const {
  if (!revealed) return "??";
  std::string s;
  s += (player == 0 ? "R" : "B");  // 红/黑 (Red/Black)
  s += PieceTypeToString(type);
  return s;
}

// --- AnqiState ---

AnqiState::AnqiState(std::shared_ptr<const Game> game) : State(game) {
  InitializeBoard();
}

// 对应 Python 中的 _initialize_board 函数。
void AnqiState::InitializeBoard() {
  ResetInternalState();

  // 1. 创建双方所有棋子
  std::vector<Piece> pieces;
  for (int p = 0; p < kNumPlayers; ++p) {
    for (int i = 0; i < kNumPieceTypes; ++i) {
      PieceType pt = static_cast<PieceType>(i);
      for (int j = 0; j < kPieceMaxCounts[i]; ++j) {
        pieces.push_back({/*player=*/(Player)p, /*type=*/pt, /*revealed=*/false});
      }
    }
  }

  // 2. 随机洗牌并放置到棋盘上
  // 对应 Python: rng.shuffle(pieces)
  std::shuffle(pieces.begin(), pieces.end(), *rng_);
  for (int i = 0; i < kTotalPositions; ++i) {
    board_[i] = pieces[i];
  }

  // 3. 根据规则，开局时随机翻开 kInitialRevealedPieces 个棋子
  // 对应 Python _initialize_board 中的新增逻辑
  if (kInitialRevealedPieces > 0) {
      std::vector<int> positions(kTotalPositions);
      std::iota(positions.begin(), positions.end(), 0);
      std::shuffle(positions.begin(), positions.end(), *rng_);

      int reveal_count = std::min(kInitialRevealedPieces, kTotalPositions);
      for(int i = 0; i < reveal_count; ++i) {
          int sq = positions[i];
          board_[sq].revealed = true;
      }
  }
}

// 对应 Python 中的 _reset_internal_state 函数。
void AnqiState::ResetInternalState() {
  board_ = {};
  current_player_ = 0;
  move_counter_ = 0;
  total_step_counter_ = 0;
  scores_ = {0, 0};
  for(auto& vec : survival_vectors_) {
    vec.fill(1.0f);
  }
  terminal_ = false;
  winner_ = kInvalidPlayer;
}

bool AnqiState::IsTerminal() const { return terminal_; }

std::vector<double> AnqiState::Returns() const {
  if (!terminal_) return {0.0, 0.0};
  if (winner_ == kInvalidPlayer) return {0.0, 0.0}; // 平局
  // OpenSpiel 惯例: 获胜方返回 1.0, 失败方返回 -1.0
  if (winner_ == 0) return {1.0, -1.0};
  return {-1.0, 1.0};
}

std::string AnqiState::ActionToString(Player player, Action action) const {
  ActionInfo info = DecodeAction(action);
  Pos p = IdxToPos(info.idx);
  if (info.type == ActionInfo::Reveal) {
    return absl::StrCat("Reveal(", p.r, ",", p.c, ")");
  } else {
    std::string dir_str;
    if (info.dir == 0) dir_str = "Up";
    else if (info.dir == 1) dir_str = "Down";
    else if (info.dir == 2) dir_str = "Left";
    else dir_str = "Right";
    std::string type_str = (info.type == ActionInfo::Move) ? "Move" : "CannonAttack";
    return absl::StrCat(type_str, "(", p.r, ",", p.c, ") ", dir_str);
  }
}

// 对应 Python 中的 _internal_apply_action 函数。
void AnqiState::DoApplyAction(Action action) {
  total_step_counter_++;
  ActionInfo info = DecodeAction(action);
  int from_idx = info.idx;

  if (info.type == ActionInfo::Reveal) {
    SPIEL_CHECK_FALSE(board_[from_idx].revealed);
    board_[from_idx].revealed = true;
    move_counter_ = 0;
  } else {  // Move or CannonAttack
    int dr = (info.dir == 0) ? -1 : (info.dir == 1) ? 1 : 0;
    int dc = (info.dir == 2) ? -1 : (info.dir == 3) ? 1 : 0;

    Pos to_pos;
    if (info.type == ActionInfo::Move) {
      Pos from_pos = IdxToPos(from_idx);
      to_pos = {from_pos.r + dr, from_pos.c + dc};
    } else {  // CannonAttack
      to_pos = GetCannonTarget(from_idx, dr, dc);
      SPIEL_CHECK_TRUE(to_pos.r != -1);  // 必须找到目标
    }
    int to_idx = PosToIdx(to_pos);

    if (board_[to_idx].player == kInvalidPlayer) {  // 移动到空格
      board_[to_idx] = board_[from_idx];
      board_[from_idx] = {kInvalidPlayer, (PieceType)-1, true}; // 标记为空格
      move_counter_++;
    } else {  // 攻击
      ResolveAttack(from_idx, to_idx);
      move_counter_ = 0;
    }
  }

  // 切换玩家 (0 -> 1, 1 -> 0)
  current_player_ = 1 - current_player_;

  // 检查游戏结束条件
  // 对应 Python 的 _check_game_over_conditions
  if (scores_[0] >= kWinningScore) {
    terminal_ = true;
    winner_ = 0;
  } else if (scores_[1] >= kWinningScore) {
    terminal_ = true;
    winner_ = 1;
  } else if (move_counter_ >= kMaxConsecutiveMovesForDraw || total_step_counter_ >= kMaxStepsPerEpisode) {
    terminal_ = true;
    winner_ = kInvalidPlayer;  // 平局
  } else if (LegalActions().empty()) {
    // 当前玩家(已经是下一位玩家)无棋可走，则对方(刚行动的玩家)获胜
    terminal_ = true;
    winner_ = 1 - current_player_;
  }
}

// 对应 Python _apply_move_action 中的吃子逻辑。
void AnqiState::ResolveAttack(int from_idx, int to_idx) {
  Piece& attacker = board_[from_idx];
  Piece& defender = board_[to_idx];
  int points = kPieceValues[static_cast<int>(defender.type)];

  // 炮攻击己方未翻开棋子，视为误伤，分数给对方。
  // Python 中此逻辑在 _apply_move_action 内实现: if defender.player == attacker.player
  // 此处实现更明确，但效果相同，因为只有炮能攻击未翻开的己方棋子。
  bool friendly_fire = (attacker.type == PieceType::kCannon &&
                        !defender.revealed &&
                        attacker.player == defender.player);

  if (friendly_fire) {
    scores_[1 - attacker.player] += points;
  } else {
    scores_[attacker.player] += points;
  }

  UpdateSurvivalVectorOnCapture(defender);

  // 执行移动和吃子
  board_[to_idx] = attacker;
  board_[from_idx] = {kInvalidPlayer, (PieceType)-1, true}; // 标记为空格
}

// 对应 Python 的 action_masks 函数。
std::vector<Action> AnqiState::LegalActions() const {
  std::vector<Action> actions;
  if (terminal_) return actions;

  // 1. 翻棋动作
  for (int i = 0; i < kTotalPositions; ++i) {
    if (!board_[i].revealed) {
      actions.push_back(EncodeAction({ActionInfo::Reveal, i, -1}));
    }
  }

  // 2. 移动/攻击动作
  GenerateMoves(current_player_, &actions);

  return actions;
}

// 对应 Python 的 _add_regular_move_masks 和 _add_cannon_attack_masks
void AnqiState::GenerateMoves(Player player,
                                 std::vector<Action>* actions) const {
  for (int i = 0; i < kTotalPositions; ++i) {
    const Piece& piece = board_[i];
    if (piece.revealed && piece.player == player) {
      Pos p = IdxToPos(i);

      // 炮的逻辑 (Cannon)
      if (piece.type == PieceType::kCannon) {
        for (int dir = 0; dir < 4; ++dir) {
          int dr = (dir == 0) ? -1 : (dir == 1) ? 1 : 0;
          int dc = (dir == 2) ? -1 : (dir == 3) ? 1 : 0;
          Pos target_pos = GetCannonTarget(i, dr, dc);
          if (target_pos.r != -1) {
            const Piece& target_piece = board_[PosToIdx(target_pos)];
            // 不能攻击已翻开的己方棋子
            // 对应 Python: valid_cannon_targets = ~self.revealed_vectors[player]
            if (!target_piece.revealed || target_piece.player != player) {
              actions->push_back(EncodeAction({ActionInfo::CannonAttack, i, dir}));
            }
          }
        }
      }
      // 其他棋子逻辑
      else {
        for (int dir = 0; dir < 4; ++dir) {
          int dr = (dir == 0) ? -1 : (dir == 1) ? 1 : 0;
          int dc = (dir == 2) ? -1 : (dir == 3) ? 1 : 0;
          Pos to_pos = {p.r + dr, p.c + dc};

          if (to_pos.r >= 0 && to_pos.r < kBoardRows && to_pos.c >= 0 &&
              to_pos.c < kBoardCols) {
            int to_idx = PosToIdx(to_pos);
            const Piece& target = board_[to_idx];

            // 移动到空格
            if (target.player == kInvalidPlayer) {
              actions->push_back(EncodeAction({ActionInfo::Move, i, dir}));
            }
            // 攻击敌方已翻开棋子
            else if (target.revealed && target.player != player && CanAttack(piece.type, target.type)) {
              actions->push_back(EncodeAction({ActionInfo::Move, i, dir}));
            }
          }
        }
      }
    }
  }
}

// 对应 Python _add_cannon_attack_masks 中的 ray-casting 逻辑。
Pos AnqiState::GetCannonTarget(int from_idx, int dr, int dc) const {
  Pos p = IdxToPos(from_idx);
  bool mount_found = false; // 是否找到炮架
  int r = p.r + dr;
  int c = p.c + dc;

  while (r >= 0 && r < kBoardRows && c >= 0 && c < kBoardCols) {
    int current_idx = PosToIdx(r, c);
    if (board_[current_idx].player != kInvalidPlayer) { // 不是空格
      if (!mount_found) {
        mount_found = true;
      } else {
        return {r, c}; // 找到炮架后的第一个棋子作为目标
      }
    }
    r += dr;
    c += dc;
  }
  return {-1, -1};  // 未找到目标
}

// 对应 Python _get_valid_target_vectors 中的特殊吃子规则。
bool AnqiState::CanAttack(PieceType attacker, PieceType defender) const {
    // 规则1: 兵可以吃将
    if (attacker == PieceType::kSoldier && defender == PieceType::kGeneral) {
        return true;
    }
    // 规则2: 将不能吃兵
    if (attacker == PieceType::kGeneral && defender == PieceType::kSoldier) {
        return false;
    }
    // 规则3: 除了炮，高等级吃低等级或同级
    return static_cast<int>(attacker) >= static_cast<int>(defender);
}

std::string AnqiState::ToString() const {
  std::string str;
  for (int r = 0; r < kBoardRows; ++r) {
    for (int c = 0; c < kBoardCols; ++c) {
      const auto& piece = board_[PosToIdx(r, c)];
      if (piece.player == kInvalidPlayer) {
          str += ".  ";
      } else {
          absl::StrAppend(&str, piece.ToString(), " ");
      }
    }
    absl::StrAppend(&str, "\n");
  }
  absl::StrAppend(&str, "Current player: ", current_player_, "\n");
  absl::StrAppend(&str, "Scores (P0 vs P1): ", scores_[0], " vs ", scores_[1], "\n");
  absl::StrAppend(&str, "Move counter: ", move_counter_, "\n");
  absl::StrAppend(&str, "Total steps: ", total_step_counter_, "\n");
  return str;
}

std::string AnqiState::ObservationString(Player player) const {
  std::string str = "Observation for player ";
  absl::StrAppend(&str, player, "\n");
  for (int r = 0; r < kBoardRows; ++r) {
    for (int c = 0; c < kBoardCols; ++c) {
      const auto& piece = board_[PosToIdx(r, c)];
      if (piece.player == kInvalidPlayer) {
          str += ".  ";
      } else if (piece.revealed) {
          absl::StrAppend(&str, piece.ToString(), " ");
      } else {
          str += "?? ";
      }
    }
    absl::StrAppend(&str, "\n");
  }
  return str;
}

// 生成与 Python get_state() 完全相同的观测张量。
// 张量是一个扁平化的 275 维向量。
void AnqiState::ObservationTensor(Player player,
                                     absl::Span<float> values) const {
  // 确认张量大小与 Python 环境的输出 (256+19=275) 一致。
  SPIEL_CHECK_EQ(values.size(),
                 kTotalPositions * (kNumPieceTypes * 2 + 2) + 3 + kSurvivalVectorSize * 2);
  std::fill(values.begin(), values.end(), 0.f);

  int offset = 0;
  // Part 1: 棋盘平面 (16 * 16 = 256)
  // 对应 _get_board_state_tensor
  // 平面顺序: [己方棋子(7), 对方棋子(7), 暗棋(1), 空格(1)]
  for (int i = 0; i < kTotalPositions; ++i) {
    const Piece& piece = board_[i];
    int base_offset = offset + i * (kNumPieceTypes * 2 + 2);
    if (piece.player != kInvalidPlayer) { // 如果格子上不是空的
      if (piece.revealed) {
        int piece_type_idx = static_cast<int>(piece.type);
        // 根据是己方还是对方棋子，选择不同的平面偏移
        int player_offset = (piece.player == player) ? 0 : kNumPieceTypes;
        values[base_offset + player_offset + piece_type_idx] = 1.0;
      } else {
        // 未翻开的棋子 (暗棋)
        values[base_offset + kNumPieceTypes * 2] = 1.0;
      }
    } else {
      // 空格
      values[base_offset + kNumPieceTypes * 2 + 1] = 1.0;
    }
  }
  offset = kTotalPositions * (kNumPieceTypes * 2 + 2);

  // Part 2 & 3: 标量特征和存活向量 (3 + 8 + 8 = 19)
  // 对应 _get_scalar_state_vector
  Player opponent = 1 - player;
  // 标量特征
  values[offset++] = static_cast<float>(scores_[player]) / kWinningScore;
  values[offset++] = static_cast<float>(scores_[opponent]) / kWinningScore;
  values[offset++] = static_cast<float>(move_counter_) / kMaxConsecutiveMovesForDraw;
  // 存活向量
  for(float val : survival_vectors_[player]) {
      values[offset++] = val;
  }
  for(float val : survival_vectors_[opponent]) {
      values[offset++] = val;
  }
}

std::unique_ptr<State> AnqiState::Clone() const {
  return std::make_unique<AnqiState>(*this);
}

void AnqiState::UndoAction(Player player, Action action) {
  // 在这个游戏中实现撤销操作非常复杂，因为存在随机性和不完美信息。
  // 因此，遵循许多 OpenSpiel 游戏实现，将其标记为未实现。
  SpielFatalError("UndoAction is not implemented.");
}

// 对应 Python _update_survival_vector_on_capture 中的索引查找逻辑。
// 使用 PIECE_SURVIVAL_VEC_INFO 字典。
int AnqiState::GetSurvivalVectorIndex(const Piece& piece) const {
    // 索引映射: 兵(0,1), 炮(2), 马(3), 车(4), 象(5), 士(6), 将(7)
    switch(piece.type) {
        case PieceType::kSoldier: {
            // 为被吃的兵找到第一个值为 1.0f 的槽位。
            int base_idx = 0;
            if (survival_vectors_[piece.player][base_idx] > 0.5f) return base_idx;
            if (survival_vectors_[piece.player][base_idx + 1] > 0.5f) return base_idx + 1;
            return base_idx; // 如果都死了，随便返回一个（理论上不应发生）
        }
        case PieceType::kCannon: return 2;
        case PieceType::kHorse: return 3;
        case PieceType::kChariot: return 4;
        case PieceType::kElephant: return 5;
        case PieceType::kAdvisor: return 6;
        case PieceType::kGeneral: return 7;
    }
    return -1; // 不应该发生
}

// 对应 Python 的 _update_survival_vector_on_capture。
void AnqiState::UpdateSurvivalVectorOnCapture(const Piece& captured_piece) {
    int index = GetSurvivalVectorIndex(captured_piece);
    if (index != -1) {
        survival_vectors_[captured_piece.player][index] = 0.0f;
    }
}


// --- AnqiGame ---
AnqiGame::AnqiGame(const GameParameters& params) : Game(kGameType, params) {}

std::vector<int> AnqiGame::ObservationTensorShape() const {
  // 棋盘平面: 16 * (己方棋子7 + 对方棋子7 + 暗棋1 + 空格1) = 16 * 16 = 256
  // 标量特征: 己方分数，对方分数，连续步数 = 3
  // 存活向量: 己方(8) + 对方(8) = 16
  // 总计: 256 + 3 + 16 = 275
  return {kTotalPositions * (kNumPieceTypes * 2 + 2) + 3 + kSurvivalVectorSize * 2};
}

}  // namespace anqi_4x4
}  // namespace open_spiel