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

#ifndef OPEN_SPIEL_GAMES_ANQI_4X4_H_
#define OPEN_SPIEL_GAMES_ANQI_4X4_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// 这是一个基于 Python 环境 'environment.py' 逻辑的4x4暗棋游戏实现。
// 游戏在一个 4x4 的棋盘上进行，目标是通过吃掉对方棋子获得指定分数。
// 这是一个不完美信息游戏，具有随机的游戏初始设置。

namespace open_spiel {
namespace anqi_4x4 {

// --- 游戏常量 ---
// 这些常量与 Python environment.py 中的定义完全对应。
constexpr int kNumPlayers = 2;
constexpr int kBoardRows = 4;
constexpr int kBoardCols = 4;
constexpr int kTotalPositions = kBoardRows * kBoardCols;
constexpr int kNumPieceTypes = 7;
constexpr int kWinningScore = 60;                   // 对应 WINNING_SCORE
constexpr int kMaxConsecutiveMovesForDraw = 24;     // 对应 MAX_CONSECUTIVE_MOVES_FOR_DRAW
constexpr int kMaxStepsPerEpisode = 100;            // 对应 MAX_STEPS_PER_EPISODE
constexpr int kInitialRevealedPieces = 2;           // 对应 INITIAL_REVEALED_PIECES
constexpr int kSurvivalVectorSize = 8;              // 存活向量维度: 兵x2, 炮x1, 马x1, 车x1, 象x1, 士x1, 将x1

// --- 动作空间定义 ---
// 将所有可能的动作映射到一个扁平的整数空间。
// 对应 Python 中的 REVEAL_ACTIONS_COUNT, REGULAR_MOVE_ACTIONS_COUNT 等。
constexpr int kRevealActionBase = 0;                     // 翻棋动作起始索引 (0-15)
constexpr int kMoveActionBase = 16;                      // 移动/普通攻击动作起始索引 (16-79)
constexpr int kCannonActionBase = 80;                    // 炮击动作起始索引 (80-143)
constexpr int kNumDistinctActions = 144;                 // 16(翻棋) + 16*4(移动) + 16*4(炮击)

// 棋子类型 (等级从低到高)
// 对应 Python 中的 PieceType Enum
enum class PieceType {
  kSoldier = 0,
  kCannon = 1,
  kHorse = 2,
  kChariot = 3,
  kElephant = 4,
  kAdvisor = 5,
  kGeneral = 6
};

// 棋子价值
// 对应 Python 中的 PIECE_VALUES
inline constexpr std::array<int, kNumPieceTypes> kPieceValues = {4, 10, 10, 10, 10, 20, 30};
// 每种棋子的最大数量 (单方)
// 对应 Python 中的 PIECE_MAX_COUNTS
inline constexpr std::array<int, kNumPieceTypes> kPieceMaxCounts = {2, 1, 1, 1, 1, 1, 1};


// 棋子结构体
// 对应 Python 中的 Piece 类
struct Piece {
  Player player;
  PieceType type;
  bool revealed = false;

  std::string ToString() const;
};

// 棋盘位置
struct Pos {
  int r, c;
  bool operator==(const Pos& other) const { return r == other.r && c == other.c; }
};

// 坐标转换
// 对应 Python 中的 POS_TO_SQ 和 SQ_TO_POS
int PosToIdx(int r, int c);
int PosToIdx(const Pos& pos);
Pos IdxToPos(int idx);


// 游戏状态类
// 对应 Python 中的 GameEnvironment 类，但只包含核心游戏逻辑，不包含训练相关的部分
class AnqiState : public State {
 public:
  AnqiState(std::shared_ptr<const Game> game);
  AnqiState(const AnqiState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;

  bool IsTerminal() const override;
  std::vector<double> Returns() const override;

  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  // 初始化和重置
  void InitializeBoard();
  void ResetInternalState();

  // 动作生成与合法性检查
  void GenerateMoves(Player player, std::vector<Action>* actions) const;
  bool CanAttack(PieceType attacker, PieceType defender) const;
  Pos GetCannonTarget(int from_idx, int dr, int dc) const;

  // 动作执行
  void ResolveAttack(int from_idx, int to_idx);
  void UpdateSurvivalVectorOnCapture(const Piece& captured_piece);
  int GetSurvivalVectorIndex(const Piece& piece) const;


  // --- 状态变量 ---
  // 这些变量对应 Python GameEnvironment 中的核心状态属性
  std::array<Piece, kTotalPositions> board_;
  Player current_player_ = 0;        // 对应 current_player (C++用0/1, Python用1/-1)
  int move_counter_ = 0;             // 对应 move_counter (连续未吃子/翻子步数)
  int total_step_counter_ = 0;       // 对应 total_step_counter (游戏总步数)
  std::array<int, kNumPlayers> scores_ = {0, 0}; // 对应 scores
  std::array<std::array<float, kSurvivalVectorSize>, kNumPlayers> survival_vectors_; // 对应 survival_vectors

  // 游戏结束状态
  bool terminal_ = false;
  Player winner_ = kInvalidPlayer;
};


// 游戏主类
class AnqiGame : public Game {
 public:
  explicit AnqiGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<AnqiState>(shared_from_this());
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1.0; }
  absl::optional<double> UtilitySum() const override { return 0.0; }
  double MaxUtility() const override { return 1.0; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return kMaxStepsPerEpisode; }
};

}  // namespace anqi_4x4
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_ANQI_4X4_H_