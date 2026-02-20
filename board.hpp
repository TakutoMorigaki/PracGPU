#pragma once


// 2048のボード構造体
struct board_2048 {
    unsigned int grid[4][4];    // ボードを表す4x4の二次元配列
    int vacant_total;           // 現在の空きマスの個数
    bool gameover_flg;          // ゲームオーバー判定フラグ
    bool move_r_flg[4][4];      // あるマスにおいて右に動けるかの判定フラグ
    bool move_l_flg[4][4];      // あるマスにおいて左に動けるかの判定フラグ
    bool move_u_flg[4][4];      // あるマスにおいて上に動けるかの判定フラグ
    bool move_d_flg[4][4];      // あるマスにおいて下に動けるかの判定フラグ
    bool marged_flg[4][4];      // すでに合成がなされているかを表すフラグ
};

// AIの行動(0:右, 1:左, 2:上, 3:下)
enum Action {
    ACT_RIGHT = 0,
    ACT_LEFT  = 1,
    ACT_UP    = 2,
    ACT_DOWN  = 3,
};

// ボードの初期化
void init_board(board_2048 &board);

// 一つのマスにおいて上下左右に動けるかを判定する
void CanMove_grid(board_2048 &board);
// 右に動けるか判定(true:動ける, false:動けない)
bool CanMove_R(board_2048 &board);
// 左に動けるか判定(true:動ける, false:動けない)
bool CanMove_L(board_2048 &board);
// 上に動けるか判定(true:動ける, false:動けない)
bool CanMove_U(board_2048 &board);
// 下に動けるか判定(true:動ける, false:動けない)
bool CanMove_D(board_2048 &board);

// ボードを右に動かす
void Move_R(board_2048 &board);
// ボードを左に動かす
void Move_L(board_2048 &board);
// ボードを上に動かす
void Move_U(board_2048 &board);
// ボードを下に動かす
void Move_D(board_2048 &board);

// AIのactionで盤面が動いたかを判定
bool apply_action(board_2048 &board, Action action);
// AIを一手進める
bool step(board_2048 &board, Action action);
// ゲームオーバーの判定
void is_gameover(board_2048 &board);
// 状態をAI用に変換
void get_state(const board_2048 &board, float state[16]);
// ランダムAIの着手方向
int ai_direc_random();

// 新しく数字をボードに追加する
void Pop_value(board_2048 &board);


// スコアの計算
int calc_score(board_2048 &board);
