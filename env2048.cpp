#include "env2048.hpp"
#include <cmath>
#include <iostream>

Env2048::Env2048(){
    init_board(board);
    last_score = 0;
}

float calc_reward(board_2048 &board, int now_score, int prev_score, bool moved){
    float reward = 0.0f;

    reward += 0.1f;

    if(!moved){
        reward -= 0.2f;
    }

    if(board.value_max == board.grid[0][0])
        reward += 0.2;

    reward += log2(now_score - prev_score + 1);

    reward += 0.25f * board.vacant_total;

    if(board.gameover_flg){
        reward -= 20.0f;
    }

    return reward;
}

std::vector<float> Env2048::reset(){
    init_board(board);
    last_score = 0;

    return GetState();
}

std::vector<float> Env2048::GetState(){
    float state[16];
    get_state(board, state);

    return std::vector<float>(state, state + 16);
}

std::vector<float> Env2048::Step(int action, float &reward, bool &done, int &score, int &max){
    int prev_score = calc_score(board);
    bool moved = step(board, (Action)action);
    int now_score = calc_score(board);
    reward = calc_reward(board, now_score, prev_score, moved);
    done = board.gameover_flg;
    score = now_score;
    max = board.value_max;

    return GetState();
}