#include "env2048.hpp"
#include <cmath>

Env2048::Env2048(){
    init_board(board);
    last_score = 0;
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

std::vector<float> Env2048::Step(int action, float &reward, bool &done){
    int prev_score = calc_score(board);
    bool moved = step(board, (Action)action);
    int now_score = calc_score(board);
    reward = now_score - prev_score;
    done = !moved;

    return GetState();
}