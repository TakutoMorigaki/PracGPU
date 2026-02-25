#pragma once

#include "board.hpp"
#include <vector>

class Env2048{
private:
    board_2048 board;
    int last_score;

public:
    Env2048();

    std::vector<float> reset();

    std::vector<float> Step(int action, float &reward, bool &done, int &score, int &max);

    std::vector<float> GetState();
};