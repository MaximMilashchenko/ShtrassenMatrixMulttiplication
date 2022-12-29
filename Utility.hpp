#pragma once

#include <iostream>
#include <random>
#include <ctime>

float** initializeMatrix(int n)
{
    float** temp = new float* [n];
    for (int i = 0; i < n; i++)
    {
        temp[i] = new float[n];
    }
    return temp;
}

void generate(float** M, int n)
{
    std::mt19937 engine;
    engine.seed(std::time(nullptr));

    std::srand(std::time(nullptr));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            M[i][j] = 1 + std::rand() % 9;
        }
    }
}

void printMatrix(float** M, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << M[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}