#pragma once

#include <immintrin.h>
#include <cstdlib>
#include <omp.h>
#include "Utility.hpp"

#define MatrixPartitionSize 8

float** addAVX(float** M1, float** M2, int n)
{
    float** temp = initializeMatrix(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j += 8)
        {
            __m256 a = _mm256_loadu_ps((const float*)&M1[i][j]);
            __m256 b = _mm256_loadu_ps((const float*)&M2[i][j]);
            __m256 c = _mm256_add_ps(a, b);
            _mm256_storeu_ps(&temp[i][j], c);
        }
    }

    return temp;
}

float** subtractAVX(float** M1, float** M2, int n)
{
    float** temp = initializeMatrix(n);
 
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j += 8)
        {
            __m256 a = _mm256_loadu_ps((const float*)&M1[i][j]);
            __m256 b = _mm256_loadu_ps((const float*)&M2[i][j]);
            __m256 c = _mm256_sub_ps(a, b);
            _mm256_storeu_ps(&temp[i][j], c);
        }
    }

    return temp;
}

float** strassenAVXMultiply(float** A, float** B, int n)
{
    if (n == MatrixPartitionSize)
    {
        float** C = initializeMatrix(MatrixPartitionSize);

        __m256 rightRow[MatrixPartitionSize];
        __m256 resultRow[MatrixPartitionSize];

        for (int i = 0; i < MatrixPartitionSize; ++i)
        {
            rightRow[i] = _mm256_loadu_ps((const float*)B[i]);
            resultRow[i] = _mm256_setzero_ps();
        }
        for (int i = 0; i < MatrixPartitionSize; ++i)
        {
            for (int j = 0; j < MatrixPartitionSize; ++j)
            {
                resultRow[i] = _mm256_add_ps(resultRow[i], _mm256_mul_ps(rightRow[j], _mm256_set1_ps(A[i][j])));
            }
            _mm256_storeu_ps((float*)C[i], resultRow[i]);
        }
        return C;
    }

    float** C = initializeMatrix(n);
    int k = n / 2;

    float** A11 = initializeMatrix(k);
    float** A12 = initializeMatrix(k);
    float** A21 = initializeMatrix(k);
    float** A22 = initializeMatrix(k);
    float** B11 = initializeMatrix(k);
    float** B12 = initializeMatrix(k);
    float** B21 = initializeMatrix(k);
    float** B22 = initializeMatrix(k);
    float** P1, **P2, **P3, **P4, **P5, **P6, **P7;

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j+=8)
        {
            _mm256_storeu_ps(&A11[i][j], _mm256_loadu_ps((const float*)&A[i][j]));
            _mm256_storeu_ps(&A12[i][j], _mm256_loadu_ps((const float*)&A[i][j+k]));
            _mm256_storeu_ps(&A21[i][j], _mm256_loadu_ps((const float*)&A[i+k][j]));
            _mm256_storeu_ps(&A22[i][j], _mm256_loadu_ps((const float*)&A[i+k][j+k]));
            _mm256_storeu_ps(&B11[i][j], _mm256_loadu_ps((const float*)&B[i][j]));
            _mm256_storeu_ps(&B12[i][j], _mm256_loadu_ps((const float*)&B[i][j + k]));
            _mm256_storeu_ps(&B21[i][j], _mm256_loadu_ps((const float*)&B[i + k][j]));
            _mm256_storeu_ps(&B22[i][j], _mm256_loadu_ps((const float*)&B[i + k][j + k]));
        }
    }

    {
        P1 = strassenAVXMultiply(A11, subtractAVX(B12, B22, k), k);
        P2 = strassenAVXMultiply(addAVX(A11, A12, k), B22, k);
        P3 = strassenAVXMultiply(addAVX(A21, A22, k), B11, k);
        P4 = strassenAVXMultiply(A22, subtractAVX(B21, B11, k), k);
        P5 = strassenAVXMultiply(addAVX(A11, A22, k), addAVX(B11, B22, k), k);
        P6 = strassenAVXMultiply(subtractAVX(A12, A22, k), addAVX(B21, B22, k), k);
        P7 = strassenAVXMultiply(subtractAVX(A11, A21, k), addAVX(B11, B12, k), k);
    }

    float** C11 = subtractAVX(addAVX(addAVX(P5, P4, k), P6, k), P2, k);
    float** C12 = addAVX(P1, P2, k);
    float** C21 = addAVX(P3, P4, k);
    float** C22 = subtractAVX(subtractAVX(addAVX(P5, P1, k), P3, k), P7, k);

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j+=8)
        {
            _mm256_storeu_ps(&C[i][j], _mm256_loadu_ps((const float*)&C11[i][j]));
            _mm256_storeu_ps(&C[i][j+k], _mm256_loadu_ps((const float*)&C12[i][j]));
            _mm256_storeu_ps(&C[i+k][j], _mm256_loadu_ps((const float*)&C21[i][j]));
            _mm256_storeu_ps(&C[i+k][j+k], _mm256_loadu_ps((const float*)&C22[i][j]));
        }
    }

    for (int i = 0; i < k; i++)
    {
        delete[] A11[i];
        delete[] A12[i];
        delete[] A21[i];
        delete[] A22[i];
        delete[] B11[i];
        delete[] B12[i];
        delete[] B21[i];
        delete[] B22[i];
        delete[] P1[i];
        delete[] P2[i];
        delete[] P3[i];
        delete[] P4[i];
        delete[] P5[i];
        delete[] P6[i];
        delete[] P7[i];
        delete[] C11[i];
        delete[] C12[i];
        delete[] C21[i];
        delete[] C22[i];
    }

    delete[] A11;
    delete[] A12;
    delete[] A21;
    delete[] A22;
    delete[] B11;
    delete[] B12;
    delete[] B21;
    delete[] B22;
    delete[] P1;
    delete[] P2;
    delete[] P3;
    delete[] P4;
    delete[] P5;
    delete[] P6;
    delete[] P7;
    delete[] C11;
    delete[] C12;
    delete[] C21;
    delete[] C22;

    return C;
}