#include "shtrassen.hpp"
#include "shtrassenAVX.hpp"

int main()
{
    int n = 512;

    float** A = initializeMatrix(n);
    float** B = initializeMatrix(n);
    generate(A, n);
    generate(B, n);
    
    /*float** C = initializeMatrix(n);
    C = strassenMultiply(A, B, n);*/

    float** CAVX = initializeMatrix(n);
    CAVX = strassenAVXMultiply(A, B, n);

    /*for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (C[i][j] - C[i][j] > 0)
            {
                std::cout << "Error! Elemnent in Matrix non equal!" << std::endl;
                std::cout << "[" << i << "][" << j << "]" << C[i][j] << " != " << CAVX[i][j] << " = " << sub << std::endl;
                return -1;
            }
        }
    }*/

    for (int i = 0; i < n; i++)
    {
        delete[] A[i];
    }
    delete[] A;

    for (int i = 0; i < n; i++)
    {
        delete[] B[i];
    }
    delete[] B;

    /*for (int i = 0; i < n; i++)
    {
        delete[] C[i];
    }
    delete[] C;*/

    for (int i = 0; i < n; i++)
    {
        delete[] CAVX[i];
    }
    delete[] CAVX;

    return 0;
}