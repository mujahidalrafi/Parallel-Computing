#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <ctime>

using namespace std;

const int length = 100;
const int threadChunk = 10;

int main()
{
    auto startTime = std::clock();

    vector<int> in(length), out(length);
    for (int i = 0; i < length; i++)
        in[i] = rand() % 10 + 1;

    #pragma omp parallel for schedule(static, threadChunk)
    for (int i = 0; i < in.size(); i++)
    {
        out[i] = in[i];
        if (i % threadChunk != 0)
            out[i] += out[i - 1];
    }

    vector<int> carries(ceil((float)length/threadChunk));
    carries[0] = 0;
    for (int i = 1; i < carries.size() ; i++)
        carries[i] = out[i * threadChunk - 1] + carries[i-1];

    #pragma omp parallel for schedule(static, threadChunk)
    for (int i = 0; i < in.size(); i++)
        out[i] += carries[i/threadChunk];

    cout << "In:" << "\t" << "Out:" << endl;
    for (int i = 0; i < length; i++)
        cout << in[i] << "\t" << out[i] << endl;

    auto endTime = std::clock();
    auto elapsedTimeMS = 1000.0 * (endTime - startTime) / CLOCKS_PER_SEC;
    std::cout << "\nExecution Time:" << elapsedTimeMS << " ms\n";

    return 0;
}
