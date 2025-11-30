#include <algorithm>
#include <cmath>
extern "C" {
void fused_kernel(float* memory, int N) {
    for (int i = 0; i < N; i++) {
        memory[1024 + i] = (((memory[0 + i] + memory[1024 + i]) * memory[0 + i]) + std::max(0.0f, memory[1024 + i]));
    }
}
}
