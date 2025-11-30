# Minimal JIT Graph Engine (C++)

A tiny demo that:
- builds a computational graph  
- allocates memory using liveness  
- generates fused C++ kernel code  
- JIT-compiles it into a `.so`  
- loads it with `dlopen()` and runs it

## Build & Run
```
g++ tinygraph.cpp -ldl -std=c++17 -O2 -o tinygraph
./tinygraph
```

## Pipeline
1. Build graph (`Add`, `Mul`, `Relu`, `Input`)
2. Memory plan → assign each node a 1024-float slot
3. Convert graph to a fused C expression
4. Generate `jit_kernel.cpp`
5. Compile:
   ```
   g++ -shared -fPIC -O3 jit_kernel.cpp -o jit_kernel.so
   ```
6. Load:
   ```cpp
   dlopen("./jit_kernel.so", RTLD_LAZY);
   dlsym(handle, "fused_kernel");
   ```
7. Execute kernel on a float arena

## Example Graph
```
A, B = inputs
T1 = A + B
T2 = T1 * A
T3 = relu(B)
Out = T2 + T3
```

## Generated Kernel (example)
```cpp
void fused_kernel(float* memory, int N) {
    for (int i = 0; i < N; i++)
        memory[out + i] = <fused expression>;
}
```

## Files
- `tinygraph.cpp` — graph + JIT engine  
- `jit_kernel.cpp` — auto-generated  
- `jit_kernel.so` — compiled kernel

