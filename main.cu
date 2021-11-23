#include <cuda_runtime.h>
#include <memory>

__device__
inline void mac_with_carry(uint64_t &lo, uint64_t &hi, const uint64_t &a, const uint64_t &b, const uint64_t &c) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GPU calculation input: a = %lx b = %lx c = %lx\n", a, b, c);
    }
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
        "madc.hi.u64 %1, %2, %3, 0;\n\t"
        :"=l"(lo), "=l"(hi): "l"(a), "l"(b), "l"(c));
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GPU calculation result: hi = %lx low = %lx\n", hi, lo);
    }
}

__global__
void test(uint64_t *out, uint32_t num){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num) {
        return;
    }
    uint64_t a = 0x42737a020c0d6393UL;
    uint64_t b = 0xffffffff00000001UL;
    uint64_t c = 0xc999e990f3f29c6dUL;
    mac_with_carry(out[tid << 1], out[(tid << 1) + 1], a, b, c);
}

int main() {
    uint64_t *d_out;
    uint32_t num = 1;
    cudaMalloc(&d_out, num * 2 * sizeof(uint64_t));
    const uint32_t BLOCK_SIZE = 256;
    uint32_t block_num = (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    test<<<block_num, BLOCK_SIZE>>>(d_out, num);
    cudaDeviceSynchronize();
    unsigned __int128 a = 0x42737a020c0d6393UL;
    unsigned __int128 b = 0xffffffff00000001UL;
    unsigned __int128 c = 0xc999e990f3f29c6dUL;
    unsigned __int128 result = a * b + c;
    printf("Cpu result: hi:%lx low:%lx\n", (uint64_t)((result >> 64) & 0xffffffffffffffffUL), (uint64_t)(result & 0xffffffffffffffffUL));
}
