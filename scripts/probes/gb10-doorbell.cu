// minimal doorbell: how fast does one store from a resident kernel reach the host?
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _r = (x); if (_r) { fprintf(stderr, "FATAL %s -> %d (%s)\n", #x, (int)_r, cudaGetErrorString(_r)); exit(1);} } while (0)
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void msleep(long ms) { struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L}; nanosleep(&ts, NULL); }

__global__ void servo_fence(volatile unsigned long long *m)   // m[0] cmd, m[1] ack, m[2] stop
{
    unsigned long long last = 0;
    while (m[2] == 0) { unsigned long long c = m[0]; if (c != last) { last = c; m[1] = c; __threadfence_system(); } }
}
__global__ void servo_wt(volatile unsigned long long *m)
{
    unsigned long long last = 0;
    while (m[2] == 0) {
        unsigned long long c = m[0];
        if (c != last) { last = c; asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(&m[1]), "l"(c) : "memory"); }
    }
}
__global__ void servo_plain(volatile unsigned long long *m)
{
    unsigned long long last = 0;
    while (m[2] == 0) { unsigned long long c = m[0]; if (c != last) { last = c; m[1] = c; } }
}

static double bench(void (*launch)(volatile unsigned long long *), unsigned long long *h,
                    const char *name, int n)
{
    memset(h, 0, 4096);
    launch(h);
    msleep(300);
    double sum = 0, mn = 1e18, mx = 0;
    for (int i = 1; i <= n; i++) {
        double a = now_us();
        h[0] = (unsigned long long)i;
        double dl = a + 2e6;
        while (h[1] != (unsigned long long)i) { if (now_us() > dl) { printf("%s TIMEOUT at %d (ack=%llu)\n", name, i, h[1]); break; } }
        double b = now_us() - a;
        sum += b; if (b < mn) mn = b; if (b > mx) mx = b;
        h[1] = 0;
    }
    h[2] = 1;
    msleep(30);
    CK(cudaDeviceSynchronize());
    printf("%-18s RTT avg %6.2f  min %6.2f  max %6.2f us\n", name, sum / n, mn, mx);
    return sum / n;
}
static void l_fence(volatile unsigned long long *h) { servo_fence<<<1, 32>>>(h); }
static void l_wt(volatile unsigned long long *h) { servo_wt<<<1, 32>>>(h); }
static void l_plain(volatile unsigned long long *h) { servo_plain<<<1, 32>>>(h); }

int main(void)
{
    unsigned long long *h = NULL;
    CK(cudaHostAlloc((void **)&h, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
    unsigned long long *d = NULL;
    CK(cudaHostGetDevicePointer((void **)&d, h, 0));
    printf("resident-kernel host<->gpu doorbell\n");
    bench(l_fence, h, "store+fence", 500);
    bench(l_wt, h, "st.global.wt", 500);
    bench(l_plain, h, "plain store", 500);
    bench(l_fence, h, "store+fence (2)", 500);
    return 0;
}
