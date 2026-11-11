// gb10-slab-read: what does it cost the GPU to read rows that something other than the GPU wrote?
//
//  The exchange's fold reads its peers' rows out of a host-mapped slab the NIC writes; the group
//  arm's fold reads device memory a GPU copy has just written. In situ the first averages 14.7 us
//  and the second 9.7. This asks whether a different kind of pinned memory would close that gap.
//
//  One resident block waits for a command word, reads `bytes` of rows and acknowledges, and the
//  host times command to acknowledgement. Every case carries the same poll and acknowledge cost
//  (the "nothing read" line), so only the differences between cases mean anything. Each round's
//  sum is checked, so a case that read stale rows would say so.
//
//  The rows are read from device memory, from cudaHostAlloc memory (what the slab is), and from
//  ordinary host memory registered afterwards with huge pages requested. Each host kind is read
//  left alone, rewritten by the host before every round, and rewritten then flushed out of the
//  CPU's caches. The last is where a NIC's write leaves a row: in memory and in nobody's cache.
//
//  What it found on GB10, 240 KiB of rows: device memory 4.6 us; cudaHostAlloc memory 10.6 us even
//  when nothing has touched it; registered memory 4.7 us left alone; and 9.8-9.9 us for both host
//  kinds once the rows were rewritten and flushed. So a row that arrives from outside costs a
//  memory read whichever pinned memory holds it, and the slab's kind is not the lever.
//
//  usage: gb10-slab-read [bytes] [rounds]
//  build: nvcc -O2 -o gb10-slab-read gb10-slab-read.cu
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _r = (x); if (_r) { fprintf(stderr, "FATAL %s -> %d (%s)\n", #x, (int)_r, cudaGetErrorString(_r)); exit(1);} } while (0)
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void msleep(long ms) { struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L}; nanosleep(&ts, NULL); }

// m[0] command, m[1] acknowledge, m[2] stop, m[3] sink.
__global__ void reader(volatile unsigned long long *m, const volatile unsigned long long *rows,
                       unsigned words)
{
    __shared__ unsigned long long command;
    __shared__ int stop;
    __shared__ unsigned long long sink[1024];
    unsigned long long last = 0;
    for (;;) {
        if (threadIdx.x == 0) {
            unsigned long long c;
            do { c = m[0]; } while (c == last && m[2] == 0);
            command = c;
            stop = m[2] != 0;
        }
        __syncthreads();
        if (stop) break;
        last = command;
        unsigned long long x = 0;
        for (unsigned i = threadIdx.x; i < words; i += blockDim.x) x ^= rows[i];
        sink[threadIdx.x] = x;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long y = 0;
            for (unsigned i = 0; i < blockDim.x; i += 64) y ^= sink[i];
            m[3] = y;
            m[1] = last;
        }
    }
}

static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// What the reader's sink must hold when every byte of the rows is `value`.
static unsigned long long expect(unsigned words, unsigned char value)
{
    unsigned long long w = 0x0101010101010101ull * value, y = 0;
    for (unsigned t = 0; t < 1024; t += 64) {
        unsigned count = t < words ? (words - t + 1023) / 1024 : 0;
        if (count & 1) y ^= w;
    }
    return y;
}

// Writes the lines back to memory and drops them from the CPU's caches, which is where a NIC's
// write leaves them: in memory and in nobody's cache.
static void flush_lines(unsigned char *p, unsigned bytes)
{
#if defined(__aarch64__)
    for (unsigned i = 0; i < bytes; i += 64) asm volatile("dc civac, %0" :: "r"(p + i) : "memory");
    asm volatile("dsb ish" ::: "memory");
#else
    (void)p; (void)bytes;
#endif
}

// rewrite: 0 leaves the rows alone, 1 has the host rewrite them before every command, 2 has it
// rewrite them and then flush them out of the CPU's caches.
static double run(const char *name, unsigned long long *h, unsigned long long *d,
                  unsigned char *host_rows, const void *dev_rows, unsigned bytes, int n, int rewrite)
{
    memset(h, 0, 64);
    reader<<<1, 1024>>>(d, (const volatile unsigned long long *)dev_rows, bytes / 8);
    msleep(200);
    double *t = (double *)malloc(sizeof(double) * n);
    int wrong = 0;
    for (int i = 1; i <= n; i++) {
        unsigned char value = rewrite ? (unsigned char)(i * 7 + 3) : 1;
        if (rewrite) memset(host_rows, value, bytes);
        if (rewrite == 2) flush_lines(host_rows, bytes);
        double a = now_us();
        h[0] = (unsigned long long)i;
        double dl = a + 2e6;
        while (h[1] != (unsigned long long)i) { if (now_us() > dl) { printf("%s TIMEOUT at %d\n", name, i); exit(2); } }
        t[i - 1] = now_us() - a;
        if (bytes && h[3] != expect(bytes / 8, value)) wrong++;
    }
    h[2] = 1;
    CK(cudaDeviceSynchronize());
    double sum = 0;
    for (int i = 16; i < n; i++) sum += t[i];
    qsort(t + 16, n - 16, sizeof(double), cmp);
    double p50 = t[16 + (n - 16) / 2];
    printf("%-46s p50 %6.2f  mean %6.2f  p10 %6.2f  p90 %6.2f us  wrong rounds %d\n", name, p50,
           sum / (n - 16), t[16 + (n - 16) / 10], t[16 + (n - 16) * 9 / 10], wrong);
    free(t);
    return p50;
}

int main(int argc, char **argv)
{
    unsigned bytes = argc > 1 ? (unsigned)atoi(argv[1]) : 245760;  // four 60 KiB rows
    int n = argc > 2 ? atoi(argv[2]) : 3000;
    unsigned long long *h, *d;
    CK(cudaHostAlloc((void **)&h, 4096, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostGetDevicePointer((void **)&d, h, 0));

    // Device rows.
    void *dev;
    CK(cudaMalloc(&dev, 4u << 20));
    CK(cudaMemset(dev, 1, 4u << 20));

    // Host-mapped rows, the slab's kind of memory.
    unsigned char *hm, *hm_dev;
    CK(cudaHostAlloc((void **)&hm, 4u << 20, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostGetDevicePointer((void **)&hm_dev, hm, 0));
    memset(hm, 1, 4u << 20);

    // Ordinary host rows with huge pages requested, registered afterwards (if the platform allows).
    unsigned char *thp = (unsigned char *)mmap(NULL, 8u << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    unsigned char *thp_rows = NULL, *thp_dev = NULL;
    if (thp != MAP_FAILED) {
        thp_rows = (unsigned char *)(((unsigned long long)thp + (2u << 20) - 1) & ~((2ull << 20) - 1));
        int adv = madvise(thp_rows, 4u << 20, MADV_HUGEPAGE);
        memset(thp_rows, 1, 4u << 20);
        cudaError_t r = cudaHostRegister(thp_rows, 4u << 20, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (r == cudaSuccess) r = cudaHostGetDevicePointer((void **)&thp_dev, thp_rows, 0);
        printf("registered host memory: madvise(MADV_HUGEPAGE)=%d cudaHostRegister=%s\n", adv, r == cudaSuccess ? "ok" : cudaGetErrorString(r));
        if (r != cudaSuccess) { thp_dev = NULL; cudaGetLastError(); }
    }

    printf("%u bytes of rows, %d rounds (first 16 dropped), command to acknowledgement\n", bytes, n);
    double base = run("nothing read", h, d, NULL, dev, 0, n, 0);
    double a = run("device memory", h, d, NULL, dev, bytes, n, 0);
    double b = run("host-mapped, left alone", h, d, hm, hm_dev, bytes, n, 0);
    double c = run("host-mapped, host rewrites every round", h, d, hm, hm_dev, bytes, n, 1);
    double c2 = run("host-mapped, rewritten and flushed", h, d, hm, hm_dev, bytes, n, 2);
    double e = 0, f = 0, f2 = 0;
    if (thp_dev) {
        e = run("registered host memory, left alone", h, d, thp_rows, thp_dev, bytes, n, 0);
        f = run("registered host memory, host rewrites", h, d, thp_rows, thp_dev, bytes, n, 1);
        f2 = run("registered host memory, rewritten and flushed", h, d, thp_rows, thp_dev, bytes, n, 2);
    }
    printf("read cost over nothing: device %.2f | host-mapped %.2f, rewritten %.2f, flushed %.2f", a - base, b - base, c - base, c2 - base);
    if (thp_dev) printf(" | registered %.2f, rewritten %.2f, flushed %.2f", e - base, f - base, f2 - base);
    printf(" us\n");
    return 0;
}
