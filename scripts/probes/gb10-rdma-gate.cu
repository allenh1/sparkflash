// regprobe: the GB10 gate for a host-proxied RDMA-write exchange.
//  1. does ibv_reg_mr accept cudaHostAlloc(MAPPED|PORTABLE) memory?
//  2. does it accept cudaMalloc device memory (the peermem question)?
//  3. can the GPU write that memory and have the NIC read exactly those bytes?
//  4. can the GPU read exactly the bytes the NIC wrote (the coherence the sum kernel needs)?
//  5. round trip: post -> peer GPU sees the arrival flag -> peer doorbell -> peer host posts reply
//     -> local GPU sees the reply. That is the loop the exchange replaces the NCCL collective with.
//
// usage: regprobe --role s|c --peer <ip> --port <p> [--dev rocep1s0f1] [--gid 3] [--bytes 10240] [--iters 2000]
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#define SLOT_BYTES (2u << 20)
#define RECV_OFF   SLOT_BYTES

static void die(const char *m) { fprintf(stderr, "FATAL: %s: %s\n", m, strerror(errno)); exit(1); }
#define CK(x) do { cudaError_t _r = (x); if (_r) { fprintf(stderr, "FATAL %s -> %d (%s)\n", #x, (int)_r, cudaGetErrorString(_r)); exit(1);} } while (0)

__global__ void fill_kernel(unsigned long long *p, unsigned long long seed, int nwords)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) p[i] = seed + (unsigned long long)i;
}

__global__ void spin_verify_kernel(const unsigned long long *payload,
                                   volatile unsigned long long *flag, unsigned long long expect,
                                   int nwords, volatile unsigned long long *doorbell,
                                   unsigned long long dbval, unsigned long long seed,
                                   unsigned int *bad, unsigned long long *spin_cyc)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (*(volatile unsigned long long *)flag != expect) { __nanosleep(32); }
        *spin_cyc = clock64() - t0;
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) {
        unsigned long long want = seed + (unsigned long long)i;
        if (*(volatile unsigned long long *)&payload[i] != want) atomicAdd(bad, 1u);
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) *doorbell = dbval;
}

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

static int tcp_connect(const char *peer, int port)
{
    int s = socket(AF_INET, SOCK_STREAM, 0); if (s < 0) die("socket");
    struct sockaddr_in a; memset(&a, 0, sizeof a); a.sin_family = AF_INET; a.sin_port = htons(port);
    inet_pton(AF_INET, peer, &a.sin_addr);
    for (int i = 0; i < 120; i++) { if (connect(s, (struct sockaddr *)&a, sizeof a) == 0) return s; usleep(500000); }
    die("connect"); return -1;
}
static int tcp_listen(int port)
{
    int s = socket(AF_INET, SOCK_STREAM, 0); if (s < 0) die("socket");
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a); a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(s, (struct sockaddr *)&a, sizeof a)) die("bind");
    if (listen(s, 1)) die("listen");
    int c = accept(s, NULL, NULL); if (c < 0) die("accept"); close(s); return c;
}
static void xchg_all(int fd, void *buf, size_t n) { char *p = (char *)buf; while (n) { ssize_t r = read(fd, p, n); if (r <= 0) die("read"); p += r; n -= (size_t)r; } }
static void xsend_all(int fd, const void *buf, size_t n) { const char *p = (const char *)buf; while (n) { ssize_t r = write(fd, p, n); if (r <= 0) die("write"); p += r; n -= (size_t)r; } }

struct endpoint { uint32_t qpn; unsigned char gid[16]; uint64_t addr; uint32_t rkey; uint32_t id; };

static int modify_qp(struct ibv_qp *qp, const struct endpoint *r, uint8_t gid_index)
{
    struct ibv_qp_attr attr; memset(&attr, 0, sizeof attr);
    attr.qp_state = IBV_QPS_INIT; attr.pkey_index = 0; attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) return 1;
    memset(&attr, 0, sizeof attr);
    attr.qp_state = IBV_QPS_RTR; attr.path_mtu = IBV_MTU_4096; attr.dest_qp_num = r->qpn;
    attr.rq_psn = 0; attr.max_dest_rd_atomic = 1; attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1; attr.ah_attr.grh.dgid = *(const union ibv_gid *)r->gid;
    attr.ah_attr.grh.sgid_index = gid_index; attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.port_num = 1;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) return 2;
    memset(&attr, 0, sizeof attr);
    attr.qp_state = IBV_QPS_RTS; attr.timeout = 14; attr.retry_cnt = 7; attr.rnr_retry = 7; attr.sq_psn = 0; attr.max_rd_atomic = 1;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) return 3;
    return 0;
}

int main(int argc, char **argv)
{
    // No peer address is baked in: which neighbour a server talks to is a deployment fact.
    const char *role = "s", *peer_ip = NULL, *dev = "rocep1s0f1";
    int port = 18615, gid_index = 3, iters = 2000;
    unsigned bytes = 10240;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--role")) role = argv[++i];
        else if (!strcmp(argv[i], "--peer")) peer_ip = argv[++i];
        else if (!strcmp(argv[i], "--port")) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dev")) dev = argv[++i];
        else if (!strcmp(argv[i], "--gid")) gid_index = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bytes")) bytes = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    }
    if (bytes + 64 > SLOT_BYTES) { fprintf(stderr, "bytes too big\n"); return 1; }
    if (peer_ip == NULL) {
        fprintf(stderr, "usage: %s --role s|r --peer A.B.C.D [--port P] [--bytes N] [--iters M] [--dev NAME] [--gid N]\n"
                        "  --peer is the other end's address on the rail being measured; there is no default\n", argv[0]);
        return 1;
    }
    int is_server = role[0] == 's';
    unsigned my_id = is_server ? 0u : 1u;
    unsigned total = bytes + 8;                 // payload + arrival flag, one WR
    int nwords = (int)(bytes / 8);

    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    struct cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("[%s] gpu0=%s unified_addressing=%d canMapHostMemory=%d integrated=%d\n",
           role, prop.name, prop.unifiedAddressing, prop.canMapHostMemory, prop.integrated);

    unsigned long long *h_buf = NULL;
    CK(cudaHostAlloc((void **)&h_buf, 2 * SLOT_BYTES, cudaHostAllocMapped | cudaHostAllocPortable));
    memset(h_buf, 0, 2 * SLOT_BYTES);
    unsigned long long *d_alias = NULL;
    CK(cudaHostGetDevicePointer((void **)&d_alias, h_buf, 0));
    printf("[%s] cudaHostAlloc(mapped) host=%p device_alias=%p equal=%d\n", role, (void *)h_buf, (void *)d_alias, h_buf == d_alias);

    void *d_malloc = NULL;
    cudaError_t e_malloc = cudaMalloc(&d_malloc, 1 << 20);
    printf("[%s] cudaMalloc rc=%d ptr=%p\n", role, (int)e_malloc, d_malloc);
    if (e_malloc) (void)cudaGetLastError();

    struct ibv_device **list = ibv_get_device_list(NULL);
    struct ibv_device *target = NULL;
    for (int i = 0; list && list[i]; i++) if (!strcmp(ibv_get_device_name(list[i]), dev)) target = list[i];
    if (!target) { fprintf(stderr, "no device %s\n", dev); return 1; }
    struct ibv_context *ctx = ibv_open_device(target);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    struct ibv_cq *cq = ibv_create_cq(ctx, 64, NULL, NULL, 0);

    int mracc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    errno = 0;
    struct ibv_mr *mr = ibv_reg_mr(pd, h_buf, 2 * SLOT_BYTES, mracc);
    printf("[%s] ibv_reg_mr(cudaHostAlloc mapped, %u B): %s %s\n", role, 2 * SLOT_BYTES,
           mr ? "OK" : "FAILED", mr ? "" : strerror(errno));
    if (mr) printf("[%s]   rkey=0x%x lkey=0x%x addr=%p\n", role, mr->rkey, mr->lkey, mr->addr);

    if (d_malloc) {
        errno = 0;
        struct ibv_mr *mrdev = ibv_reg_mr(pd, d_malloc, 1 << 20, mracc);
        printf("[%s] ibv_reg_mr(cudaMalloc device, 1 MiB): %s %s\n", role,
               mrdev ? "OK" : "FAILED", mrdev ? "" : strerror(errno));
        if (mrdev) {
            // if it registers, can the NIC actually read it? (write 10 KB from it to the peer later)
            printf("[%s]   device rkey=0x%x\n", role, mrdev->rkey);
        }
    }
    if (!mr) { fprintf(stderr, "host registration failed; the gate is closed\n"); return 2; }

    struct ibv_qp_init_attr ia; memset(&ia, 0, sizeof ia);
    ia.send_cq = cq; ia.recv_cq = cq; ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = 64; ia.cap.max_recv_wr = 8; ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
    struct ibv_qp *qp = ibv_create_qp(pd, &ia);
    if (!qp) die("ibv_create_qp");

    union ibv_gid gid; memset(&gid, 0, sizeof gid);
    if (ibv_query_gid(ctx, 1, gid_index, &gid)) die("ibv_query_gid");
    char gs[64]; inet_ntop(AF_INET, gid.raw + 12, gs, sizeof gs);
    printf("[%s] qpn=0x%x gid[%d]=%s\n", role, qp->qp_num, gid_index, gs);

    struct endpoint me; memset(&me, 0, sizeof me);
    me.qpn = qp->qp_num; memcpy(me.gid, gid.raw, 16);
    me.addr = (uint64_t)(uintptr_t)((char *)h_buf + RECV_OFF);   // what peers write into
    me.rkey = mr->rkey; me.id = my_id;

    struct endpoint them; memset(&them, 0, sizeof them);
    int fd = is_server ? tcp_listen(port) : tcp_connect(peer_ip, port);
    if (is_server) { xchg_all(fd, &them, sizeof them); xsend_all(fd, &me, sizeof me); }
    else           { xsend_all(fd, &me, sizeof me); xchg_all(fd, &them, sizeof them); }
    printf("[%s] peer qpn=0x%x id=%u addr=0x%llx rkey=0x%x\n", role, them.qpn, them.id, (unsigned long long)them.addr, them.rkey);
    int rc = modify_qp(qp, &them, (uint8_t)gid_index);
    if (rc) { fprintf(stderr, "modify_qp failed at step %d\n", rc); return 3; }

    unsigned long long *send = h_buf;                       // payload + flag
    unsigned long long *d_send = d_alias;
    volatile unsigned long long *r_flag = (volatile unsigned long long *)((char *)h_buf + RECV_OFF + bytes);
    volatile unsigned long long *r_door = (volatile unsigned long long *)((char *)h_buf + RECV_OFF + bytes + 8);
    unsigned long long *d_r_pay = (unsigned long long *)((char *)d_alias + RECV_OFF);
    unsigned long long *d_r_flag = (unsigned long long *)((char *)d_alias + RECV_OFF + bytes);
    unsigned long long *d_r_door = (unsigned long long *)((char *)d_alias + RECV_OFF + bytes + 8);
    unsigned long long *d_r_spin = (unsigned long long *)((char *)d_alias + RECV_OFF + bytes + 16);
    unsigned int *d_r_bad = (unsigned int *)((char *)d_alias + RECV_OFF + bytes + 24);
    unsigned int h_bad = 0;
    CK(cudaMemset(d_r_bad, 0, 4));

    // coherence check 1: GPU writes, the CPU must see it
    fill_kernel<<<1, 256>>>(d_send, 0x1122334455667788ull, 64);
    CK(cudaDeviceSynchronize());
    printf("[%s] gpu->cpu: word0=0x%llx expect=0x1122334455667788 %s\n", role,
           (unsigned long long)send[0], send[0] == 0x1122334455667788ull ? "coherent" : "STALE");

    double rtt_min = 1e18, rtt_sum = 0, post_min = 1e18, post_sum = 0;
    int rtt_n = 0, post_n = 0;
    unsigned long long spin_max = 0;
    for (int it = 0; it < iters; it++) {
        int i_send = is_server ? (it % 2 == 0) : (it % 2 == 1);
        unsigned long long seq = (unsigned long long)(it / 2 + 1);
        if (i_send) {
            unsigned long long seed = ((unsigned long long)my_id << 32) | seq;
            fill_kernel<<<1, 256>>>(d_send, seed, nwords);
            CK(cudaDeviceSynchronize());
            send[bytes / 8] = seed;                          // the arrival flag the peer will see
            struct ibv_sge sge = { .addr = (uint64_t)(uintptr_t)send, .length = total, .lkey = mr->lkey };
            struct ibv_send_wr wr; memset(&wr, 0, sizeof wr);
            wr.wr_id = seq; wr.opcode = IBV_WR_RDMA_WRITE; wr.sg_list = &sge; wr.num_sge = 1;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = them.addr; wr.wr.rdma.rkey = them.rkey;
            double t0 = now_us();
            struct ibv_send_wr *bad = NULL;
            if (ibv_post_send(qp, &wr, &bad)) die("post_send");
            double tp = now_us();
            struct ibv_wc wc; int n;
            do { n = ibv_poll_cq(cq, 1, &wc); } while (n == 0);
            double t1 = now_us();
            if (wc.status != IBV_WC_SUCCESS) { fprintf(stderr, "wc status %d\n", wc.status); return 4; }
            post_sum += t1 - tp; post_min = (t1 - tp) < post_min ? t1 - tp : post_min; post_n++;

            // the peer's reply: same shape, seed built from the peer's id
            unsigned long long expect = ((unsigned long long)them.id << 32) | seq;
            *r_door = 0;
            spin_verify_kernel<<<1, 256>>>(d_r_pay, d_r_flag, expect, nwords, d_r_door, seq, expect, d_r_bad, d_r_spin);
            while (*(volatile unsigned long long *)r_door != seq) { /* host spin on the GPU doorbell */ }
            double t2 = now_us();
            rtt_sum += t2 - t0; rtt_min = (t2 - t0) < rtt_min ? t2 - t0 : rtt_min; rtt_n++;
            unsigned long long spin = 0; CK(cudaMemcpy(&spin, d_r_spin, 8, cudaMemcpyDeviceToHost));
            if (spin > spin_max) spin_max = spin;
        } else {
            unsigned long long expect = ((unsigned long long)them.id << 32) | seq;
            *r_door = 0;
            spin_verify_kernel<<<1, 256>>>(d_r_pay, d_r_flag, expect, nwords, d_r_door, seq, expect, d_r_bad, d_r_spin);
            while (*(volatile unsigned long long *)r_door != seq) { }
            // reply immediately, exactly as the proxy will
            unsigned long long rseed = ((unsigned long long)my_id << 32) | seq;
            fill_kernel<<<1, 256>>>(d_send, rseed, nwords);
            CK(cudaDeviceSynchronize());
            send[bytes / 8] = rseed;
            struct ibv_sge sge = { .addr = (uint64_t)(uintptr_t)send, .length = total, .lkey = mr->lkey };
            struct ibv_send_wr wr; memset(&wr, 0, sizeof wr);
            wr.wr_id = seq; wr.opcode = IBV_WR_RDMA_WRITE; wr.sg_list = &sge; wr.num_sge = 1;
            wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = them.addr; wr.wr.rdma.rkey = them.rkey;
            struct ibv_send_wr *bad2 = NULL;
            struct ibv_wc wc2; int n2;
            if (ibv_post_send(qp, &wr, &bad2)) die("post_send");
            do { n2 = ibv_poll_cq(cq, 1, &wc2); } while (n2 == 0);
            if (wc2.status != IBV_WC_SUCCESS) { fprintf(stderr, "reply wc status %d\n", wc2.status); return 4; }
        }
        if (it % 400 == 0) {
            CK(cudaMemcpy(&h_bad, d_r_bad, 4, cudaMemcpyDeviceToHost));
            printf("[%s] it=%d bad=%u flag=%llu door=%llu\n", role, it, h_bad, *r_flag, *r_door);
            fflush(stdout);
        }
    }
    CK(cudaMemcpy(&h_bad, d_r_bad, 4, cudaMemcpyDeviceToHost));
    printf("[%s] RESULT payload_bad_words=%u verifies=%s\n", role, h_bad, h_bad == 0 ? "YES" : "NO");
    printf("[%s] post->cq  min %.2f us avg %.2f us (n=%d)\n", role, post_min, post_sum / (post_n ? post_n : 1), post_n);
    if (rtt_n) printf("[%s] full RTT  min %.2f us avg %.2f us (n=%d)  gpu spin_max %llu cycles\n",
                      role, rtt_min, rtt_sum / rtt_n, rtt_n, spin_max);
    close(fd);
    return h_bad == 0 ? 0 : 5;
}
