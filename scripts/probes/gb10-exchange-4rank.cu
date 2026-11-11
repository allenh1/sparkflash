// gb10-exchange-4rank: the direct-RDMA exchange on the real fabric, four ranks, measured.
//
//  This is the collective the serving step would run in place of its NCCL group: every rank owns
//  one registered pinned slot per peer, publishes its row's flag, posts one RDMA write per peer,
//  and waits until every peer's flag is >= its own sequence number. The host spin is the same work
//  the sum kernel does over the same words, so the number this prints is the exchange's cost.
//
//  Two things here are load-bearing and were bugs first:
//   - the arrival wait is a predicate over every peer, re-evaluated, and it compares >= and not
//     ==. Counting transitions instead of re-evaluating hangs the moment one peer is lapped, and
//     --trace showed exactly that: a rank that reads flag 2 while it waits for 1 must count it.
//   - the GID is chosen by local IP (--myip) and not by a fixed table index. A wrong index builds
//     a QP that connects, accepts writes and never delivers, which looks like a lost packet.
//
//  usage: gb10-exchange-4rank --rank R --peer ip0 ip1 ip2 ip3 --myip A.B.C.D --bytes N --iters M
//         [--port P] [--dev rocep1s0f1] [--gid N] [--trace N] [--signal-all]
//  Every rank owns one registered slot per peer; a round is: publish seq in the send row's
//  flag word, post one RDMA write to each peer, then wait until every peer's flag is >= seq.
//  The wait is >= and not ==: a reader that gets lapped (or descheduled between posting and
//  reading) must not deadlock on a round that has already been overwritten. Laps are counted
//  and reported, because a lap is a probe artifact, not a property of the fabric.
//  Reports, in us, time to the first peer's arrival and to the last, with percentiles.
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

#define WORLD 4
#define SLOT (1u << 20)
#define STRIDE (SLOT + 64)

static void die(const char *m) { fprintf(stderr, "FATAL %s: %s\n", m, strerror(errno)); exit(1); }
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void xall(int fd, void *b, size_t n) { char *p = (char *)b; while (n) { ssize_t r = read(fd, p, n); if (r <= 0) die("read"); p += r; n -= (size_t)r; } }
static void sendall(int fd, const void *b, size_t n) { const char *p = (const char *)b; while (n) { ssize_t r = write(fd, p, n); if (r <= 0) die("write"); p += r; n -= (size_t)r; } }

struct ep { uint32_t qpn[WORLD]; unsigned char gid[16]; uint64_t addr; uint32_t rkey; uint32_t rank; };

static int gid_is_ipv4(const union ibv_gid *g, uint32_t ip_host_order)
{
    const unsigned char *r = (const unsigned char *)g->raw;
    for (int i = 0; i < 10; i++) if (r[i] != 0) return 0;
    if (r[10] != 0xff || r[11] != 0xff) return 0;
    uint32_t v = ((uint32_t)r[12] << 24) | ((uint32_t)r[13] << 16) | ((uint32_t)r[14] << 8) | r[15];
    return v == ip_host_order;
}

static int find_gid_slot(struct ibv_context *ctx, uint32_t ip)
{
    int best = -1;
    for (int i = 0; i < 32; i++) {
        union ibv_gid g;
        if (ibv_query_gid(ctx, 1, i, &g)) break;
        int all_zero = 1, all_ff = 1;
        for (int k = 0; k < 16; k++) { if (g.raw[k]) all_zero = 0; if (g.raw[k] != 0xff) all_ff = 0; }
        if (all_zero || all_ff) continue;
        if (gid_is_ipv4(&g, ip)) { best = i; }
    }
    return best;
}

static int qp_to_rtr_rts(struct ibv_qp *qp, const struct ep *r, uint8_t gid_index, int my_rank)
{
    struct ibv_qp_attr a; memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) return 1;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_4096; a.dest_qp_num = r->qpn[my_rank]; a.rq_psn = 0;
    a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1; a.ah_attr.grh.dgid = *(const union ibv_gid *)r->gid;
    a.ah_attr.grh.sgid_index = gid_index; a.ah_attr.grh.hop_limit = 64; a.ah_attr.port_num = 1;
    if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) return 2;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = 0; a.max_rd_atomic = 1;
    if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) return 3;
    return 0;
}

static const char *wc_status(enum ibv_wc_status s)
{
    switch (s) {
    case IBV_WC_SUCCESS: return "SUCCESS";
    case IBV_WC_LOC_LEN_ERR: return "LOC_LEN_ERR";
    case IBV_WC_LOC_QP_OP_ERR: return "LOC_QP_OP_ERR";
    case IBV_WC_LOC_EEC_OP_ERR: return "LOC_EEC_OP_ERR";
    case IBV_WC_LOC_PROT_ERR: return "LOC_PROT_ERR";
    case IBV_WC_WR_FLUSH_ERR: return "WR_FLUSH_ERR";
    case IBV_WC_MW_BIND_ERR: return "MW_BIND_ERR";
    case IBV_WC_BAD_RESP_ERR: return "BAD_RESP_ERR";
    case IBV_WC_LOC_ACCESS_ERR: return "LOC_ACCESS_ERR";
    case IBV_WC_REM_INV_REQ_ERR: return "REM_INV_REQ_ERR";
    case IBV_WC_REM_ACCESS_ERR: return "REM_ACCESS_ERR";
    case IBV_WC_REM_OP_ERR: return "REM_OP_ERR";
    case IBV_WC_RETRY_EXC_ERR: return "RETRY_EXC_ERR";
    case IBV_WC_RNR_RETRY_EXC_ERR: return "RNR_RETRY_EXC_ERR";
    case IBV_WC_LOC_RDD_VIOL_ERR: return "LOC_RDD_VIOL_ERR";
    case IBV_WC_REM_INV_RD_REQ_ERR: return "REM_INV_RD_REQ_ERR";
    case IBV_WC_REM_ABORT_ERR: return "REM_ABORT_ERR";
    case IBV_WC_INV_EECN_ERR: return "INV_EECN_ERR";
    case IBV_WC_INV_EEC_STATE_ERR: return "INV_EEC_STATE_ERR";
    case IBV_WC_FATAL_ERR: return "FATAL_ERR";
    case IBV_WC_RESP_TIMEOUT_ERR: return "RESP_TIMEOUT_ERR";
    case IBV_WC_GENERAL_ERR: return "GENERAL_ERR";
    default: return "?";
    }
}

int main(int argc, char **argv)
{
    int rank = 0, port = 18700, iters = 200, gid_index = -1;
    // No addresses are baked in: the rail a rank is on is a deployment fact, not a property of the
    // probe, and a default here would be a real host's address in the repository.
    const char *peerip[WORLD] = {NULL, NULL, NULL, NULL};
    const char *dev = "rocep1s0f1";
    const char *myip = NULL;
    int trace = 0, signal_all = 0;
    int have_peers = 0;
    unsigned bytes = 10240;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rank")) rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--peer")) { for (int k = 0; k < WORLD; k++) peerip[k] = argv[++i]; have_peers = 1; }
        else if (!strcmp(argv[i], "--port")) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bytes")) bytes = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dev")) dev = argv[++i];
        else if (!strcmp(argv[i], "--gid")) gid_index = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--myip")) myip = argv[++i];
        else if (!strcmp(argv[i], "--trace")) trace = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--signal-all")) signal_all = 1;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (bytes + 8 > SLOT) { fprintf(stderr, "bytes too big\n"); return 1; }
    if (!have_peers) {
        fprintf(stderr, "usage: %s --rank R --peer ip0 ip1 ip2 ip3 [--myip A.B.C.D] [--bytes N] [--iters M]\n"
                        "  --peer is the rank-order address table for the rail being measured; there is no default\n", argv[0]);
        return 1;
    }

    unsigned char *base = NULL;
    if (posix_memalign((void **)&base, 4096, STRIDE * WORLD)) die("memalign");
    memset(base, 0, STRIDE * WORLD);
    for (unsigned r = 0; r < WORLD; r++) *(unsigned long long *)(base + r * STRIDE + bytes) = 0;

    struct ibv_device **list = ibv_get_device_list(NULL);
    struct ibv_device *tgt = NULL;
    for (int i = 0; list && list[i]; i++) if (!strcmp(ibv_get_device_name(list[i]), dev)) tgt = list[i];
    if (!tgt) { fprintf(stderr, "no device %s\n", dev); return 1; }
    struct ibv_context *ctx = ibv_open_device(tgt);
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    struct ibv_cq *cq = ibv_create_cq(ctx, 256, NULL, NULL, 0);
    struct ibv_mr *mr = ibv_reg_mr(pd, base, STRIDE * WORLD,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!mr) { fprintf(stderr, "reg_mr failed: %s\n", strerror(errno)); return 2; }

    // GID by local IP: the fixed index is a guess, and a wrong one silently builds a QP that
    // never delivers.
    if (myip) {
        uint32_t ip; if (inet_pton(AF_INET, myip, &ip) != 1) die("myip");
        uint32_t host = ntohl(ip);
        int slot = find_gid_slot(ctx, host);
        if (slot < 0) { fprintf(stderr, "no GID matches %s\n", myip); return 3; }
        printf("[%d] gid for %s is index %d\n", rank, myip, slot);
        gid_index = slot;
    }
    if (gid_index < 0) gid_index = 3;

    struct ibv_qp_init_attr ia; memset(&ia, 0, sizeof ia);
    ia.send_cq = cq; ia.recv_cq = cq; ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = 256; ia.cap.max_recv_wr = 8; ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
    struct ibv_qp *qp[WORLD]; memset(qp, 0, sizeof qp);
    for (int p = 0; p < WORLD; p++) { if (p == rank) continue; qp[p] = ibv_create_qp(pd, &ia); if (!qp[p]) die("create_qp"); }

    union ibv_gid gid; memset(&gid, 0, sizeof gid);
    if (ibv_query_gid(ctx, 1, gid_index, &gid)) die("query_gid");
    struct ep me; memset(&me, 0, sizeof me);
    me.addr = (uint64_t)(uintptr_t)base; me.rkey = mr->rkey; me.rank = (uint32_t)rank;
    memcpy(me.gid, gid.raw, 16);
    for (int p = 0; p < WORLD; p++) if (p != rank) me.qpn[p] = qp[p]->qp_num;

    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons(port + rank); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(lfd, (struct sockaddr *)&a, sizeof a)) die("bind");
    if (listen(lfd, WORLD)) die("listen");

    struct ep peers[WORLD]; memset(peers, 0, sizeof peers);
    int accepted = 0;
    for (int p = rank + 1; p < WORLD; p++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in b; memset(&b, 0, sizeof b);
        b.sin_family = AF_INET; b.sin_port = htons(port + p); inet_pton(AF_INET, peerip[p], &b.sin_addr);
        int ok = 0;
        for (int t = 0; t < 120; t++) { if (connect(fd, (struct sockaddr *)&b, sizeof b) == 0) { ok = 1; break; } usleep(500000); }
        if (!ok) { fprintf(stderr, "[%d] connect to %d at %s failed\n", rank, p, peerip[p]); return 4; }
        sendall(fd, &me, sizeof me);
        struct ep theirs; xall(fd, &theirs, sizeof theirs);
        peers[theirs.rank] = theirs;
        close(fd);
    }
    while (accepted < rank) {
        int c = accept(lfd, NULL, NULL); if (c < 0) die("accept");
        struct ep theirs; xall(c, &theirs, sizeof theirs);
        sendall(c, &me, sizeof me);
        peers[theirs.rank] = theirs;
        close(c);
        accepted++;
    }
    close(lfd);
    for (int p = 0; p < WORLD; p++) {
        if (p == rank) continue;
        if (peers[p].qpn[rank] == 0) { fprintf(stderr, "rank %d: no endpoint for peer %d\n", rank, p); return 5; }
        int rc = qp_to_rtr_rts(qp[p], &peers[p], (uint8_t)gid_index, rank);
        if (rc) { fprintf(stderr, "[%d] modify_qp peer %d step %d: %s\n", rank, p, rc, strerror(errno)); return 6; }
    }
    printf("[%d] mesh up: gid=%d bytes=%u slot=%u\n", rank, gid_index, bytes, STRIDE);
    for (int p = 0; p < WORLD; p++) {
        if (p == rank) continue;
        printf("[%d]   peer %d: my qp %u -> dest qp %u, gid %02x%02x:%02x%02x\n", rank, p,
               qp[p]->qp_num, peers[p].qpn[rank], peers[p].gid[12], peers[p].gid[13], peers[p].gid[14], peers[p].gid[15]);
    }

    unsigned long long seq = 0;
    unsigned long long rx[WORLD]; memset(rx, 0, sizeof rx);
    double *first_v = malloc(sizeof(double) * (size_t)iters);
    double *last_v = malloc(sizeof(double) * (size_t)iters);
    int n = 0, laps = 0;
    unsigned long long worst_lap = 0;
    double warm = 16;   // the first rounds pay for QP warm-up and are not part of the number
    for (int it = 0; it < iters; it++) {
        seq++;
        *(unsigned long long *)(base + rank * STRIDE + bytes) = seq;
        double t0 = now_us();
        for (int p = 0; p < WORLD; p++) {
            if (p == rank) continue;
            struct ibv_sge sge = { .addr = (uint64_t)(uintptr_t)(base + rank * STRIDE),
                                   .length = bytes + 8, .lkey = mr->lkey };
            struct ibv_send_wr wr; memset(&wr, 0, sizeof wr);
            wr.wr_id = seq; wr.opcode = IBV_WR_RDMA_WRITE; wr.sg_list = &sge; wr.num_sge = 1;
            wr.send_flags = (signal_all || seq % 8 == 0) ? IBV_SEND_SIGNALED : 0;
            wr.wr.rdma.remote_addr = peers[p].addr + rank * STRIDE;
            wr.wr.rdma.rkey = peers[p].rkey;
            struct ibv_send_wr *bad = NULL;
            int prc = ibv_post_send(qp[p], &wr, &bad);
            if (prc) { fprintf(stderr, "[%d] post_send peer %d it %d: %s\n", rank, p, it, strerror(prc)); return 7; }
        }
        {
            struct ibv_wc wc[16];
            int got = ibv_poll_cq(cq, 16, wc);
            for (int i = 0; i < got; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    printf("[%d] it=%d BAD COMPLETION qp %u status %s (%d) vendor %#x\n", rank, it,
                           wc[i].qp_num, wc_status(wc[i].status), (int)wc[i].status, wc[i].vendor_err);
                    printf("[%d]   that qp belongs to peer(s):", rank);
                    for (int p = 0; p < WORLD; p++) if (p != rank && qp[p]->qp_num == wc[i].qp_num) printf(" %d", p);
                    printf("\n");
                    fflush(stdout);
                    return 8;
                }
            }
        }
        double t_first = 0;
        double dl = t0 + 2e6;
        // Arrival is a predicate over all peers, re-evaluated every poll, and it is >= seq and
        // not == seq: a peer that has already published a later round has arrived, and a reader
        // that gets lapped must not wait for a round that is gone. Counting transitions instead
        // of re-evaluating the predicate hangs the moment one peer is lapped.
        int seen = 0;
        for (;;) {
            struct ibv_wc wc[16];
            int got = ibv_poll_cq(cq, 16, wc);
            for (int i = 0; i < got; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    printf("[%d] it=%d BAD COMPLETION qp %u status %s (%d) vendor %#x\n", rank, it,
                           wc[i].qp_num, wc_status(wc[i].status), (int)wc[i].status, wc[i].vendor_err);
                    for (int p = 0; p < WORLD; p++) if (p != rank && qp[p]->qp_num == wc[i].qp_num)
                        printf("[%d]   that is peer %d\n", rank, p);
                    return 8;
                }
            }
            seen = 0;
            for (int p = 0; p < WORLD; p++) {
                if (p == rank) continue;
                unsigned long long v = *(volatile unsigned long long *)(base + p * STRIDE + bytes);
                if (v >= seq) {
                    if (rx[p] < v) rx[p] = v;
                    if (seen == 0 && t_first == 0) t_first = now_us();
                    seen++;
                    if (v > seq) {
                        laps++;
                        if (v - seq > worst_lap) worst_lap = v - seq;
                    }
                }
            }
            if (seen == WORLD - 1) break;
            if (now_us() > dl) {
                fprintf(stderr, "[%d] timeout it=%d seen=%d/%d; flags:", rank, it, seen, WORLD - 1);
                for (int p = 0; p < WORLD; p++) {
                    if (p == rank) continue;
                    fprintf(stderr, " p%d=%llu", p, *(volatile unsigned long long *)(base + p * STRIDE + bytes));
                }
                fprintf(stderr, " (want >= %llu)\n", seq);
                return 9;
            }
        }
        double t_last = now_us();
        if (trace && it < trace) {
            printf("[%d] it=%d seq=%llu first %.2f last %.2f flags", rank, it, seq, t_first - t0, t_last - t0);
            for (int p = 0; p < WORLD; p++) if (p != rank)
                printf(" p%d=%llu", p, *(volatile unsigned long long *)(base + p * STRIDE + bytes));
            printf("\n");
        }
        if (it >= warm) {
            first_v[n] = t_first - t0;
            last_v[n] = t_last - t0;
            n++;
        }
    }
    int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
    qsort(first_v, (size_t)n, sizeof(double), cmp);
    qsort(last_v, (size_t)n, sizeof(double), cmp);
    double fs = 0, ls = 0;
    for (int i = 0; i < n; i++) { fs += first_v[i]; ls += last_v[i]; }
    printf("[%d] exchange %u B: first avg %.2f p50 %.2f p99 %.2f | all peers avg %.2f p50 %.2f p99 %.2f max %.2f us (n=%d laps=%d worst=%llu)\n",
           rank, bytes, fs / n, first_v[n / 2], first_v[(int)(n * 0.99)], ls / n, last_v[n / 2],
           last_v[(int)(n * 0.99)], last_v[n - 1], n, laps, worst_lap);
    return 0;
}
