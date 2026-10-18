/*
 * The three libibverbs entry points this backend needs that are `static inline` in
 * <infiniband/verbs.h> and therefore not exported by libibverbs.so.1 (measured against
 * rdma-core 39: nm -D resolves 16 of the 19 entry points the backend uses, missing exactly
 * ibv_post_send, ibv_post_recv and ibv_poll_cq).
 *
 * The inline wrappers dispatch through the QP's provider ops; no libibverbs symbol is called
 * here, so this shim links nothing against libibverbs and the run-time dlopen still owns the
 * device context. Building it needs the header; see build.rs for the ENOSYS fallback.
 */
#include <infiniband/verbs.h>

int sparkflash_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                             struct ibv_send_wr **bad) {
    return ibv_post_send(qp, wr, bad);
}

int sparkflash_ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                             struct ibv_recv_wr **bad) {
    return ibv_post_recv(qp, wr, bad);
}

int sparkflash_ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    return ibv_poll_cq(cq, num_entries, wc);
}
