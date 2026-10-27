// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <stdio.h>
#include <stddef.h>
#include <infiniband/verbs.h>
int main(void){
#define P(s,f) printf("%s.%s %zu\n", #s, #f, offsetof(struct s, f))
#define PS(s) printf("sizeof %s %zu align %zu\n", #s, sizeof(s), _Alignof(s))
  PS(union ibv_gid); PS(struct ibv_sge); PS(struct ibv_send_wr); PS(struct ibv_recv_wr); PS(struct ibv_wc); PS(struct ibv_port_attr);
  PS(struct ibv_qp_cap); PS(struct ibv_global_route); PS(struct ibv_ah_attr); PS(struct ibv_qp_init_attr); PS(struct ibv_qp_attr);
  PS(struct ibv_qp); PS(struct ibv_mr); PS(struct ibv_mw_bind_info);
  P(ibv_send_wr, next); P(ibv_send_wr, sg_list); P(ibv_send_wr, num_sge); P(ibv_send_wr, opcode);
  P(ibv_send_wr, send_flags); P(ibv_send_wr, imm_data); P(ibv_send_wr, wr); P(ibv_send_wr, qp_type);
  P(ibv_wc, status); P(ibv_wc, opcode); P(ibv_wc, byte_len); P(ibv_wc, imm_data); P(ibv_wc, qp_num);
  P(ibv_wc, src_qp); P(ibv_wc, wc_flags); P(ibv_wc, pkey_index); P(ibv_wc, sl); P(ibv_wc, dlid_path_bits);
  P(ibv_port_attr, state); P(ibv_port_attr, max_mtu); P(ibv_port_attr, active_mtu); P(ibv_port_attr, gid_tbl_len);
  P(ibv_port_attr, pkey_tbl_len); P(ibv_port_attr, active_speed); P(ibv_port_attr, phys_state);
  P(ibv_port_attr, link_layer); P(ibv_port_attr, flags); P(ibv_port_attr, port_cap_flags2);
  P(ibv_qp_init_attr, send_cq); P(ibv_qp_init_attr, cap); P(ibv_qp_init_attr, qp_type); P(ibv_qp_init_attr, sq_sig_all);
  P(ibv_global_route, flow_label); P(ibv_global_route, sgid_index); P(ibv_global_route, hop_limit); P(ibv_global_route, traffic_class);
  P(ibv_ah_attr, dlid); P(ibv_ah_attr, sl); P(ibv_ah_attr, is_global); P(ibv_ah_attr, port_num);
  P(ibv_qp_attr, cur_qp_state); P(ibv_qp_attr, path_mtu); P(ibv_qp_attr, qkey); P(ibv_qp_attr, rq_psn);
  P(ibv_qp_attr, sq_psn); P(ibv_qp_attr, dest_qp_num); P(ibv_qp_attr, qp_access_flags); P(ibv_qp_attr, cap);
  P(ibv_qp_attr, ah_attr); P(ibv_qp_attr, alt_ah_attr); P(ibv_qp_attr, pkey_index); P(ibv_qp_attr, en_sqd_async_notify);
  P(ibv_qp_attr, max_rd_atomic); P(ibv_qp_attr, min_rnr_timer); P(ibv_qp_attr, port_num); P(ibv_qp_attr, timeout);
  P(ibv_qp_attr, retry_cnt); P(ibv_qp_attr, rnr_retry); P(ibv_qp_attr, alt_timeout); P(ibv_qp_attr, rate_limit);
  P(ibv_qp, pd); P(ibv_qp, srq); P(ibv_qp, handle); P(ibv_qp, qp_num); P(ibv_qp, state); P(ibv_qp, qp_type);
  P(ibv_mr, addr); P(ibv_mr, length); P(ibv_mr, handle); P(ibv_mr, lkey); P(ibv_mr, rkey);
  printf("WC RECV %d RECV_RDMA_WITH_IMM %d SEND %d RDMA_WRITE %d\n", IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM, IBV_WC_SEND, IBV_WC_RDMA_WRITE);
  printf("QPT_RC %d PORT_ACTIVE %d MTU1024 %d WR_IMM %d SIGNALED %d\n", IBV_QPT_RC, IBV_PORT_ACTIVE, IBV_MTU_1024, IBV_WR_RDMA_WRITE_WITH_IMM, IBV_SEND_SIGNALED);
  printf("STATE init %d rtr %d rts %d err %d\n", IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS, IBV_QPS_ERR);
  printf("MASKS state %d pkey %d port %d access %d av %d mtu %d dest %d rqpsn %d maxdest %d minrnr %d sqpsn %d timeout %d retry %d rnr %d maxrd %d\n",
    IBV_QP_STATE, IBV_QP_PKEY_INDEX, IBV_QP_PORT, IBV_QP_ACCESS_FLAGS, IBV_QP_AV, IBV_QP_PATH_MTU, IBV_QP_DEST_QPN, IBV_QP_RQ_PSN, IBV_QP_MAX_DEST_RD_ATOMIC, IBV_QP_MIN_RNR_TIMER, IBV_QP_SQ_PSN, IBV_QP_TIMEOUT, IBV_QP_RETRY_CNT, IBV_QP_RNR_RETRY, IBV_QP_MAX_QP_RD_ATOMIC);
  return 0;
}
