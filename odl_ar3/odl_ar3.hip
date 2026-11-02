// odl_ar3.hip -- N-rank (full-mesh, N <= 4) SUM all-reduce over the OdinLink (odl_tb5) stream API
// for the decode-sized collectives of the exllamav3 cross-host TP backend at world size >= 3
// (model_tp_odl.py, EXL3_TP_ODL_AR3=1). The 2-rank odl_ar2.hip generalised to a mesh in ONE hop:
//
//   stream:  [stage src -> pinned send slot + doorbell] -> [wait for every peer + canonical sum]
//   CPU:     one progress thread per peer cable: doorbell -> stream_send of the send slot to that
//            peer; nonblocking stream_recv of that peer's message into its pinned receive slot,
//            published with a release store to the slot flag the GPU kernel spins on
//
// Every rank sends its WHOLE contribution to every peer over the direct cable to that peer (the
// 3-box mesh is a triangle: one cable per peer), so an all-reduce is one network hop instead of
// RCCL's 2(N-1) ring steps, and both directions of every cable carry traffic at the same time.
// Each rank then sums all N contributions in canonical rank order (r = 0, 1, ..., N-1) with fp32
// accumulation; every rank adds the same bits in the same order, so the reduced tensors are
// bit-identical across ranks, which the replicated MoE routing relies on.
//
// Fully stream-ordered: nothing here blocks the host, so the caller needs no device sync after an
// all-reduce (consumers on the same stream are ordered behind the wait+sum kernel).
//
// Transport: OdinLink streams in latency mode (messages <= 64 KiB; the driver fragments them at
// ~4 KiB inside the ioctl and submits without waiting for TX completion). Streams deliver complete
// messages in order, so message k on a cable is round k. The same stream id (ODL3_SID, default
// 243) is opened on both ends of every cable: stream ids are per device, and each stream has
// exactly one sender (the peer at the other end of that cable).
//
// Peer discovery: odl3_open() opens the stream on every /dev/odl_tb5_N whose link is up;
// odl3_handshake() (called after a barrier, so every rank has its streams open before anything
// is sent -- the driver drops messages for streams that are not open) sends one ping carrying the
// rank on every cable and reads the peer's ping back, which yields the cable -> peer-rank map at
// run time. No per-boot device map is needed (device indices follow XDomain probe order and
// change across reloads).
//
// Slot reuse (NSLOTS = 2, same argument as odl_ar2 extended to N ranks): a peer only sends round
// R + 2 after its wait+sum(R + 1), which needs our round R + 1, which (stream order) is staged
// after our wait+sum(R) finished reading receive slot R % 2 of every peer. Our stage(R + 2)
// overwrites send slot R % 2 only after our wait+sum(R + 1), which needs every peer's R + 1,
// which each peer sends only after receiving our R -- so every progress thread has posted R.
// A progress thread may see the doorbell advance by two rounds at once; the per-slot byte counts
// keep both sizes.
//
// dtype 3 (fp32 in / fp32 out over a bf16 wire): contributions are rounded to bf16 (own one too,
// since the peers only ever see the bf16 version), summed in fp32 in rank order, and the result is
// rounded to bf16 and widened back -- the same rounding contract as the RCCL path it replaces
// (bf16 all-reduce of the bf16-cast tensor), with one rounding of the sum instead of one per ring
// step.
//
// Build: odl_ar3/build.sh (hipcc from the project venv, -lodl_tb5 from ~/.cache/odinlink).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <odl_tb5/odl_tb5.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <dirent.h>

#define ODL3_MAX_RANKS 4
#define ODL3_MAX_PEERS (ODL3_MAX_RANKS - 1)
#define ODL3_MAX_DEVS 8
#define ODL3_MAX_BYTES (64u << 10)            /* latency path of the driver */
#define ODL3_NSLOTS 3                         /* 3: a slot is never written while the GPU may read it */
#define ODL3_HDR 64                           /* per-message header ahead of the payload */
#define ODL3_SLOT (ODL3_MAX_BYTES + ODL3_HDR)  /* slot stride (send and receive) */
#define ODL3_MSG_DATA 1u
#define ODL3_MSG_NACK 2u
#define ODL3_MSG_MAGIC 0x4f443344u          /* "OD3D" */
#define ODL3_PING_MAGIC 0x4f444c34u          /* "ODL4" */

struct odl3_ping { uint32_t magic, rank, world, max_bytes, nslots, pad; };
struct odl3_msg_hdr { uint32_t magic, type; uint64_t round; };   /* DATA: payload follows at ODL3_HDR */

struct odl3_shared {
    uint64_t doorbell;                         /* last staged round, GPU release-store */
    uint32_t slot_nbytes[ODL3_NSLOTS];         /* wire bytes of the round staged in each slot */
    int kernel_err;                            /* wait timeout latch (GPU writes) */
    int thread_err;                            /* progress-thread error latch */
    uint64_t pad[4];
    uint64_t recv_flag[ODL3_MAX_PEERS][ODL3_NSLOTS * 8];   /* [peer][slot * 8]: own cache line */
};

struct odl3_peer {
    odl_tb5_t h;
    int dev;
    int rank;                                  /* peer rank on this cable (-1 = none found) */
    uint8_t sid;
    char *recv_buf, *recv_buf_dev;             /* NSLOTS * MAX_BYTES, pinned */
    pthread_t tid;
    int started;
    /* big (prefill-sized) path: a second, BLOCKING handle on the same cable with its own stream */
    odl_tb5_t bh;
    uint8_t bsid;
    char *brs, *brs_dev;                       /* reduce-scatter: our shard as this peer sent it */
    char *bag, *bag_dev;                       /* all-gather: this peer's reduced shard */
    pthread_t stid, rtid;
    int bstarted;
};

static struct odl3_peer devs_[ODL3_MAX_DEVS];
static int ndevs_;
static struct odl3_peer *peer_of_rank_[ODL3_MAX_RANKS];   /* rank -> cable (NULL = self) */
static int my_rank_ = -1, world_ = 0, npeers_ = 0;
static char *send_buf_, *send_buf_dev_;
static struct odl3_shared *shr_, *shr_dev_;
static uint64_t host_seq_;
static int inited_, opened_;
static uint64_t hot_ns_ = 50000000ull;          /* ODL3_HOT_US: hot-spin window after activity */
static volatile int progress_exit_;
static uint8_t sid_ = 243;
static uint64_t nack_ns_ = 10000000ull;          /* ODL3_NACK_US: re-request a missing round after this */
static uint64_t test_drop_;                       /* ODL3_TEST_DROP=N: skip the FIRST send of every Nth round (recovery test) */
static uint64_t st_nacks_, st_resends_, st_dups_, st_aheads_;   /* recovery counters (all peers) */

extern "C" {
uint32_t odl3_max_bytes(void) { return ODL3_MAX_BYTES; }
int odl3_last_error(void) {
    int e = shr_ ? (shr_->kernel_err | shr_->thread_err) : 0;
    if (shr_) { shr_->kernel_err = 0; shr_->thread_err = 0; }
    return e;
}
uint64_t odl3_seq(void) { return host_seq_; }
int odl3_peer_rank_of_dev(int i) { return (i >= 0 && i < ndevs_) ? devs_[i].rank : -1; }
int odl3_dev_index(int i) { return (i >= 0 && i < ndevs_) ? devs_[i].dev : -1; }
int odl3_num_devs(void) { return ndevs_; }
/* recovery counters: NACKs sent, rounds resent, duplicates dropped, rounds received ahead of a gap */
void odl3_stats(uint64_t *out4) {
    out4[0] = __atomic_load_n(&st_nacks_, __ATOMIC_RELAXED); out4[1] = __atomic_load_n(&st_resends_, __ATOMIC_RELAXED);
    out4[2] = __atomic_load_n(&st_dups_, __ATOMIC_RELAXED); out4[3] = __atomic_load_n(&st_aheads_, __ATOMIC_RELAXED);
}
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* ---- progress: one thread per peer cable ----
 *
 * Loss recovery. The raw DMA path can drop a frame (RX CRC error, or a frame lost with no error
 * flag); the driver then discards the whole message (fragment gap) and nothing retransmits it.
 * Every message therefore carries {round} in a header, and the receiver of round k NACKs it when
 *   - round k + 1 arrives first (a certain gap: rounds travel in order on a stream), or
 *   - it has waited ODL3_NACK_US for round k while its own GPU already needs it.
 * The sender resends round k from its send slot, which is provably intact: slot k % 3 is only
 * rewritten by stage(k + 3), which needs every peer's round k + 2, which no peer can send before
 * receiving our round k. Duplicates (a resend racing a late original) are dropped by round number.
 * Receiving straight into slot (rcvd + 1) % 3 is always safe: with 3 slots the GPU can only be
 * reading rounds rcvd - 1 and rcvd. A round that arrives ahead of a gap (at most one: round
 * rcvd + 3 cannot be sent before our wait+sum(rcvd + 1)) is moved to its own slot and published
 * right after the missing one. NACK and DATA share the cable's stream; this thread is the only
 * sender on our side of it, as the driver's per-stream reassembly requires. */
static int send_msg(struct odl3_peer *p, const void *buf, uint32_t len) {
    uint64_t t0 = now_ns();
    for (;;) {
        int ret = odl_tb5_stream_send(p->h, p->sid, sid_, buf, len);
        if (ret == 0) return 0;
        /* -EINTR: a signal or ptrace stop (profiler, debugger) interrupted the ioctl; the receiver
         * resets its reassembly at the next MSG_START, so resending the whole message is safe */
        if (ret != -EAGAIN && ret != -EINTR) { fprintf(stderr, "odl_ar3: stream_send to rank %d failed: %d\n", p->rank, ret); return -1; }
        if (now_ns() - t0 > 2000000000ull) { fprintf(stderr, "odl_ar3: stream_send to rank %d starved\n", p->rank); return -1; }
    }
}

static int post_round(struct odl3_peer *p, uint64_t s, int resend) {
    uint32_t slot = (uint32_t)(s % ODL3_NSLOTS);
    uint32_t nbytes = __atomic_load_n(&shr_->slot_nbytes[slot], __ATOMIC_ACQUIRE);
    char *msg = send_buf_ + (uint64_t)slot * ODL3_SLOT;
    struct odl3_msg_hdr *h = (struct odl3_msg_hdr *)msg;   /* host-written; the GPU writes only the payload */
    h->magic = ODL3_MSG_MAGIC; h->type = ODL3_MSG_DATA; h->round = s;
    if (test_drop_ && !resend && s % test_drop_ == 0) return 0;   /* simulated loss: the peer must NACK it */
    return send_msg(p, msg, ODL3_HDR + nbytes);
}

static int send_nack(struct odl3_peer *p, uint64_t r) {
    struct odl3_msg_hdr h = { ODL3_MSG_MAGIC, ODL3_MSG_NACK, r };
    __atomic_add_fetch(&st_nacks_, 1, __ATOMIC_RELAXED);
    return send_msg(p, &h, sizeof h);
}

struct thread_arg { struct odl3_peer *p; int pidx; };
static struct thread_arg targs_[ODL3_MAX_PEERS];

static void *progress_main(void *arg) {
    struct odl3_peer *p = ((struct thread_arg *)arg)->p;
    int pidx = ((struct thread_arg *)arg)->pidx;
    uint64_t posted = 0, rcvd = 0, ahead = 0, last_act = now_ns();
    uint64_t wait_since = 0, last_nack = 0;
    while (!progress_exit_) {
        uint64_t s = __atomic_load_n(&shr_->doorbell, __ATOMIC_ACQUIRE);
        if (s > posted) {
            while (posted < s) {
                posted++;
                if (post_round(p, posted, 0)) { shr_->thread_err = 1; break; }
            }
            last_act = now_ns();
        }
        /* receive whatever comes next into the slot of the next expected round */
        uint32_t slot = (uint32_t)((rcvd + 1) % ODL3_NSLOTS);
        char *dst = p->recv_buf + (uint64_t)slot * ODL3_SLOT;
        uint8_t src_id = 0;
        uint32_t actual = 0;
        int ret = odl_tb5_stream_recv(p->h, p->sid, dst, ODL3_SLOT, &src_id, &actual);
        if (ret == 0) {
            struct odl3_msg_hdr h;
            memcpy(&h, dst, sizeof h);
            last_act = now_ns();
            if (src_id != sid_ || actual < sizeof h || h.magic != ODL3_MSG_MAGIC) {
                fprintf(stderr, "odl_ar3: malformed message from rank %d (src %u, %u bytes)\n", p->rank, src_id, actual);
                shr_->thread_err = 1;
            } else if (h.type == ODL3_MSG_NACK) {
                /* resend if staged and still in its slot; a round not staged yet goes out when it
                 * is, and one older than the slot window the peer provably has already */
                if (h.round >= 1 && h.round <= posted && h.round + ODL3_NSLOTS > posted) {
                    __atomic_add_fetch(&st_resends_, 1, __ATOMIC_RELAXED);
                    if (post_round(p, h.round, 1)) shr_->thread_err = 1;
                }
            } else if (h.round == rcvd + 1) {
                __atomic_store_n(&shr_->recv_flag[pidx][slot * 8], rcvd + 1, __ATOMIC_RELEASE);
                rcvd++;
                wait_since = 0;
                if (ahead == rcvd + 1) {
                    uint32_t as = (uint32_t)(ahead % ODL3_NSLOTS);
                    __atomic_store_n(&shr_->recv_flag[pidx][as * 8], ahead, __ATOMIC_RELEASE);
                    rcvd = ahead;
                    ahead = 0;
                }
            } else if (h.round == rcvd + 2 && !ahead) {
                /* round rcvd + 1 was lost: keep this one in its own slot and ask for the missing one */
                uint32_t as = (uint32_t)(h.round % ODL3_NSLOTS);
                memcpy(p->recv_buf + (uint64_t)as * ODL3_SLOT, dst, actual);
                ahead = h.round;
                __atomic_add_fetch(&st_aheads_, 1, __ATOMIC_RELAXED);
                if (send_nack(p, rcvd + 1)) shr_->thread_err = 1;
                last_nack = now_ns();
            } else if (h.round <= rcvd || h.round == ahead) {
                __atomic_add_fetch(&st_dups_, 1, __ATOMIC_RELAXED);    /* a resend racing a late original */
            } else {
                fprintf(stderr, "odl_ar3: round %llu from rank %d while expecting %llu\n",
                        (unsigned long long)h.round, p->rank, (unsigned long long)(rcvd + 1));
                shr_->thread_err = 1;
            }
        } else if (ret != -EAGAIN && ret != -EINTR) {
            fprintf(stderr, "odl_ar3: stream_recv from rank %d failed: %d\n", p->rank, ret);
            shr_->thread_err = 1;
        }
        /* timeout NACK: our GPU needs round rcvd + 1 (we staged it) and it has not come */
        uint64_t now = now_ns();
        if (s > rcvd) {
            if (!wait_since) wait_since = now;
            uint64_t since = last_nack > wait_since ? last_nack : wait_since;
            if (now - since > nack_ns_) {
                if (send_nack(p, rcvd + 1)) shr_->thread_err = 1;
                last_nack = now;
            }
        } else {
            wait_since = 0;
        }
        if (now - last_act > hot_ns_ && s <= rcvd) usleep(200);
    }
    return NULL;
}

/* ---- GPU kernels ---- */
struct odl3_srcs { const void *p[ODL3_MAX_RANKS]; };   /* per rank: receive slot, or NULL = own src */

/* stage: own contribution (wire dtype) into the send slot, then ring the doorbell (one block) */
template <int DT>
__global__ void odl3_stage_kernel(char *__restrict__ slot_buf, const void *__restrict__ src, uint32_t n,
                                  struct odl3_shared *sh, uint32_t slot, uint64_t seq) {
    if (DT == 3) {
        __hip_bfloat16 *o = (__hip_bfloat16 *)slot_buf;
        const float *s = (const float *)src;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) o[i] = __float2bfloat16(s[i]);
    } else if (DT == 2) {
        float *o = (float *)slot_buf;
        const float *s = (const float *)src;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) o[i] = s[i];
    } else {
        uint16_t *o = (uint16_t *)slot_buf;
        const uint16_t *s = (const uint16_t *)src;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) o[i] = s[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t wire = n * (DT == 2 ? 4u : 2u);
        __hip_atomic_store(&sh->slot_nbytes[slot], wire, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&sh->doorbell, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__device__ __forceinline__ bool odl3_wait(uint64_t *flag, uint64_t seq, int *err) {
    long iters = 0;
    for (;;) {
        uint64_t v = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (v >= seq) return true;
        if (++iters > (2L << 30)) { *err = 1; return false; }
        __builtin_amdgcn_s_sleep(2);
    }
}

template <int DT> __device__ __forceinline__ float odl3_load(const void *p, uint32_t i) {
    if (DT == 0 || DT == 3) return __bfloat162float(((const __hip_bfloat16 *)p)[i]);
    if (DT == 1) return __half2float(((const __half *)p)[i]);
    return ((const float *)p)[i];
}

/* own contribution, exactly as the peers see it on the wire */
template <int DT> __device__ __forceinline__ float odl3_own(const void *src, uint32_t i) {
    if (DT == 3) return __bfloat162float(__float2bfloat16(((const float *)src)[i]));
    return odl3_load<DT>(src, i);
}

/* thread 0 waits for every peer's message of this round, then all threads sum the N contributions
 * in canonical rank order into dst (in place allowed: each element is read before it is written) */
template <int DT>
__global__ void odl3_wait_sum_kernel(void *dst, const void *src, struct odl3_srcs rs, int world,
                                     uint32_t n, uint64_t *flags, int npeers, uint64_t seq, int *err) {
    if (threadIdx.x == 0) {
        for (int p = 0; p < npeers; p++)
            if (!odl3_wait(flags + p * (ODL3_NSLOTS * 8), seq, err)) break;
    }
    __syncthreads();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (int r = 0; r < world; r++)
            acc += rs.p[r] ? odl3_load<DT>(rs.p[r], i) : odl3_own<DT>(src, i);
        if (DT == 3)      ((float *)dst)[i] = __bfloat162float(__float2bfloat16(acc));
        else if (DT == 0) ((__hip_bfloat16 *)dst)[i] = __float2bfloat16(acc);
        else if (DT == 1) ((__half *)dst)[i] = __float2half(acc);
        else              ((float *)dst)[i] = acc;
    }
}

/* ======================================================================================
 * Big (prefill-sized) path: mesh reduce-scatter + all-gather, fp32 in/out over a bf16 wire
 *
 * The tensor is cut into `world` contiguous shards; rank r owns shard r. Reduce-scatter: every
 * rank sends shard p of its bf16-staged tensor to rank p over the cable to p and sums the
 * world contributions of its own shard in canonical rank order. All-gather: every rank sends
 * its reduced shard to every peer. Per cable and direction that is 2/world of the tensor,
 * against 2(world-1)/world for RCCL's ring on one direction of each cable, and both directions
 * of all cables carry traffic at once. Stream-ordered like the small path:
 *   [stage fp32->bf16] [door rs] [wait rs + sum own shard -> bag + dst] [door ag] [wait ag -> dst]
 * One sender and one receiver thread per peer (blocking handle, stream ODL3_BIG_SID, default
 * 244); the receiver learns each op's size from a header message.
 *
 * Buffer reuse (single-buffered except the reduced shard, double-buffered by op parity): our
 * stage(k+1) follows our final(k), which needs every peer's all-gather(k), which each peer
 * sends after its reduce(k), which needs our reduce-scatter(k) -- so our senders are done
 * with the staged tensor. A peer's reduce-scatter(k+1) into brs follows its final(k), which
 * needs our all-gather(k), sent after our reduce(k) finished reading brs. A peer's
 * all-gather(k+1) into bag follows its reduce(k+1), which needs our reduce-scatter(k+1),
 * sent after our stage(k+1), which follows our final(k) (the last reader of bag). Our reduced
 * shard of op k+2 overwrites bown[k & 1] after our final(k+1), which needs the peers'
 * all-gather(k+1), which each peer sends only after its final(k) -- after receiving our
 * all-gather(k) completely.
 * ====================================================================================== */
#define ODL3_BIG_MSG (1u << 20)
#define ODL3_BIG_MAGIC 0x4f444c42u                /* "ODLB" */
struct odl3_big_hdr { uint32_t magic, phase; uint64_t op, n; };

struct odl3_big_shared {
    uint64_t door_rs;                          /* last op whose bf16 tensor is staged */
    uint64_t door_ag;                          /* last op whose reduced own shard is in bown */
    uint64_t n[2];                             /* elements of op, by parity */
    uint64_t pad[4];
    uint64_t rs_flag[ODL3_MAX_PEERS * 8];      /* [pidx * 8]: last op whose shard from pidx landed */
    uint64_t ag_flag[ODL3_MAX_PEERS * 8];
};

static int big_req_, big_on_;
static uint64_t big_max_n_ = 2048ull * 5120ull;   /* ODL3_BIG_MAX_ELEMS */
static uint64_t big_shard_max_;
static char *bsend_, *bsend_dev_;              /* the staged bf16 tensor */
static char *bown_[2], *bown_dev_[2];          /* our reduced shard, by op parity */
static struct odl3_big_shared *bsh_, *bsh_dev_;
static uint64_t big_seq_;
static uint8_t big_sid_ = 244;

__host__ __device__ static inline void odl3_shard(uint64_t n, int world, int r, uint64_t *off, uint64_t *len) {
    uint64_t per = ((n + (uint64_t)world - 1) / (uint64_t)world + 63ull) & ~63ull;
    uint64_t o = per * (uint64_t)r, e;
    if (o > n) o = n;
    e = o + per;
    if (e > n) e = n;
    *off = o; *len = e - o;
}

static int big_send(struct odl3_peer *p, const void *buf, uint64_t len) {
    const char *c = (const char *)buf;
    while (len) {
        uint32_t m = len > ODL3_BIG_MSG ? ODL3_BIG_MSG : (uint32_t)len;
        int ret = odl_tb5_stream_send(p->bh, p->bsid, big_sid_, c, m);
        if (ret == -EINTR || ret == -EAGAIN) continue;      /* receiver resets at the next MSG_START */
        if (ret < 0) { fprintf(stderr, "odl_ar3: big send to rank %d failed: %d\n", p->rank, ret); return -1; }
        c += m; len -= m;
    }
    return 0;
}

static int big_recv(struct odl3_peer *p, void *buf, uint64_t len) {
    char *c = (char *)buf;
    while (len) {
        uint8_t src = 0; uint32_t got = 0;
        uint32_t want = len > ODL3_BIG_MSG ? ODL3_BIG_MSG : (uint32_t)len;
        int ret = odl_tb5_stream_recv(p->bh, p->bsid, c, want, &src, &got);
        if (ret == -EINTR || ret == -EAGAIN) continue;      /* nothing dequeued */
        if (ret < 0) return ret;
        if (got == 0 || got > want) { fprintf(stderr, "odl_ar3: big recv from rank %d: bad length %u\n", p->rank, got); return -1; }
        c += got; len -= got;
    }
    return 0;
}

struct big_arg { struct odl3_peer *p; int pidx; };
static struct big_arg bargs_[ODL3_MAX_PEERS];

static void *big_send_main(void *arg) {
    struct odl3_peer *p = ((struct big_arg *)arg)->p;
    uint64_t done = 0;
    while (!progress_exit_) {
        if (__atomic_load_n(&bsh_->door_rs, __ATOMIC_ACQUIRE) <= done) { usleep(20); continue; }
        uint64_t op = done + 1;
        uint64_t n = __atomic_load_n(&bsh_->n[op & 1], __ATOMIC_ACQUIRE);
        uint64_t off, len, moff, mlen;
        odl3_shard(n, world_, p->rank, &off, &len);
        odl3_shard(n, world_, my_rank_, &moff, &mlen);
        struct odl3_big_hdr h = { ODL3_BIG_MAGIC, 0, op, n };
        if (big_send(p, &h, sizeof h) || (len && big_send(p, bsend_ + off * 2, len * 2))) { shr_->thread_err = 1; break; }
        while (!progress_exit_ && __atomic_load_n(&bsh_->door_ag, __ATOMIC_ACQUIRE) < op) usleep(10);
        if (progress_exit_) break;
        h.phase = 1;
        if (big_send(p, &h, sizeof h) || (mlen && big_send(p, bown_[op & 1], mlen * 2))) { shr_->thread_err = 1; break; }
        done = op;
    }
    return NULL;
}

static void *big_recv_main(void *arg) {
    struct odl3_peer *p = ((struct big_arg *)arg)->p;
    int pidx = ((struct big_arg *)arg)->pidx;
    uint64_t expect = 1;
    while (!progress_exit_) {
        for (uint32_t phase = 0; phase < 2; phase++) {
            struct odl3_big_hdr h;
            int ret = big_recv(p, &h, sizeof h);
            if (ret < 0) { if (!progress_exit_) { fprintf(stderr, "odl_ar3: big recv from rank %d failed: %d\n", p->rank, ret); shr_->thread_err = 1; } return NULL; }
            if (h.magic != ODL3_BIG_MAGIC || h.phase != phase || h.op != expect || h.n > big_max_n_) {
                fprintf(stderr, "odl_ar3: big header from rank %d out of sequence (op %llu/%llu phase %u)\n",
                        p->rank, (unsigned long long)h.op, (unsigned long long)expect, h.phase);
                shr_->thread_err = 1; return NULL;
            }
            uint64_t off, len;
            odl3_shard(h.n, world_, phase == 0 ? my_rank_ : p->rank, &off, &len);
            if (len && big_recv(p, phase == 0 ? p->brs : p->bag, len * 2) < 0) {
                if (!progress_exit_) { fprintf(stderr, "odl_ar3: big payload from rank %d failed\n", p->rank); shr_->thread_err = 1; }
                return NULL;
            }
            __atomic_store_n(phase == 0 ? &bsh_->rs_flag[pidx * 8] : &bsh_->ag_flag[pidx * 8], expect, __ATOMIC_RELEASE);
        }
        expect++;
    }
    return NULL;
}

__global__ void odl3_big_stage_kernel(__hip_bfloat16 *__restrict__ out, const float *__restrict__ src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        out[i] = __float2bfloat16(src[i]);
    __threadfence_system();
}

__global__ void odl3_big_door_kernel(uint64_t *door, uint64_t *nslot, uint64_t n, uint64_t op) {
    __threadfence_system();
    if (nslot) __hip_atomic_store(nslot, n, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(door, op, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* own shard: canonical-order sum of every rank's bf16 contribution -> bf16 bown + fp32 dst */
__global__ void odl3_big_reduce_kernel(float *dst, const float *src, __hip_bfloat16 *bown, struct odl3_srcs rs,
                                       int world, uint64_t off, uint64_t len, uint64_t *rs_flags, int npeers,
                                       uint64_t op, int *err) {
    if (threadIdx.x == 0)
        for (int p = 0; p < npeers; p++)
            if (!odl3_wait(rs_flags + p * 8, op, err)) break;
    __syncthreads();
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < len; i += (uint64_t)gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int r = 0; r < world; r++)
            acc += rs.p[r] ? __bfloat162float(((const __hip_bfloat16 *)rs.p[r])[i])
                           : __bfloat162float(__float2bfloat16(src[off + i]));
        __hip_bfloat16 v = __float2bfloat16(acc);
        bown[i] = v;
        dst[off + i] = __bfloat162float(v);
    }
    __threadfence_system();
}

struct odl3_big_ag { const __hip_bfloat16 *p[ODL3_MAX_PEERS]; uint64_t off[ODL3_MAX_PEERS], len[ODL3_MAX_PEERS]; };

__global__ void odl3_big_final_kernel(float *dst, struct odl3_big_ag ag, int npeers, uint64_t *ag_flags, uint64_t op, int *err) {
    if (threadIdx.x == 0)
        for (int p = 0; p < npeers; p++)
            if (!odl3_wait(ag_flags + p * 8, op, err)) break;
    __syncthreads();
    for (int p = 0; p < npeers; p++)
        for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < ag.len[p]; i += (uint64_t)gridDim.x * blockDim.x)
            dst[ag.off[p] + i] = __bfloat162float(ag.p[p][i]);
}

static int odl3_big_open_dev(struct odl3_peer *p) {
    if (odl_tb5_open(&p->bh, p->dev) < 0) { fprintf(stderr, "odl_ar3: big open dev %d failed\n", p->dev); return -1; }
    uint8_t got = 0;
    int ret = odl_tb5_stream_open(p->bh, big_sid_, &got);
    if (ret < 0 || got != big_sid_) { fprintf(stderr, "odl_ar3: dev %d big stream_open(%u) failed: %d\n", p->dev, big_sid_, ret); return -1; }
    p->bsid = got;
    if (hipHostMalloc((void **)&p->brs, big_shard_max_ * 2, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc((void **)&p->bag, big_shard_max_ * 2, hipHostMallocDefault) != hipSuccess ||
        hipHostGetDevicePointer((void **)&p->brs_dev, p->brs, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **)&p->bag_dev, p->bag, 0) != hipSuccess)
        { fprintf(stderr, "odl_ar3: big pinned receive buffers failed\n"); return -1; }
    return 0;
}

static int odl3_big_alloc(void) {
    uint64_t off, len;
    odl3_shard(big_max_n_, world_, 0, &off, &len);
    big_shard_max_ = len;
    if (hipHostMalloc((void **)&bsend_, big_max_n_ * 2, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc((void **)&bown_[0], big_shard_max_ * 2, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc((void **)&bown_[1], big_shard_max_ * 2, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc((void **)&bsh_, sizeof *bsh_, hipHostMallocDefault) != hipSuccess ||
        hipHostGetDevicePointer((void **)&bsend_dev_, bsend_, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **)&bown_dev_[0], bown_[0], 0) != hipSuccess ||
        hipHostGetDevicePointer((void **)&bown_dev_[1], bown_[1], 0) != hipSuccess ||
        hipHostGetDevicePointer((void **)&bsh_dev_, (void *)bsh_, 0) != hipSuccess)
        { fprintf(stderr, "odl_ar3: big pinned buffers failed\n"); return -1; }
    memset(bsh_, 0, sizeof *bsh_);
    return 0;
}

extern "C" {
int odl3_big_ready(void) { return big_on_; }
uint64_t odl3_big_max_elems(void) { return big_on_ ? big_max_n_ : 0; }
}

/* ---- public API ---- */

/* Open every /dev/odl_tb5_N whose link comes up within wait_ms, open stream ODL3_SID on it (nonblocking
 * fd) and allocate the pinned slots. The caller then runs a cross-rank barrier, odl3_handshake(), a
 * second barrier and odl3_start(). Returns the number of usable cables, or < 0 on error. */
extern "C" int odl3_open(int rank, int world, int wait_ms) {
    if (opened_) return ndevs_;
    if (world < 2 || world > ODL3_MAX_RANKS || rank < 0 || rank >= world) {
        fprintf(stderr, "odl_ar3: bad rank %d / world %d\n", rank, world); return -1;
    }
    const char *e;
    if ((e = getenv("ODL3_SID"))) sid_ = (uint8_t)atoi(e);
    if ((e = getenv("ODL3_HOT_US"))) hot_ns_ = (uint64_t)atoll(e) * 1000ull;
    if ((e = getenv("ODL3_NACK_US")) && atoll(e) > 0) nack_ns_ = (uint64_t)atoll(e) * 1000ull;
    if ((e = getenv("ODL3_TEST_DROP")) && atoll(e) > 0) test_drop_ = (uint64_t)atoll(e);
    if (test_drop_ && (e = getenv("ODL3_TEST_DROP_RANK")) && atoi(e) != rank) test_drop_ = 0;   /* drop on one rank only */
    if ((e = getenv("ODL3_BIG"))) big_req_ = atoi(e) != 0;
    if ((e = getenv("ODL3_BIG_SID"))) big_sid_ = (uint8_t)atoi(e);
    if ((e = getenv("ODL3_BIG_MAX_ELEMS")) && atoll(e) > 0) big_max_n_ = (uint64_t)atoll(e);
    my_rank_ = rank; world_ = world; ndevs_ = 0;

    /* candidate devices: ODL3_DEVS="0,1" or every /dev/odl_tb5_N */
    int cand[ODL3_MAX_DEVS], ncand = 0;
    if ((e = getenv("ODL3_DEVS")) && *e) {
        char buf[64]; strncpy(buf, e, sizeof buf - 1); buf[sizeof buf - 1] = 0;
        for (char *t = strtok(buf, ","); t && ncand < ODL3_MAX_DEVS; t = strtok(NULL, ",")) cand[ncand++] = atoi(t);
    } else {
        for (int i = 0; i < ODL3_MAX_DEVS; i++) {
            char path[64]; snprintf(path, sizeof path, "/dev/odl_tb5_%d", i);
            if (access(path, F_OK) == 0) cand[ncand++] = i;
        }
    }
    for (int c = 0; c < ncand; c++) {
        struct odl3_peer *p = &devs_[ndevs_];
        memset(p, 0, sizeof *p);
        p->rank = -1; p->dev = cand[c];
        if (odl_tb5_open(&p->h, p->dev) < 0) { fprintf(stderr, "odl_ar3: open dev %d failed\n", p->dev); continue; }
        if (odl_tb5_wait_peer(p->h, wait_ms) < 0) { fprintf(stderr, "odl_ar3: dev %d link not ready, skipped\n", p->dev); odl_tb5_close(p->h); continue; }
        uint8_t got = 0;
        int ret = odl_tb5_stream_open(p->h, sid_, &got);
        if (ret < 0 || got != sid_) { fprintf(stderr, "odl_ar3: dev %d stream_open(%u) failed: %d\n", p->dev, sid_, ret); odl_tb5_close(p->h); return -1; }
        p->sid = got;
        int fd = odl_tb5_get_fd(p->h);
        if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK) < 0) { perror("odl_ar3: O_NONBLOCK"); return -1; }
        if (hipHostMalloc((void **)&p->recv_buf, ODL3_NSLOTS * ODL3_SLOT, hipHostMallocDefault) != hipSuccess ||
            hipHostGetDevicePointer((void **)&p->recv_buf_dev, p->recv_buf, 0) != hipSuccess)
            { fprintf(stderr, "odl_ar3: pinned receive slots failed\n"); return -1; }
        memset(p->recv_buf, 0, ODL3_NSLOTS * ODL3_SLOT);
        ndevs_++;
    }
    if (ndevs_ < world - 1) { fprintf(stderr, "odl_ar3: %d usable cable(s), need %d\n", ndevs_, world - 1); return -1; }
    if (hipHostMalloc((void **)&send_buf_, ODL3_NSLOTS * ODL3_SLOT, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc((void **)&shr_, sizeof *shr_, hipHostMallocDefault) != hipSuccess ||
        hipHostGetDevicePointer((void **)&send_buf_dev_, send_buf_, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **)&shr_dev_, (void *)shr_, 0) != hipSuccess)
        { fprintf(stderr, "odl_ar3: pinned send slots failed\n"); return -1; }
    memset(send_buf_, 0, ODL3_NSLOTS * ODL3_SLOT);
    memset(shr_, 0, sizeof *shr_);
    /* big path: its streams must be open on every rank before barrier 1, like the small ones */
    if (big_req_) {
        int ok = odl3_big_alloc() == 0;
        for (int i = 0; ok && i < ndevs_; i++) ok = odl3_big_open_dev(&devs_[i]) == 0;
        big_on_ = ok;
        if (!ok) fprintf(stderr, "odl_ar3: big path unavailable on rank %d\n", rank);
    }
    opened_ = 1;
    return ndevs_;
}

/* Ping every cable with our rank and read each peer's ping back (call between two cross-rank
 * barriers). Builds the rank -> cable map. Returns 0 when world - 1 distinct peers were found. */
extern "C" int odl3_handshake(int timeout_ms) {
    if (!opened_) return -1;
    struct odl3_ping mine = { ODL3_PING_MAGIC, (uint32_t)my_rank_, (uint32_t)world_, ODL3_MAX_BYTES, ODL3_NSLOTS, 0 };
    for (int i = 0; i < ndevs_; i++)
        if (odl_tb5_stream_send(devs_[i].h, devs_[i].sid, sid_, &mine, sizeof mine) != 0)
            fprintf(stderr, "odl_ar3: ping send on dev %d failed\n", devs_[i].dev);
    int found = 0;
    uint64_t t0 = now_ns();
    while (found < world_ - 1 && now_ns() - t0 < (uint64_t)timeout_ms * 1000000ull) {
        for (int i = 0; i < ndevs_; i++) {
            if (devs_[i].rank >= 0) continue;
            struct odl3_ping theirs; uint8_t sidr = 0; uint32_t alen = 0;
            int r = odl_tb5_stream_recv(devs_[i].h, devs_[i].sid, &theirs, sizeof theirs, &sidr, &alen);
            if (r == -EAGAIN || r == -EINTR) continue;
            if (r < 0) { fprintf(stderr, "odl_ar3: ping recv on dev %d failed: %d\n", devs_[i].dev, r); return -1; }
            if (alen != sizeof theirs || sidr != sid_ || theirs.magic != ODL3_PING_MAGIC ||
                theirs.world != (uint32_t)world_ || theirs.max_bytes != ODL3_MAX_BYTES ||
                theirs.nslots != ODL3_NSLOTS || theirs.rank >= (uint32_t)world_ || theirs.rank == (uint32_t)my_rank_) {
                fprintf(stderr, "odl_ar3: bad ping on dev %d (len %u rank %u world %u)\n",
                        devs_[i].dev, alen, theirs.rank, theirs.world);
                return -1;
            }
            if (peer_of_rank_[theirs.rank]) { fprintf(stderr, "odl_ar3: rank %u reachable on two cables\n", theirs.rank); return -1; }
            devs_[i].rank = (int)theirs.rank;
            peer_of_rank_[theirs.rank] = &devs_[i];
            found++;
        }
        if (found < world_ - 1) usleep(100);
    }
    if (found < world_ - 1) { fprintf(stderr, "odl_ar3: handshake found %d of %d peers\n", found, world_ - 1); return -1; }
    return 0;
}

extern "C" int odl3_start(void) {
    if (!opened_) return -1;
    if (inited_) return 0;
    host_seq_ = 0;
    progress_exit_ = 0;
    npeers_ = 0;
    for (int r = 0; r < world_; r++) {
        if (r == my_rank_) continue;
        struct odl3_peer *p = peer_of_rank_[r];
        if (!p) { fprintf(stderr, "odl_ar3: no cable to rank %d\n", r); return -1; }
        targs_[npeers_].p = p;
        targs_[npeers_].pidx = npeers_;
        if (pthread_create(&p->tid, NULL, progress_main, &targs_[npeers_])) { fprintf(stderr, "odl_ar3: progress thread failed\n"); return -1; }
        p->started = 1;
        npeers_++;
    }
    if (big_on_) {
        for (int i = 0; i < npeers_; i++) {
            struct odl3_peer *p = targs_[i].p;
            bargs_[i].p = p;
            bargs_[i].pidx = i;
            if (pthread_create(&p->stid, NULL, big_send_main, &bargs_[i]) ||
                pthread_create(&p->rtid, NULL, big_recv_main, &bargs_[i]))
                { fprintf(stderr, "odl_ar3: big threads failed\n"); return -1; }
            p->bstarted = 1;
        }
    }
    inited_ = 1;
    fprintf(stderr, "odl_ar3: rank%d ready, world %d, sid %u, hot %llu us, big %s; peers:", my_rank_, world_, sid_,
            (unsigned long long)(hot_ns_ / 1000), big_on_ ? "on" : "off");
    for (int i = 0; i < npeers_; i++) fprintf(stderr, " rank%d@dev%d", targs_[i].p->rank, targs_[i].p->dev);
    fprintf(stderr, "\n");
    return 0;
}

/* Enqueue one all-reduce on `stream`: dst = sum over ranks of src, stream-ordered, in place allowed.
 * dtype: 0 bf16, 1 fp16, 2 fp32, 3 fp32 over a bf16 wire. nbytes is the size of src. */
extern "C" int odl3_enqueue(void *stream, void *dst, const void *src, uint32_t nbytes, int dtype) {
    if (!inited_) return -1;
    uint32_t elt = (dtype == 2 || dtype == 3) ? 4u : 2u;
    uint32_t n = nbytes / elt;
    uint32_t wire = n * (dtype == 2 ? 4u : 2u);
    if (wire == 0 || wire > ODL3_MAX_BYTES) return -1;
    host_seq_++;
    uint64_t s = host_seq_;
    uint32_t slot = (uint32_t)(s % ODL3_NSLOTS);
    hipStream_t hs = (hipStream_t)stream;
    char *send_slot = send_buf_dev_ + (uint64_t)slot * ODL3_SLOT + ODL3_HDR;
    struct odl3_srcs rs;
    for (int r = 0; r < ODL3_MAX_RANKS; r++) rs.p[r] = NULL;
    for (int i = 0; i < npeers_; i++)
        rs.p[targs_[i].p->rank] = targs_[i].p->recv_buf_dev + (uint64_t)slot * ODL3_SLOT + ODL3_HDR;
    uint64_t *flags = &shr_dev_->recv_flag[0][slot * 8];
    int *err = &shr_dev_->kernel_err;
    uint32_t blocks = (n + 8191) / 8192;
    if (blocks < 1) blocks = 1;
    if (blocks > 8) blocks = 8;
    switch (dtype) {
    case 0:
        hipLaunchKernelGGL(odl3_stage_kernel<0>, dim3(1), dim3(1024), 0, hs, send_slot, src, n, shr_dev_, slot, s);
        hipLaunchKernelGGL(odl3_wait_sum_kernel<0>, dim3(blocks), dim3(1024), 0, hs, dst, src, rs, world_, n, flags, npeers_, s, err);
        break;
    case 1:
        hipLaunchKernelGGL(odl3_stage_kernel<1>, dim3(1), dim3(1024), 0, hs, send_slot, src, n, shr_dev_, slot, s);
        hipLaunchKernelGGL(odl3_wait_sum_kernel<1>, dim3(blocks), dim3(1024), 0, hs, dst, src, rs, world_, n, flags, npeers_, s, err);
        break;
    case 2:
        hipLaunchKernelGGL(odl3_stage_kernel<2>, dim3(1), dim3(1024), 0, hs, send_slot, src, n, shr_dev_, slot, s);
        hipLaunchKernelGGL(odl3_wait_sum_kernel<2>, dim3(blocks), dim3(1024), 0, hs, dst, src, rs, world_, n, flags, npeers_, s, err);
        break;
    case 3:
        hipLaunchKernelGGL(odl3_stage_kernel<3>, dim3(1), dim3(1024), 0, hs, send_slot, src, n, shr_dev_, slot, s);
        hipLaunchKernelGGL(odl3_wait_sum_kernel<3>, dim3(blocks), dim3(1024), 0, hs, dst, src, rs, world_, n, flags, npeers_, s, err);
        break;
    default:
        return -1;
    }
    return (hipGetLastError() == hipSuccess) ? 0 : -1;
}

/* Enqueue one big all-reduce (fp32 in/out over a bf16 wire, dtype 3 only) on `stream`: mesh
 * reduce-scatter + all-gather, stream-ordered, in place allowed. nbytes is the size of src. */
extern "C" int odl3_enqueue_big(void *stream, void *dst, const void *src, uint64_t nbytes, int dtype) {
    if (!inited_ || !big_on_ || dtype != 3) return -1;
    uint64_t n = nbytes / 4;
    if (n == 0 || n > big_max_n_) return -1;
    big_seq_++;
    uint64_t op = big_seq_;
    hipStream_t hs = (hipStream_t)stream;
    uint64_t off, len;
    odl3_shard(n, world_, my_rank_, &off, &len);
    struct odl3_srcs rs;
    struct odl3_big_ag ag;
    for (int r = 0; r < ODL3_MAX_RANKS; r++) rs.p[r] = NULL;
    for (int i = 0; i < npeers_; i++) {
        struct odl3_peer *p = targs_[i].p;
        rs.p[p->rank] = p->brs_dev;
        ag.p[i] = (const __hip_bfloat16 *)p->bag_dev;
        odl3_shard(n, world_, p->rank, &ag.off[i], &ag.len[i]);
    }
    int *err = &shr_dev_->kernel_err;
    uint32_t gs = (uint32_t)((n + 1023) / 1024); if (gs > 240) gs = 240;
    uint32_t gr = (uint32_t)((len + 1023) / 1024); if (gr < 1) gr = 1; if (gr > 120) gr = 120;
    hipLaunchKernelGGL(odl3_big_stage_kernel, dim3(gs), dim3(1024), 0, hs, (__hip_bfloat16 *)bsend_dev_, (const float *)src, n);
    hipLaunchKernelGGL(odl3_big_door_kernel, dim3(1), dim3(1), 0, hs, &bsh_dev_->door_rs, &bsh_dev_->n[op & 1], n, op);
    hipLaunchKernelGGL(odl3_big_reduce_kernel, dim3(gr), dim3(1024), 0, hs, (float *)dst, (const float *)src,
                       (__hip_bfloat16 *)bown_dev_[op & 1], rs, world_, off, len, bsh_dev_->rs_flag, npeers_, op, err);
    hipLaunchKernelGGL(odl3_big_door_kernel, dim3(1), dim3(1), 0, hs, &bsh_dev_->door_ag, (uint64_t *)NULL, (uint64_t)0, op);
    hipLaunchKernelGGL(odl3_big_final_kernel, dim3(gr), dim3(1024), 0, hs, (float *)dst, ag, npeers_, bsh_dev_->ag_flag, op, err);
    return (hipGetLastError() == hipSuccess) ? 0 : -1;
}

/* Stop the progress threads and close the cables; call once all enqueued work has completed. */
extern "C" void odl3_shutdown(void) {
    if (inited_ && (st_nacks_ | st_resends_ | st_dups_ | st_aheads_))
        fprintf(stderr, "odl_ar3: rank%d recovered losses: %llu NACKs sent, %llu rounds resent, "
                "%llu duplicates dropped, %llu early rounds held\n", my_rank_,
                (unsigned long long)st_nacks_, (unsigned long long)st_resends_,
                (unsigned long long)st_dups_, (unsigned long long)st_aheads_);
    if (inited_) {
        progress_exit_ = 1;
        for (int i = 0; i < npeers_; i++)
            if (targs_[i].p->started) { pthread_join(targs_[i].p->tid, NULL); targs_[i].p->started = 0; }
        for (int i = 0; i < npeers_; i++) {
            struct odl3_peer *p = targs_[i].p;
            if (!p->bstarted) continue;
            /* the receiver blocks in stream_recv: closing the stream wakes it with -ESHUTDOWN */
            odl_tb5_stream_close(p->bh, p->bsid);
            p->bsid = 0;
            pthread_join(p->stid, NULL);
            pthread_join(p->rtid, NULL);
            p->bstarted = 0;
        }
        inited_ = 0;
    }
    if (opened_) {
        for (int i = 0; i < ndevs_; i++) {
            odl_tb5_stream_close(devs_[i].h, devs_[i].sid);
            odl_tb5_close(devs_[i].h);
            if (devs_[i].bh) {
                if (devs_[i].bsid) odl_tb5_stream_close(devs_[i].bh, devs_[i].bsid);
                odl_tb5_close(devs_[i].bh);
                devs_[i].bh = NULL;
            }
        }
        for (int r = 0; r < ODL3_MAX_RANKS; r++) peer_of_rank_[r] = NULL;
        ndevs_ = 0;
        opened_ = 0;
        big_on_ = 0;
    }
}
