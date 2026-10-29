// odl_ar2.hip -- 2-rank SUM all-reduce over the OdinLink (odl_tb5) stream API, for the decode-sized
// collectives of the exllamav3 cross-host TP backend (model_tp_odl.py). Adapted from
// vllm-strix-halo/odinlink/ar2/odl_ar2.hip, which carries the same decode all-reduces for the vLLM
// stack on these boxes.
//
//   stream:  [stage src -> pinned send slot] -> [doorbell] -> [wait + add into dst]
//   CPU:     progress thread polls the doorbell, moves the bytes over OdinLink, publishes the
//            peer's message with a release store to the slot flag the GPU kernel spins on
//
// Fully stream-ordered: nothing here blocks the host, so the caller needs no device sync after an
// all-reduce (consumers on the same stream are ordered behind the wait+add kernel).
//
// Transport: OdinLink streams in latency mode. One ODL_TB5_IOCTL_STREAM_SEND per round; messages
// up to 64 KiB are fragmented at 4024 bytes inside the ioctl and submitted without waiting for TX
// completion. Receive is a nonblocking STREAM_RECV polled by the progress thread; the ioctl copies
// the reassembled message straight into the GPU-visible pinned receive slot (UMA zero-copy).
// Streams deliver complete messages in order, so message k on the stream is round k.
//
// Slot reuse: the peer only sends round N + 2 after its wait+add(N + 1) completed, which needs our
// N + 1, which (stream order) needs our wait+add(N) to have finished reading receive slot N, so an
// arriving message never overwrites a slot the GPU still reads.
//
// Changes against the vLLM original:
//  * the byte count of each round lives in a per-slot array written by the doorbell kernel ahead
//    of its release store. The original packed it into the doorbell word, so a progress thread
//    that saw two doorbells at once replayed both rounds with the LATER size (under-sending the
//    first when the sizes differ). At most two rounds can be pending (the flow control above), so
//    two slots are enough to keep both sizes
//  * dtype 3: fp32 in / fp32 out over a bf16 wire, bit-identical to the RCCL path it replaces
//    (fp32 -> bf16, bf16 all-reduce with fp32 accumulation, bf16 -> fp32): the stage kernel
//    converts into the send slot and rings the doorbell itself, the wait+add kernel converts back
//  * configurable stream ids (ODL2_SID0 / ODL2_SID1, default 241 / 242, clear of the RCCL net
//    plugin's auto-assigned low ids) and progress-thread hot-spin window (ODL2_HOT_US, default
//    50 ms, so the first all-reduce of a step does not pay a backoff sleep after sampling)
//
// Build: odl_ar2/build.sh (hipcc from the project venv, -lodl_tb5 from ~/.cache/odinlink).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <odl_tb5/odl_tb5.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#define ODL2_MAX_BYTES (1u << 20)
#define ODL2_NSLOTS 2 /* send and receive are double-buffered */
#define ODL2_PING_MAGIC 0x4f444c33u /* "ODL3" */

static odl_tb5_t H;
static uint8_t my_sid, peer_sid;
static char *send_buf, *recv_buf;          /* host pointers (pinned)   */
static char *send_buf_dev, *recv_buf_dev;  /* device views of the same */
static volatile uint64_t *recv_flag;       /* host view, after the receive slots */
static uint64_t *recv_flag_dev;            /* device view                        */
static uint64_t host_seq;                  /* caller-side round counter          */
static int inited;
static uint64_t hot_ns = 50000000ull;

struct odl2_shared {
    uint64_t doorbell;                 /* last staged round, GPU release-store */
    uint32_t slot_nbytes[ODL2_NSLOTS]; /* bytes of the round staged in each slot */
    int kernel_err;                    /* wait+add timeout latch (GPU writes)  */
    int thread_err;                    /* progress-thread error latch          */
};
static struct odl2_shared *shr;       /* host view   */
static struct odl2_shared *shr_dev;   /* device view */

static pthread_t progress_tid;
static volatile int progress_exit;

extern "C" {
uint32_t odl2_max_bytes(void) { return ODL2_MAX_BYTES; }
int odl2_last_error(void) {
    int e = shr ? (shr->kernel_err | shr->thread_err) : 0;
    if (shr) { shr->kernel_err = 0; shr->thread_err = 0; }
    return e;
}
uint64_t odl2_seq(void) { return host_seq; }
}

/* ---- TCP rendezvous / barrier (rank 1 listens, rank 0 connects) ---- */
struct xchg { uint32_t magic; uint32_t rank; uint32_t max_bytes; uint32_t nslots; };
static int rdv_fd = -1;

static int tcp_connect_rdv(int rank, const char *peer_ip, int port) {
    int fd = -1;
    if (rank == 1) {
        int lfd = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = {};
        a.sin_family = AF_INET; a.sin_port = htons(port);
        a.sin_addr.s_addr = INADDR_ANY;
        if (bind(lfd, (struct sockaddr*)&a, sizeof a) || listen(lfd, 1)) { perror("odl_ar2: bind/listen"); close(lfd); return -1; }
        fd = accept(lfd, NULL, NULL);
        close(lfd);
        if (fd < 0) { perror("odl_ar2: accept"); return -1; }
    } else {
        for (int tries = 0; tries < 1200; tries++) {
            fd = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in a = {};
            a.sin_family = AF_INET; a.sin_port = htons(port);
            inet_pton(AF_INET, peer_ip, &a.sin_addr);
            if (connect(fd, (struct sockaddr*)&a, sizeof a) == 0) break;
            close(fd); fd = -1;
            usleep(100000);
        }
        if (fd < 0) { fprintf(stderr, "odl_ar2: connect to %s:%d failed\n", peer_ip, port); return -1; }
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    rdv_fd = fd;
    return 0;
}

static int tcp_barrier(struct xchg *mine, struct xchg *theirs) {
    if (write(rdv_fd, mine, sizeof *mine) != (ssize_t)sizeof *mine) { perror("odl_ar2: barrier write"); return -1; }
    size_t got = 0;
    while (got < sizeof *theirs) {
        ssize_t r = read(rdv_fd, (char*)theirs + got, sizeof *theirs - got);
        if (r <= 0) { perror("odl_ar2: barrier read"); return -1; }
        got += r;
    }
    return 0;
}

/* ---- progress thread: doorbell -> stream_send; poll stream_recv ---- */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static int post_round(uint64_t s) {
    uint32_t slot = (uint32_t)(s % ODL2_NSLOTS);
    uint32_t nbytes = __atomic_load_n(&shr->slot_nbytes[slot], __ATOMIC_ACQUIRE);
    const char *src = send_buf + (uint64_t)slot * ODL2_MAX_BYTES;
    uint64_t t0 = now_ns();
    for (;;) {
        int ret = odl_tb5_stream_send(H, my_sid, peer_sid, src, nbytes);
        if (ret == 0) return 0;
        if (ret != -EAGAIN) { fprintf(stderr, "odl_ar2: stream_send failed: %d\n", ret); return -1; }
        if (now_ns() - t0 > 2000000000ull) { fprintf(stderr, "odl_ar2: stream_send starved (frame pool)\n"); return -1; }
    }
}

static int poll_recv(uint64_t next_seq) {
    uint32_t slot = (uint32_t)(next_seq % ODL2_NSLOTS);
    char *dst = recv_buf + (uint64_t)slot * ODL2_MAX_BYTES;
    uint8_t src_id = 0;
    uint32_t actual = 0;
    int ret = odl_tb5_stream_recv(H, my_sid, dst, ODL2_MAX_BYTES, &src_id, &actual);
    if (ret == -EAGAIN) return 0;
    if (ret < 0) { fprintf(stderr, "odl_ar2: stream_recv failed: %d\n", ret); return -1; }
    if (src_id != peer_sid) { fprintf(stderr, "odl_ar2: unexpected src stream %u\n", src_id); return -1; }
    /* publish: data is in the slot before the flag store (release) */
    __atomic_store_n((uint64_t *)&recv_flag[slot], next_seq, __ATOMIC_RELEASE);
    return 1;
}

static void *progress_main(void *unused) {
    (void)unused;
    uint64_t posted = 0, rcvd = 0, last_act = now_ns();
    while (!progress_exit) {
        uint64_t s = __atomic_load_n(&shr->doorbell, __ATOMIC_ACQUIRE);
        if (s > posted) {
            while (posted < s) {
                posted++;
                if (post_round(posted)) { shr->thread_err = 1; break; }
            }
            last_act = now_ns();
        }
        int r = poll_recv(rcvd + 1);
        if (r > 0) { rcvd++; last_act = now_ns(); }
        else if (r < 0) { shr->thread_err = 1; }
        if (now_ns() - last_act > hot_ns) usleep(200);
    }
    return NULL;
}

/* ---- GPU kernels ---- */
__global__ void odl2_doorbell_kernel(struct odl2_shared *sh, uint32_t slot, uint32_t nbytes, uint64_t seq) {
    if (threadIdx.x == 0) {
        __hip_atomic_store(&sh->slot_nbytes[slot], nbytes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&sh->doorbell, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

/* dtype 3 stage: fp32 -> bf16 into the send slot, then ring the doorbell (one block) */
__global__ void odl2_stage_bf16_kernel(__hip_bfloat16 *__restrict__ slot_buf, const float *__restrict__ src,
                                       uint32_t n, struct odl2_shared *sh, uint32_t slot, uint64_t seq) {
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
        slot_buf[i] = __float2bfloat16(src[i]);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_store(&sh->slot_nbytes[slot], n * 2u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&sh->doorbell, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__device__ __forceinline__ bool odl2_wait(uint64_t *flag, uint64_t seq, int *err) {
    long iters = 0;
    for (;;) {
        uint64_t v = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (v >= seq) return true;
        if (++iters > (2L << 30)) { *err = 1; return false; }
        __builtin_amdgcn_s_sleep(2);
    }
}

/* thread 0 spins on the peer flag, then all threads add the pinned receive slot into dst
 * (bail after ~seconds so a dead peer latches an error instead of wedging the GPU) */
template <typename T, typename ACC>
__global__ void odl2_wait_add_kernel(T *__restrict__ dst, const T *__restrict__ src, const T *__restrict__ recv,
                                     uint32_t n, uint64_t *flag, uint64_t seq, int *err) {
    if (threadIdx.x == 0) odl2_wait(flag, seq, err);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = (T)((ACC)src[i] + (ACC)recv[i]);
}

/* dtype 3: dst = float(bf16(float(bf16(src)) + float(recv_bf16))), as RCCL's bf16 sum would round */
__global__ void odl2_wait_add_bf16wire_kernel(float *dst, const float *src, const __hip_bfloat16 *__restrict__ recv,
                                              uint32_t n, uint64_t *flag, uint64_t seq, int *err) {
    if (threadIdx.x == 0) odl2_wait(flag, seq, err);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float a = __bfloat162float(__float2bfloat16(src[i]));
        float b = __bfloat162float(recv[i]);
        dst[i] = __bfloat162float(__float2bfloat16(a + b));
    }
}

/* ---- public API ---- */
extern "C" int odl2_init(int rank, int dev_index, const char *peer_ip, int port) {
    if (inited) return 0;
    if (rank != 0 && rank != 1) { fprintf(stderr, "odl_ar2: rank must be 0/1\n"); return -1; }
    const char *e;
    int sid0 = (e = getenv("ODL2_SID0")) ? atoi(e) : 241;
    int sid1 = (e = getenv("ODL2_SID1")) ? atoi(e) : 242;
    if ((e = getenv("ODL2_HOT_US"))) hot_ns = (uint64_t)atoll(e) * 1000ull;

    int ret = odl_tb5_open(&H, dev_index);
    if (ret < 0) { fprintf(stderr, "odl_ar2: odl_tb5_open(%d) failed: %d\n", dev_index, ret); return -1; }
    ret = odl_tb5_wait_peer(H, 15000);
    if (ret < 0) { fprintf(stderr, "odl_ar2: wait_peer failed: %d (link not ready?)\n", ret); return -1; }

    uint8_t want = (uint8_t)(rank == 0 ? sid0 : sid1);
    peer_sid = (uint8_t)(rank == 0 ? sid1 : sid0);
    ret = odl_tb5_stream_open(H, want, &my_sid);
    if (ret < 0 || my_sid != want) { fprintf(stderr, "odl_ar2: stream_open(%u) failed: %d\n", want, ret); return -1; }

    int fd = odl_tb5_get_fd(H);
    if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK) < 0) { perror("odl_ar2: fcntl O_NONBLOCK"); return -1; }

    if (hipHostMalloc((void**)&send_buf, ODL2_NSLOTS * ODL2_MAX_BYTES, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "odl_ar2: hipHostMalloc send failed\n"); return -1; }
    if (hipHostMalloc((void**)&recv_buf, ODL2_NSLOTS * ODL2_MAX_BYTES + 4096, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "odl_ar2: hipHostMalloc recv failed\n"); return -1; }
    if (hipHostMalloc((void**)&shr, sizeof *shr, hipHostMallocDefault) != hipSuccess)
        { fprintf(stderr, "odl_ar2: hipHostMalloc shr failed\n"); return -1; }
    memset(send_buf, 0, ODL2_NSLOTS * ODL2_MAX_BYTES);
    memset(recv_buf, 0, ODL2_NSLOTS * ODL2_MAX_BYTES + 4096);
    memset(shr, 0, sizeof *shr);
    recv_flag = (volatile uint64_t *)(recv_buf + ODL2_NSLOTS * ODL2_MAX_BYTES);

    if (hipHostGetDevicePointer((void**)&send_buf_dev, send_buf, 0) != hipSuccess ||
        hipHostGetDevicePointer((void**)&recv_buf_dev, recv_buf, 0) != hipSuccess ||
        hipHostGetDevicePointer((void**)&shr_dev, (void*)shr, 0) != hipSuccess)
        { fprintf(stderr, "odl_ar2: hipHostGetDevicePointer failed\n"); return -1; }
    recv_flag_dev = (uint64_t *)(recv_buf_dev + ODL2_NSLOTS * ODL2_MAX_BYTES);

    /* barrier 1: both streams are open before anyone sends */
    struct xchg mine = { ODL2_PING_MAGIC, (uint32_t)rank, ODL2_MAX_BYTES, ODL2_NSLOTS }, theirs;
    if (tcp_connect_rdv(rank, peer_ip, port)) return -1;
    if (tcp_barrier(&mine, &theirs)) return -1;
    if (theirs.magic != ODL2_PING_MAGIC || theirs.rank != (uint32_t)(1 - rank) ||
        theirs.max_bytes != ODL2_MAX_BYTES || theirs.nslots != ODL2_NSLOTS)
        { fprintf(stderr, "odl_ar2: rendezvous mismatch\n"); return -1; }

    /* stream self-test: one small message each way over the odl link */
    uint64_t ping = ((uint64_t)ODL2_PING_MAGIC << 32) | (uint32_t)rank;
    if (odl_tb5_stream_send(H, my_sid, peer_sid, &ping, sizeof ping) != 0)
        { fprintf(stderr, "odl_ar2: self-test send failed\n"); return -1; }
    uint64_t pong = 0; uint8_t sidr = 0; uint32_t alen = 0;
    uint64_t t0 = now_ns();
    for (;;) {
        int r = odl_tb5_stream_recv(H, my_sid, &pong, sizeof pong, &sidr, &alen);
        if (r == 0) break;
        if (r != -EAGAIN) { fprintf(stderr, "odl_ar2: self-test recv failed: %d\n", r); return -1; }
        if (now_ns() - t0 > 5000000000ull) { fprintf(stderr, "odl_ar2: self-test timeout (no odl traffic from peer)\n"); return -1; }
    }
    if (alen != sizeof pong || sidr != peer_sid || pong != (((uint64_t)ODL2_PING_MAGIC << 32) | (uint32_t)(1 - rank)))
        { fprintf(stderr, "odl_ar2: self-test payload mismatch\n"); return -1; }

    /* barrier 2: both self-tests done, stream queues empty -> data phase */
    if (tcp_barrier(&mine, &theirs)) return -1;
    close(rdv_fd); rdv_fd = -1;

    host_seq = 0;
    progress_exit = 0;
    if (pthread_create(&progress_tid, NULL, progress_main, NULL)) { fprintf(stderr, "odl_ar2: progress thread failed\n"); return -1; }
    inited = 1;
    fprintf(stderr, "odl_ar2: rank%d ready (dev=%d sid=%u peer_sid=%u hot=%llu us)\n",
            rank, dev_index, my_sid, peer_sid, (unsigned long long)(hot_ns / 1000));
    return 0;
}

/* Enqueue one all-reduce on `stream`: dst = src + peer(src), stream-ordered, in place allowed
 * (dst == src). dtype: 0 bf16, 1 fp16, 2 fp32, 3 fp32 over a bf16 wire. nbytes is the size of
 * src (for dtype 3, 4 bytes per element). */
extern "C" int odl2_enqueue(void *stream, void *dst, const void *src, uint32_t nbytes, int dtype) {
    if (!inited) return -1;
    uint32_t wire = dtype == 3 ? nbytes / 2 : nbytes;
    if (wire > ODL2_MAX_BYTES) return -1;
    host_seq++;
    uint64_t s = host_seq;
    uint32_t slot = (uint32_t)(s % ODL2_NSLOTS);
    hipStream_t hs = (hipStream_t)stream;
    char *send_slot = send_buf_dev + (uint64_t)slot * ODL2_MAX_BYTES;
    const char *recv_slot = recv_buf_dev + (uint64_t)slot * ODL2_MAX_BYTES;
    uint64_t *flag = recv_flag_dev + slot;
    int *err = &shr_dev->kernel_err;

    if (dtype == 3) {
        uint32_t n = nbytes / 4;
        hipLaunchKernelGGL(odl2_stage_bf16_kernel, dim3(1), dim3(1024), 0, hs,
                           (__hip_bfloat16 *)send_slot, (const float *)src, n, shr_dev, slot, s);
        hipLaunchKernelGGL(odl2_wait_add_bf16wire_kernel, dim3(1), dim3(1024), 0, hs,
                           (float *)dst, (const float *)src, (const __hip_bfloat16 *)recv_slot, n, flag, s, err);
        return (hipGetLastError() == hipSuccess) ? 0 : -1;
    }

    if (hipMemcpyAsync(send_slot, src, nbytes, hipMemcpyDeviceToDevice, hs) != hipSuccess) return -1;
    hipLaunchKernelGGL(odl2_doorbell_kernel, dim3(1), dim3(1), 0, hs, shr_dev, slot, nbytes, s);
    switch (dtype) {
    case 0:
        hipLaunchKernelGGL((odl2_wait_add_kernel<__hip_bfloat16, float>), dim3(1), dim3(1024), 0, hs,
                           (__hip_bfloat16*)dst, (const __hip_bfloat16*)src, (const __hip_bfloat16*)recv_slot,
                           nbytes / 2, flag, s, err);
        break;
    case 1:
        hipLaunchKernelGGL((odl2_wait_add_kernel<__half, float>), dim3(1), dim3(1024), 0, hs,
                           (__half*)dst, (const __half*)src, (const __half*)recv_slot, nbytes / 2, flag, s, err);
        break;
    case 2:
        hipLaunchKernelGGL((odl2_wait_add_kernel<float, float>), dim3(1), dim3(1024), 0, hs,
                           (float*)dst, (const float*)src, (const float*)recv_slot, nbytes / 4, flag, s, err);
        break;
    default:
        return -1;
    }
    return (hipGetLastError() == hipSuccess) ? 0 : -1;
}

/* Stop the progress thread and close the device; call once all enqueued work has completed. */
extern "C" void odl2_shutdown(void) {
    if (!inited) return;
    progress_exit = 1;
    pthread_join(progress_tid, NULL);
    odl_tb5_stream_close(H, my_sid);
    odl_tb5_close(H);
    inited = 0;
}
