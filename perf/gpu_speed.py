import torch, time, glob, threading
torch.cuda.set_device(0)
def sclk():
    for f in glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk"):
        return " ".join(l.strip() for l in open(f) if "*" in l)
res = {}
# bandwidth: GEMV over a 1 GiB fp16 matrix
W = torch.randn(32768, 16384, device="cuda", dtype=torch.half)
x = torch.randn(16384, 1, device="cuda", dtype=torch.half)
for _ in range(5): y = W @ x
torch.cuda.synchronize()
seen = []
def poll():
    for _ in range(20):
        seen.append(sclk()); time.sleep(0.05)
th = threading.Thread(target=poll); th.start()
t0 = time.perf_counter(); n = 50
for _ in range(n): y = W @ x
torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
th.join()
res["gemv_GBps"] = W.numel() * 2 / dt / 1e9
# compute: big fp16 GEMM
A = torch.randn(8192, 8192, device="cuda", dtype=torch.half)
for _ in range(3): C = A @ A
torch.cuda.synchronize(); t0 = time.perf_counter(); n = 10
for _ in range(n): C = A @ A
torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
res["gemm_TFLOPS"] = 2 * 8192**3 / dt / 1e12
# launch latency: tiny kernels back to back
z = torch.zeros(16, device="cuda")
torch.cuda.synchronize(); t0 = time.perf_counter(); n = 5000
for _ in range(n): z.add_(1.0)
t1 = time.perf_counter(); torch.cuda.synchronize(); t2 = time.perf_counter()
res["launch_us"] = (t1 - t0) / n * 1e6
res["tiny_exec_us"] = (t2 - t0) / n * 1e6
# CPU single-thread python speed
t0 = time.perf_counter(); s = 0
for i in range(3_000_000): s += i * i
res["py_loop_s"] = time.perf_counter() - t0
print({k: round(v, 2) for k, v in res.items()}, "sclk under load:", sorted(set(seen)))
