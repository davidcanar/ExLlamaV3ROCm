import torch, time, glob, threading
torch.cuda.set_device(0)
dev = glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk")[0].rsplit("/", 1)[0]
def rd(f):
    try: return open(f"{dev}/{f}").read().strip()
    except Exception as e: return f"n/a ({e.__class__.__name__})"
hw = glob.glob(f"{dev}/hwmon/hwmon*")[0]
def hwrd(f):
    try: return open(f"{hw}/{f}").read().strip()
    except Exception: return "n/a"
seen, pw = [], []
stop = False
def poll():
    while not stop:
        seen.append([l.strip() for l in open(f"{dev}/pp_dpm_sclk") if "*" in l][0])
        pw.append(hwrd("power1_average") if hwrd("power1_average") != "n/a" else hwrd("power1_input"))
        time.sleep(0.05)
A = torch.randn(8192, 8192, device="cuda", dtype=torch.half)
th = threading.Thread(target=poll); th.start()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 4:
    C = A @ A; torch.cuda.synchronize()
torch.cuda.synchronize()
stop = True; th.join()
mhz = [int(s.split(":")[1].strip().split("M")[0]) for s in seen]
print("sclk under GEMM: min", min(mhz), "max", max(mhz), "median", sorted(mhz)[len(mhz)//2],
      "| power (uW):", sorted(pw)[len(pw)//2] if pw else "n/a",
      "| perf_level:", rd("power_dpm_force_performance_level"),
      "| power1_cap:", hwrd("power1_cap"), "cap_max:", hwrd("power1_cap_max"),
      "| od:", rd("pp_od_clk_voltage").replace("\n", " ")[:200])
