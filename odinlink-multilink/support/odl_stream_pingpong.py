#!/usr/bin/env python3
"""Auto-ID stream ping-pong over libodl_tb5 (series-correct data-path test).

The old C odl_pingpong hardcodes server stream 1 / client stream 2. The current
driver allocates stream ids from 20..255 and ignores a fixed request, so that
tool cannot work any more. This one does what the RCCL plugin does: both sides
open an auto-assigned stream and advertise its id in-band.

  server <dev> <iters>   prints its stream id, then echoes every message back
  client <dev> <srv_sid> <iters>

Usage: odl_stream_pingpong.py server 0 200
       odl_stream_pingpong.py client 0 <sid printed by server> 200
"""
import ctypes, struct, sys, time

lib = ctypes.CDLL("/usr/local/lib/odinlink/libodl_tb5.so.0", use_errno=True)
lib.odl_tb5_stream_recv.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_void_p,
                                    ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8),
                                    ctypes.POINTER(ctypes.c_uint32)]
lib.odl_tb5_stream_send.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8,
                                    ctypes.c_void_p, ctypes.c_uint32]


def open_dev(dev):
    h = ctypes.c_void_p()
    rc = lib.odl_tb5_open(ctypes.byref(h), ctypes.c_int(dev))
    if rc < 0:
        sys.exit(f"open dev{dev}: {rc}")
    sid = ctypes.c_uint8(0)
    rc = lib.odl_tb5_stream_open(h, ctypes.c_uint8(0), ctypes.byref(sid))
    if rc < 0:
        sys.exit(f"stream_open dev{dev}: {rc} errno {ctypes.get_errno()}")
    return h, sid.value


def recv(h, sid, n):
    buf = ctypes.create_string_buffer(n)
    src = ctypes.c_uint8(0)
    got = ctypes.c_uint32(0)
    rc = lib.odl_tb5_stream_recv(h, sid, buf, n, ctypes.byref(src), ctypes.byref(got))
    if rc < 0:
        sys.exit(f"recv(sid={sid}) failed: {rc}")
    return buf.raw[:got.value], src.value


def send(h, sid, dst, payload):
    rc = lib.odl_tb5_stream_send(h, sid, dst, payload, len(payload))
    if rc < 0:
        sys.exit(f"send(sid={sid} dst={dst}) failed: {rc}")


def main():
    mode, dev = sys.argv[1], int(sys.argv[2])
    h, sid = open_dev(dev)

    if mode == "server":
        iters = int(sys.argv[3])
        print(f"SERVER_SID={sid}", flush=True)
        for _ in range(iters):
            data, src = recv(h, sid, 4096)
            send(h, sid, src, data)          # echo back to the sender's stream
        print(f"server echoed {iters} messages", flush=True)
        return

    srv = int(sys.argv[3]); iters = int(sys.argv[4])
    payload = b"x" * 4032
    rtts = []
    for i in range(iters):
        t0 = time.monotonic()
        send(h, sid, srv, payload)
        data, src = recv(h, sid, 4096)
        if len(data) != len(payload):
            sys.exit(f"short echo: {len(data)} != {len(payload)}")
        rtts.append((time.monotonic() - t0) * 1e6)
    rtts.sort()
    p = lambda q: rtts[min(len(rtts) - 1, int(q * len(rtts)))]
    print(f"RESULT dev={dev} iters={iters} min={rtts[0]:.1f} p50={p(.5):.1f} "
          f"p90={p(.9):.1f} max={rtts[-1]:.1f} us")


if __name__ == "__main__":
    main()
