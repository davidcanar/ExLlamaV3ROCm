#!/usr/bin/env python3
"""Discover which /dev/odl_tb5_N reaches which peer box (the per-boot cable map).

Run on every box at about the same time (each run lasts --seconds):
    python3 odl_peer_probe.py --name box1 --ip 10.0.2.1
Opens stream --sid on every odl device, then for --seconds re-sends a small ping carrying
(name, ip) every 100 ms on every device (messages to a stream the peer has not opened yet are
dropped by the driver, so the pings repeat) while polling each device for the peer's ping.
Prints one line per device:  dev<N> -> <peer name> <peer ip>   and an ODL_RCCL_PEERS string.
Uses only a spare stream id; nothing else on the link is touched. Servers must be stopped.
"""
import argparse
import ctypes
import errno
import fcntl
import glob
import os
import struct
import time

LIB = os.path.expanduser("~/.cache/odinlink/build/lib/libodl_tb5.so")
MAGIC = 0x4F444C50  # "ODLP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ip", required=True)
    ap.add_argument("--sid", type=int, default=250)
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()

    lib = ctypes.CDLL(LIB)
    lib.odl_tb5_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.odl_tb5_wait_peer.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.odl_tb5_stream_open.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint8)]
    lib.odl_tb5_stream_close.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
    lib.odl_tb5_stream_send.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_uint32]
    lib.odl_tb5_stream_recv.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_uint32,
                                        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32)]
    lib.odl_tb5_get_fd.argtypes = [ctypes.c_void_p]
    lib.odl_tb5_close.argtypes = [ctypes.c_void_p]

    devs = []
    for path in sorted(glob.glob("/dev/odl_tb5_*")):
        idx = int(path.rsplit("_", 1)[1])
        h = ctypes.c_void_p()
        if lib.odl_tb5_open(ctypes.byref(h), idx) < 0:
            print(f"dev{idx}: open failed")
            continue
        if lib.odl_tb5_wait_peer(h, 3000) < 0:
            print(f"dev{idx}: link not ready")
            lib.odl_tb5_close(h)
            continue
        got = ctypes.c_uint8(0)
        rc = lib.odl_tb5_stream_open(h, a.sid, ctypes.byref(got))
        if rc < 0 or got.value != a.sid:
            print(f"dev{idx}: stream_open({a.sid}) failed rc={rc}")
            lib.odl_tb5_close(h)
            continue
        fd = lib.odl_tb5_get_fd(h)
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
        devs.append([idx, h, None])

    name = a.name.encode()[:31]
    ipb = a.ip.encode()[:15]
    ping = struct.pack("<I32s16s", MAGIC, name, ipb)
    pbuf = ctypes.create_string_buffer(ping, len(ping))
    rbuf = ctypes.create_string_buffer(256)
    src = ctypes.c_uint8(0)
    alen = ctypes.c_uint32(0)
    t_end = time.monotonic() + a.seconds
    next_ping = 0.0
    while time.monotonic() < t_end:
        now = time.monotonic()
        if now >= next_ping:
            for d in devs:
                lib.odl_tb5_stream_send(d[1], a.sid, a.sid, pbuf, len(ping))
            next_ping = now + 0.1
        for d in devs:
            while True:
                rc = lib.odl_tb5_stream_recv(d[1], a.sid, rbuf, 256, ctypes.byref(src), ctypes.byref(alen))
                if rc != 0:
                    break
                if alen.value == len(ping):
                    m, pn, pip = struct.unpack("<I32s16s", rbuf.raw[:len(ping)])
                    if m == MAGIC and d[2] is None:
                        d[2] = (pn.rstrip(b"\0").decode(), pip.rstrip(b"\0").decode())
        time.sleep(0.005)
    peers = []
    for idx, h, who in devs:
        if who:
            print(f"dev{idx} -> {who[0]} {who[1]}")
            peers.append(f"{who[1]}={idx}")
        else:
            print(f"dev{idx} -> (no ping received)")
        lib.odl_tb5_stream_close(h, a.sid)
        lib.odl_tb5_close(h)
    print("ODL_RCCL_PEERS=" + ",".join(peers))


if __name__ == "__main__":
    main()
