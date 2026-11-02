# OdinLink multi-link fabric (3+ boxes)

How to build, install and verify the multi-cable OdinLink mesh. This is the
runbook behind `patches/linux/0001` + `patches/odinlink/0004`/`0005`
(upstream PR: Geramy/OdinLink-Five#37, issue #35).

## What the patches change

| Patch | Layer | Why |
|---|---|---|
| `patches/linux/0001-thunderbolt-source-xdomain-protocol-handlers.patch` | kernel core | Handlers get the **source XDomain** of an XDomain packet, resolved in the receiving controller's own domain. Adds `struct tb_protocol_handler_xd` + `tb_register_protocol_handler_xd()`/`_unregister_`; **leaves `struct tb_protocol_handler` byte-identical to mainline** and keeps the stock dispatch path. |
| `patches/odinlink/0004-odl-tb5-demux-by-source-xdomain.patch` | driver | Demux LOGIN/LOGOUT by source XDomain instead of route, so each point-to-point peer completes its own handshake. Falls back to route demux when built against a stock core. |
| `patches/odinlink/0005-odl-tb5-bound-connect-retries.patch` | driver | Bounds the connect-retry loop (a `-EINVAL` there used to retry at 1 Hz forever and wedge the USB4 controller) and allocates a legal in-HopID for the `bind_any`/Mac-sink path. |

### Why the core struct must stay identical

The in-tree `thunderbolt_net` embeds `struct tb_protocol_handler` in its private
data (`net->handler`), and distros ship it prebuilt. Inserting fields mid-struct
makes a patched core read tbnet's fields at the wrong offsets; the corruption
lands on `net->napi.state`:

```
kernel BUG at net/core/dev.c:7671!   (BUG_ON(!test_bit(NAPI_STATE_SCHED)))
  napi_enable -> tbnet_open -> __dev_open
```

…under `rtnl_lock`, so **plugging a cable hangs the entire machine**. That is
exactly the failure mode this design avoids, and `check-tb-handler-abi.sh`
enforces it at build time.

## Build

Needs `kernel-devel-$(uname -r)`, `patch`, `rpm2cpio`, `cpio`, `gcc`, network
on first run (kernel SRPM is fetched from kojipkgs, kernel.org as fallback).

```bash
odinlink-multilink/build.sh              # or: build.sh <kver>
```

Outputs `~/.cache/odl-multi/<kver>/out/{thunderbolt.ko,odl_tb5.ko,lib/}`.

## Install (every box, before connecting cables)

```bash
sudo odinlink-multilink/install.sh <kver>
```

Installs:

- `thunderbolt.ko` → `/lib/modules/<kver>/updates/thunderbolt/` — `depmod`
  prefers `updates/`, and `updates/` is also the only directory depmod gives
  priority to by default. Verify: `modinfo -F filename thunderbolt` must print
  the `updates/` path.
- `odl_tb5.ko` + `libodl_tb5.so` → `/usr/local/lib/odinlink/`,
  `odl-state` → `/usr/local/bin/`, loader → `/usr/local/sbin/odl-swap.sh`,
  unit → `/etc/systemd/system/odinlink.service` (enabled).
- Also refreshes `~/.cache/odinlink/{driver,build/lib}` for manual loads.

**Keep thunderbolt out of the initramfs.** An initramfs copy shadows
`updates/` and silently pins an older core across reboots:

```bash
printf 'omit_drivers+=" thunderbolt "\n' | sudo tee /etc/dracut.conf.d/60-odl-multilink.conf
sudo dracut --force --regenerate-all
```

## Activate without a reboot

A reboot is simplest. Without one, everything that depends on the core has to
be unloaded first — note `typec` → `typec_ucsi` depends on `thunderbolt`, so
`rmmod thunderbolt` otherwise fails silently and `modprobe` becomes a no-op
leaving the **old** core running:

```bash
sudo odinlink-multilink/support/reload-node.sh
```

It unloads `odl_tb5`, `thunderbolt_net`, `typec_*`, `typec_ucsi`, `typec`, unbinds
the NHI controllers, swaps the core, rebinds, and loads the driver with
`bind_any=0`. If `typec_ucsi` is pinned by hardware (it usually is), reboot.

## Kernel updates

`thunderbolt.ko` and `odl_tb5.ko` are built for one kernel, and a kernel update
installs a new default kernel without them. Stage both before rebooting, on every
box:

```bash
sudo dnf upgrade --refresh                       # new kernel + kernel-devel <new-kver>
odinlink-multilink/build.sh <new-kver>           # e.g. 7.2.14-200.fc44.x86_64
sudo odinlink-multilink/install.sh <new-kver>    # core -> /lib/modules/<new-kver>/updates
```

Before the reboot check that `sudo grubby --default-kernel` is the new kernel, that
`sudo grubby --info=/boot/vmlinuz-<new-kver>` still carries the kernel command line,
and that `sudo lsinitrd /boot/initramfs-<new-kver>.img | grep /thunderbolt.ko`
prints nothing (`omit_drivers`). Stop the model server first: `install.sh` rewrites
`libodl_tb5.so` in place.

`/usr/local/lib/odinlink/odl_tb5.ko` is one path for every kernel, so once a new
kernel is staged the old one has no matching driver. To boot back into it, run
`sudo odinlink-multilink/install.sh <old-kver>` (its build stays in
`~/.cache/odl-multi/<old-kver>`).

After the reboot:

- `odl-state` on every box. A box that booted after its peers had given up
  (`giving up after 20 login attempts` in dmesg) leaves those links idle; run
  `sudo systemctl restart odinlink` on all boxes at about the same time.
- `tools/ds41/tp3_start.sh probe` (or `exl3ds4.sh probe` for two boxes) must see
  every cable from both ends.
- `perf/tp3/rccl_stall.sh 12` before putting the server back.
- Userspace (`rebuild-plugin.sh`, `odl_ar3/build.sh`) does not depend on the kernel;
  rebuild it only to pick up a new compiler, on every box.

Moving the 3-box rig from 7.1.8/7.2.13 to 7.2.14 on 2026-11-02: the 7.2.14
`drivers/thunderbolt` is identical to 7.2.13's, and `patches/linux/0001` applies with
`--fuzz=0`. 7.2 on both ends of a cable exposed the `rx_repost` race fixed by
`patches/odinlink/0013` (next section).

## Loader defaults

`odl-swap.sh` loads with **`bind_any=0`** and `ring=1024`, `e2e=1`.

`bind_any=1` additionally attaches to hosts that do not advertise OdinLink, as
a Mac/bind_any "sink". That path guesses the peer TX HopID as `1`, which
`tb_xdomain_alloc_in_hopid()` rejects (`TB_PATH_MIN_HOPID` is 8) — the source of
`failed to allocate input HopID: -22`. Only enable it (`ODL_BIND_ANY=1`) for a
real Mac sink.

## IP over the mesh (control plane)

Each trunk is point-to-point and every link is in `10.0.2.0/24`, so each
interface carries **only a `/32` route to its direct peer**. Two interfaces with
equal `/24` connected routes make replies leave on the wrong cable — the classic
"ping works one way" symptom.

`support/setup-fabric-net.sh <box1|box2|box3>` creates NetworkManager profiles
**bound to the interface MAC** rather than to `thunderbolt0/1`: those netdev
names follow XDomain-appearance order and can swap between boots, which would
otherwise move addresses onto the wrong cable.

Reference 3-box wiring (cable any port; the script encodes it by MAC):

| Link | Endpoints | Addresses |
|---|---|---|
| Box1 ↔ Box2 | Box1 `tb0` ↔ Box2 `tb0` | 10.0.2.1 ↔ 10.0.2.2 |
| Box1 ↔ Box3 | Box1 `tb1` ↔ Box3 `tb1` | 10.0.2.4 ↔ 10.0.2.3 |
| Box2 ↔ Box3 | Box2 `tb1` ↔ Box3 `tb0` | 10.0.2.5 ↔ 10.0.2.6 |

## Verify

```bash
# devices: one per link, all must be ready
odl-state                       # expect dev0: state=ready ... devN: state=ready
ls /dev/odl_tb5_*

# no transport-level regressions
sudo dmesg | grep -icE 'kernel BUG|napi_enable|NULL pointer'     # -> 0
sudo dmesg | grep -E 'odl_tb5: (entering READY|giving up)'

# IP mesh: each box must reach its direct peers
ping -c3 10.0.2.2 ; ping -c3 10.0.2.3

# real data path over a link (server on one box, client on the other)
odl_pingpong server <dev> 4096 200 20
odl_pingpong client <dev> 4096 200 20     # expect p50 ~33 us on Strix Halo
```

### Device↔peer mapping is NOT stable

Device indices are assigned in XDomain probe order, so they change across
reboots and module reloads. Observed on the same 3-box mesh, before and after
a synchronized reload:

```
first boot : Box1 dev0 = Box2, Box1 dev1 = Box3
after      : Box1 dev0 = Box3, Box1 dev1 = Box2     <-- swapped
```

Never hardcode a mapping (that is also why `ODL_RCCL_PEERS` is passed
explicitly rather than inferred). Re-derive it whenever the driver reloads:
start a server on one peer device and try each local device, one at a time
(`support/odl_stream_pingpong.py server|client`; the server prints its
auto-assigned stream id, which the client takes as an argument). The pair that
completes is the mapping. Mixing it up looks like a hang, because a stream id
sent on the wrong cable has no matching stream on the peer.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `kernel BUG at net/core/dev.c:7671`, `napi_enable`, box hangs on cable plug | A core was built with fields added **inside** `struct tb_protocol_handler`. Old/new ABI mix. Rebuild with `patches/linux/0001` and check `odinlink-multilink/check-tb-handler-abi.sh`. |
| `failed to allocate input HopID: -22` + hang | Pre-0005 driver in the `bind_any` sink path loop, or `bind_any=1` on an all-OdinLink fabric. Use `bind_any=0`. |
| Second link's login times out ×20, `/dev/odl_tb5_1` missing on the peer | Route-based demux (driver built without `0004`, or core without `patches/linux/0001`). Check `dmesg` for `source-XDomain protocol demux`. |
| `modprobe thunderbolt` resolves to `kernel/...`, not `updates/...` | `updates/` not populated, or an initramfs copy is winning. Re-install and regenerate the initramfs with `omit_drivers`. |
| `rmmod thunderbolt` says "in use by: typec" | Unload `typec_ucsi` → `typec` first (`support/reload-node.sh`), or reboot. |
| All devices READY, but the cable probe (`tp3_start.sh probe`) sees each cable in one direction only, and dmesg shows `can_send=0 ... free=<256 ... rx_posted=~4000` | `odl_tb5` without `patches/odinlink/0013`: concurrent RX reposts drained the frame pool. Rebuild and install the driver, then `systemctl restart odinlink` on every box. |
| A box logs `giving up after 20 login attempts ... staying idle` after a reboot | Its peer booted later. `sudo systemctl restart odinlink` on all boxes together. |
| `Unknown symbol tb_register_protocol_handler_xd` | Driver built for the xd API, running core is stock. Install the patched core (or rebuild the driver against stock headers for single-cable use). |

## Multi-link and TP>2

The fabric is ready for TP=3 and `patches/odinlink/0006` makes the RCCL plugin
ready as well. Two changes:

1. **One handle per device.** The plugin used to keep a single global
   `g_handle`, opened with whichever device RCCL mentioned first and then
   reused for every peer — on a 3-node mesh all collectives went down one
   cable. Handles are now per `/dev/odl_tb5_N` with per-device refcounts.
2. **Peer-aware device selection.** `listen()` advertises this rank's link
   address in the NCCL handle (in the free bytes after `hw_id`, marked with
   `0xA5`; old peers simply ignore it), and `connect()` uses it to pick the
   local cable that reaches that peer:

   ```bash
   # on every rank; the address must be the one on the links it serves
   export ODL_RCCL_LOCAL_IP=10.0.2.1
   # peer address -> local /dev/odl_tb5_N, comma separated
   export ODL_RCCL_PEERS="10.0.2.2=0,10.0.2.3=1"
   ```

   With neither variable set, behaviour is unchanged (RCCL's device index is
   used), so a 2-node deployment needs no changes at all.

Reference values for the 3-box mesh in this runbook:

| Rank | Box | `ODL_RCCL_LOCAL_IP` | `ODL_RCCL_PEERS` |
|---|---|---|---|
| 0 | Box1 | 10.0.2.1 | `10.0.2.2=0,10.0.2.3=1` |
| 1 | Box2 | 10.0.2.2 | `10.0.2.1=0,10.0.2.6=1` |
| 2 | Box3 | 10.0.2.3 | `10.0.2.4=0,10.0.2.5=1` |

(The device numbers come from the device↔peer map above: Box1 `dev0`=Box2,
`dev1`=Box3; Box2 `dev0`=Box1, `dev1`=Box3; Box3 `dev0`=Box2, `dev1`=Box1 —
note Box3's numbering is the reverse of its address order, which is exactly
why this map is explicit rather than inferred.)

`ODL_RCCL_DEVICES` still filters which devices are exposed at all; leave it
unset on a multi-link host so both cables are visible.

## RCCL with more than one OdinLink device (TP>2)

Two NCCL-side details bite as soon as a host exposes two odl devices; neither
shows up with one cable, so a working 2-node setup proves nothing about them.

**1. Every net device needs a unique identity.** NCCL builds one topology node
per net device and matches them by the properties the plugin reports. Returning
the same `name`/`pciPath` for both (and `guid = dev`, i.e. 0 for the first)
collapses them, and the first collective then fails with:

```
ncclInternalError: Could not find NET with id 0
```

`patches/odinlink/0007` gives each device a distinct name
(`OdinLink-TB5/odl_tb5_N`), pciPath and non-zero guid.

**2. Kill leftovers before rerunning.** A launcher killed by `timeout` does not
kill the remote python: orphaned ranks keep retrying bootstrap and holding
streams, and the next run then hangs against them. Symptoms are several
`rccl3_allreduce.py` processes per box (`pgrep -af rccl3_allreduce`), all stuck
after `ODL_TB5 net plugin loaded`. `support/rccl3_run.sh` now pre-cleans, uses a
random bootstrap port per run, and traps EXIT to clean again.

Bootstrap must ride the LAN, not the 10.0.2.x mesh: that mesh is
point-to-point, so one master address is not reachable from every box.
`NCCL_SOCKET_IFNAME` is per box and is not uniform either (a USB NIC comes up
as `enp198s0f0u2u4`), so the launcher detects each box's LAN interface.

## Virtual single device with per-connection negotiation (0009) — WORKING

`0009-rccl-plugin-virtual-device-negotiation.patch` replaces the parked WIP:
NCCL sees ONE net device; every connection is negotiated per peer and then
carries its own dedicated data stream.

- The driver reassembles received messages with per-stream state, so a data
  stream must have exactly one sender. Every connection therefore opens its
  own stream pair (connector S, acceptor R).
- A per-cable CONTROL stream carries only single-frame messages (no
  reassembly → safe to multiplex): `hello{S,C,L}` from the connector
  ("I send from S, grant me on control stream C, for listener L") and
  `grant{S,R}` back. A dispatcher thread per cable is the sole reader.
- The listener tag L matters: NCCL exchanges each listen handle with ONE
  specific peer, and without it a 3-box mesh crossed connections and
  double-counted contributions (all-reduce returned 10 instead of 6).
- connect() returns before the grant arrives; the send worker blocks on
  dst_ready (published by the control dispatcher), so NCCL can post sends
  immediately.
- Watch the two g_lock re-entrancy traps: listen() runs under g_lock and the
  control-plane setup must use the `_locked` handle helpers.

Verified: 2-rank and 3-rank all-reduce `exact=True` (repeat runs), zero
transfers failed, fabric healthy afterwards. `ODL_RCCL_DEVICES` pinning is no
longer needed (and should stay unset so all cables are visible).


## OPEN: rccl3_allreduce microbenchmark hangs at 3 ranks (regression)

`support/rccl3_run.sh` passed 3×/3 at 3 ranks on 2026-11-01 (23:37–23:47, 4 MiB,
`exact=True`, all rc=0) and has hung on every attempt on 2026-11-02, at 10 KiB,
64 KiB and 4 MiB, with and without `odl_trace_rx`, `NCCL_MAX_NCHANNELS=1`,
and after an `odl_tb5` reload on Box3. The same boxes, plugin and driver pass:

- `support/../r2` 2-rank allreduce (11.7 us @ 4 MiB) — every time
- the **TP=3 model server** end to end: 2.9bpw and 3.5bpw, 2K→512K ctx,
  DSpark 27.3 tok/s, needle-correct at 38K tokens

so the fabric and the negotiated plugin are healthy for the real workload; the
failure is specific to torch.distributed's 3-rank communicator shape.

Signature (per-box plugin traces, `ODL_DEBUG=1`, saved in
`odl-multi/debug-3rank-1110/`): asymmetric per-pair channel counts
(box1: 2 connects →Box2, 4 →Box3; box3: 4 →Box1, 4 →Box2), hellos/grants all
complete with zero drops, then data completes on a subset of channels
(xfer DONE 4/12/16 per box) and NCCL's watchdog aborts. No kernel-level drops.
Suspects: listener-tag (lidx) collisions across the two peers' handle exchanges
(lidx is a per-process counter; two boxes can present the same lidx to a third),
or a channel-count mismatch in connect/accept pairing that the model runtime's
fixed channel pattern never triggers.

### Update 2026-11-02 afternoon: three separate causes

1. **Stale cable map (fixed).** After the 10:42 synchronized `odl_tb5` reload, Box2's devices came
   back swapped (dev0 = Box3, dev1 = Box1) while `tp3_worker.sh` still mapped
   `10.0.2.1=0,10.0.2.3=1`. With a stale `ODL_RCCL_PEERS` the plugin opens a connection on the
   cable to the *other* peer; its hello lands on that box's control stream (id 20 on every device)
   and is accepted there, because listener tags are per-process counters that collide across
   boxes. That produces exactly the asymmetric per-pair channel counts above, and the world-3
   warmup hangs. `tools/ds41/odl_peer_probe.py` now derives the map on all three boxes at once and
   `tools/ds41/tp3_start.sh` hands each rank its own `ODL_RCCL_PEERS` on every start
   (`tp3_start.sh probe` prints it). With correct maps the world-3 warmup passes.
2. **Fixed (patches/odinlink/0012): intermittent stall at the first large collective.** With
   correct maps, about 1 start in 6 stalled in the first collective that connects new channels
   or trees (the warm-up prefill, or the first 20 MiB all-reduce): every rank inside the same
   collective, the RCCL proxy spinning, no driver message, and in the `ODL_DEBUG=1` trace one
   send stuck with `dst=0` — its grant never delivered. The plugin's connector registry was
   indexed by the local send stream id alone, while stream ids are allocated per device and both
   cables hand out the same low ids: two pending connects with the same id on different cables
   overwrote each other's entry and one grant went to the wrong comm. 0012 keys it by
   (device, stream id). `perf/tp3/rccl_stall.sh` reproduces it on bare ranks in ~2 s per attempt
   (2 of 12 hung before, 0 of 20 after, collision present in 12 of the 20). It was independent
   of odl_ar3 (hung with it off too).
   The `rccl3_allreduce` "hang" above was its exit, not the data path: rank 0 printed
   `exact=True` and every transfer in the 10:12 traces completed; the script then never returned
   from teardown. (Its 12.1 us per 4 MiB is the enqueue time: there is no sync before the timer
   stops.)
3. **Frame loss on the raw DMA path.** Under concurrent load all three boxes logged
   `RX CRC error ... frame dropped` followed by `stream N fragment gap ... dropping message`, often
   two boxes in the same second. The driver drops the whole message and nothing retransmits it,
   so an RCCL collective that loses a frame hangs. The odl_ar3 decode all-reduce recovers its own
   messages (NACK + resend); RCCL traffic does not yet.

Never attach `py-spy` (or any ptrace tool) to a rank without `--nonblocking`: the stop interrupts
the plugin's blocking `STREAM_RECV`, the driver returns `-EINTR`, the plugin fails the transfer and
NCCL raises `ncclSystemError`.
