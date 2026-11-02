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
