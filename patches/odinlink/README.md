# patches/odinlink — local OdinLink-Five driver + RCCL v7 plugin diff

The local modification of OdinLink-Five against upstream. `0001` is a 5-file
diff:

    git diff 4534f58 -- \
        driver/odl_tb5_core.h driver/odl_tb5_params.c driver/odl_tb5_proto.c \
        driver/odl_tb5_ring_dma.c rccl/src/odl_tb5_plugin.c

Upstream / base:

- Repository: <https://github.com/Geramy/OdinLink-Five>
- Base commit: `4534f58` (merge of PR #26, `fix/issue-4-probe-logging`)

What the diff does (summary):

- `driver/odl_tb5_ring_dma.c` (+ the small core.h/params/proto touches):
  stream protocol v2 — an 8-byte stream header carrying `frag_idx` with
  per-frame tails — so one logical message may span multiple DMA frames
  (1 MiB default payload, reassembly against the receive pools, explicit
  receive-message lifetime handling via `odl_tb5_rx_msg_free()`), plus the
  associated ring/pool fixes to keep multi-frame traffic stable.
- `rccl/src/odl_tb5_plugin.c`: RCCL v7 network plugin changes — sends
  1 MiB multi-frame messages by default (`ODL_MSG_DEFAULT`), keeps legacy
  4000-byte single-frame mode available via `ODL_RCCL_CHUNK=4000`, and adds
  the stream-scoped send/recv paths the v2 framing enables. Validated with
  multi-frame + stream-scoped sync + four RCCL channels (`NCCL_MAX_NCHANNELS=4`)
  against the full-model and DSpark regressions.

`0002-rccl-plugin-ODL_RCCL_DEVICES-allow-list.patch` applies on top of
`0001` (plugin only, MIT): `ODL_RCCL_DEVICES="0[,1...]"` limits the
`/dev/odl_tb5_N` devices the plugin offers to RCCL, with an RCCL-device to
`/dev/odl_tb5_N` map in the shared-handle open. A second Thunderbolt host on
another USB4 port (a third machine) shows up as another odl_tb5 device; RCCL
would otherwise see an asymmetric NIC count across the ranks and could open
the wrong link. The launchers set `ODL_RCCL_DEVICES=0`; unset keeps the old
behaviour.

`0003-odl-tb5-rx-copy-and-stale-tunnel-fixes.patch` applies on top of
`0002` (driver only, GPL-2.0). Three fixes, all from a hang seen under memory
pressure on 2026-10-28:

- when the spare RX assembly buffers run dry (routine during big RCCL
  transfers), the message copy used an order-8 `kmalloc(GFP_ATOMIC)` that fails
  on a fragmented host, dropping the message and hanging both RCCL ranks. The
  RX callback runs in a workqueue, so the copy is now `kvmalloc(GFP_KERNEL)`;
- module exit freed `bind_any` devices without disabling their XDomain paths,
  which left the DMA tunnel in the thunderbolt core (the next load failed
  `enable_paths` with -ENOMEM until a reboot) and a devm action pointing into
  the unloaded module. Those devices now go through the normal destroy path;
- before enabling paths, a stale DMA tunnel on the instance's own NHI rings is
  torn down (rings are exclusive; HopIDs are wildcards, rings are not, so
  ThunderboltIP's tunnel is never touched). This recovers a link left in that
  state by an older build.

After rebuilding, reload the module on both boxes with the server stopped
(`sudo rmmod odl_tb5`, then load it as before). Validated with
`perf/run_stress.sh` (1–64 MiB all-reduces, exact results, with
`rx_asm_pool=0` and with CPU load on both boxes) and full model serving.

Apply with `git apply` (plain diffs, no commit identity to preserve):

    git clone https://github.com/Geramy/OdinLink-Five odinlink
    cd odinlink
    git checkout 4534f58
    git apply /path/to/patches/odinlink/*.patch      # 0001, 0002, 0003 in order

`setup.sh` performs the clone/patch/build automatically (default source
location `$HOME/.cache/odinlink`, override with `EXL3_ODL_SRC`; the built
plugin path expected by the launchers defaults to
`$HOME/.cache/odinlink/build`, override with `EXL3_ODL_BUILD`).

License notes (from the upstream project):

- `driver/` — kernel module `odl_tb5.ko`, **GPL-2.0**. The driver files this
  patch touches are derivative work of GPL-2.0 code and remain GPL-2.0.
- Userspace components — `lib/`, `rccl/` (the RCCL v7 plugin
  `librccl_net_odl_tb5.so`), `nccl/`, `cli/`, `daemon/`, `tray/`, `verbs/`:
  **MIT**. The plugin half of this patch applies to MIT code.


`0004-odl-tb5-demux-by-source-xdomain.patch` and
`0005-odl-tb5-bound-connect-retries.patch` add multi-link support on top of
`0001`-`0003` (both apply cleanly at `4534f58` after the earlier three):

- `0004` demuxes XDomain LOGIN/LOGOUT by **source XDomain** instead of by
  route. Two point-to-point links put every peer at the same route string
  (`X-2`), so a route-blind handler dispatches the second peer's login to the
  wrong device and that link never completes. Uses
  `tb_register_protocol_handler_xd()` when the core provides it and falls back
  to the route handler otherwise, so a single-cable build against a stock
  kernel still works. Upstream: Geramy/OdinLink-Five issue #35, PR #37.
- `0005` bounds the connect-retry loop. A failed DMA path setup after a
  successful login used to re-handshake once per second forever (the retry
  counter only tracked failed *logins*); on Strix Halo that `tb_cfg_*` storm
  takes the USB4 controller — and the GPU, sharing the power/firmware domain —
  down. It also gives the `bind_any`/Mac-sink path a legal input HopID instead
  of the hardcoded `1`, which is below `TB_PATH_MIN_HOPID` (8) and produced
  `failed to allocate input HopID: -22`.

Both require the kernel-side change in
`patches/linux/0001-thunderbolt-source-xdomain-protocol-handlers.patch`.
That patch adds a **separate** `struct tb_protocol_handler_xd` and registration
API and deliberately leaves `struct tb_protocol_handler` byte-identical to
mainline: the in-tree `thunderbolt_net` embeds that struct in its private data,
so adding fields mid-struct corrupts `net->napi.state` and plugging a cable
hangs the machine (`napi_enable` BUG under `rtnl_lock`).
`odinlink-multilink/check-tb-handler-abi.sh` enforces the layout at build time.

`0012-rccl-plugin-per-cable-connector-registry.patch` (plugin only, MIT) fixes
the intermittent TP=3 RCCL hang at the first collective that connects new
channels or trees. The control dispatcher delivers grants through a connector
registry that was indexed by the local send stream id alone, but stream ids are
allocated per odl device and every cable hands out the same low ids (20, 21,
...). When two pending connects to different peers got the same id, the second
overwrote the first, one peer's grant was published to the wrong comm (which
could then send to the wrong stream) and the other connection's send worker
waited for its grant forever (an `ODL_DEBUG=1` trace shows a send stuck with
`dst=0`). The registry is now keyed by (odl device, stream id). Reproduced with
`perf/tp3/rccl_stall.sh` (the TP=3 start-up collectives on bare ranks, ~2 s per
attempt): 2 of 12 starts hung before, 0 of 20 after, with the collision present
in 12 of those 20 traces.

`0013-odl-tb5-rx-repost-reserve-before-post.patch` (driver) fixes one-way links
after both ends of a cable moved to a 7.2 kernel. `odl_tb5_rx_repost()` read
`rx_posted` once and then counted locally, but it runs both from process context
(stream open) and from the RX completion path, which refills after every frame.
A stream opened while its peer was already sending got refilled by both at once
and posted about 2 x `rx_target` (4063 of the 4096 pool frames in one trace),
leaving fewer than `ODL_TB5_TX_POOL_RESERVE` frames free: every send on that
cable then failed `can_send` until inbound frames returned their slots, about one
per peer ping. The DMA handshake still passes, so `odl-state` reports READY and
only the data path is one-way - the box that opened its stream last cannot send.
Each frame is now reserved in `rx_posted` before it is posted. Found 2026-11-02
right after moving all three boxes to 7.2.14: `tools/ds41/tp3_start.sh probe`
showed data flowing only from the lower- to the higher-numbered box on every
cable (the probe starts on box1 first).

Bring-up, addressing scheme and verification:
[`odinlink-multilink/docs/RUNBOOK.md`](../../odinlink-multilink/docs/RUNBOOK.md).

Only this diff is redistributed here; the OdinLink-Five source itself is
fetched from upstream by `setup.sh`. See the repository NOTICE.
