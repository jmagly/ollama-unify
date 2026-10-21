# ollama-unify

> Classify the host, select the safest available Ollama backend, contain memory pressure, and consolidate scattered model stores — without assuming one vendor or one machine shape.

## The problem

Ollama silently ends up with multiple disconnected model libraries on most boxes:

- `~/.ollama/models` — where `ollama serve` from your shell defaults
- `/usr/share/ollama/.ollama/models` — where `ollama.service` (systemd, as user `ollama`) defaults
- `/srv/ollama/models`, `/var/lib/ollama/...` — wherever `OLLAMA_MODELS=` was set in `/etc/default/ollama` or a systemd drop-in
- Whatever your container, agent runtime, or eval harness happens to set when it spawns `ollama serve`

The result: `ollama list` on port 11434 shows a different set of models than `ollama list` on port 11436. Disk fills up with duplicate blobs. Cold-load thrashes the wrong drive. The systemd daemon "runs fine" but serves nothing because all the manifests live in your home directory.

## What it does

`ollama-unify` first classifies the operating system, architecture, effective RAM, cgroup/container boundary, CPU, service manager, installed model payloads, recent Ollama memory projections, and every accelerator backend it can interrogate. It then chooses CUDA, ROCm, Vulkan, Metal, or CPU according to the capabilities actually present and derives scheduler, device-memory, and host-memory limits from those measurements.

The model-store workflow remains independent: it scans every place Ollama stores models, shows exactly what is where, and interactively migrates everything into one canonical location. On compatible systemd GPU hosts, the classified policy also installs a local lease negotiator in front of Ollama. It drains and unloads Ollama before a cooperative external CUDA workload loads, then lets Ollama refit against the VRAM that actually remains. The containment policy refuses unsafe starts, gives Ollama low CPU/I/O priority, asks `systemd-oomd` to kill it under sustained memory pressure, and leaves it stopped after a failure instead of replaying a destructive request loop.

```
  ___  _ _                                       _  __
 / _ \| | | __ _ _ __ ___   __ _    _   _ _ __ (_)/ _|_   _
| | | | | |/ _` | '_ ` _ \ / _` |  | | | | '_ \| | |_| | | |
| |_| | | | (_| | | | | | | (_| |  | |_| | | | | |  _| |_| |
 \___/|_|_|\__,_|_| |_| |_|\__,_|   \__,_|_| |_|_|_|  \__, |
                                                      |___/

Scanning for ollama model stores…

  #   Path                                          Size     Filesystem             Manifests Blobs     Owner
  ──  ────                                          ────     ──────────             ───────── ─────     ─────
  [1] /home/you/.ollama/models                      183G     /dev/nvme0n1p2 (/)     37        74        you:you
  [2] /srv/ollama/models                            73G      /dev/nvme1n1p1 (/srv)  0         12        ollama:ollama

Detected daemons:
  • ollama.service active (PID 696536)
  • 7 manual 'ollama serve' processes (PIDs: 100362, 422930, ...)

Host classification
  Platform: Ubuntu 24.04 LTS (Linux/x86_64; none)
  CPU: 16-Core Workstation CPU — 32 logical cores
  Memory: 65536 MiB (/proc/meminfo)
  Class: workstation; service manager: systemd 255

Accelerator classification
  [cuda/dedicated] GPU 0: NVIDIA 24GB Accelerator, 24576 MiB, compute 8.6, GPU-… (dedicated)
  [cuda/constrained] GPU 1: NVIDIA 2GB Display Adapter, 2048 MiB, compute 6.1 — below 4096 MiB safety floor

Selected Ollama safety policy
  Backend: cuda (dedicated) — highest-confidence native NVIDIA backend
  Device memory: live free-VRAM telemetry; no guessed fixed carve-out
  Model scan: largest installed inference payload 14336 MiB (/srv/ollama/models/manifests/...)
  Ollama history: largest observed host projection 6144 MiB
  Host memory: 2048 MiB startup headroom; throttle at 6144 MiB; hard cap at 14336 MiB; 51200 MiB remains outside the cgroup
  GPU policy: maximum safe GPU layers; spread=0; pageable/cgroup-bounded CPU overflow only
  GPU host paths: unified spill and pinned-host buffers disabled
  Scheduler: 1 model, 1 parallel request, 8192-token context, queue 64

Available mount points for unified storage:
  Filesystem      Size  Used Avail Use% Mounted on
  /dev/nvme0n1p2 1.8T  1.5T  232G  87% /
  /dev/nvme1n1p1 1.8T  100G  1.7T   6% /srv

Where to unify all models? [/srv/ollama/models]: ↵

Update ollama.service via drop-in to use OLLAMA_MODELS=/srv/ollama/models? [Y]: ↵
Apply this classified backend and OOM safety policy to ollama.service? [Y]: ↵
Service runs as 'ollama'. Change to 'you' for single-user simplification? [Y]: ↵
Replace each original store path with a symlink to /srv/ollama/models? [Y]: ↵
Add 'export OLLAMA_MODELS=/srv/ollama/models' to /home/you/.bashrc? [Y]: ↵

Plan
  Destination: /srv/ollama/models
    /home/you/.ollama/models  →  /srv/ollama/models   (183G, strategy: rsync)
    /srv/ollama/models        →  (orphan blobs, archiving)
  Total to move: 183G
  • Point ollama.service at the unified model store
  • Install classified backend/device/host-memory OOM guardrails
  • Change service User/Group to you
  • Symlink originals → destination
  • Add OLLAMA_MODELS export to /home/you/.bashrc
  • Stop running daemons, perform transfer, restart ollama.service

Proceed? [N]: y
```

## Features

- **Discovery** — finds every store referenced by shell env, `/etc/default/ollama`, `/etc/environment`, systemd unit + drop-ins, and live `ollama runner` cmdlines
- **Interactive destination picker** — shows free space per mount, suggests the best candidate
- **Host classifier** — reports OS/architecture, CPU/core count, physical and effective RAM, container/VM status, service manager, and memory-risk tier
- **Backend classifier** — inventories and selects CUDA, ROCm, Vulkan, Metal, or CPU without embedding host-specific UUIDs, GPU counts, or RAM totals
- **Hardware-aware transfer** — picks the fastest supported method automatically:
  - **same filesystem** → `mv` (instant, just a rename)
  - **reflink-capable** (btrfs / XFS / ZFS) → `cp --reflink=auto` (instant CoW)
  - **cross-drive** → local-copy-tuned `rsync`, enabling optimization flags only when that installed rsync supports them
- **Content-addressed dedup** — blobs are SHA-256-named, so `rsync --ignore-existing` collapses duplicates across stores for free
- **Systemd integration** — installs a drop-in to set `OLLAMA_MODELS` and (optionally) change the service `User/Group` to your user for cleaner single-user setups
- **Role-aware accelerator selection** — prefers dedicated devices; display/shared GPUs remain usable when they are the only eligible accelerator
- **Generic capability floors** — classifies constrained or legacy devices from memory and compute telemetry rather than special-casing product names
- **Backend isolation and preflight** — emits the correct visibility variables for the selected backend and validates CUDA, ROCm, or Vulkan telemetry before systemd starts Ollama
- **GPU-first bounded-overflow mode** — delegates placement to Ollama's live-VRAM scheduler and fits the maximum safe number of layers into available VRAM without forcing every selected device into every load; overflow uses ordinary pageable memory inside the service cgroup, never CUDA unified-memory spill, registered host mappings, pinned-host buffers, or swap
- **Live device capacity** — lets Ollama schedule against current free-VRAM telemetry instead of subtracting a guessed percentage; an explicit `OLLAMA_SAFE_VRAM_RESERVE_MIB` remains available when another workload needs a fixed carve-out
- **Dynamic external-GPU negotiation** — installs a streaming API proxy and local lease broker; cooperative workloads load first, after which Ollama automatically refits into the remaining VRAM and moves the unmatched layers to cgroup-bounded pageable RAM
- **Agent and Docker auto-discovery** — registers `docker gpu` as a Docker CLI plugin, publishes a machine-readable discovery manifest and well-known HTTP endpoint, and adds an idempotent managed CUDA policy to the invoking user's existing global Codex instructions
- **Unregistered CUDA quarantine** — unknown CUDA processes block new placement and all broker-controlled weight eviction on their GPUs. Resident lanes remain pinned until those processes exit; unrelated GPUs remain available. Acquire a scoped lease before CUDA initialization.
- **Upgrade-safe pinning** — detects when an Ollama upgrade clears the pinned loopback backend and would drop the daemon back onto the negotiator's public port; `--check-update`, `--reconcile`, and `--update-ollama` repair the drift and cycle the proxy and backend in the only safe order, and an installed path-unit watchdog does the same for upgrades taken outside this script
- **Measured host OOM containment** — sets `MemoryHigh` from the largest recent Ollama host-memory projection and `MemoryMax` from the larger of that projection or the largest installed inference payload; the unallocated host RAM is the result, not a target selected by the script
- **Pressure-aware fail-closed startup** — installs a systemd condition that skips startup without marking the unit failed when `MemAvailable` cannot cover the empty API daemon's bounded startup headroom or memory PSI is already unsafe; model lanes retain live host-memory admission and cgroup containment
- **Proactive pressure killing** — configures `systemd-oomd` to kill Ollama at sustained memory pressure or swap exhaustion before the host becomes unusable
- **CPU and storage protection** — caps Ollama at four logical CPU cores by default and gives its CPU and I/O cgroups low weight, keeping interactive work responsive during cold loads
- **Bounded scheduling** — defaults to one loaded model and one parallel request per process, admits managed models at their verified maximum context against memory budgets, derives the base-backend default and queue limits from effective RAM, and enables Flash Attention plus `q8_0` KV cache
- **Backward-compat symlinks** — replaces each original path with a symlink to the unified store, so hardcoded references in agents, scripts, and `OLLAMA_MODELS=` overrides keep working
- **Shell rc update** — appends `export OLLAMA_MODELS=…` to your `.bashrc` / `.zshrc` / `config.fish` so new shells use the canonical path natively
- **Safety first** — never deletes data. Originals are renamed to `.bak` (or `.orphan-blobs` for stores with no manifests). You reclaim the space manually when you've confirmed everything works.

CUDA admission also checks the NVIDIA driver's `gpu_recovery_action` before granting leases, preparing growth, starting managed lanes, or admitting inference (including reuse of resident lanes). A selected GPU requiring reset or drain blocks new work; a reported node reboot blocks work even if that GPU is outside the selected set. Missing selected devices and failed recovery queries also block admission. Discovery and status expose `gpu_health` and coordination warnings. Confirmed recovery requirements return HTTP 503 with `gpu_recovery_required` and `retryable: false`; unavailable telemetry returns `gpu_health_unavailable`. Existing owners can still heartbeat and release leases, and model-unload requests remain available. The broker never resets GPUs or reboots the host automatically. Drivers predating the recovery query retain their previous admission behavior and report that this protection is unavailable. This check contains an already reported driver fault; it cannot prevent the first hardware failure or repair an NVLink bridge.

## Quick start

```bash
# inspect first (read-only — no daemons stopped, no changes made until you confirm)
curl -sSL https://raw.githubusercontent.com/robit-man/ollama-unify/main/ollama-unify.sh -o ollama-unify.sh
chmod +x ollama-unify.sh
./ollama-unify.sh
```

Or clone:

```bash
git clone https://github.com/robit-man/ollama-unify.git
cd ollama-unify
./ollama-unify.sh
```

Classify a machine without requiring migration tools, sudo, a running Ollama service, or any changes:

```bash
./ollama-unify.sh --classify
```

Preview the generated backend and OOM policy:

```bash
./ollama-unify.sh --safety-preview
```

Install or refresh only the systemd safety policy without scanning or migrating model stores:

```bash
./ollama-unify.sh --install-safety
```

On a dedicated CUDA/ROCm systemd host, that command also installs and boot-enables `ollama-unify-negotiator.service` and the `ollama-unify-gpu-lease` client. Ollama moves to the loopback backend on port `11436`; the negotiator preserves the previously configured public address on port `11434`.

### Dynamic GPU leases

Deployment agents can discover the broker without knowing this repository:

```bash
docker --help                  # lists: gpu*  Negotiate CUDA VRAM with Ollama
docker gpu discover           # machine-readable host policy and selected GPUs
docker gpu status             # live leases, models, VRAM, and cgroup RAM
```

For a long-running Docker ASR/TTS stack, launch it through the broker and use a readiness check that succeeds only after its CUDA models are resident:

```bash
docker gpu run \
  --owner asr-tts \
  --justification 'serve the shared ASR and TTS endpoints' \
  --expected-duration 14400 \
  --vram-mib 8192 \
  --gpu GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee \
  --ready-command 'curl -fsS http://127.0.0.1:8080/health/ready' \
  -- docker compose up
```

`--gpu` creates a scope that is exclusive against other external leases. While that reservation is pending or revoking, broker-owned Ollama lanes can continue only on other selected GPUs. After the workload calls `ready`, its CUDA footprint is stable, so the broker can place managed Ollama lanes in measured free VRAM on a single scoped GPU with the configured reserve margin. A scope with two or more GPUs is exclusive for its whole lifetime: the broker retires Ollama lanes on those GPUs at acquire and never places, loads, or reclaims lanes there until release. Multi-GPU owners (tensor parallel, NCCL, CUDA peer access) drive peer-to-peer traffic across NVLink/PCIe, and starting or killing a colocated Ollama lane during that traffic preceded a fatal NVLink Xid 74 and a host lockup that required a hard reboot. The workload must call `prepare` before later VRAM growth; that transition drains the affected capacity first. The wrapper gives the child exactly the scoped UUIDs through `CUDA_VISIBLE_DEVICES`, heartbeats its lease, and releases it after the foreground command stops. The declared MiB value validates that the selected devices have enough live free memory; it is not an enforceable fractional VRAM carve-out. Without an explicit GPU scope, the broker retains the conservative host-wide drain for compatibility because it cannot prove where the external process will allocate.

For workloads managed by another supervisor, use the explicit lifecycle:

1. `token=$(docker gpu acquire --owner <name> --justification '<purpose>' --expected-duration <seconds> --gpu <uuid> --token-only)` — reserves the specified whole GPU. Add `--gpu` again for each additional device.
2. Start the external workload with `CUDA_VISIBLE_DEVICES` set to exactly those UUIDs, then wait until its CUDA models are fully loaded.
3. `docker gpu ready <token>` — marks the external allocation active and stable. Single-GPU scopes become eligible for live-free-VRAM Ollama placement; multi-GPU scopes stay exclusive; legacy unscoped leases reopen Ollama here.
4. Before increasing the workload's VRAM use, call `prepare <token>`, resize it, then call `ready <token>` again.
5. Stop the external workload, ensuring its CUDA allocation is gone, then call `release <token>` so Ollama can reload and expand.

Lease registration requires a specific owner. A meaningful purpose
(`--justification`) and expected duration (`--expected-duration`, in seconds)
are recommended when setting up workloads, but both are optional. Basic
acquisition works with `docker gpu acquire --owner <name> --gpu <uuid>`;
`docker gpu run` also accepts omitted metadata while still requiring readiness.

Discovery advertises required and recommended fields separately. Acquisition
returns non-blocking recommendations in `public_lease.recommendations`, and
the CLI prints them on stderr without changing its exit status or token-only
stdout. Unknown purpose and release time stay unknown, including for existing
leases. Expected duration is coordination metadata, separate from heartbeat TTL
and pending deadlines; omission does not disable expiry or cleanup.

`docker gpu discover`, HTTP discovery, and `docker gpu status` publish token-free
summaries of owner, purpose, GPU UUIDs, and expected release time, with a
coordination warning when leases exist. Agents must inspect those summaries
before acquiring. Scope conflicts, VRAM capacity, and GPU health checks still
block unsafe admission. Conflicting acquisition errors identify the current
lessee and horizon.

Lease heartbeat clients tolerate a bounded negotiator service restart. They
retry the same persisted token and never acquire a replacement scope or change
the assigned GPU UUIDs. The reconnect window defaults to 90 seconds and is
also capped at half of the lease TTL. Set
`OLLAMA_SAFE_HEARTBEAT_RECONNECT_GRACE` when installing to tune the ceiling.
An explicit broker rejection still fails immediately. If the broker remains
unavailable past the bounded window, `docker gpu run` terminates its child so
the external CUDA allocation fails closed.

To preserve a running workload while upgrading a legacy host-wide lease, use `docker gpu scope <token> --gpu <uuid> [--gpu <uuid> ...]`. The broker accepts the live transition only when all foreign CUDA growth since acquire is contained in the requested external-owner-exclusive scope and the scope can satisfy the original reservation. It then reopens measured free VRAM for broker-owned lanes without stopping the external process. Foreign CUDA on an unreserved GPU quarantines it without evicting resident weights.

Use `docker gpu status` to see leases, drain state, loaded Ollama models, foreign CUDA processes, per-GPU memory, and Ollama cgroup memory. The original `ollama-unify-gpu-lease` command remains available when Docker CLI discovery is not applicable. `num_gpu` in the Ollama API means GPU-offloaded model layers—not the number of physical GPUs. The script keeps every selected accelerator visible; on a three-A100 host Ollama may dynamically use one, two, or all three.

### Fixed GPU groups and flexible services

An enabled GPU can still contain another workload. A lease records its owner,
current placement and reserved capacity; it does not disable that card.
The default `--placement fixed` reserves the exact `--gpu` set. Use it for a
workload requiring a particular NVLink pair or another hardware constraint.
It is never silently moved to different GPUs.

A restartable service can explicitly opt into `--placement flexible`. Its
repeated `--gpu` options are ordered **candidates**, and the broker assigns
exactly one of them. The assigned UUID is returned in `lease.gpu_uuids` and is
the only GPU made visible to its workers. Prefer the standalone GPU first to
keep a bridged pair available:

```bash
docker gpu run --owner voice-service --placement flexible --priority 0 \
  --gpu GPU-STANDALONE --gpu GPU-PAIR-A --gpu GPU-PAIR-B \
  --vram-mib 45000 \
  --drain-command '/srv/voice/gpu-control drain' \
  --resume-command '/srv/voice/gpu-control resume' \
  --ready-command '/srv/voice/gpu-control ready' \
  -- /srv/voice/start-foreground

# A separately started trainer requires this exact pair.
docker gpu run --owner trainer --priority 100 \
  --gpu GPU-PAIR-A --gpu GPU-PAIR-B --vram-mib 100000 \
  --ready-command '/srv/trainer/ready' -- /srv/trainer/start-foreground
```

Priorities range from -1000 to 1000, default 0. A strictly higher-priority
request may ask a flexible owner to relocate; equal-priority, fixed and
non-cooperative legacy owners keep their reservations. The broker first
reserves an eligible destination with verified capacity. The flexible
supervisor stops new application work, finishes active requests through its
drain hook, gracefully stops its own workers, and waits for their exit. The
broker independently verifies that the source CUDA allocation has disappeared
before changing the same lease to the reserved destination. Only after the
workers restart there and pass readiness can the pinned requester proceed.
No live CUDA context is retargeted, and relocation never forcibly kills a worker.

The drain hook must prevent new requests and report success only when current
work is safe to stop. A busy hook can return nonzero and be polled again. The
resume hook reopens admission if the requesting client disconnects or times
out before restart. Readiness must verify resident workers and reopen normal
service admission after a successful restart. The child receives
`OLLAMA_UNIFY_GPU_LEASE_SUPERVISED=1` and `OLLAMA_UNIFY_GPU_LEASE`; it must not
acquire, heartbeat or release a second lease owned by the wrapper.

Independent supervisors can use flexible `acquire`, inspect their heartbeat's
`relocation` field, drain their workers, then call
`docker gpu relocate TOKEN --handoff-id ID`. They must use the returned exact
UUID and call `ready` after residency. Flexible scopes cannot be widened or
retargeted through `scope`. Destinations and waiting intent are public without
lease tokens. A broker restart cancels waiting intent, while committed lease
placement and priority remain persisted. The existing heartbeat reconnect
window preserves live owners during broker updates.
If cancellation arrives during graceful stop, the supervisor reloads its
original, still-reserved GPU through `prepare` and the normal health checks.

If there is no safe destination, an owner is busy, or the bounded handoff
expires, the broker does not grant overlapping ownership. A higher priority
does not create VRAM or override a fixed owner's hardware requirements.
Multi-GPU groups remain exclusive throughout peer traffic; another service
cannot trigger model load/unload on one member while that group is active.

### Opt-in lower-priority cache evacuation

An operator can declare broker-owned **idle singleton caches** movable and give
an ordinary external lease request a higher priority. Undeclared caches are
protected. This opt-in path copies and verifies each cache at an eligible
destination before retiring its original process group; it does not merely
unload lower-priority weights. Without these options, existing acquisition
behavior is unchanged.

```bash
# Inspect owners, health, and selected UUIDs before requesting CUDA capacity.
docker gpu discover

# Declare this exact model movable to these selected GPUs. Hard model GPU
# policies still apply; the declaration cannot widen them.
docker gpu set-cache-policy MODEL:TAG --movable --priority 10 \
  --gpu GPU-DESTINATION-A --gpu GPU-DESTINATION-B

# Reserve a different target for a higher-priority external workload.
docker gpu acquire --owner voice-service \
  --justification 'run the scoped voice service' --expected-duration 3600 \
  --vram-mib 8192 --gpu GPU-TARGET --priority 20 \
  --evacuation-id voice-start-001
```

The response has `lease.state: "pending"` and a token. It is **not** a ready
external workload. Initialize the child with exactly the returned GPU UUIDs,
then call `ready TOKEN`, heartbeat, and release only after its CUDA allocation
exits. `docker gpu run` accepts the same `--priority` and `--evacuation-id`
options and retains its normal readiness/heartbeat/child-exit lifecycle.
Before growing an active allocation, use
`docker gpu prepare TOKEN --priority 20 --evacuation-id voice-grow-002`, grow it,
then call `ready TOKEN` again. A failed evacuation does not revoke the existing
active lease. An evacuation ID explicitly requests copy-first managed-cache
evacuation; an ordinary acquisition priority only participates in cooperative
external-lease relocation and does not authorize cache eviction by itself.

Priorities are integers from -1000 to 1000; only strictly lower-priority caches
may move. The broker drains admitted work to completion, preserves queued and
retained request identities/bodies, and refuses to move a cache with queued
demand. No admitted inference is replayed or cancelled for this feature. A
moved cache has a new lane/runtime generation; an old warm certificate never
authorizes the replacement. Clients must obtain a fresh proof for future work.

Planning uses live physical memory and reservation accounting. It may first
move a different eligible cache to consolidate room, with at most eight caches
and a bounded search. Each destination is rechecked before loading, uses the
same model digest/context, and must report full GPU allocation and exact
process-group placement. There must be one spare managed server slot for the
copy. Existing peer groups are indivisible and cannot be repartitioned or
packed into a singleton by evacuation. External leases, quarantined GPUs,
health faults, hard GPU policies, uncertain process exit, insufficient room,
and protected caches produce a refusal; there is no CPU or alternate-GPU
fallback. All source/destination promises remain charged until their whole
process groups exit.

`docker gpu status` exposes token-free `cache_residency.policies` and
`cache_residency.operations`, including source/destination identities and the
transaction phase. Policy and operation state is stored in the private,
fsynced `cache-residency.json` next to the lease state (override with
`OLLAMA_UNIFY_CACHE_RESIDENCY_STATE`). Evacuation children wait on a private
startup pipe until their PID/start/process group has been durably recorded;
parent failure before that acknowledgement exits the wrapper before native
Ollama can load CUDA. A stable evacuation ID binds the exact request and priority. Repeating a successful request returns its existing
linked lease; changing its body is rejected. A failed or recovered operation
requires a new ID for a new request, so a lost response never silently starts
another move.

Use `docker gpu cancel-evacuation ID` to request rollback before grant. Rollback
warms the original cache before retiring its destination. If restoration or
process exit cannot be proven, status remains `restoration_pending` and the
affected scopes remain blocked. An interrupted broker reloads that obligation
as `recovery_pending`; it never adopts or kills an old PID or replays a request.
After **all recorded process generations have exited**, an operator can call
`docker gpu recover-evacuation ID` to restore the archived cache identities
under the current hard policies. Recovery grants no new lease. If policy has
changed, recovery refuses; cancel the obsolete obligation only after its
recorded processes have exited. Ordinary lease heartbeat/release semantics
still apply to any already persisted grant.

The journal holds up to 256 operations. After settlement and release of its
linked lease, `docker gpu archive-evacuation ID` returns the token-free record
and removes it from the live journal. Save that returned record for audit;
never reuse an archived operation ID. Remove a movable declaration with
`docker gpu set-cache-policy MODEL:TAG --priority 10` (without `--movable`).

### Broker-owned parallel Ollama lanes

Local clients that need concurrent model processes can ask the public broker to create capacity before starting their work:

```bash
curl -fsS http://127.0.0.1:11434/.well-known/ollama-unify-gpu-negotiator/capacity \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:35b","parallel":3,"gpu_uuids":["GPU-uuid"]}'
```

An embedding client can state its endpoint explicitly when it prewarms a lane by adding `"endpoint":"/api/embed"`. The broker validates this field. Embedding warm loads use `/api/embed` with `input:[]`, which loads the native runner without computing a vector; native residency and physical placement still must pass before readiness. This needs a native version that schedules a runner before returning empty embeddings (verified in stock 0.4.0 and 0.35.0). Stock 0.3.0 returns before loading, so the broker refuses missing residency instead of falling back to embedding computation. It also infers embedding-only and reranking-only warm-up contracts from local model capabilities when the field is absent. Lazy capacity for an ordinary inference request always uses that request's endpoint family. Discovery metadata publishes the reserved private port range so cooperating clients never start a competing Ollama process on a broker lane.

`gpu_uuids` is an optional ordered hard allowlist. When present, the broker
reuses and creates lanes only on the ordered intersection of that list and its
live selected GPUs; it never falls back to another GPU. This lets long-lived
clients survive a selected-device removal without weakening their constraint.
The first available UUID is preferred. Ordinary inference requests use the
equivalent `X-Ollama-Unify-GPU-UUIDs` comma-separated header, so lazy lane
creation obeys the same constraint. An empty allowlist or one with no available
member is rejected. Omitting the field/header retains automatic placement.

The broker looks up the installed model size and capabilities through `/api/tags`, adds configurable model and VRAM margins, checks live free memory, reserves host headroom, and admits the request only if every missing lane fits. Models that fit one allowed GPU retain a single-GPU `ollama serve` process and private loopback port. Placement spreads these lanes across available GPUs first, then co-locates additional processes only while the conservative live-VRAM budget still fits. A lane's VRAM promise remains committed until its complete process group exits, including surviving runner descendants. The broker loads the requested model through its native completion, embedding, or reranking endpoint and verifies residency before it marks the lane ready. Clients continue sending inference to public port `11434`; private lanes cannot be selected or bypassed. An ordinary inference request lazily creates capacity when needed. Model tags that omit `:latest` share the same lane identity; separately named aliases remain distinct.

Aggregate artifact size remains the default weight estimate. A known omni sidecar bundle can be excluded only when the local manifest digest, every layer's size and identity, and native `/api/show` model/projector/adapter blob paths agree exactly. This follows the stock [Ollama loader and Modelfile serialization](https://github.com/ollama/ollama/blob/v0.35.0/server/images.go): the custom bundle is not a runtime model, projector or adapter layer. Unknown or draft layers, missing local files, unavailable metadata, symlinked layer or manifest files, or changed file/directory identities retain the full aggregate estimate. Model overhead, operator context floors, parallel context reserves and GPU headroom remain unchanged. This changes estimates for new admissions; it does not shrink a resident lane's committed reservation.

When the model estimate exceeds every allowed GPU's physical capacity but fits their combined capacity, the broker can create one lane spanning an ordered GPU group. The entire capacity of every member is reserved exclusively before model loading. Every intersecting idle singleton lane is drained first; active lanes, external leases, unregistered CUDA contexts, or unavailable telemetry defer placement with a retryable error. A model larger than the allowed aggregate physical capacity receives a permanent capacity error. The child receives exactly the group's UUIDs through `CUDA_VISIBLE_DEVICES`, with `OLLAMA_SCHED_SPREAD=1`; Ollama chooses layer distribution from device memory. The broker does not assume equally sized weight shards. `num_gpu=-1` still controls offloaded layers, not the number of physical devices.

Managed lanes default to full GPU admission. A client can explicitly permit pageable CPU model layers with `cpu_offload:true` on capacity requests and `X-Ollama-Unify-CPU-Offload: allow` on inference requests (`deny` or no header keeps the default). Discovery advertises `cpu_offload_capacity_field` and `cpu_offload_header`. Full GPU placement remains preferred; offload is considered only when the exact model/context estimate exceeds the physical capacity of the entire allowed GPU scope. It never widens client/model/request GPU policies or replaces an unavailable GPU with CPU-only inference. Every allowed GPU is reserved exclusively, including a one-GPU scope, and native loading must show positive VRAM on exactly that scope. Ollama chooses automatic layer fitting; clients must not use `num_gpu` as a device count.

CPU offload conservatively reserves the entire model/context estimate plus per-lane host workspace, rather than guessing a favorable GPU/CPU split. Admission checks physical `MemAvailable`, the managed broker process's actual cgroup-v2 `memory.high`/`memory.max` headroom through every ancestor, and outstanding lane promises. Unlimited cgroups remain bounded by physical host availability; unavailable telemetry defers admission. Overflow uses ordinary pageable RAM with pinned-host registration and CUDA unified-memory spill disabled. This deliberately rejects some workloads that a tighter measured split might fit. Busy offload slots queue subsequent requests; the default mode cannot reuse an offload lane. Logical retries/resumes preserve the original permission, and full-GPU warm certificates cannot certify CPU-offloaded lanes.

To request a group even when the real model and context fit one GPU, explicitly set `placement:"exclusive_group"` on the capacity API. `gpu_uuids` must then be an ordered array of at least two distinct broker-selected GPUs, and every member must satisfy the model's hard GPU policy. No unavailable or excluded member is removed, substituted, or treated as an optional preference. The whole ordered group is used; a singleton, subgroup, or differently ordered group cannot satisfy this request. One exact group supports at most the configured `instance_parallel` slots. Missing or `"auto"` placement retains the ordinary fitting-singleton/oversized-group behavior; unsupported placement values fail explicitly.

```sh
# After discovery and authorized hardware setup, request only the designated UUIDs.
curl -fsS http://127.0.0.1:11434/.well-known/ollama-unify-gpu-negotiator/capacity \
  -H 'Content-Type: application/json' \
  -d '{"model":"robit/qwen3.8-27b-obliterated-e03:27b","parallel":1,"endpoint":"/api/chat","placement":"exclusive_group","gpu_uuids":["GPU-FIRST-DESIGNATED-UUID","GPU-SECOND-DESIGNATED-UUID"]}'
```

This preserves the exact installed model digest, genuine memory estimate, and verified context. It reserves every member's full physical capacity and uses the existing whole-process stop and peer-safety rules. Readiness requires native full GPU residency plus observed positive process-group VRAM on every requested GPU and no other GPU. If Ollama does not actually distribute the model across the complete group, capacity fails; this API never manufactures a tensor split from model-size hints or CPU fallback. It does not prove that a previously faulting GPU or interconnect is healthy. Actual designated-GPU inspection remains mandatory before token generation.

Use the same `placement:"exclusive_group"` and exact `gpu_uuids` with `warm_admission_proof:true` to inspect this group without creating capacity or refreshing idle lifetime. Proof issuance and certified inference recheck current actual per-member residency. The existing warm admission certificate selects only the original certified lane generation; a later singleton or replacement group never joins it. Ordinary inference without a warm certificate still follows the existing allowlist contract, so clients requiring the verified group must use that certificate and its unchanged ordered GPU header.

A grouped lane becomes ready only after native model residency reports full GPU allocation and NVIDIA process telemetry confirms nonzero allocations by its process group on every reserved GPU, with none outside the scope. Capacity and status expose `gpu_uuids`, full-card `reserved_mib_by_gpu`, and measured `observed_vram_mib_by_gpu`; the legacy `gpu_uuid` remains the first member. A request or operator policy must allow every group member to reuse it. An idle group with no queued demand may be retired as a whole to make room for a different model, including a small embedding model. No member becomes available until the complete process group exits; active, loading, externally leased, or quarantined groups remain protected. A scoped external lease touching even one member drains the whole group before acquisition. Cancellation, explicit unload, policy changes, and retirement retain every reservation until the whole process group is confirmed gone.

Before the first native model-load request, including a zero-token load, the broker reopens the spawned server's actual PID/start/process-group identity and `/proc` CUDA scope. Its `CUDA_VISIBLE_DEVICES` must exactly match the ordered reserved UUIDs, and the process generation must remain unchanged across that read. Missing, duplicate, unreadable, changed, or foreign scope fails before any model POST; the existing whole-process cleanup and reservation rules still apply. Full environment values are never included in this failure diagnostic. Actual GPU residency is independently checked after loading and before certified inference.

Managed process teardown holds the lease transition lock through its final safety check and process exit. Pending and revoking scopes, active external multi-GPU scopes, and all live unscoped leases defer teardown. Unscoped leases retain the host-wide drain while active and exclude other external owners. Releasing a lease verifies that external CUDA allocations have settled before removing coordination state; it does not evict models alongside the releasing owner. A cancellation worker that finds a transition in progress keeps its exact active request and reservations and retries without blocking on the transition lock.

Grouped children discard inherited `GGML_CUDA_P2P` and `GGML_CUDA_ENABLE_UNIFIED_MEMORY` and retain `GGML_CUDA_NO_PINNED=1`, using Ollama's normal layer split and host-staged transfers. A value of `0` is not used to disable a presence-based GGML flag. These controls preserve quarantine and driver recovery checks; they do not establish that a previously faulting GPU, NVLink, PCIe, or IOMMU path is safe. CPU fixtures verify scheduling and lifecycle behavior, while hardware validation still requires an authorized, healthy, exclusively reserved scope.

Managed completion lanes default to the exact installed model's verified maximum context. The old RAM-derived `OLLAMA_UNIFY_MAX_CONTEXT` remains a base-backend default and does not silently cap a managed model at 8192. Before admission, the broker binds `/api/show` context and attention metadata to the exact `/api/tags` SHA-256 digest. It reserves model overhead, VRAM headroom, each parallel slot's KV cache, workspace, and recurrent state for supported hybrid models. KV accounting uses `q8_0` block sizes, or `f16` when the model cannot use quantized Flash Attention. Qwen3.5 full-attention intervals are accounted separately from recurrent layers. Missing memory metadata produces `model_context_memory_unverified`; it does not authorize an unbudgeted large context. Insufficient live capacity waits for resources, while an estimate exceeding the allowed aggregate physical capacity fails explicitly. Context is never silently reduced to make placement succeed.

Existing digest-bound `OLLAMA_UNIFY_MODEL_CONTEXT_PROFILES` supply a verified memory floor; available geometry always contributes its own estimate. An unchanged exact profile lacking geometry can retain the original conservative operator allowance of at least one MiB per 16 context tokens, with that provenance reported explicitly. Automatic mode upgrades a legacy smaller context to the model maximum and adds the required memory allowance; it cannot use this fallback to grow an unknown layout. A deliberate operator ceiling uses `OLLAMA_UNIFY_CONTEXT_HARD_LIMIT` (installer override `OLLAMA_SAFE_CONTEXT_HARD_LIMIT`); discovery reports that reduction as `operator_hard_context_limit` alongside the model maximum. `OLLAMA_UNIFY_AUTO_MODEL_CONTEXT=0` (installer override `OLLAMA_SAFE_AUTO_MODEL_CONTEXT=0`) retains the older fixed-profile/default behavior for compatibility. Live discovery accepts `?model=<exact-tag>` to resolve metadata without loading the model, and publishes `model_context_resolutions`, effective digest-bound profiles, the hard ceiling, and whether profiles are memory floors.

Warm-up and native inference use the same admitted context. The lane is ready only after `/api/ps` confirms the exact digest and per-slot context, and reported VRAM must fit its reservation. A resident lane keeps its original profile: a changed artifact or context policy excludes it from new requests, waits for active work, and retires the complete idle process group before replacement. OpenAI-compatible requests use the lane's admitted server context; Ollama's OpenAI request schema cannot override a model's explicit `num_ctx`. If that model parameter differs from the admitted context, the broker returns `model_context_openai_mismatch` and directs the caller to the native API or an aligned exact-model wrapper. It does not claim that an ignored JSON option changed the runtime context.

Inference uses one broker-owned queue and one capacity reconciler. FIFO order is strict within each model. The broker queue has a hard configurable ceiling (`OLLAMA_UNIFY_POOL_MAX_QUEUE`, default 64); excess callers receive an immediate retryable 503 instead of expanding memory and admission work without bound. A request for another model can use its own available warm lane while an older model waits. During a GPU-capacity retry window, later cold work may also use free GPUs outside every older request's effective client/model/request scope. This placement preserves the original FIFO ticket and allowlist, keeps server slots for older models, and cannot reclaim a resident lane or enable CPU offload. The older head gets the first capacity retry when its deadline arrives or a blocked lease clears. Speculative startup and HTTP operations share a five-second budget; failed attempts back off for 30 seconds. Cleanup keeps the existing guarded process teardown and surviving reservations, while foreground loads retain their normal deadlines. The default process ceiling is twice the selected GPU count, while every new lane remains hard-gated by current free VRAM and host-memory reserves. Multiple managed processes can therefore share one GPU when they fit. When every selected GPU is held by a pending or revoking scoped lease, the reconciler parks on lease-state changes instead of retrying an impossible lane placement every two seconds. When a new model does not fit, the broker reclaims the least-recently-used idle lane that has no queued demand, without disturbing active or demanded lanes. Disconnected waiters are removed before they can create phantom inference. Permanent admission failures are returned immediately and removed from the queue, so one incompatible or missing model cannot starve later requests. Runtime status and the well-known endpoint report queue depth, ceiling, age, per-model demand, phases, logical resumes, stale removals, duplicate attempts, and aggregate admission/cancellation/failure timing. Responses return `X-Ollama-Unify-Lane`, `X-Ollama-Unify-Request-Id`, exact admission-only `X-Ollama-Unify-Queue-Ms`, current `X-Ollama-Unify-Queue-Position`, and opaque monotonic `X-Ollama-Unify-Queue-Ticket` headers. The ticket proves order without exposing prompt content. Idle lanes are unloaded and stopped automatically, including embedding-only models that reject the generation endpoint. Stable new or increased foreign CUDA allocations affect only broker lanes on selected GPUs; process departure and unselected-GPU churn do not destroy the pool. A scoped cooperative lease temporarily drains the pool while its reservation is committed. Pending and revoking scopes block their GPUs. Active single-GPU scopes admit Ollama lanes against measured free VRAM because later external growth requires a new `prepare` transition. Active multi-GPU scopes stay blocked so lane churn never runs concurrently with the owner's peer-to-peer traffic. A legacy unscoped lease keeps the host-wide drain. Explicit release compares aggregate per-GPU foreign use with the pre-lease baseline and allows a small telemetry tolerance (`OLLAMA_UNIFY_FOREIGN_RELEASE_TOLERANCE_MIB`, default 256 MiB) so harmless driver/runtime accounting drift cannot pin an otherwise released scope forever. Broker shutdown also terminates its complete child process groups, and lanes are intentionally ephemeral across restart.

Every admitted backend call is also an exact, renewable lifecycle lease. The
broker renews it when the backend connection is established, response headers
arrive, and each response chunk arrives. A silent backend expires after
`OLLAMA_UNIFY_REQUEST_ACTIVITY_TTL` (default: the broker drain timeout, 300
seconds). A disconnected non-logical request is cancelled immediately. A
logical request may continue for `OLLAMA_UNIFY_REQUEST_DETACHED_TTL` seconds
(default 30), renewed by backend progress, so a quick completion remains
replayable without allowing abandoned generation to live forever. Transport
cancellation gets `OLLAMA_UNIFY_REQUEST_CANCEL_GRACE` seconds (default 5); a
managed lane that still does not terminate receives a process-group SIGTERM
before its reservation can be released. Automatic CUDA cancellation never
escalates to SIGKILL: an unresponsive parent or surviving runner retains the
whole lane's reservation and blocks replacement until the process group exits.
Other backends retain bounded SIGKILL escalation. Cancellation retains the
exact backend socket even when a `Connection: close` response transfers socket
ownership away from HTTPConnection. Each process-group stop attempt has a
bounded, automatically renewed deadline; an unproven stop keeps its reservation
charged. Release is
keyed by the original admission, so delayed handler
cleanup cannot decrement a newer request. Status and discovery expose only
aggregate phases, ages, renewals, disconnects, expiries, cancellations, and
forced lane stops—never prompts, fingerprints, or logical IDs.
Stop diagnostics identify a busy transition, a specific safety guard, or a
surviving process group, and discovery exposes `cancel_process_policy`.
This teardown policy does not certify GPU, PCIe or NVLink hardware stability.

Operators can give an exact tagged model a soft ordered GPU preference through
`OLLAMA_UNIFY_MODEL_GPU_PREFERENCES`, for example
`{"model:tag":["GPU-uuid-for-first-choice"]}`. The scheduler chooses the first
preferred selected GPU with live capacity. If every preferred GPU is blocked or
cannot fit the lane, ordinary capacity-aware placement remains available. The
live discovery document reports the active preference map.

Clients can label admission with `X-Ollama-Unify-Logical-Request-Id`, `X-Ollama-Unify-Workload-Class` (`foreground`, `interactive-control`, or `background`), and `X-Ollama-Unify-Queue-Policy` (`wait` or `yield`). `X-Ollama-Unify-Admission-Wait-Ms` bounds one HTTP admission attempt; it does not limit model loading or generation after admission. With the default `wait` policy and a logical request ID, an expired bounded attempt leaves one detached waiter and its accepted, clamped body in the original FIFO position for `OLLAMA_UNIFY_POOL_RESUME_TTL` seconds (default 30). Resume with the same logical ID, the same HTTP method/path, `X-Ollama-Unify-Resume-Request: true`, and no request body. The broker forwards its retained body, so a reconnect does not resend or duplicate a large prompt. Retained request memory is bounded by `OLLAMA_UNIFY_RETAINED_REQUEST_MAX_BODY_BYTES` (default 16 MiB) and `OLLAMA_UNIFY_RETAINED_REQUEST_MAX_TOTAL_BYTES` (default 128 MiB). A changed full-body request under an active ID returns `logical_request_conflict`; an overlapping attempt returns `logical_request_in_progress`; body-free resume after cancellation or expiry returns a typed non-retryable tombstone. An explicit full-body request can start a new attempt after that tombstone. The `yield` policy removes the waiter at the admission deadline.

Clients that independently verify native runner hardware can opt into **warm-only admission**. First use ordinary capacity setup when authorized, inspect actual native runner PIDs/start times, model blobs, GPU placement, full residency and context, then POST the capacity endpoint with `warm_admission_proof:true`, the same `model`, `parallel`, `endpoint`, and explicit ordered `gpu_uuids`. This proof-only call inspects existing resident lanes; it never loads a model, creates capacity, or renews idle lifetime. It returns `503 warm_preflight_required` when the requested warm capacity cannot be certified.

A successful response adds `warm_admission` (schema `io.ollama-unify.warm-admission.v1`) containing the broker boot identity, canonical model/digest/context, ordered GPU allowlist, and opaque `{id,generation_id}` lane identities. `warm_admission_lanes` separately exposes each lane's `server_process` and `runtime_processes` as `{pid,start_time_ticks}`, stable `native_ps` fields (`name`, `digest`, `context_length`, `size`, `size_vram`), and its exact scope. Clients must join these descriptors to their independently observed hardware proof; the certificate alone does not prove physical GPU placement. No private port is a routing input.

Embedding-only artifacts can also be certified without a completion context profile. The broker reopens the current installed digest and capabilities and binds the positive native resident context, subject to the operator hard limit. Warm embedding requests use that certified context when legacy default context preparation would differ; an explicitly conflicting context is refused before retention. Completion-capable artifacts still require their exact completion profile.

Send JSON-encoded `warm_admission` in `X-Ollama-Unify-Warm-Admission`, together with the matching `X-Ollama-Unify-GPU-UUIDs`. Under the pool lock, admission checks current process/native residency identities and claims only matching ready certified lanes. Retired members may disappear; at least one certified member must remain eligible. Busy certified lanes queue normally. This request cannot create, reload, reclaim, migrate to, or select an uncertified lane, including a lane another client adds concurrently. Native prediction/sampling options remain variable; load-affecting or unknown options, runner-variant changes, context changes, and explicit unload are refused. The fixed certified context and automatic `num_gpu:-1` are allowed. Unlabelled clients keep their existing behavior.

An initial stale certificate returns `503 warm_preflight_required` with explicit `admission_retained:false` and `backend_started:false` **before logical registration or body retention**. Only this disposition permits fresh zero-token setup/physical recertification and retry of the same pending logical ID. After retention or admission the exact certificate is part of the request fingerprint; changing or removing it conflicts. Body-free resume may omit it and inherits the original proof. A stale retained request reports `admission_retained:true` and never widens its lane set. Completed responses replay the original bound response without requiring current residency; expired/cancelled bound requests retain their certificate restriction for the ordinary tombstone lifetime. Normal idle reaping remains enabled.

Embedding endpoints (`/api/embed`, `/api/embeddings`, and `/v1/embeddings`) default to `background` and `yield` when the workload-class header is absent or blank. These requests may reuse a warm embedding lane or load into free capacity, but receive an immediate retryable `503 background_capacity_deferred` if placement would require reclaiming resident lanes. Explicit workload-class and queue-policy headers retain their existing meaning: an embedding caller can request `foreground` admission, and an explicit `wait` still permits reclamation. Chat and generation defaults remain unchanged. Discovery publishes these embedding defaults under `parallel_pool.admission_protocol.unlabelled_embedding_defaults`.

After a logical inference request completes, the broker retains a bounded byte-for-byte response replay for `OLLAMA_UNIFY_COMPLETED_RESPONSE_TTL` seconds (default 120). The cache is limited by `OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_ENTRIES` (default 64), `OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_BODY_BYTES` (default 8 MiB), and `OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_TOTAL_BYTES` (default 64 MiB). An identical logical ID and request fingerprint replays the original HTTP status, end-to-end headers, and body without backend admission or generation and adds `X-Ollama-Unify-Response-Replayed: true`. A changed request under the same retained ID fails with `logical_request_conflict`. If a completed body exceeds a retention bound, the broker keeps a content-free tombstone and returns non-retryable `completed_response_unavailable` on replay instead of generating again. Discovery and status expose only limits and aggregate counters. They never expose request bodies, response bodies, logical IDs, or fingerprints from the completed-response cache.

Admission failures return an explicit `reason_code`, boolean `retryable`, JSON `retry_after_ms` when retryable, and matching `Retry-After`, `X-Ollama-Unify-Retryable`, and `X-Ollama-Unify-Reason-Code` headers. A retained deadline also returns current `queue_position`, stable `queue_ticket`, and matching headers. Current reason codes distinguish `queue_admission_timeout`, `queue_full`, `lease_transition`, generic `lane_capacity_wait`, `reclaimable_placement_wait` for a busy broker lane that can later make room, `host_memory_unavailable`, permanent capacity or model refusal, retained-request lifecycle states, `backend_start_failed`, and invalid or conflicting request contracts. A queue deadline can also include `cause_reason_code`, which preserves the capacity condition observed by the reconciler. Permanent placement failures do not carry `Retry-After`.

A pending lease transition also has an absolute `OLLAMA_UNIFY_PENDING_TIMEOUT` deadline, which defaults to 300 seconds and is not extended by heartbeats. At the deadline, the broker changes the lease to `revoking` and rejects `ready` and `heartbeat`. A scoped lease keeps only its reserved GPUs unavailable, so inference continues on unreserved devices. A legacy unscoped lease fails queued inference with an explicit `503` and keeps Ollama globally drained. The external supervisor must stop the CUDA workload and call `release`; only a verified return to the pre-lease foreign-GPU baseline removes the reservation.

The same discovery document is installed at `/usr/local/share/ollama-unify/gpu-negotiator.json` and served at `/.well-known/ollama-unify-gpu-negotiator` on the public Ollama address. Human-readable cross-agent instructions are installed at `/usr/local/share/ollama-unify/AGENTS.md`. If the invoking account already has `~/.codex`, the installer maintains a marked block in `~/.codex/AGENTS.md`; set `OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY=0` to opt out without disabling Docker or machine-readable discovery.

### Tray indicator

On desktop hosts the installer adds a system tray indicator (`/usr/local/libexec/ollama-unify-tray`, GTK with Ayatana AppIndicator) that shows who is using the broker. Each GPU, lease, and broker-owned Ollama lane gets a submenu:

- **GPUs** — every CUDA device the broker reports, whatever its size or count, with used and free VRAM, PCI bus, lease holder, Ollama lanes, and other CUDA processes labelled by their systemd unit or container.
- **Leases** — owner, state, GPUs (multi-GPU scopes are marked exclusive), requested VRAM, justification, age, expected release, and heartbeat. Actions follow the lease's state: *Mark ready*, *Prepare for resize*, *Renew heartbeat*, *Revoke lease*, *Release lease*, and *Force release*.
- **Ollama lanes** — model, GPU, in-flight requests, and reserved VRAM, with *Stop lane* (idle lanes only) and *Force stop lane*.
- **Allowed GPUs** — each lane has an *Allowed GPUs* submenu with a check item per brokered GPU. Toggling sets that model's GPU allowlist (`set_model_gpus` on the control socket, persisted next to the lease state). Placement and lane selection follow it, idle lanes on a newly disallowed GPU move at once, and busy lanes move after their requests finish. Requests whose own GPU constraint shares no GPU with the policy get `409 gpu_policy_conflict`. Allowing every GPU clears the policy.
- **GPUs in lease** — each pending or active lease has a matching submenu that adds or removes GPUs from its scope through the broker's scope checks. GPUs held by other leases are shown but cannot be selected, and a lease's last GPU cannot be removed.
- **Clients → Allowed GPUs** — select one or several GPUs for that client, allow all, or uncheck every GPU to pause new GPU requests. Active requests finish; queued and future requests follow the new selection without stopping shared lanes used by other clients. Model policies, explicit request scopes and lease reservations still apply. Multiple selections are an allowlist; they do not force a model to span every selected card. Selecting none returns a non-retryable `403 client_gpu_disabled` for new inference/capacity work and does not force CPU inference. Metadata remains available.

Destructive actions ask for confirmation, and failures open a dialog that shows the broker's error. Release and prepare wait as long as the broker's configured drain, unload, and settle deadlines allow. The icon switches to a warning when a lease is pending, revoking, or overdue, when the broker is draining, or when the Ollama backend is down.

Every menu label is pixel-measured against the narrowest active monitor work
area. Oversized labels are shortened with an ellipsis before GTK lays out the
menu, with the complete value retained as a tooltip, so a long client, model,
unit, or lease name cannot push the tray menu across a screen boundary.

The indicator talks to the control socket, so only members of the broker's access group can use it. That group already has full lease control through the socket. The socket also accepts `revoke` (by token) and `stop_lane` (by lane id, `force` to interrupt in-flight requests). The user unit `ollama-unify-tray.service` is enabled globally for `graphical-session.target`, and the broker unit's `ExecStartPost` starts it in every active desktop session of an access-group member, so the indicator comes up with the broker. Choosing *Quit indicator* stops it until the next login or broker start.

### Cooperative external-service handoff (opt-in)

An independently supervised, single-GPU service can acquire with
`--yield-on-request`. This does not enable preemption for other owners and is
not supported for unscoped/multi-GPU leases or cache-evacuation acquisition.
The opted-in owner must watch `lease_handoff_requests` in live discovery and
honor waiting, unexpired intent naming its owner and exact GPU scope in
`incumbents`. The capability is advertised at
`lease_policy.cooperative_handoff.schema` as
`io.ollama-unify.cooperative-handoff.v1`.

A valid conflicting `acquire` waits for up to `OLLAMA_UNIFY_HANDOFF_TIMEOUT`
(default180 seconds; bounded1–600). The incumbent remains active and must keep
heartbeating while it drains/stops its CUDA workload, then explicitly release.
The broker grants no overlapping lease and never forcibly revokes the incumbent
for this request. Another external requester cannot steal the intended scope
between release and admission. An existing lease owner cannot wait for a
handoff, preventing hold-and-wait cycles. Invalid metadata, impossible hardware
capacity, non-opted-in owners and pending/revoking incumbents do not trigger yield.

Intent contains no lease credentials and is cancelled on requester disconnect,
timeout or daemon restart. Owner opt-in is persisted with the lease. A successful
handoff still runs ordinary health, memory, transition and readiness gates; it is
not a guarantee of admission. Services choose their own separately acquired
replacement scope. There is no zero-downtime CUDA migration guarantee. Update
both daemon and CLI together. Acquire's client timeout includes the protocol's
maximum handoff wait even when the CLI does not inherit systemd configuration.
`docker gpu run` does not opt in: only supervisors that implement the cooperative
shutdown contract should do so.

### Client attribution

The broker records which application sends each request and which Ollama lane serves it. Each client connection is identified once, without extra privileges:

- **Declared name** — a client can send `X-Ollama-Unify-Client: NAME` to name itself. The name takes precedence over everything else.
- **Local processes** — socket diagnostics (`ss -e`) give the caller's UID and cgroup, which identifies its systemd unit, login scope, or Docker container. The exact PID is resolved for the broker's own user; other users' callers are narrowed to the processes in their unit.
- **Bridge-network containers** — the peer IP is mapped to a container name through the Docker API when the broker's user can read the Docker socket.
- **Anything else** — the remote address and `User-Agent`.

Only process names are recorded, never command lines, which can contain secrets. Client attribution is an expiring observability cache: `OLLAMA_UNIFY_CLIENT_HISTORY_TTL` defaults to 3600 seconds and is renewed automatically whenever that client sends a request. When activity stops, the top-level client record, its per-lane history, and each lane's *Used by* entry expire naturally. An in-flight request remains attributed even when it runs beyond the TTL, then becomes eligible for expiry after it exits. Broker status publishes `expires_at` on retained attribution rows and `client_history_policy` with the effective TTL and bounds. The cache keeps at most 256 clients (`OLLAMA_UNIFY_CLIENT_HISTORY_LIMIT`), 32 lanes per client (`OLLAMA_UNIFY_CLIENT_LANE_HISTORY_LIMIT`), and eight clients per lane. Broker status (`clients`) includes request counts per model and retained lane; each lane reports `triggered_by`, the client whose request started it, and its recent clients. The tray shows *Started for* and *Used by* in each lane's submenu, and a *Clients* section lists the 20 most recent applications with their identity and lanes.

Client GPU policies persist independently of attribution history and daemon
restarts. Configured clients remain in the indicator after their recent usage
expires, so paused clients can be enabled again. The control socket accepts
`{"action":"set_client_gpus","client_key":"app:NAME","gpu_uuids":["GPU-UUID"]}`;
use `[]` to pause or `null` to clear the policy and allow all broker-selected
GPUs. A disjoint explicit request returns `409 client_gpu_policy_conflict`;
exact groups are never silently shrunk. Individual tray checkboxes use atomic
per-GPU edits so rapid selections do not
overwrite each other while the indicator waits for its next status refresh.
The control request is
`{"action":"set_client_gpu","client_key":"app:NAME","gpu_uuid":"GPU-UUID","allowed":false}`.
Policies follow the existing client
identity key, so applications sharing a declared name share that policy. Send
distinct `X-Ollama-Unify-Client` names when applications need independent
controls. This is cooperative placement, not an authentication boundary.

### Manual inference hold

Set `OLLAMA_UNIFY_INFERENCE_HOLD_FILE` in the negotiator's service environment
to an operator-controlled file path. While that file exists, new native and
OpenAI inference requests, capacity preparation, managed-lane creation and
model selection fail with HTTP 503 and `reason_code=operator_inference_hold`.
Metadata and GPU lease-control APIs remain available. Existing executing
requests are not forcibly terminated, and all selected GPUs remain selected.

Creating or removing the file takes effect dynamically. Its text supplies the
operator reason; an unreadable file fails closed. The daemon publishes the
effective hold through discovery/status, and `docker gpu discover` uses that
daemon value even when the CLI has a different environment. Removing a hold
admits waiting/background clients immediately, so release it deliberately.

### Surviving Ollama upgrades

Hosts with an executable `/usr/local/sbin/ollama-version-guard` delegate release
admission to that host-owned manager. On those hosts, `--update-ollama` refuses
before downloading or stopping either service, including in noninteractive
and assume-yes modes. The generated reconciliation helper executes the
manager's `ensure` operation and propagates failure; it never accepts a changed
binary as a new approved version or falls through to the legacy restart path.
Manual `--reconcile` also refuses its legacy service cycle and directs the
operator to `sudo ollama-version-guard ensure`.
Install and authorize updates through that manager's documented workflow.
The manager itself is supplied by the host; this integration does not install
it, grant upgrade permission or certify a new release's CUDA behavior.

The following legacy upgrade behavior applies to hosts without that manager.

The official Ollama installer rewrites `/etc/systemd/system/ollama.service` and restarts the daemon. The unit it writes carries no `OLLAMA_HOST`, so the pinned loopback backend exists only in the late-priority ollama-unify drop-in. If that drop-in is cleared or drifts, Ollama falls back to its built-in `0.0.0.0:11434` and collides head-on with the negotiator that already owns that address — whichever process loses the bind race dies.

Three commands close that gap:

```bash
./ollama-unify.sh --check-update     # installed vs latest release, plus any pinning drift (read-only)
./ollama-unify.sh --reconcile        # repin and restart the pair after an out-of-band upgrade
./ollama-unify.sh --update-ollama    # update inside a safe stop → repin → start envelope
```

`--update-ollama` stops the negotiator first so the public address is free while the installer runs, lets the official installer do its work, reapplies the pinned safety policy, then restarts the pair in the only safe order — proxy down, backend up, proxy up — and verifies `/api/tags` end to end before reporting success. If the installer fails, the previous policy is reapplied and the stack is brought back up.

`--install-safety` also arms a watchdog for upgrades taken outside this script:

- `/usr/local/libexec/ollama-unify-reconcile` — a standalone repair helper with no dependency on this repository. It reads the pinned addresses from `/usr/local/share/ollama-unify/state.env`, re-asserts `OLLAMA_HOST` in the drop-in and `OLLAMA_UNIFY_BACKEND` in the negotiator config, and cycles the pair in order. If the drop-in has vanished entirely it restores the backend pin alone and logs that `--install-safety` is needed to rebuild the full containment policy.
- `ollama-unify-reconcile.path` — a systemd path unit watching the Ollama binary. Any upgrade that rewrites it triggers the helper, which no-ops when there is no drift.

Print shell-compatible exports for a manual server or a non-systemd supervisor:

```bash
./ollama-unify.sh --print-env
# Review first; then, for the current shell only:
eval "$(./ollama-unify.sh --print-env)"
```

## Requirements

| Required | Optional |
| --- | --- |
| `bash` (4+) and `awk` for classification | `rsync`, `find`, `stat`, `du`, and `df` for model migration |
| `python3` for the generated streaming GPU negotiator on systemd GPU hosts | `curl` for external-workload readiness checks and post-migration API verification |
| | `sudo` (only if writing to protected paths or touching systemd) |
| | `systemctl` (required for service integration and cgroup OOM containment) |
| | `nvidia-smi` for CUDA telemetry |
| | `amd-smi` + `jq`, or `rocminfo`, for ROCm telemetry |
| | `vulkaninfo` for Vulkan telemetry |
| | `system_profiler` for macOS Metal classification |

The fixture suite exercises CUDA dedicated/display/constrained devices, multi-GPU ROCm, integrated Vulkan, Apple Metal, CPU-only hosts, and the negotiator's streaming proxy/drain/lease/resize/release lifecycle. Classification is designed to degrade cleanly on Linux, macOS, FreeBSD, WSL, VMs, and containers. Native Windows requires WSL because this is a Bash utility. systemd cgroup containment and automatic negotiator installation remain Linux-specific.

## Safety guarantees

- **No data is ever deleted.** Source stores are renamed (`<path>.bak` for stores with manifests, `<path>.orphan-blobs` for stores with only orphan blobs). You reclaim the space yourself with `rm -rf` after verifying.
- **The migration flow stops daemons only after you confirm** the plan summary. `--install-safety` is an explicit maintenance command and may briefly restart an active Ollama service so the new boundary takes effect.
- **Idempotent re-runs.** Running the script a second time on a unified setup detects "only one store," reclassifies the host, and can update the systemd safety profile without moving data.
- **Sudo is only requested if needed.** If your destination is in your home dir, no daemons are running, and you skip the systemd step, the script runs with zero privilege escalation.
- **On supported systemd versions, ordinary userspace OOMs are scoped to Ollama.** `MemoryHigh` throttles and reclaims first; `systemd-oomd` reacts to sustained PSI; `MemoryMax` is the last line of defense. `OOMPolicy=stop`, `KillMode=control-group`, and the default `Restart=on-success` keep OOM/driver failures fail-closed while recovering an unexpected clean daemon exit. Explicit `systemctl stop` remains stopped.

## What the script does, step by step

1. Scans the filesystem and current environment for ollama model directories
2. Inspects each one: size, filesystem, owner, manifest/blob counts
3. Detects active daemons (`ollama.service` + manual `ollama serve` processes)
4. Prompts you to pick a unified destination (default = the largest store)
5. Classifies the host and all detectable CUDA, ROCm, Vulkan, Metal, and CPU candidates
6. Selects a backend and derives device, scheduler, and host-memory limits
7. Asks opt-in questions for the systemd model path, OOM guardrails, symlinks, shell rc, and service user
8. Shows the full plan and waits for your final confirmation
9. Stops all daemons cleanly
10. Transfers each non-destination store using the optimal strategy for that source→dest fs pair
11. Archives originals as `.bak` (or `.orphan-blobs`)
12. Normalizes ownership and permissions on the unified destination
13. Installs the memory-pressure preflight and a late-priority systemd drop-in (if requested)
14. Creates backward-compat symlinks (if requested)
15. Appends the env export to your shell rc (if requested)
16. Restarts `ollama.service`
17. Verifies via manifest count and `/api/tags`
18. Prints a backup summary with reclamation commands

## Hardware classification and OOM policy

Heavy inference can exhaust discrete VRAM, shared/unified memory, pinned host pages, swap, or all of them together. The classifier does not assume that every GPU is a dedicated CUDA card—or that a GPU exists at all.

### Automatic backend order

| Candidate | Detection and classification | Selection policy |
| --- | --- |
| CUDA | `nvidia-smi`: UUID, VRAM, display role, compute capability | Dedicated devices first; display devices only when no dedicated CUDA device qualifies; default floor 4 GiB and compute 5.x |
| ROCm | `amd-smi --json` or `rocminfo`: UUID/ordinal, product, VRAM when available | Native AMD candidate; distinguishes discrete, shared/constrained, and unknown-memory devices |
| Vulkan | `vulkaninfo --summary`: device index, name, discrete/integrated/virtual type | Discrete first, then shared/integrated; uses conservative memory estimates because standard summary telemetry lacks free VRAM |
| Metal | macOS `system_profiler` and Apple Silicon architecture | Preferred automatically on macOS; GPU and CPU share unified memory |
| CPU | OS CPU and effective-memory telemetry | Always-available fallback with GPU backends disabled where the platform supports those controls |

On macOS, `auto` prefers Metal. Elsewhere it chooses CUDA, then ROCm, then Vulkan, then CPU. Override an available backend with `OLLAMA_SAFE_BACKEND=cuda|rocm|vulkan|metal|cpu`.

### Dynamic limits

| Guardrail | Derived behavior |
| --- | --- |
| Effective RAM | Physical RAM reduced to the current cgroup limit when running inside a constrained container or service |
| Host class | `<8 GiB` constrained, `<32 GiB` personal, `<128 GiB` workstation, otherwise memory-rich server |
| Installed payload | Sums inference-bearing model/projector/adapter/tensor layers per manifest and selects the largest installed manifest footprint |
| Host throttle | Largest `projected to use … MiB of host memory` value found in the last 30 days of `ollama.service` journal history; falls back to the installed-payload boundary when no projection exists |
| Host hard cap | Larger of the largest installed inference payload and largest observed host projection; the script fails closed instead of inventing a cap when neither a model nor an explicit operator limit exists |
| Host remainder | Effective RAM minus the hard cap. This is an informational outside-cgroup budget; it is not incorrectly treated as memory the empty API daemon must allocate at startup |
| Device capacity | Current free VRAM as measured by Ollama at load time; no automatic fixed carve-out is subtracted a second time |
| Dedicated multi-GPU | Base Ollama retains automatic placement. Managed lanes prefer one fitting GPU; oversized models use exclusive ordered groups with forced spreading and verified placement |
| Physical GPU count | Every selected dedicated accelerator remains in the visibility list. API `num_gpu=-1` selects the number of offloaded layers automatically; it does not reduce the host to one physical GPU |
| External lease | Ollama drains and unloads, the external workload allocates first, then Ollama reloads against remaining VRAM; release repeats the cycle so Ollama can expand |
| Dedicated GPU load | Automatic maximum GPU layers, layer splitting and runner fitting enabled, unified-memory spill absent, pinned-host allocation disabled, cgroup-bounded pageable CPU overflow allowed, and cgroup swap disabled |
| Loaded models | 1 on every hardware shape by default; an override is required to allow simultaneous resident models |
| Parallel requests | 1 per model |
| Base-backend context default | 2,048 below 8 GiB RAM; 4,096 below 16 GiB; 8,192 otherwise (CPU hosts below 32 GiB never exceed 4,096 by default); managed completion lanes use verified model maxima and memory admission |
| Request queue | 8, 16, or 64 according to effective RAM |
| Idle retention | 5 minutes rather than indefinitely |
| KV cache | Flash Attention plus `q8_0`, reducing cache growth compared with `f16` |
| Start preflight | Refuses startup when `MemAvailable` is below 2 GiB of bounded daemon headroom (configurable and capped by `MemoryMax`) or at/above 20% full memory PSI over 10 seconds; each model lane separately checks live host memory before load |
| CPU and I/O | 400% CPU quota (capped to host capacity), CPU weight 10, I/O weight 10, and nice level 10 |
| systemd containment | Version-gated `MemoryHigh`, `MemoryMax`, `MemorySwapMax`, `OOMPolicy`, `ManagedOOMMemoryPressure`, and `ManagedOOMSwap` |
| Failure behavior | `Restart=on-success` by default: unexpected clean daemon exits recover, while an OOM, driver failure, or refused preflight remains stopped until the cause is resolved and `systemctl start ollama` is run |

These defaults are intentionally fail-closed. Ollama documents that parallel processing multiplies context allocation, so 32K context × 3 parallel requests can require roughly 96K tokens of context memory per loaded model. Ollama recommends UUIDs for NVIDIA and AMD selection because numeric ordering may vary. llama.cpp enables unified memory by the presence of `GGML_CUDA_ENABLE_UNIFIED_MEMORY`; the generated policy therefore removes it instead of assigning `0`. See Ollama's [concurrency and memory guidance](https://docs.ollama.com/faq), [GPU/backend guidance](https://docs.ollama.com/gpu), llama.cpp's [CUDA build/runtime guidance](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#unified-memory), and AMD's [ROCm isolation guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html).

The defaults can be changed for a single run:

```bash
OLLAMA_SAFE_CONTEXT_LENGTH=16384 \
OLLAMA_SAFE_MAX_LOADED_MODELS=1 \
OLLAMA_SAFE_HOST_RESERVE_MIB=98304 \
OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT=20 \
OLLAMA_SAFE_CPU_QUOTA_PERCENT=400 \
OLLAMA_SAFE_BACKEND=cuda \
./ollama-unify.sh
```

Supported overrides are `OLLAMA_SAFE_BACKEND`, `OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB`, `OLLAMA_SAFE_MIN_GPU_MEMORY_MIB`, `OLLAMA_SAFE_MIN_COMPUTE_MAJOR`, `OLLAMA_SAFE_MODEL_STORE`, `OLLAMA_SAFE_LARGEST_MODEL_MIB`, `OLLAMA_SAFE_OBSERVED_HOST_MIB`, `OLLAMA_SAFE_VRAM_RESERVE_MIB`, `OLLAMA_SAFE_HOST_RESERVE_MIB`, `OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB`, `OLLAMA_SAFE_HOST_MEMORY_MAX_MIB`, `OLLAMA_SAFE_STARTUP_HEADROOM_MIB`, `OLLAMA_SAFE_CONTEXT_LENGTH`, `OLLAMA_SAFE_NUM_PARALLEL`, `OLLAMA_SAFE_MAX_LOADED_MODELS`, `OLLAMA_SAFE_MAX_QUEUE`, `OLLAMA_SAFE_KEEP_ALIVE`, `OLLAMA_SAFE_SWAP_MAX`, `OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT`, `OLLAMA_SAFE_CPU_QUOTA_PERCENT`, `OLLAMA_SAFE_CPU_WEIGHT`, `OLLAMA_SAFE_IO_WEIGHT`, `OLLAMA_SAFE_RESTART_POLICY` (`no`, `on-success`, or `on-failure`), `OLLAMA_SAFE_NEGOTIATOR_LISTEN`, `OLLAMA_SAFE_NEGOTIATOR_GROUP`, `OLLAMA_SAFE_NEGOTIATOR_DRAIN_TIMEOUT`, `OLLAMA_SAFE_NEGOTIATOR_PENDING_TIMEOUT`, `OLLAMA_SAFE_NEGOTIATOR_UNLOAD_TIMEOUT`, `OLLAMA_SAFE_NEGOTIATOR_LEASE_TTL`, `OLLAMA_SAFE_NEGOTIATOR_ANON_POLL`, `OLLAMA_SAFE_NEGOTIATOR_ANON_SETTLE`, `OLLAMA_SAFE_NEGOTIATOR_ANON_MAX_DRAIN`, `OLLAMA_SAFE_POOL_ENABLED`, `OLLAMA_SAFE_POOL_MAX_SERVERS`, `OLLAMA_SAFE_POOL_PORT_START`, `OLLAMA_SAFE_POOL_INSTANCE_PARALLEL`, `OLLAMA_SAFE_POOL_RESUME_TTL`, `OLLAMA_SAFE_POOL_IDLE_TIMEOUT`, `OLLAMA_SAFE_POOL_READY_TIMEOUT`, `OLLAMA_SAFE_POOL_LOAD_TIMEOUT`, `OLLAMA_SAFE_REQUEST_ACTIVITY_TTL`, `OLLAMA_SAFE_REQUEST_DETACHED_TTL`, `OLLAMA_SAFE_REQUEST_CANCEL_GRACE`, `OLLAMA_SAFE_POOL_VRAM_RESERVE_MIB`, `OLLAMA_SAFE_POOL_HOST_RESERVE_MIB`, `OLLAMA_SAFE_POOL_MODEL_OVERHEAD_PERCENT`, `OLLAMA_SAFE_POOL_OLLAMA_BINARY`, and `OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY` (`0` or `1`).

`OLLAMA_CONTEXT_LENGTH` is normally only a server default. Managed completion lanes resolve and reserve the verified model maximum, subject to an explicit hard ceiling, as described above. Native positive `num_gpu`/`main_gpu` overrides are replaced with automatic live fitting. The RAM-derived context ceiling still applies to unmanaged base-backend requests and compatibility mode. Clients that bypass the proxy and contact the loopback backend directly can bypass those request-level checks; the one-model scheduler, PSI kill, hard cgroup boundary, and failure-only stop policy remain the final containment layer. Under launchd, rc.d, WSL without systemd, or a manually launched server, classification and policy generation still work, but the generated environment must be integrated manually and native cgroup containment is unavailable.

## Common scenarios

**"My shell sees one set of models, my systemd daemon sees another."**

That's the canonical case. The script's default flow handles it.

**"Which backend will this machine use?"**

Run `./ollama-unify.sh --classify`. It reports every candidate, why a device is dedicated/shared/constrained, the selected backend, effective-memory tier, and derived limits. `--safety-preview` also prints the exact policy; `--print-env` emits shell-compatible exports for a manual server or another supervisor.

**"My manual `ollama serve` is much slower than the systemd one."**

That commonly means the manual process inherited different backend visibility or library paths and fell back to CPU. Prefer the managed service so its generated policy applies. On CUDA, if you must launch manually, select inference GPUs by UUID rather than exposing every numeric device:

```bash
export CUDA_VISIBLE_DEVICES=GPU-uuid-for-compute-device-1,GPU-uuid-for-compute-device-2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Discover UUIDs with `nvidia-smi -L`. Do not include a low-memory desktop adapter merely because it appears in `nvidia-smi`; mixed display/inference visibility can make discovery and allocation less predictable.

**"My models are already unified; I only want the crash protection."**

Run the script normally. A single detected store no longer exits immediately: the script skips migration and offers the dynamic systemd safety profile. Use `--safety-preview` first if you only want to inspect the generated policy.

**"I want models on a dedicated drive."**

Mount your fast drive somewhere (e.g., `/srv` or `/data`), pre-create the directory, and pass that path at the "Where to unify all models?" prompt. Cross-filesystem transfers use the fastest locally supported rsync options.

**"I have a container/Docker setup."**

The script handles host-level stores. If Ollama runs in a container, bind-mount the unified path into the container after migration:

```yaml
# docker-compose.yml
volumes:
  - /srv/ollama/models:/root/.ollama/models
```

## Limitations

- **Classification is broader than service integration.** CUDA, ROCm, Vulkan, Metal, CPU, containers, WSL, macOS, and FreeBSD are classified best-effort; automatic cgroup installation currently targets `ollama.service` on systemd.
- **Native Windows needs WSL.** The project is Bash-based and does not install a Windows service policy.
- **ROCm and Vulkan telemetry varies by driver generation.** Missing memory telemetry is reported explicitly instead of being presented as an exact capacity.
- **Single host only.** No remote/multi-host orchestration. For distributed Ollama deployments, run the script per host.
- **Doesn't handle in-flight model pulls.** If `ollama pull` is mid-download when daemons stop, restart it after migration.
- **Systemd containment only covers `ollama.service`.** Manually launched `ollama serve` processes do not inherit the cgroup limits or generated environment.
- **Unregistered CUDA cannot be admitted atomically.** The broker quarantines observed unknown processes and defers model loading, idle eviction, replacement, and forced lane teardown on those GPUs. Managed models have native expiry disabled so that their unload follows the same guard. Detection cannot prevent an undeclared first allocation or a process arriving after the final telemetry check; acquire/prepare a scoped lease before initializing or growing CUDA. Backend crashes and actions outside the broker remain outside this protection. Managed child output and UUID/model/PID lifecycle events are retained in the service journal.
- **No userspace policy can guarantee recovery from every driver, firmware, PSU, or hardware failure.** These controls reduce oversubscription and contain normal service OOMs; persistent kernel or accelerator-driver warnings still require platform investigation.

Run the regression suites with `./tests/test-classifier.sh`, `./tests/test-transfer.sh`, and `./tests/test-negotiator.sh`.

## Contributing

Issues and PRs welcome at <https://github.com/robit-man/ollama-unify>.

## License

MIT — see [LICENSE](LICENSE).
