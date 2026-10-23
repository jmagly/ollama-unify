# Dedicated confidential project profile

This is an opt-in profile for **one authenticated principal per dedicated broker instance**.
The default shared broker behavior is unchanged. The profile provides an application boundary;
GPU isolation, transport, user separation, network isolation and retention controls require
host deployment. It does not make active data inaccessible to root or physically isolate GPU memory.

## Configuration

Render the Python broker with `source ollama-unify.sh; render_gpu_negotiator_script`, then invoke
that executable with `serve`. Configure through a private `OLLAMA_UNIFY_CONFIG` environment file
or process environment. Do not put a token in either environment or the configuration file.

| Setting | Required value or purpose |
| --- | --- |
| `OLLAMA_UNIFY_CONFIDENTIAL` | `1`; invalid boolean values fail startup. Project/token/approved-model settings while disabled also fail startup. |
| `OLLAMA_UNIFY_PROJECT_ID` | Fixed safe identifier, 1–64 letters/digits/dot/underscore/hyphen; starts with a letter/digit. |
| `OLLAMA_UNIFY_TOKEN_FILE` | Absolute protected file path; regular, no symlink, one hard link, owned by root or service UID, mode `0400` or `0600`. A narrowly checked systemd 255 exception permits root:root `0440` only at the exact `$CREDENTIALS_DIRECTORY/broker-token` under `/run/credentials/<unit>.service`, with a root:root `0550` credential directory and protected nonsymlink root-owned ancestors. Token is 43–128 URL-safe ASCII characters, generated from at least 32 random bytes. A newline is accepted. Rotate by replacing file and restarting broker. |
| `OLLAMA_UNIFY_APPROVED_MODELS` | Nonempty JSON map of canonical model tags such as `{"approved-model:latest":"<64 lowercase SHA256 hex>"}`. Digests must be approved separately; never fabricate them from names. |
| `OLLAMA_UNIFY_BACKEND_TYPE` | `cuda`. |
| `OLLAMA_UNIFY_POOL_ENABLED` | `1`. |
| `OLLAMA_UNIFY_SELECTED_GPUS` | Explicit comma-separated UUIDs allocated to this project. |
| `OLLAMA_UNIFY_BACKEND` | Dedicated trusted metadata Ollama endpoint on loopback; default `127.0.0.1:11436`. |
| `OLLAMA_UNIFY_LISTEN` | Dedicated frontend TCP endpoint on loopback, if not using UDS. |
| `OLLAMA_UNIFY_HTTP_SOCKET` | Recommended: absolute dedicated frontend Unix socket, distinct from control socket. Created `0660` with broker primary group. TCP frontend is not started when set. |
| `OLLAMA_UNIFY_SOCKET` | Separate trusted administrator control socket; created `0600`. Never expose to an agent. |
| `OLLAMA_UNIFY_POOL_PORT_START` | Dedicated loopback worker port range start. |
| `OLLAMA_UNIFY_LEASE_STATE` | Dedicated lease state path; do not use a shared broker path. |
| `OLLAMA_UNIFY_MODEL_POLICY_STATE` | Dedicated policy state path. |
| `OLLAMA_UNIFY_CACHE_RESIDENCY_STATE` | Dedicated residency state path. |
| `OLLAMA_UNIFY_MODELS` | Dedicated immutable/read-only approved model store, also used by metadata backend. |
| `OLLAMA_UNIFY_CHILD_HOME` | Dedicated disposable worker home, without Ollama cloud credentials. |
| `OLLAMA_UNIFY_CLIENT_READ_TIMEOUT` | Total request header and total body deadlines, default 15 seconds; clamped 1–60 seconds. |
| `OLLAMA_UNIFY_MAX_CONNECTIONS` | Concurrent frontend connections, default 16; clamped 1–128. Excess connections close without creating another request thread. |

Model SHA256 accepts Ollama API responses with or without `sha256:` prefix. Both metadata tags
and the **actual admitted worker's resident model** must match before client inference dispatch.
No runtime model mutation is permitted: a read-only model store closes the check/load race
against untrusted workloads; host administrators remain trusted.

## HTTP boundary

Clients send `Authorization: Bearer <token>`. Exactly one header is required. Authentication runs
before discovery, body reading and inference policy checks. Only exact inference paths listed
by `INFERENCE_PATHS` with POST and authenticated `GET /healthz` are permitted. Health returns
only `{"ok":true}`; this is frontend liveness, not a claim of GPU readiness. Discovery, capacity,
model management, model listing, running-model listing and query-bearing URLs are forbidden.

Declared client labels and User-Agent cannot select identity: every request is attributed to
`project:<project-id>`. Logical-request, resume, warm-admission and GPU-placement headers are
rejected, disabling request replay/retention paths for HTTP clients. Default non-logical
cancellation, lease accounting and GPU health checks are preserved. Prompts/responses necessarily
exist in memory while active; zeroization of Python memory, GPU memory or operating-system
buffers is not guaranteed.

Malformed, duplicate or negative Content-Length, any transfer encoding, oversized bodies and
Expect handling are rejected. Header/body read deadlines and connection limits bound resource
use. Inference must be valid JSON with an approved model. Duplicate JSON keys and case aliases
for model/options/keep_alive are rejected to avoid Python/Go parser differences. Lifecycle
`keep_alive` and placement options are forbidden; sampling/context options have an explicit
allowlist. Inference payload contents remain untrusted input to the model.

Only Content-Type and Accept request headers reach workers. Tokens, cookies, declared identity
and arbitrary headers are stripped. Backend HTTP errors and broker exceptions produce typed,
generic JSON errors. Successful inference output is returned as provided by the model. Only
Content-Type is forwarded from backend response headers, with broker-generated operational
headers. Child stdout/stderr are discarded in this profile. Broker logging retains level and
timestamp with a fixed content-free event message; it intentionally sacrifices per-error detail.
Do not attach another logging handler without the confidentiality filter.

Workers drop inherited `OLLAMA_*` environment variables and force `OLLAMA_NO_CLOUD=1`. This is
additional protection, not an egress boundary: verify the pinned Ollama version supports the flag
and enforce network isolation independently. Remote-model metadata is rejected.

## Host acceptance requirements

Run broker and workers in a private network namespace without external routes. Put agents in
a **different** network namespace; sharing the broker namespace lets agents bypass authentication
through raw loopback worker ports. Share only the frontend Unix socket directory with agents,
using a dedicated group. Keep the control socket, backend and worker ports inaccessible. Put
disposable workspaces/state/homes on controlled temporary storage, disable swap/core dumps and
content-bearing service logs, enforce source/agent tool egress rules, and verify cleanup after
normal exit, crash and reboot. These are deployment responsibilities and are not established by
source tests. SSH/Cloudflare or WireGuard transport terminates at a controlled frontend bridge;
it must not grant access to backend/control endpoints.

Pre-client-data acceptance includes real GPU allocation verification, approved artifacts and
hashes, access from the selected transport, rejected cross-project/backend/control access,
network denial from agent and worker contexts, content-free logs, crash cleanup and agreed
retention. Do not claim those controls are operational until tests on the target host pass.

## CPU verification

```
bash -c 'source ollama-unify.sh; render_gpu_negotiator_script' > /tmp/test-broker
chmod 0755 /tmp/test-broker
python3 tests/test-negotiator-confidential.py /tmp/test-broker "$PWD/tests/fixtures/bin"
bash tests/test-negotiator.sh
```

The confidential suite exercises real proxy authentication before body reading, endpoint and
framing restrictions, model allowlist/digest rejection, JSON alias rejection, real CPU-fixture
inference, fixed attribution, stripped credentials, empty replay cache, generic errors,
content-free logs including uncaught main/thread exceptions, header/body deadlines, connection exhaustion/recovery, UDS health and strict
startup validation. It does not establish production hardware, filesystem or network behavior.

## Holding a dedicated physical GPU against a shared host broker

If the private broker runs beside the existing shared broker, first acquire an **explicit
exclusive external lease from the shared broker**, using the dedicated physical UUID:

```
ollama-unify-gpu-negotiator acquire --owner private-project --gpu GPU-UUID --exclusive \
  --ttl 300 --justification 'Dedicated private inference instance'
```

This command is an example interface, not a deployed configuration. Keep the returned token
in a root-protected owner service; do not publish command output or put the token in process
arguments. Maintain heartbeats and call `ready` after the private owner is prepared. The control
JSON equivalent adds the strict Boolean `"exclusive":true` to `action:"acquire"`.
`run --exclusive` provides the same fixed-scope policy for a supervised child.

The option drains intersecting shared managed lanes before granting the lease. It rejects
existing foreign CUDA contexts on that scope, and keeps the GPU unavailable to shared broker
placement even after `ready`. Plain single-GPU leases retain their existing free-VRAM sharing
semantics; multi-GPU leases remain intrinsically exclusive. `exclusive` is persisted with the
lease and appears in raw control status. Public `lease_summaries` report effective exclusivity.
The owner, token, heartbeat, foreign baseline, TTL and scope survive broker restart.

Explicit exclusive leases require a fixed nonempty scope and cannot combine flexible placement,
cooperative yield or cache evacuation. A live exclusive lease cannot change scope; release it
before acquiring another GPU. There is no forced migration. Release, including `--force`, must
verify that **no CUDA contexts of any ownership remain** on the reserved GPU, including canonical/base/orphan managed contexts and contexts reporting
zero MiB. The dedicated raw telemetry query does not apply managed-process exclusions. Revoke expiry retains the explicit lease while CUDA remains or telemetry is unavailable;
expired owner registration alone cannot free the dedicated GPU. Existing non-opt-in lease release
and abandonment behavior is unchanged.

The separate canonical GPU lock and exclusive lease are complementary: the lock coordinates
project deployment, while the shared broker lease prevents its own scheduler from placing lanes
on the private GPU. Neither prevents a privileged administrator or uncooperative process from
bypassing coordination. Verify the expected lease owner, exact UUID, active state, strict
`exclusive:true`, current heartbeat/TTL, and a protected token hash before granting private access.


## Model residency and CPU overflow

The confidential profile rejects the CPU-offload admission header, including an explicit
`deny` header. It also refuses the internal CPU-overflow capacity path. Approved model layers
must be fully resident on the selected GPUs: positive finite total-model and GPU-resident byte
counts are checked at warm-up and again before dispatch. A partial GPU load or CPU-offload lane
cannot receive a confidential inference request. Oversized models fail admission; select a
smaller approved model rather than expanding into host RAM.

CPU execution for scheduling, tokenization and application work remains normal. This policy
prohibits overflow of model layers into host RAM; it does not claim the CPU is unused. Dedicated
development users should have no routine sudo access. Host root administrators remain trusted
and can inspect active data; this profile does not provide hardware confidential computing.
