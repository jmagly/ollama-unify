# Explicit exclusive GPU lease cleanup

An explicit `--exclusive` lease reserves its fixed GPU scope against both other
external leases and broker-owned model placement. The reservation stays in force
after the owner calls `ready`; positive-budget non-exclusive single-GPU leases
retain their ordinary sharing behavior.

Heartbeat expiry and a revoke deadline do not establish that an exclusive owner
has finished using CUDA. Before abandoning an expired explicit exclusive lease,
the broker requires a successful raw CUDA-context query showing no contexts on
its GPU scope. This includes zero-MiB contexts, managed or orphaned contexts, and
child contexts created after the owner's readiness snapshot. Unavailable or
invalid telemetry retains the reservation. It can therefore remain quarantined
beyond the lease TTL when an owner crashes or stops renewing.

Release, including forced release, retains the same all-context requirement.
Operators should stop the owning workload, verify every scoped CUDA context has
exited, then release the reservation. This change does not add a GPU reset or a
forced process kill. A broker lease coordinates cooperating processes; it cannot
prevent a privileged administrator or uncooperative process from accessing a GPU.

Tray lease details and GPU-selection menus identify explicit single-GPU reservations as
exclusive, matching multi-GPU reservations. The absence of a second GPU in a
lease is not evidence that it can share with other work.

## Regression checks

`tests/test-negotiator-exclusive.py` exercises acquisition, readiness, fixed
scope, conflicting placements, persistence and all-context cleanup, including
zero-MiB and unavailable-telemetry expiry cases. The tray model test checks the
single-GPU exclusive labels without a display.

```sh
bash -c 'source ollama-unify.sh; render_gpu_negotiator_script' > /tmp/test-broker
chmod 0755 /tmp/test-broker
python3 tests/test-negotiator-exclusive.py /tmp/test-broker "$PWD/tests/fixtures/bin"
```
