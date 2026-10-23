#!/usr/bin/env bash
# ollama-unify — consolidate scattered ollama model stores into one canonical location
# https://github.com/robit-man/ollama-unify
#
# Detects every ollama model directory referenced by:
#   - $HOME/.ollama/models (per-user default)
#   - /usr/share/ollama/.ollama/models (system-user default)
#   - /etc/default/ollama, /etc/environment, systemd unit env (OLLAMA_MODELS)
#   - Live `ollama runner` cmdlines
# Then interactively unifies them at a destination of your choosing, picks the
# fastest available transfer method (same-fs mv / reflink / NVMe rsync), and
# optionally rewires systemd + shell rc + backward-compat symlinks. It can also
# install dynamic GPU/host-memory guardrails that contain Ollama OOM failures.
#
# Safety: never deletes data. Renames originals to .bak / .orphan-blobs for you
# to remove after verifying.

set -euo pipefail

# ───────────────────────────────────────────────────────────────────── colors
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'; C_GRN=$'\033[32m'
  C_YEL=$'\033[33m'; C_CYN=$'\033[36m'; C_RST=$'\033[0m'
else
  C_DIM=; C_BOLD=; C_RED=; C_GRN=; C_YEL=; C_CYN=; C_RST=
fi
say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n%s%s%s\n' "$C_BOLD$C_CYN" "$*" "$C_RST"; }
ok()   { printf '%s✓%s %s\n' "$C_GRN" "$C_RST" "$*"; }
warn() { printf '%s!%s %s\n' "$C_YEL" "$C_RST" "$*" >&2; }
err()  { printf '%s✗%s %s\n' "$C_RED" "$C_RST" "$*" >&2; }
ask()  { local prompt="$1" default="${2:-}" reply; if [ -n "$default" ]; then
           read -r -p "$prompt [$default]: " reply < /dev/tty || true
           printf '%s' "${reply:-$default}"
         else
           read -r -p "$prompt: " reply < /dev/tty || true
           printf '%s' "$reply"
         fi; }
confirm() { local reply; reply=$(ask "$1" "${2:-N}"); case "${reply,,}" in y|yes|true|1) return 0 ;; *) return 1 ;; esac; }

# ───────────────────────────────────────────────────────────────────── banner
banner() {
  cat <<'BANNER'
  ___  _ _                                       _  __
 / _ \| | | __ _ _ __ ___   __ _    _   _ _ __ (_)/ _|_   _
| | | | | |/ _` | '_ ` _ \ / _` |  | | | | '_ \| | |_| | | |
| |_| | | | (_| | | | | | | (_| |  | |_| | | | | |  _| |_| |
 \___/|_|_|\__,_|_| |_| |_|\__,_|   \__,_|_| |_|_|_|  \__, |
                                                      |___/
BANNER
  printf '%sUnify scattered Ollama model stores into one canonical location.%s\n\n' "$C_DIM" "$C_RST"
}

# ───────────────────────────────────────────────────── prerequisite checks
require() { command -v "$1" >/dev/null 2>&1 || { err "missing required command: $1"; exit 2; }; }
require_migration_tools() {
  require rsync; require du; require df; require find; require stat; require awk; require sort
}

HAS_SUDO=0; command -v sudo >/dev/null 2>&1 && HAS_SUDO=1
HAS_SYSTEMD=0; SYSTEMD_VERSION=0
if command -v systemctl >/dev/null 2>&1 && systemctl --version >/dev/null 2>&1; then
  HAS_SYSTEMD=1
  SYSTEMD_VERSION=$(systemctl --version | awk 'NR==1 {print $2; exit}')
  [[ "$SYSTEMD_VERSION" =~ ^[0-9]+$ ]] || SYSTEMD_VERSION=0
fi
HAS_CURL=0; command -v curl >/dev/null 2>&1 && HAS_CURL=1

# ─────────────────────────────────── portable host + accelerator classifier
# The safety layer always produces a scheduler/host-memory profile. Accelerator
# discovery is capability-driven and degrades through CUDA → ROCm → Vulkan →
# Metal → CPU unless OLLAMA_SAFE_BACKEND explicitly selects an available one.
SAFETY_READY=0
HOST_OS=""
HOST_ARCH=""
HOST_NAME=""
HOST_CPU=""
HOST_CPU_CORES=0
HOST_VIRTUALIZATION="none"
HOST_SERVICE_MANAGER="none"
HOST_MEMORY_SOURCE=""
SAFETY_PHYSICAL_MEMORY_MIB=0
SAFETY_HOST_TOTAL_MIB=0
SAFETY_HOST_CLASS=""
SAFETY_BACKEND="cpu"
SAFETY_BACKEND_CLASS="fallback"
SAFETY_BACKEND_REASON=""
SAFETY_DEVICE_COUNT=0
SAFETY_SHARED_ACCELERATOR=0
SAFETY_MIN_DEVICE_MEMORY_MIB=0
SAFETY_AGGREGATE_DEVICE_MEMORY_MIB=0
SAFETY_DEVICE_MEMORY_KNOWN=0
SAFETY_DEDICATED_VRAM_RATIO_PERCENT=0
SAFETY_VRAM_RESERVE_MIB=0
SAFETY_VRAM_RESERVE_BYTES=0
SAFETY_HOST_RESERVE_MIB=0
SAFETY_HOST_MEMORY_HIGH_MIB=0
SAFETY_HOST_MEMORY_MAX_MIB=0
SAFETY_STARTUP_HEADROOM_MIB=0
SAFETY_LARGEST_MODEL_MIB=0
SAFETY_LARGEST_MODEL_SOURCE=""
SAFETY_OBSERVED_HOST_MIB=0
SAFETY_HOST_LIMIT_SOURCE=""
SAFETY_GPU_PREFERRED=0
SAFETY_SCHED_SPREAD=0
SAFETY_CONTEXT_LENGTH=0
SAFETY_NUM_PARALLEL=0
SAFETY_MAX_LOADED_MODELS=0
SAFETY_MAX_QUEUE=0
SAFETY_KEEP_ALIVE=""
SAFETY_SWAP_MAX=""
SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT=20
SAFETY_CPU_QUOTA_PERCENT=400
SAFETY_CPU_WEIGHT=10
SAFETY_IO_WEIGHT=10
SAFETY_RESTART_POLICY="on-success"
SAFETY_PREFLIGHT_PATH="/usr/local/libexec/ollama-unify-memory-preflight"
SAFETY_GPU_PREFLIGHT_PATH="/usr/local/libexec/ollama-unify-gpu-preflight"
SAFETY_NEGOTIATOR_ENABLED=0
SAFETY_NEGOTIATOR_PATH="/usr/local/libexec/ollama-unify-gpu-negotiator"
SAFETY_NEGOTIATOR_CLI_PATH="/usr/local/bin/ollama-unify-gpu-lease"
SAFETY_NEGOTIATOR_CONFIG_PATH="/etc/default/ollama-unify-negotiator"
SAFETY_NEGOTIATOR_UNIT_PATH="/etc/systemd/system/ollama-unify-negotiator.service"
SAFETY_NEGOTIATOR_SOCKET="/run/ollama-unify/gpu-negotiator.sock"
SAFETY_OLLAMA_BACKEND="127.0.0.1:11436"
SAFETY_DOCKER_PLUGIN_PATH="/usr/local/lib/docker/cli-plugins/docker-gpu"
SAFETY_TRAY_PATH="/usr/local/libexec/ollama-unify-tray"
SAFETY_TRAY_UNIT_PATH="/etc/systemd/user/ollama-unify-tray.service"
SAFETY_LEGACY_DOCKER_PLUGIN_PATH="/usr/local/lib/docker/cli-plugins/docker-gpu-lease"
SAFETY_DISCOVERY_DIR="/usr/local/share/ollama-unify"
SAFETY_DISCOVERY_PATH="/usr/local/share/ollama-unify/gpu-negotiator.json"
SAFETY_AGENT_INSTRUCTIONS_PATH="/usr/local/share/ollama-unify/AGENTS.md"
SAFETY_STATE_PATH="/usr/local/share/ollama-unify/state.env"
SAFETY_RECONCILE_HELPER_PATH="/usr/local/libexec/ollama-unify-reconcile"
SAFETY_RECONCILE_SERVICE_PATH="/etc/systemd/system/ollama-unify-reconcile.service"
SAFETY_RECONCILE_PATH_UNIT_PATH="/etc/systemd/system/ollama-unify-reconcile.path"
SAFETY_OLLAMA_RELEASE_API="https://api.github.com/repos/ollama/ollama/releases/latest"
SAFETY_OLLAMA_INSTALL_URL="https://ollama.com/install.sh"

CUDA_TOOL=""; CUDA_COUNT=0; CUDA_MIN_VRAM_MIB=0; CUDA_TOTAL_VRAM_MIB=0; CUDA_SHARED=0
ROCM_TOOL=""; ROCM_COUNT=0; ROCM_MIN_VRAM_MIB=0; ROCM_TOTAL_VRAM_MIB=0; ROCM_KNOWN_VRAM_COUNT=0; ROCM_SHARED=0
VULKAN_TOOL=""; VULKAN_COUNT=0; VULKAN_SHARED=0
METAL_COUNT=0
declare -a ACCELERATOR_SUMMARIES=() SAFETY_DEVICE_IDS=() SAFETY_SELECTED_SUMMARIES=()
declare -a SAFETY_PREFLIGHT_DIRECTIVES=()
declare -a CUDA_IDS=() CUDA_SUMMARIES=() CUDA_PREFLIGHT=()
declare -a ROCM_IDS=() ROCM_SUMMARIES=() ROCM_PREFLIGHT=()
declare -a VULKAN_IDS=() VULKAN_SUMMARIES=() VULKAN_PREFLIGHT=()
declare -a METAL_SUMMARIES=()

trim_ws() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

require_uint_value() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || { err "$name must be an unsigned integer (got: $value)"; exit 2; }
}

detect_host_profile() {
  HOST_OS=$(uname -s 2>/dev/null || printf 'Unknown')
  HOST_ARCH=$(uname -m 2>/dev/null || printf 'unknown')
  HOST_NAME=$(hostname 2>/dev/null || printf 'unknown')
  HOST_CPU_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
  [[ "$HOST_CPU_CORES" =~ ^[0-9]+$ ]] || HOST_CPU_CORES=1

  case "$HOST_OS" in
    Linux)
      if [ -r /etc/os-release ]; then
        HOST_NAME=$(awk -F= '/^PRETTY_NAME=/{v=substr($0,index($0,"=")+1); gsub(/^"|"$/,"",v); print v; exit}' /etc/os-release)
      fi
      HOST_CPU=$(awk -F: '/^(model name|Hardware)[[:space:]]*:/{v=$2; sub(/^[[:space:]]+/,"",v); print v; exit}' /proc/cpuinfo 2>/dev/null)
      [ -n "$HOST_CPU" ] || HOST_CPU="$HOST_ARCH CPU"
      if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
        HOST_VIRTUALIZATION="wsl"
      elif [ -e /.dockerenv ]; then
        HOST_VIRTUALIZATION="container"
      elif command -v systemd-detect-virt >/dev/null 2>&1; then
        local detected_virt=""
        if detected_virt=$(systemd-detect-virt 2>/dev/null); then
          HOST_VIRTUALIZATION="$detected_virt"
        else
          HOST_VIRTUALIZATION="none"
        fi
      fi
      if [ -d /run/systemd/system ] && [ "$HAS_SYSTEMD" = 1 ]; then HOST_SERVICE_MANAGER="systemd"; fi
      ;;
    Darwin)
      HOST_NAME="macOS $(sw_vers -productVersion 2>/dev/null || true)"
      HOST_CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || printf '%s CPU' "$HOST_ARCH")
      HOST_SERVICE_MANAGER="launchd"
      ;;
    FreeBSD)
      HOST_NAME="FreeBSD $(uname -r 2>/dev/null || true)"
      HOST_CPU=$(sysctl -n hw.model 2>/dev/null || printf '%s CPU' "$HOST_ARCH")
      HOST_SERVICE_MANAGER="rc.d"
      ;;
    *) HOST_CPU="$HOST_ARCH CPU" ;;
  esac

  local physical_mib=0
  if [ -r /proc/meminfo ]; then
    physical_mib=$(awk '/^MemTotal:/ { print int($2 / 1024); exit }' /proc/meminfo)
    HOST_MEMORY_SOURCE="/proc/meminfo"
  elif command -v sysctl >/dev/null 2>&1; then
    local memory_bytes
    memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || sysctl -n hw.physmem 2>/dev/null || printf '0')
    if [[ "$memory_bytes" =~ ^[0-9]+$ ]]; then physical_mib=$((memory_bytes / 1024 / 1024)); fi
    HOST_MEMORY_SOURCE="sysctl"
  fi
  if ! [[ "$physical_mib" =~ ^[0-9]+$ ]] || [ "$physical_mib" -lt 1024 ]; then
    physical_mib=4096
    HOST_MEMORY_SOURCE="conservative 4 GiB fallback"
  fi
  SAFETY_PHYSICAL_MEMORY_MIB="$physical_mib"
  SAFETY_HOST_TOTAL_MIB="$physical_mib"

  local limit_file="" limit_raw="" limit_mib=0
  if [ -r /sys/fs/cgroup/memory.max ]; then
    limit_file=/sys/fs/cgroup/memory.max
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    limit_file=/sys/fs/cgroup/memory/memory.limit_in_bytes
  fi
  if [ -n "$limit_file" ]; then
    limit_raw=$(tr -d '[:space:]' < "$limit_file")
    if [[ "$limit_raw" =~ ^[0-9]+$ ]]; then
      limit_mib=$(awk -v bytes="$limit_raw" 'BEGIN { printf "%.0f", bytes / 1048576 }')
      if [ "$limit_mib" -ge 1024 ] && [ "$limit_mib" -lt "$SAFETY_HOST_TOTAL_MIB" ]; then
        SAFETY_HOST_TOTAL_MIB="$limit_mib"
        HOST_MEMORY_SOURCE="$HOST_MEMORY_SOURCE, constrained by cgroup"
      fi
    fi
  fi
  if [ -n "${OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB "$OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB"
    [ "$OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB" -ge 1024 ] \
      || { err "OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB must be at least 1024"; exit 2; }
    SAFETY_HOST_TOTAL_MIB="$OLLAMA_SAFE_EFFECTIVE_MEMORY_MIB"
    HOST_MEMORY_SOURCE="explicit override"
  fi

  case "$SAFETY_HOST_TOTAL_MIB" in
    ''|*[!0-9]*) SAFETY_HOST_CLASS="unknown" ;;
    *)
      if [ "$SAFETY_HOST_TOTAL_MIB" -lt 8192 ]; then SAFETY_HOST_CLASS="constrained"
      elif [ "$SAFETY_HOST_TOTAL_MIB" -lt 32768 ]; then SAFETY_HOST_CLASS="personal"
      elif [ "$SAFETY_HOST_TOTAL_MIB" -lt 131072 ]; then SAFETY_HOST_CLASS="workstation"
      else SAFETY_HOST_CLASS="memory-rich server"
      fi
      ;;
  esac
}

detect_cuda_devices() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  CUDA_TOOL=$(command -v nvidia-smi)
  [[ "$CUDA_TOOL" == /* ]] || { CUDA_TOOL=""; return 0; }
  local inventory=""
  inventory=$("$CUDA_TOOL" --query-gpu=index,uuid,name,display_active,memory.total,compute_cap \
    --format=csv,noheader,nounits 2>/dev/null) || return 0

  local min_vram="${OLLAMA_SAFE_MIN_GPU_MEMORY_MIB:-4096}"
  local min_compute="${OLLAMA_SAFE_MIN_COMPUTE_MAJOR:-5}"
  require_uint_value OLLAMA_SAFE_MIN_GPU_MEMORY_MIB "$min_vram"
  require_uint_value OLLAMA_SAFE_MIN_COMPUTE_MAJOR "$min_compute"
  local -a dedicated_ids=() dedicated_summaries=() shared_ids=() shared_summaries=()
  local dedicated_min=0 dedicated_total=0 shared_min=0 shared_total=0

  while IFS=',' read -r raw_index raw_uuid raw_name raw_display raw_vram raw_compute; do
    local index uuid name display vram compute major role summary
    index=$(trim_ws "$raw_index"); uuid=$(trim_ws "$raw_uuid"); name=$(trim_ws "$raw_name")
    display=$(trim_ws "$raw_display"); vram=$(trim_ws "$raw_vram"); compute=$(trim_ws "$raw_compute")
    major="${compute%%.*}"
    if ! [[ "$vram" =~ ^[0-9]+$ && "$major" =~ ^[0-9]+$ && "$uuid" == GPU-* ]]; then
      ACCELERATOR_SUMMARIES+=("[cuda/unusable] GPU $index: $name — incomplete CUDA telemetry")
    elif [ "$major" -lt "$min_compute" ]; then
      ACCELERATOR_SUMMARIES+=("[cuda/legacy] GPU $index: $name, ${vram} MiB, compute $compute — below compute ${min_compute}.x")
    elif [ "$vram" -lt "$min_vram" ]; then
      ACCELERATOR_SUMMARIES+=("[cuda/constrained] GPU $index: $name, ${vram} MiB, compute $compute — below ${min_vram} MiB safety floor")
    else
      if [ "$display" = "Enabled" ]; then role="shared-display"; else role="dedicated"; fi
      summary="GPU $index: $name, ${vram} MiB, compute $compute, $uuid ($role)"
      ACCELERATOR_SUMMARIES+=("[cuda/$role] $summary")
      if [ "$role" = "dedicated" ]; then
        dedicated_ids+=("$uuid"); dedicated_summaries+=("$summary")
        dedicated_total=$((dedicated_total + vram))
        if [ "$dedicated_min" -eq 0 ] || [ "$vram" -lt "$dedicated_min" ]; then dedicated_min="$vram"; fi
      else
        shared_ids+=("$uuid"); shared_summaries+=("$summary")
        shared_total=$((shared_total + vram))
        if [ "$shared_min" -eq 0 ] || [ "$vram" -lt "$shared_min" ]; then shared_min="$vram"; fi
      fi
    fi
  done <<< "$inventory"

  if [ ${#dedicated_ids[@]} -gt 0 ]; then
    CUDA_IDS=("${dedicated_ids[@]}"); CUDA_SUMMARIES=("${dedicated_summaries[@]}")
    CUDA_MIN_VRAM_MIB="$dedicated_min"; CUDA_TOTAL_VRAM_MIB="$dedicated_total"; CUDA_SHARED=0
  elif [ ${#shared_ids[@]} -gt 0 ]; then
    CUDA_IDS=("${shared_ids[@]}"); CUDA_SUMMARIES=("${shared_summaries[@]}")
    CUDA_MIN_VRAM_MIB="$shared_min"; CUDA_TOTAL_VRAM_MIB="$shared_total"; CUDA_SHARED=1
  fi
  CUDA_COUNT=${#CUDA_IDS[@]}
  local uuid
  for uuid in "${CUDA_IDS[@]}"; do
    CUDA_PREFLIGHT+=("ExecStartPre=$SAFETY_GPU_PREFLIGHT_PATH $CUDA_TOOL $uuid")
  done
}

detect_rocm_devices() {
  local inventory="" tool=""
  if command -v amd-smi >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
    tool=$(command -v amd-smi)
    local amd_json=""
    amd_json=$("$tool" static --json 2>/dev/null) || amd_json=""
    if [ -n "$amd_json" ]; then
      inventory=$(printf '%s\n' "$amd_json" | jq -r '
        (if type == "array" then to_entries elif type == "object" then to_entries else [] end)[] |
        .key as $ordinal | .value as $g |
        [($g.gpu // $g.gpu_id // $g.id // $ordinal),
         ($g.uuid // $g.asic.uuid // $g.gpu_uuid // ""),
         ($g.asic.market_name // $g.asic.name // $g.board.product_name // $g.name // "AMD GPU"),
         ($g.vram.size.value // $g.vram.size // 0)] | @tsv' 2>/dev/null || true)
    fi
  fi
  if [ -z "$inventory" ] && command -v rocminfo >/dev/null 2>&1; then
    tool=$(command -v rocminfo)
    inventory=$("$tool" 2>/dev/null | awk '
      function flush() { if (is_gpu) { print ordinal "\t" uuid "\t" market "\t0"; ordinal++ } }
      /^[[:space:]]*Agent[[:space:]][0-9]+/ { flush(); is_gpu=0; uuid=""; market="AMD ROCm GPU"; next }
      /^[[:space:]]*Device Type:/ { if ($NF == "GPU") is_gpu=1; next }
      /^[[:space:]]*Uuid:/ { uuid=$NF; next }
      /^[[:space:]]*Marketing Name:/ { sub(/^[^:]*:[[:space:]]*/,""); market=$0; next }
      END { flush() }' || true)
  fi
  [ -n "$inventory" ] || return 0
  ROCM_TOOL="$tool"
  local ordinal=0
  while IFS=$'\t' read -r raw_id raw_uuid raw_name raw_vram; do
    local id uuid name vram role summary
    id=$(trim_ws "$raw_id"); uuid=$(trim_ws "$raw_uuid"); name=$(trim_ws "$raw_name"); vram=$(trim_ws "$raw_vram")
    [[ "$vram" =~ ^[0-9]+$ ]] || vram=0
    if [ -n "$uuid" ] && [ "$uuid" != "N/A" ]; then id="$uuid"; else id="$ordinal"; fi
    if [ "$vram" -eq 0 ]; then role="unknown-memory"; ROCM_SHARED=1
    elif [ "$vram" -lt 4096 ]; then role="shared-or-constrained"; ROCM_SHARED=1
    else role="discrete"; fi
    summary="GPU $ordinal: $name"
    [ "$vram" -gt 0 ] && summary="$summary, ${vram} MiB"
    summary="$summary, id $id ($role)"
    ROCM_IDS+=("$id"); ROCM_SUMMARIES+=("$summary")
    ACCELERATOR_SUMMARIES+=("[rocm/$role] $summary")
    if [ "$vram" -gt 0 ] && { [ "$ROCM_MIN_VRAM_MIB" -eq 0 ] || [ "$vram" -lt "$ROCM_MIN_VRAM_MIB" ]; }; then
      ROCM_MIN_VRAM_MIB="$vram"
    fi
    if [ "$vram" -gt 0 ]; then
      ROCM_TOTAL_VRAM_MIB=$((ROCM_TOTAL_VRAM_MIB + vram))
      ROCM_KNOWN_VRAM_COUNT=$((ROCM_KNOWN_VRAM_COUNT + 1))
    fi
    ordinal=$((ordinal + 1))
  done <<< "$inventory"
  ROCM_COUNT=${#ROCM_IDS[@]}
  if [ "$ROCM_COUNT" -gt 0 ]; then
    if [[ "$ROCM_TOOL" == */amd-smi ]]; then ROCM_PREFLIGHT+=("ExecStartPre=$ROCM_TOOL list")
    else ROCM_PREFLIGHT+=("ExecStartPre=$ROCM_TOOL"); fi
  fi
}

detect_vulkan_devices() {
  command -v vulkaninfo >/dev/null 2>&1 || return 0
  VULKAN_TOOL=$(command -v vulkaninfo)
  [[ "$VULKAN_TOOL" == /* ]] || { VULKAN_TOOL=""; return 0; }
  local inventory=""
  inventory=$("$VULKAN_TOOL" --summary 2>/dev/null | awk '
    function flush() { if (seen && name != "") print idx "\t" name "\t" dtype }
    /^[[:space:]]*GPU[0-9]+:/ { flush(); idx=$1; sub(/^GPU/,"",idx); sub(/:$/,"",idx); name=""; dtype="unknown"; seen=1; next }
    /^[[:space:]]*deviceName[[:space:]]*=/ { sub(/^[^=]*=[[:space:]]*/,""); name=$0; next }
    /^[[:space:]]*deviceType[[:space:]]*=/ { sub(/^[^=]*=[[:space:]]*/,""); dtype=$0; next }
    END { flush() }' || true)
  [ -n "$inventory" ] || { VULKAN_TOOL=""; return 0; }
  local -a discrete_ids=() discrete_summaries=() shared_ids=() shared_summaries=()
  while IFS=$'\t' read -r id name dtype; do
    local role summary
    case "$dtype" in
      *DISCRETE_GPU*) role="discrete" ;;
      *INTEGRATED_GPU*|*VIRTUAL_GPU*) role="shared" ;;
      *CPU*) ACCELERATOR_SUMMARIES+=("[vulkan/cpu-device] GPU $id: $name — not an accelerator"); continue ;;
      *) role="unknown" ;;
    esac
    summary="GPU $id: $name ($role, memory telemetry unavailable)"
    ACCELERATOR_SUMMARIES+=("[vulkan/$role] $summary")
    if [ "$role" = "discrete" ]; then discrete_ids+=("$id"); discrete_summaries+=("$summary")
    else shared_ids+=("$id"); shared_summaries+=("$summary"); fi
  done <<< "$inventory"
  if [ ${#discrete_ids[@]} -gt 0 ]; then
    VULKAN_IDS=("${discrete_ids[@]}"); VULKAN_SUMMARIES=("${discrete_summaries[@]}"); VULKAN_SHARED=0
  else
    VULKAN_IDS=("${shared_ids[@]}"); VULKAN_SUMMARIES=("${shared_summaries[@]}"); VULKAN_SHARED=1
  fi
  VULKAN_COUNT=${#VULKAN_IDS[@]}
  [ "$VULKAN_COUNT" -gt 0 ] && VULKAN_PREFLIGHT+=("ExecStartPre=$VULKAN_TOOL --summary")
}

detect_metal_devices() {
  [ "$HOST_OS" = "Darwin" ] || return 0
  if command -v system_profiler >/dev/null 2>&1; then
    while IFS= read -r name; do
      [ -n "$name" ] || continue
      METAL_SUMMARIES+=("$name (Metal, unified/shared memory)")
      ACCELERATOR_SUMMARIES+=("[metal/unified] $name")
    done < <(system_profiler SPDisplaysDataType 2>/dev/null | awk -F: '/Chipset Model:/{sub(/^[[:space:]]+/,"",$2); print $2}')
  fi
  if [ ${#METAL_SUMMARIES[@]} -eq 0 ] && [ "$HOST_ARCH" = "arm64" ]; then
    METAL_SUMMARIES+=("Apple Silicon GPU (Metal, unified memory)")
    ACCELERATOR_SUMMARIES+=("[metal/unified] Apple Silicon GPU")
  fi
  METAL_COUNT=${#METAL_SUMMARIES[@]}
}

detect_unconfigured_accelerators() {
  [ ${#ACCELERATOR_SUMMARIES[@]} -eq 0 ] || return 0
  [ "$HOST_OS" = "Linux" ] || return 0
  if command -v lspci >/dev/null 2>&1; then
    while IFS= read -r device; do
      [ -n "$device" ] || continue
      ACCELERATOR_SUMMARIES+=("[pci/unconfigured] $device — no usable Ollama backend telemetry")
    done < <(lspci 2>/dev/null | awk 'tolower($0) ~ /vga compatible controller|3d controller|display controller/ {$1=""; sub(/^[[:space:]]+/,""); print}')
  fi
  if [ ${#ACCELERATOR_SUMMARIES[@]} -eq 0 ] && compgen -G '/dev/dri/renderD*' >/dev/null 2>&1; then
    ACCELERATOR_SUMMARIES+=("[drm/unconfigured] render nodes exist, but neither ROCm nor Vulkan telemetry is available")
  fi
}

select_safety_backend() {
  local requested="${OLLAMA_SAFE_BACKEND:-auto}"
  requested="${requested,,}"
  case "$requested" in auto|cuda|rocm|vulkan|metal|cpu) ;; *) err "OLLAMA_SAFE_BACKEND must be auto, cuda, rocm, vulkan, metal, or cpu"; exit 2 ;; esac
  if [ "$requested" = "auto" ]; then
    if [ "$HOST_OS" = "Darwin" ] && [ "$METAL_COUNT" -gt 0 ]; then requested="metal"
    elif [ "$CUDA_COUNT" -gt 0 ]; then requested="cuda"
    elif [ "$ROCM_COUNT" -gt 0 ]; then requested="rocm"
    elif [ "$VULKAN_COUNT" -gt 0 ]; then requested="vulkan"
    else requested="cpu"; fi
  fi

  case "$requested" in
    cuda)
      [ "$CUDA_COUNT" -gt 0 ] || { err "CUDA was requested but no eligible CUDA device was classified"; exit 2; }
      SAFETY_DEVICE_IDS=("${CUDA_IDS[@]}"); SAFETY_SELECTED_SUMMARIES=("${CUDA_SUMMARIES[@]}")
      SAFETY_PREFLIGHT_DIRECTIVES=("${CUDA_PREFLIGHT[@]}")
      SAFETY_DEVICE_COUNT="$CUDA_COUNT"
      SAFETY_AGGREGATE_DEVICE_MEMORY_MIB="$CUDA_TOTAL_VRAM_MIB"; SAFETY_DEVICE_MEMORY_KNOWN=1
      SAFETY_MIN_DEVICE_MEMORY_MIB="$CUDA_MIN_VRAM_MIB"; SAFETY_SHARED_ACCELERATOR="$CUDA_SHARED"
      SAFETY_BACKEND_CLASS=$([ "$CUDA_SHARED" = 1 ] && printf 'shared-display' || printf 'dedicated')
      SAFETY_BACKEND_REASON="highest-confidence native NVIDIA backend"
      ;;
    rocm)
      [ "$ROCM_COUNT" -gt 0 ] || { err "ROCm was requested but no ROCm device was classified"; exit 2; }
      SAFETY_DEVICE_IDS=("${ROCM_IDS[@]}"); SAFETY_SELECTED_SUMMARIES=("${ROCM_SUMMARIES[@]}")
      SAFETY_PREFLIGHT_DIRECTIVES=("${ROCM_PREFLIGHT[@]}")
      SAFETY_DEVICE_COUNT="$ROCM_COUNT"
      if [ "$ROCM_KNOWN_VRAM_COUNT" -eq "$ROCM_COUNT" ]; then
        SAFETY_AGGREGATE_DEVICE_MEMORY_MIB="$ROCM_TOTAL_VRAM_MIB"; SAFETY_DEVICE_MEMORY_KNOWN=1
      fi
      SAFETY_MIN_DEVICE_MEMORY_MIB="$ROCM_MIN_VRAM_MIB"; SAFETY_SHARED_ACCELERATOR="$ROCM_SHARED"
      SAFETY_BACKEND_CLASS=$([ "$ROCM_SHARED" = 1 ] && printf 'shared/integrated' || printf 'discrete')
      SAFETY_BACKEND_REASON="native AMD ROCm backend"
      ;;
    vulkan)
      [ "$VULKAN_COUNT" -gt 0 ] || { err "Vulkan was requested but no Vulkan accelerator was classified"; exit 2; }
      SAFETY_DEVICE_IDS=("${VULKAN_IDS[@]}"); SAFETY_SELECTED_SUMMARIES=("${VULKAN_SUMMARIES[@]}")
      SAFETY_PREFLIGHT_DIRECTIVES=("${VULKAN_PREFLIGHT[@]}")
      SAFETY_DEVICE_COUNT="$VULKAN_COUNT"
      SAFETY_SHARED_ACCELERATOR="$VULKAN_SHARED"; SAFETY_MIN_DEVICE_MEMORY_MIB=0
      SAFETY_BACKEND_CLASS=$([ "$VULKAN_SHARED" = 1 ] && printf 'shared/integrated' || printf 'discrete')
      SAFETY_BACKEND_REASON="portable GPU fallback with approximate memory telemetry"
      ;;
    metal)
      [ "$METAL_COUNT" -gt 0 ] || { err "Metal was requested but no Metal device was classified"; exit 2; }
      SAFETY_SELECTED_SUMMARIES=("${METAL_SUMMARIES[@]}")
      SAFETY_DEVICE_COUNT="$METAL_COUNT"
      SAFETY_SHARED_ACCELERATOR=1; SAFETY_MIN_DEVICE_MEMORY_MIB=0
      SAFETY_BACKEND_CLASS="unified-memory"; SAFETY_BACKEND_REASON="native Apple Metal backend"
      ;;
    cpu)
      SAFETY_DEVICE_COUNT=0; SAFETY_SHARED_ACCELERATOR=1; SAFETY_MIN_DEVICE_MEMORY_MIB=0
      SAFETY_SELECTED_SUMMARIES=("$HOST_CPU (${HOST_CPU_CORES} logical cores)")
      SAFETY_BACKEND_CLASS="host-memory"; SAFETY_BACKEND_REASON="no eligible accelerator or explicit CPU selection"
      ;;
  esac
  SAFETY_BACKEND="$requested"
}

detect_model_memory_profile() {
  SAFETY_LARGEST_MODEL_MIB=0
  SAFETY_LARGEST_MODEL_SOURCE=""
  SAFETY_OBSERVED_HOST_MIB=0

  if [ -n "${OLLAMA_SAFE_LARGEST_MODEL_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_LARGEST_MODEL_MIB "$OLLAMA_SAFE_LARGEST_MODEL_MIB"
    SAFETY_LARGEST_MODEL_MIB="$OLLAMA_SAFE_LARGEST_MODEL_MIB"
    SAFETY_LARGEST_MODEL_SOURCE="explicit override"
  else
    local -a roots=() candidates=()
    local candidate existing duplicate
    if [ -n "${OLLAMA_SAFE_MODEL_STORE:-}" ]; then
      candidates+=("$OLLAMA_SAFE_MODEL_STORE")
    else
      [ -n "${OLLAMA_MODELS:-}" ] && candidates+=("$OLLAMA_MODELS")
      candidates+=("/srv/ollama/models" "${HOME}/.ollama/models" \
        "/usr/share/ollama/.ollama/models" "/var/lib/ollama/.ollama/models" "/root/.ollama/models")
      candidates+=("${STORE_PATHS[@]:-}")

      local configured=""
      if [ "$HAS_SYSTEMD" = 1 ]; then
        configured=$(systemctl show ollama.service -p Environment --value 2>/dev/null \
          | grep -o 'OLLAMA_MODELS=[^[:space:]]*' | tail -n 1 || true)
        [ -n "$configured" ] && candidates+=("${configured#OLLAMA_MODELS=}")
      fi
      for existing in /etc/default/ollama /etc/environment; do
        [ -r "$existing" ] || continue
        while IFS= read -r configured; do
          configured="${configured#OLLAMA_MODELS=}"
          configured="${configured%\"}"; configured="${configured#\"}"
          configured="${configured%\'}"; configured="${configured#\'}"
          [ -n "$configured" ] && candidates+=("$configured")
        done < <(grep -E '^[[:space:]]*(export[[:space:]]+)?OLLAMA_MODELS=' "$existing" 2>/dev/null \
          | sed -E 's/^[[:space:]]*(export[[:space:]]+)?OLLAMA_MODELS=//' || true)
      done
    fi

    for candidate in "${candidates[@]}"; do
      [ -d "$candidate/manifests" ] || continue
      duplicate=0
      for existing in "${roots[@]:-}"; do
        [ "$candidate" = "$existing" ] && { duplicate=1; break; }
      done
      [ "$duplicate" = 1 ] || roots+=("$candidate")
    done

    local manifest bytes mib max_bytes=0
    for candidate in "${roots[@]:-}"; do
      while IFS= read -r -d '' manifest; do
        # Sum only inference payloads in each manifest. This includes split
        # model/projector/adapter/tensor layers while excluding templates,
        # licenses, prompts, and other metadata.
        bytes=$(awk 'BEGIN { RS="}" }
          /"mediaType"[[:space:]]*:[[:space:]]*"application\/vnd\.ollama\.image\.(model|projector|adapter|tensor)"/ {
            record=$0
            sub(/^.*"size"[[:space:]]*:[[:space:]]*/, "", record)
            sub(/[^0-9].*$/, "", record)
            if (record ~ /^[0-9]+$/) sum += record
          }
          END { printf "%.0f", sum + 0 }' "$manifest" 2>/dev/null || printf '0')
        [[ "$bytes" =~ ^[0-9]+$ ]] || bytes=0
        if [ "$bytes" -gt "$max_bytes" ]; then
          max_bytes="$bytes"
          SAFETY_LARGEST_MODEL_SOURCE="$manifest"
        fi
      done < <(find "$candidate/manifests" -type f -print0 2>/dev/null)
    done
    if [ "$max_bytes" -gt 0 ]; then
      mib=$(((max_bytes + 1048575) / 1048576))
      SAFETY_LARGEST_MODEL_MIB="$mib"
    fi
  fi

  if [ -n "${OLLAMA_SAFE_OBSERVED_HOST_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_OBSERVED_HOST_MIB "$OLLAMA_SAFE_OBSERVED_HOST_MIB"
    SAFETY_OBSERVED_HOST_MIB="$OLLAMA_SAFE_OBSERVED_HOST_MIB"
  elif command -v journalctl >/dev/null 2>&1; then
    local observed=""
    observed=$(journalctl -u ollama.service --no-pager --since '-30 days' --grep 'host memory' -n 512 2>/dev/null \
      | awk '/projected to use [0-9]+ MiB of host memory/ {
          line=$0
          sub(/^.*projected to use /, "", line)
          sub(/ .*/, "", line)
          if (line + 0 > max) max=line + 0
        }
        END { print max + 0 }' || true)
    [[ "$observed" =~ ^[0-9]+$ ]] && SAFETY_OBSERVED_HOST_MIB="$observed"
  fi
}

build_resource_limits() {
  # Host limits are measurements, not percentages selected for a particular
  # machine: normal pressure begins at the largest host projection Ollama has
  # reported, and the hard boundary is the largest installed inference payload.
  SAFETY_DEDICATED_VRAM_RATIO_PERCENT=0
  if [ "$SAFETY_DEVICE_MEMORY_KNOWN" = 1 ] && [ "$SAFETY_SHARED_ACCELERATOR" = 0 ] \
    && [ "$SAFETY_AGGREGATE_DEVICE_MEMORY_MIB" -gt 0 ]; then
    SAFETY_DEDICATED_VRAM_RATIO_PERCENT=$((SAFETY_AGGREGATE_DEVICE_MEMORY_MIB * 100 / SAFETY_HOST_TOTAL_MIB))
  fi

  local host_max=0 host_high=0
  if [ -n "${OLLAMA_SAFE_HOST_RESERVE_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_HOST_RESERVE_MIB "$OLLAMA_SAFE_HOST_RESERVE_MIB"
    [ "$OLLAMA_SAFE_HOST_RESERVE_MIB" -lt "$SAFETY_HOST_TOTAL_MIB" ] \
      || { err "OLLAMA_SAFE_HOST_RESERVE_MIB must be smaller than effective host memory"; exit 2; }
    host_max=$((SAFETY_HOST_TOTAL_MIB - OLLAMA_SAFE_HOST_RESERVE_MIB))
    host_high="$host_max"
    SAFETY_HOST_LIMIT_SOURCE="explicit host reserve"
  else
    [ "$SAFETY_LARGEST_MODEL_MIB" -gt 0 ] || {
      err "no installed Ollama inference payload was found; refusing to invent a host-memory limit"
      err "install a model, set OLLAMA_SAFE_MODEL_STORE, or explicitly set OLLAMA_SAFE_HOST_RESERVE_MIB"
      exit 2
    }
    host_max="$SAFETY_LARGEST_MODEL_MIB"
    [ "$SAFETY_OBSERVED_HOST_MIB" -gt "$host_max" ] && host_max="$SAFETY_OBSERVED_HOST_MIB"
    host_high="$SAFETY_OBSERVED_HOST_MIB"
    [ "$host_high" -gt 0 ] || host_high="$host_max"
    SAFETY_HOST_LIMIT_SOURCE="installed manifests and Ollama journal projections"
  fi
  if [ -n "${OLLAMA_SAFE_HOST_MEMORY_MAX_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_HOST_MEMORY_MAX_MIB "$OLLAMA_SAFE_HOST_MEMORY_MAX_MIB"
    host_max="$OLLAMA_SAFE_HOST_MEMORY_MAX_MIB"
    SAFETY_HOST_LIMIT_SOURCE="explicit host-memory boundary"
  fi
  if [ -n "${OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB:-}" ]; then
    require_uint_value OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB "$OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB"
    host_high="$OLLAMA_SAFE_HOST_MEMORY_HIGH_MIB"
  fi
  if [ "$host_max" -le 0 ] || [ "$host_max" -ge "$SAFETY_HOST_TOTAL_MIB" ]; then
    err "derived host-memory hard cap (${host_max} MiB) must be between 1 MiB and effective host RAM"
    exit 2
  fi
  if [ "$host_high" -le 0 ] || [ "$host_high" -gt "$host_max" ]; then
    err "host-memory throttle (${host_high} MiB) must be between 1 MiB and the hard cap"
    exit 2
  fi
  SAFETY_HOST_MEMORY_MAX_MIB="$host_max"
  SAFETY_HOST_MEMORY_HIGH_MIB="$host_high"
  SAFETY_HOST_RESERVE_MIB=$((SAFETY_HOST_TOTAL_MIB - SAFETY_HOST_MEMORY_MAX_MIB))

  local default_startup_headroom=2048
  [ "$host_max" -lt "$default_startup_headroom" ] \
    && default_startup_headroom="$host_max"
  SAFETY_STARTUP_HEADROOM_MIB="${OLLAMA_SAFE_STARTUP_HEADROOM_MIB:-$default_startup_headroom}"
  require_uint_value OLLAMA_SAFE_STARTUP_HEADROOM_MIB "$SAFETY_STARTUP_HEADROOM_MIB"
  if [ "$SAFETY_STARTUP_HEADROOM_MIB" -lt 1 ] \
    || [ "$SAFETY_STARTUP_HEADROOM_MIB" -gt "$SAFETY_HOST_MEMORY_MAX_MIB" ]; then
    err "OLLAMA_SAFE_STARTUP_HEADROOM_MIB must be between 1 MiB and the host-memory hard cap"
    exit 2
  fi

  # Ollama already measures live free VRAM when it schedules a load. Do not
  # subtract a guessed percentage a second time. A fixed carve-out exists only
  # when the operator explicitly requests one.
  SAFETY_VRAM_RESERVE_MIB=0
  if [ -n "${OLLAMA_SAFE_VRAM_RESERVE_MIB:-}" ] && [ "$SAFETY_BACKEND" != "cpu" ] && [ "$SAFETY_BACKEND" != "metal" ]; then
    require_uint_value OLLAMA_SAFE_VRAM_RESERVE_MIB "$OLLAMA_SAFE_VRAM_RESERVE_MIB"
    SAFETY_VRAM_RESERVE_MIB="$OLLAMA_SAFE_VRAM_RESERVE_MIB"
  fi
  if [ "$SAFETY_MIN_DEVICE_MEMORY_MIB" -gt 0 ] \
    && [ "$SAFETY_VRAM_RESERVE_MIB" -gt $((SAFETY_MIN_DEVICE_MEMORY_MIB / 2)) ]; then
    err "OLLAMA_SAFE_VRAM_RESERVE_MIB cannot exceed half of the smallest selected device"
    exit 2
  fi
  SAFETY_VRAM_RESERVE_BYTES=$((SAFETY_VRAM_RESERVE_MIB * 1024 * 1024))

  SAFETY_GPU_PREFERRED=0
  # Do not force every selected accelerator to participate in every load.
  # Ollama's native scheduler can then place against live free VRAM and only
  # split a model when the load actually requires multiple devices.
  SAFETY_SCHED_SPREAD=0
  if [[ "$SAFETY_BACKEND" =~ ^(cuda|rocm)$ ]] && [ "$SAFETY_SHARED_ACCELERATOR" = 0 ]; then
    SAFETY_GPU_PREFERRED=1
  fi
  if [ "$SAFETY_GPU_PREFERRED" = 1 ] && [ "$HOST_OS" = "Linux" ] \
    && [ "$HOST_SERVICE_MANAGER" = "systemd" ]; then
    SAFETY_NEGOTIATOR_ENABLED=1
  fi

  local default_context=8192 default_queue=64 default_models=1
  if [ "$SAFETY_HOST_TOTAL_MIB" -lt 8192 ]; then default_context=2048; default_queue=8
  elif [ "$SAFETY_HOST_TOTAL_MIB" -lt 16384 ]; then default_context=4096; default_queue=16; fi
  if [ "$SAFETY_BACKEND" = "cpu" ] && [ "$SAFETY_HOST_TOTAL_MIB" -lt 32768 ] && [ "$default_context" -gt 4096 ]; then
    default_context=4096
  fi

  SAFETY_CONTEXT_LENGTH="${OLLAMA_SAFE_CONTEXT_LENGTH:-$default_context}"
  SAFETY_NUM_PARALLEL="${OLLAMA_SAFE_NUM_PARALLEL:-1}"
  SAFETY_MAX_QUEUE="${OLLAMA_SAFE_MAX_QUEUE:-$default_queue}"
  SAFETY_KEEP_ALIVE="${OLLAMA_SAFE_KEEP_ALIVE:-5m}"
  SAFETY_MAX_LOADED_MODELS="${OLLAMA_SAFE_MAX_LOADED_MODELS:-$default_models}"
  if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then SAFETY_SWAP_MAX="${OLLAMA_SAFE_SWAP_MAX:-0}"
  elif [ "$SAFETY_HOST_TOTAL_MIB" -lt 16384 ]; then SAFETY_SWAP_MAX="${OLLAMA_SAFE_SWAP_MAX:-2G}"
  else SAFETY_SWAP_MAX="${OLLAMA_SAFE_SWAP_MAX:-8G}"; fi
  SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT="${OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT:-20}"
  SAFETY_CPU_QUOTA_PERCENT="${OLLAMA_SAFE_CPU_QUOTA_PERCENT:-400}"
  SAFETY_CPU_WEIGHT="${OLLAMA_SAFE_CPU_WEIGHT:-10}"
  SAFETY_IO_WEIGHT="${OLLAMA_SAFE_IO_WEIGHT:-10}"
  SAFETY_RESTART_POLICY="${OLLAMA_SAFE_RESTART_POLICY:-on-success}"

  local host_cpu_capacity=$((HOST_CPU_CORES * 100))
  [ "$host_cpu_capacity" -ge 100 ] || host_cpu_capacity=100

  require_uint_value OLLAMA_SAFE_CONTEXT_LENGTH "$SAFETY_CONTEXT_LENGTH"
  require_uint_value OLLAMA_SAFE_NUM_PARALLEL "$SAFETY_NUM_PARALLEL"
  require_uint_value OLLAMA_SAFE_MAX_QUEUE "$SAFETY_MAX_QUEUE"
  require_uint_value OLLAMA_SAFE_MAX_LOADED_MODELS "$SAFETY_MAX_LOADED_MODELS"
  require_uint_value OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT "$SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT"
  require_uint_value OLLAMA_SAFE_CPU_QUOTA_PERCENT "$SAFETY_CPU_QUOTA_PERCENT"
  require_uint_value OLLAMA_SAFE_CPU_WEIGHT "$SAFETY_CPU_WEIGHT"
  require_uint_value OLLAMA_SAFE_IO_WEIGHT "$SAFETY_IO_WEIGHT"
  [ "$SAFETY_NUM_PARALLEL" -ge 1 ] || { err "OLLAMA_SAFE_NUM_PARALLEL must be at least 1"; exit 2; }
  [ "$SAFETY_MAX_LOADED_MODELS" -ge 1 ] || { err "OLLAMA_SAFE_MAX_LOADED_MODELS must be at least 1"; exit 2; }
  if [ "$SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT" -lt 1 ] \
    || [ "$SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT" -gt 100 ]; then
    err "OLLAMA_SAFE_MEMORY_PRESSURE_LIMIT_PERCENT must be between 1 and 100"; exit 2
  fi
  [ "$SAFETY_CPU_QUOTA_PERCENT" -ge 100 ] \
    || { err "OLLAMA_SAFE_CPU_QUOTA_PERCENT must be at least 100"; exit 2; }
  [ "$SAFETY_CPU_QUOTA_PERCENT" -gt "$host_cpu_capacity" ] && SAFETY_CPU_QUOTA_PERCENT="$host_cpu_capacity"
  if [ "$SAFETY_CPU_WEIGHT" -lt 1 ] || [ "$SAFETY_CPU_WEIGHT" -gt 10000 ]; then
    err "OLLAMA_SAFE_CPU_WEIGHT must be between 1 and 10000"; exit 2
  fi
  if [ "$SAFETY_IO_WEIGHT" -lt 1 ] || [ "$SAFETY_IO_WEIGHT" -gt 10000 ]; then
    err "OLLAMA_SAFE_IO_WEIGHT must be between 1 and 10000"; exit 2
  fi
  case "$SAFETY_RESTART_POLICY" in
    no|on-success|on-failure) ;;
    *) err "OLLAMA_SAFE_RESTART_POLICY must be no, on-success, or on-failure"; exit 2 ;;
  esac
  [[ "$SAFETY_KEEP_ALIVE" =~ ^[0-9]+(ms|s|m|h)$ ]] || { err "OLLAMA_SAFE_KEEP_ALIVE must be a finite duration such as 5m"; exit 2; }
  [[ "$SAFETY_SWAP_MAX" =~ ^(0|[0-9]+[KMGT])$ ]] || { err "OLLAMA_SAFE_SWAP_MAX must be 0 or a systemd size such as 8G"; exit 2; }
}

build_safety_profile() {
  [ "$SAFETY_READY" = 1 ] && return 0
  detect_host_profile
  detect_cuda_devices
  detect_rocm_devices
  detect_vulkan_devices
  detect_metal_devices
  detect_unconfigured_accelerators
  select_safety_backend
  detect_model_memory_profile
  build_resource_limits
  SAFETY_READY=1
}

print_safety_profile() {
  hdr "Host classification"
  say "  Platform: $HOST_NAME ($HOST_OS/$HOST_ARCH; $HOST_VIRTUALIZATION)"
  say "  CPU: $HOST_CPU — $HOST_CPU_CORES logical cores"
  if [ "$SAFETY_PHYSICAL_MEMORY_MIB" -ne "$SAFETY_HOST_TOTAL_MIB" ]; then
    say "  Memory: ${SAFETY_PHYSICAL_MEMORY_MIB} MiB physical; ${SAFETY_HOST_TOTAL_MIB} MiB effective ($HOST_MEMORY_SOURCE)"
  else
    say "  Memory: ${SAFETY_HOST_TOTAL_MIB} MiB ($HOST_MEMORY_SOURCE)"
  fi
  if [ "$HOST_SERVICE_MANAGER" = "systemd" ]; then
    say "  Class: $SAFETY_HOST_CLASS; service manager: systemd $SYSTEMD_VERSION"
  else
    say "  Class: $SAFETY_HOST_CLASS; service manager: $HOST_SERVICE_MANAGER"
  fi

  hdr "Accelerator classification"
  if [ ${#ACCELERATOR_SUMMARIES[@]} -gt 0 ]; then printf '  %s\n' "${ACCELERATOR_SUMMARIES[@]}"
  else say "  No usable accelerator telemetry; CPU fallback is available."; fi

  hdr "Selected Ollama safety policy"
  say "  Backend: $SAFETY_BACKEND ($SAFETY_BACKEND_CLASS) — $SAFETY_BACKEND_REASON"
  printf '  Device: %s\n' "${SAFETY_SELECTED_SUMMARIES[@]}"
  if [ "$SAFETY_VRAM_RESERVE_MIB" -gt 0 ]; then
    say "  Device memory: explicit ${SAFETY_VRAM_RESERVE_MIB} MiB carve-out per selected accelerator"
  elif [ "$SAFETY_DEVICE_COUNT" -gt 0 ]; then
    say "  Device memory: live free-VRAM telemetry; no guessed fixed carve-out"
  fi
  if [ "$SAFETY_DEVICE_MEMORY_KNOWN" = 1 ] && [ "$SAFETY_SHARED_ACCELERATOR" = 0 ]; then
    say "  Aggregate dedicated device memory: ${SAFETY_AGGREGATE_DEVICE_MEMORY_MIB} MiB across ${SAFETY_DEVICE_COUNT} accelerator(s); ${SAFETY_DEDICATED_VRAM_RATIO_PERCENT}% of host RAM"
  fi
  if [ "$SAFETY_LARGEST_MODEL_MIB" -gt 0 ]; then
    say "  Model scan: largest installed inference payload ${SAFETY_LARGEST_MODEL_MIB} MiB ($SAFETY_LARGEST_MODEL_SOURCE)"
  fi
  if [ "$SAFETY_OBSERVED_HOST_MIB" -gt 0 ]; then
    say "  Ollama history: largest observed host projection ${SAFETY_OBSERVED_HOST_MIB} MiB"
  fi
  say "  Host memory: ${SAFETY_STARTUP_HEADROOM_MIB} MiB startup headroom; throttle at ${SAFETY_HOST_MEMORY_HIGH_MIB} MiB; hard cap at ${SAFETY_HOST_MEMORY_MAX_MIB} MiB; ${SAFETY_HOST_RESERVE_MIB} MiB remains outside the cgroup"
  say "  Host limit basis: $SAFETY_HOST_LIMIT_SOURCE"
  say "  Scheduler: ${SAFETY_MAX_LOADED_MODELS} model(s), ${SAFETY_NUM_PARALLEL} parallel request(s), ${SAFETY_CONTEXT_LENGTH}-token context, queue ${SAFETY_MAX_QUEUE}"
  if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then
    say "  GPU policy: native live-VRAM placement; forced spread disabled; pageable/cgroup-bounded CPU overflow only"
    say "  GPU host paths: unified spill and pinned-host buffers disabled"
  fi
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    say "  GPU negotiator: cooperative leases plus anonymous-process rebalance; Ollama refits after external allocation"
  fi
  if [ "$HOST_SERVICE_MANAGER" = "systemd" ]; then
    say "  Containment: memory PSI ${SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT}%; CPU quota ${SAFETY_CPU_QUOTA_PERCENT}%; restart $SAFETY_RESTART_POLICY"
  fi
  if [ "$HOST_SERVICE_MANAGER" = "systemd" ] && [ "$SYSTEMD_VERSION" -lt 231 ]; then
    warn "systemd $SYSTEMD_VERSION is too old for MemoryHigh/MemoryMax; scheduler limits still apply."
  elif [ "$HOST_SERVICE_MANAGER" != "systemd" ]; then
    warn "Native cgroup OOM containment is unavailable under $HOST_SERVICE_MANAGER; scheduler limits still apply."
  fi
}

csv_from_array() { local IFS=,; printf '%s' "$*"; }

render_safety_environment_directives() {
  local ids
  ids=$(csv_from_array "${SAFETY_DEVICE_IDS[@]}")
  case "$SAFETY_BACKEND" in
    cuda)
      printf '%s\n' "Environment=\"CUDA_VISIBLE_DEVICES=$ids\"" "Environment=\"HIP_VISIBLE_DEVICES=-1\"" \
        "Environment=\"ROCR_VISIBLE_DEVICES=-1\"" "Environment=\"GPU_DEVICE_ORDINAL=-1\"" \
        "Environment=\"GGML_VK_VISIBLE_DEVICES=-1\"" "Environment=\"OLLAMA_VULKAN=0\"" "Environment=\"OLLAMA_IGPU_ENABLE=0\""
      ;;
    rocm)
      printf '%s\n' "Environment=\"ROCR_VISIBLE_DEVICES=$ids\"" \
        "Environment=\"GGML_VK_VISIBLE_DEVICES=-1\"" "Environment=\"OLLAMA_VULKAN=0\"" \
        "Environment=\"OLLAMA_IGPU_ENABLE=$SAFETY_SHARED_ACCELERATOR\""
      ;;
    vulkan)
      printf '%s\n' "Environment=\"CUDA_VISIBLE_DEVICES=-1\"" "Environment=\"HIP_VISIBLE_DEVICES=-1\"" \
        "Environment=\"ROCR_VISIBLE_DEVICES=-1\"" "Environment=\"GPU_DEVICE_ORDINAL=-1\"" \
        "Environment=\"GGML_VK_VISIBLE_DEVICES=$ids\"" "Environment=\"OLLAMA_VULKAN=1\"" \
        "Environment=\"OLLAMA_IGPU_ENABLE=$SAFETY_SHARED_ACCELERATOR\""
      ;;
    cpu)
      printf '%s\n' "Environment=\"CUDA_VISIBLE_DEVICES=-1\"" "Environment=\"HIP_VISIBLE_DEVICES=-1\"" \
        "Environment=\"ROCR_VISIBLE_DEVICES=-1\"" "Environment=\"GPU_DEVICE_ORDINAL=-1\"" \
        "Environment=\"GGML_VK_VISIBLE_DEVICES=-1\"" "Environment=\"OLLAMA_VULKAN=0\"" "Environment=\"OLLAMA_IGPU_ENABLE=0\""
      ;;
  esac
  printf '%s\n' \
    "Environment=\"OLLAMA_MAX_LOADED_MODELS=${SAFETY_MAX_LOADED_MODELS}\"" \
    "Environment=\"OLLAMA_NUM_PARALLEL=${SAFETY_NUM_PARALLEL}\"" \
    "Environment=\"OLLAMA_SCHED_SPREAD=${SAFETY_SCHED_SPREAD}\"" \
    "Environment=\"OLLAMA_CONTEXT_LENGTH=${SAFETY_CONTEXT_LENGTH}\"" \
    "Environment=\"OLLAMA_KEEP_ALIVE=${SAFETY_KEEP_ALIVE}\"" \
    "Environment=\"OLLAMA_MAX_QUEUE=${SAFETY_MAX_QUEUE}\"" \
    "Environment=\"OLLAMA_FLASH_ATTENTION=1\"" \
    "Environment=\"OLLAMA_KV_CACHE_TYPE=q8_0\""
  if [ "$SAFETY_VRAM_RESERVE_MIB" -gt 0 ]; then
    printf '%s\n' "Environment=\"OLLAMA_GPU_OVERHEAD=${SAFETY_VRAM_RESERVE_BYTES}\""
  fi
  if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then
    printf '%s\n' \
      "Environment=\"LLAMA_ARG_N_GPU_LAYERS=auto\"" \
      "Environment=\"LLAMA_ARG_SPLIT_MODE=layer\"" \
      "Environment=\"LLAMA_ARG_FIT=on\"" \
      "Environment=\"GGML_CUDA_NO_PINNED=1\""
  fi
}

render_safety_shell_exports() {
  if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then
    printf '%s\n' "unset GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_CUDA_REGISTER_HOST LLAMA_ARG_FIT_TARGET"
  else
    printf '%s\n' "unset GGML_CUDA_NO_PINNED LLAMA_ARG_N_GPU_LAYERS LLAMA_ARG_SPLIT_MODE LLAMA_ARG_FIT LLAMA_ARG_FIT_TARGET"
  fi
  if [ "$SAFETY_VRAM_RESERVE_MIB" -eq 0 ]; then
    printf '%s\n' "unset OLLAMA_GPU_OVERHEAD"
  fi
  if [ "$SAFETY_BACKEND" = "rocm" ]; then
    printf '%s\n' "unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL"
  fi
  local line assignment
  while IFS= read -r line; do
    [[ "$line" == Environment=\"*\" ]] || continue
    assignment="${line#Environment=\"}"
    assignment="${assignment%\"}"
    printf 'export %q\n' "$assignment"
  done < <(render_safety_environment_directives)
}

render_safety_preflight_script() {
  cat <<'PREFLIGHT'
#!/bin/sh
# Generated by ollama-unify. Refuse to start the empty Ollama API daemon when
# MemAvailable cannot cover its bounded startup headroom or the host is stalled
# in reclaim. Model lanes have separate live host-memory admission checks, and
# MemoryMax remains the cgroup hard boundary. On modern systemd this runs as an
# ExecCondition, so a refusal skips startup without marking the unit failed.
set -eu

required_mib=${1:-}
pressure_limit=${2:-}
case "$required_mib:$pressure_limit" in
  *[!0-9:]*|:|*:)
    echo "ollama-unify preflight: invalid required memory or pressure limit" >&2
    exit 75
    ;;
esac

mem_available_kib=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo 2>/dev/null || true)
case "$mem_available_kib" in
  ''|*[!0-9]*)
    echo "ollama-unify preflight: cannot read MemAvailable" >&2
    exit 75
    ;;
esac
mem_available_mib=$((mem_available_kib / 1024))
if [ "$mem_available_mib" -lt "$required_mib" ]; then
  echo "ollama-unify preflight: refusing start; ${mem_available_mib} MiB available, ${required_mib} MiB startup headroom required" >&2
  exit 75
fi

if [ -r /proc/pressure/memory ]; then
  full_avg10=$(awk '/^full / { for (i=1; i<=NF; i++) if ($i ~ /^avg10=/) { sub(/^avg10=/, "", $i); print $i; exit } }' /proc/pressure/memory)
  if [ -n "$full_avg10" ] && awk -v actual="$full_avg10" -v limit="$pressure_limit" 'BEGIN { exit !(actual >= limit) }'; then
    echo "ollama-unify preflight: refusing start; memory full avg10=${full_avg10}% (limit ${pressure_limit}%)" >&2
    exit 75
  fi
fi
exit 0
PREFLIGHT
}

render_gpu_preflight_script() {
  cat <<'PREFLIGHT'
#!/bin/sh
# Generated by ollama-unify. Validate only the selected CUDA UUID. Some
# nvidia-smi releases return 255 for the whole inventory when an unrelated GPU
# is unavailable even though they still print healthy rows. The selected GPU
# is safe to use only when its complete telemetry row is present.
set -eu

tool=${1:-}
wanted=${2:-}
if [ -z "$tool" ] || [ -z "$wanted" ] || [ ! -x "$tool" ]; then
  echo "ollama-unify GPU preflight: invalid tool or selected UUID" >&2
  exit 75
fi

inventory=$(
  "$tool" --query-gpu=uuid,memory.total,compute_cap \
    --format=csv,noheader,nounits 2>/dev/null || true
)
if printf '%s\n' "$inventory" | awk -F, -v wanted="$wanted" '
  {
    uuid=$1; total=$2; compute=$3
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", total)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", compute)
    if (uuid == wanted && total ~ /^[0-9]+$/ && compute ~ /^[0-9]+([.][0-9]+)?$/) {
      found=1
    }
  }
  END { exit !found }
'; then
  exit 0
fi

echo "ollama-unify GPU preflight: selected CUDA device unavailable: $wanted" >&2
exit 75
PREFLIGHT
}

render_gpu_negotiator_script() {
  cat <<'NEGOTIATOR'
#!/usr/bin/env python3
# Generated by ollama-unify. This process owns Ollama's public HTTP port,
# drains in-flight requests around cooperative GPU leases, unloads resident
# runners, and lets Ollama refit from live VRAM after external allocation.
from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import http.server
import json
import logging
import math
import os
import pwd
import re
import secrets
import select
import shlex
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs


def env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
        return value if value >= 0 else default
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def model_context_profiles(raw: str) -> dict[str, dict[str, object]]:
    """Parse operator-scoped fixed context and memory admission profiles."""
    profiles = json.loads(raw or "{}")
    if not isinstance(profiles, dict):
        raise ValueError("model context profiles must be a JSON object")
    for model, profile in profiles.items():
        if not isinstance(model, str) or ":" not in model or not isinstance(profile, dict):
            raise ValueError("model context profile requires an exact tagged model")
        if set(profile) != {"context_length", "extra_vram_mib", "model_digest"}:
            raise ValueError("model context profile has unexpected fields")
        context = profile["context_length"]
        reserve = profile["extra_vram_mib"]
        digest = profile["model_digest"]
        if type(context) is not int or not 1 <= context <= 2 ** 31 - 1:
            raise ValueError("model context profile length is outside supported bounds")
        if type(reserve) is not int or reserve <= 0:
            raise ValueError("model context profile memory allowance is insufficient")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("model context profile requires an exact SHA-256 digest")
    return profiles


def effective_model_context_profile(model: str) -> dict | None:
    model = canonical_model_tag(model)
    return RESOLVED_MODEL_CONTEXT_PROFILES.get(model) or MODEL_CONTEXT_PROFILES.get(model)


def estimate_model_context_memory(info: dict, context: int) -> dict:
    """Bound KV and fp32 recurrent state from the exact GGUF metadata."""
    architecture = info.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise PermanentCapacityError("model architecture is unavailable for context memory admission", 422,
                                     "model_context_memory_unverified")
    def positive(key: str) -> int:
        value = info.get(f"{architecture}.{key}")
        if type(value) is not int or value <= 0:
            raise PermanentCapacityError(f"model context memory metadata is unavailable: {architecture}.{key}",
                                         422, "model_context_memory_unverified")
        return value
    blocks = positive("block_count")
    kv_heads = positive("attention.head_count_kv")
    key = info.get(f"{architecture}.attention.key_length")
    value = info.get(f"{architecture}.attention.value_length")
    if type(key) is not int or key <= 0 or type(value) is not int or value <= 0:
        embedding, heads = positive("embedding_length"), positive("attention.head_count")
        if embedding % heads:
            raise PermanentCapacityError("model attention dimensions cannot be verified", 422,
                                         "model_context_memory_unverified")
        key = value = embedding // heads
    attention_blocks = blocks
    recurrent_bytes = 0
    interval = info.get(f"{architecture}.full_attention_interval")
    has_ssm = any(name.startswith(f"{architecture}.ssm.") for name in info)
    if interval is not None or has_ssm:
        if architecture not in ("qwen35", "qwen35moe") or type(interval) is not int or interval <= 0:
            raise PermanentCapacityError("hybrid context memory layout cannot be verified", 422,
                                         "model_context_memory_unverified")
        attention_blocks = math.ceil(blocks / interval)
        inner, state = positive("ssm.inner_size"), positive("ssm.state_size")
        groups, convolution = positive("ssm.group_count"), positive("ssm.conv_kernel")
        recurrent_bytes = (blocks - attention_blocks) * (
            inner * state + (inner + 2 * groups * state) * max(0, convolution - 1)
        ) * 4
    # Match Ollama's model-level Flash Attention eligibility. A requested
    # quantized cache falls back to f16 when that model cannot use it.
    quantized = (architecture in ("qwen35", "qwen35moe", "qwen3next")
                 or (architecture != "gemma2" and key == value))
    quantized = quantized and f"{architecture}.pooling_type" not in info
    # q8_0 stores 32 int8 values plus a two-byte scale per block. Round
    # dimensions independently so non-multiples cannot under-reserve storage.
    token_bytes = ((math.ceil(key / 32) + math.ceil(value / 32)) * 34
                   if quantized else (key + value) * 2)
    kv_bytes = context * attention_blocks * kv_heads * token_bytes
    kv_mib = math.ceil(kv_bytes / (1024 * 1024))
    recurrent_mib = math.ceil(recurrent_bytes / (1024 * 1024))
    # Reserve a bounded workspace in addition to the existing model/VRAM
    # margins; each parallel slot owns its complete context allocation.
    reserve = math.ceil((kv_mib + recurrent_mib + 1024) / 256) * 256
    return {"extra_vram_mib": reserve, "kv_cache_type": "q8_0" if quantized else "f16",
            "kv_cache_mib": kv_mib, "recurrent_state_mib": recurrent_mib,
            "attention_blocks": attention_blocks}


def resolve_model_context_profile(model: str, match: dict | None = None) -> dict | None:
    """Resolve an exact installed artifact's maximum before managed admission."""
    model = canonical_model_tag(model)
    configured = MODEL_CONTEXT_PROFILES.get(model)
    if not configured and (not POOL_ENABLED or not AUTO_MODEL_CONTEXT):
        return None
    with MODEL_CONTEXT_LOCK:
        cached = RESOLVED_MODEL_CONTEXT_PROFILES.get(model)
        resolution = MODEL_CONTEXT_RESOLUTIONS.get(model, {})
        if (match is None and cached and time.monotonic() - resolution.get("checked_at", 0) < 5):
            return cached
        if match is None:
            tags = backend_json("GET", "/api/tags", timeout=10.0).get("models", [])
            match = next((item for item in tags if isinstance(item, dict)
                          and canonical_model_tag(str(item.get("name") or item.get("model") or "")) == model), None)
        if not isinstance(match, dict):
            raise PermanentCapacityError(f"model {model!r} is not installed", 404, "model_not_installed")
        digest = match.get("digest")
        if configured and digest != configured["model_digest"]:
            raise PermanentCapacityError("model context profile digest differs from installed artifact", 422,
                                         "model_context_identity_mismatch")
        tag_capabilities = match.get("capabilities")
        if not configured and isinstance(tag_capabilities, list) and tag_capabilities and "completion" not in tag_capabilities:
            return None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PermanentCapacityError("model context requires an exact installed SHA-256 digest", 422,
                                         "model_context_identity_mismatch")
        if cached and cached["model_digest"] == digest:
            return cached
        metadata = backend_json("POST", "/api/show", {"model": model}, timeout=10.0)
        capabilities = metadata.get("capabilities", tag_capabilities)
        if not configured and isinstance(capabilities, list) and capabilities and "completion" not in capabilities:
            return None
        current_tags = backend_json("GET", "/api/tags", timeout=10.0).get("models", [])
        current = next((item for item in current_tags if isinstance(item, dict)
                        and canonical_model_tag(str(item.get("name") or item.get("model") or "")) == model), {})
        if current.get("digest") != digest:
            raise CapacityError("model artifact changed during context verification", 503,
                                "model_context_identity_changed")
        info = metadata.get("model_info", {})
        architecture = info.get("general.architecture") if isinstance(info, dict) else None
        maximum = info.get(f"{architecture}.context_length") if isinstance(info, dict) else None
        if type(maximum) is not int or not 1 <= maximum <= 2 ** 31 - 1:
            raise PermanentCapacityError("model maximum context cannot be verified from exact metadata", 422,
                                         "model_context_limit_unverified")
        if configured and configured["context_length"] > maximum:
            raise PermanentCapacityError("model context profile exceeds verified model context metadata", 422,
                                         "model_context_limit_unverified")
        context = configured["context_length"] if configured and not AUTO_MODEL_CONTEXT else maximum
        reason = "operator_model_profile" if configured and context < maximum else None
        if HARD_MAX_CONTEXT > 0 and context > HARD_MAX_CONTEXT:
            context, reason = HARD_MAX_CONTEXT, "operator_hard_context_limit"
        try:
            memory = estimate_model_context_memory(info, context)
            memory["memory_estimate_source"] = "exact_model_metadata"
        except PermanentCapacityError:
            # Retain the original conservative operator contract only when
            # this exact artifact is not being grown. A small manual reserve
            # must never bypass available KV geometry or authorize growth.
            if (not configured or context > configured["context_length"]
                    or configured["extra_vram_mib"] < math.ceil(context / 16)):
                raise
            memory = {"memory_estimate_source": "operator_digest_bound_reserve"}
        profile = {"context_length": context,
                   "extra_vram_mib": max(configured["extra_vram_mib"] if configured else 0,
                                         memory.get("extra_vram_mib", 0)),
                   "model_digest": digest}
        RESOLVED_MODEL_CONTEXT_PROFILES[model] = profile
        parameters = metadata.get("parameters", "")
        configured_context = re.search(r"(?m)^num_ctx\s+(\d+)\s*$", parameters) if isinstance(parameters, str) else None
        MODEL_CONTEXT_RESOLUTIONS[model] = {
            "model_max_context": maximum, "context_length": context, "model_digest": digest,
            "source": "operator_profile" if configured else "exact_model_metadata",
            "limit_reason": reason, "checked_at": time.monotonic(), **memory,
            "legacy_profile_upgraded": bool(configured and context > configured["context_length"]),
            "openai_context_compatible": configured_context is None or int(configured_context[1]) == context,
        }
        return profile


def verified_model_context(model: str, resident: list[dict], profile: dict | None = None) -> int | None:
    """Require runtime evidence of a profiled lane's admitted per-slot context."""
    profile = profile or effective_model_context_profile(model)
    if profile is None:
        return None
    resolved = next((item for item in resident if isinstance(item, dict)
                     and canonical_model_tag(str(item.get("name") or item.get("model") or "")) == model), {})
    actual = resolved.get("context_length")
    if resolved.get("digest") != profile["model_digest"]:
        raise PermanentCapacityError("managed model artifact differs from admitted context identity", 422,
                                     "model_context_identity_mismatch")
    if type(actual) is not int or actual != profile["context_length"]:
        raise PermanentCapacityError(
            "managed model did not resolve the admitted per-slot context", 422,
            "model_context_runtime_mismatch",
        )
    return actual


def env_owner_gpu_scopes(name: str) -> dict[str, list[str]]:
    scopes: dict[str, list[str]] = {}
    for item in os.environ.get(name, "").split(";"):
        owner, separator, values = item.partition("=")
        gpu_uuids = [value.strip() for value in values.split(",") if value.strip()]
        if separator and owner.strip() and gpu_uuids:
            scopes[owner.strip()] = list(dict.fromkeys(gpu_uuids))
    return scopes


def env_model_gpu_preferences(
    name: str, selected_gpus: list[str],
) -> dict[str, list[str]]:
    """Parse soft, ordered per-model GPU placement preferences."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    preferences = json.loads(raw)
    if not isinstance(preferences, dict):
        raise ValueError("model GPU preferences must be a JSON object")
    selected = set(selected_gpus)
    normalized: dict[str, list[str]] = {}
    for model, values in preferences.items():
        if not isinstance(model, str) or ":" not in model:
            raise ValueError("model GPU preference requires an exact tagged model")
        if not isinstance(values, list) or not values:
            raise ValueError("model GPU preference requires a non-empty GPU list")
        gpu_uuids = list(dict.fromkeys(values))
        if any(not isinstance(value, str) or not value for value in gpu_uuids):
            raise ValueError("model GPU preference contains an invalid GPU UUID")
        if selected and any(value not in selected for value in gpu_uuids):
            raise ValueError("model GPU preference contains an unselected GPU")
        normalized[model] = gpu_uuids
    return normalized


def load_environment_file(path: str) -> None:
    """Load the installer's simple quoted KEY=VALUE file for standalone CLI calls."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("OLLAMA_UNIFY_") or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def split_address(value: str) -> tuple[str, int]:
    value = value.removeprefix("http://").removeprefix("https://").rstrip("/")
    if value.startswith("[") and "]:" in value:
        host, port = value[1:].split("]:", 1)
        return host, int(port)
    if ":" not in value:
        return value, 11434
    host, port = value.rsplit(":", 1)
    return host, int(port)


load_environment_file(os.environ.get(
    "OLLAMA_UNIFY_CONFIG", "/etc/default/ollama-unify-negotiator"
))
BACKEND_HOST, BACKEND_PORT = split_address(os.environ.get("OLLAMA_UNIFY_BACKEND", "127.0.0.1:11436"))
LISTEN_HOST, LISTEN_PORT = split_address(os.environ.get("OLLAMA_UNIFY_LISTEN", "127.0.0.1:11434"))
CONTROL_SOCKET = os.environ.get("OLLAMA_UNIFY_SOCKET", "/run/ollama-unify/gpu-negotiator.sock")
INFERENCE_HOLD_FILE = os.environ.get("OLLAMA_UNIFY_INFERENCE_HOLD_FILE", "")
DRAIN_TIMEOUT = env_float("OLLAMA_UNIFY_DRAIN_TIMEOUT", 300.0)
PENDING_TIMEOUT = env_float("OLLAMA_UNIFY_PENDING_TIMEOUT", 300.0)
UNLOAD_TIMEOUT = env_float("OLLAMA_UNIFY_UNLOAD_TIMEOUT", 120.0)
# A revoked lease waits for its owner to free CUDA memory and release. A dead
# owner never returns, so revocation needs its own deadline. Past it, the
# broker abandons the lease and reclaims the scope; live free-VRAM telemetry,
# not the lease table, remains authoritative for placement.
REVOKE_TIMEOUT = env_float("OLLAMA_UNIFY_REVOKE_TIMEOUT", 300.0)
DEFAULT_LEASE_TTL = env_int("OLLAMA_UNIFY_LEASE_TTL", 300)
HEARTBEAT_TIMEOUT = env_float("OLLAMA_UNIFY_HEARTBEAT_TIMEOUT", 10.0)
HANDOFF_TIMEOUT = max(1.0, min(600.0, env_float("OLLAMA_UNIFY_HANDOFF_TIMEOUT", 180.0)))
HEARTBEAT_RECONNECT_GRACE = env_float(
    "OLLAMA_UNIFY_HEARTBEAT_RECONNECT_GRACE", 90.0
)
MAX_CONTEXT = env_int("OLLAMA_UNIFY_MAX_CONTEXT", 0)
MODEL_CONTEXT_PROFILES = model_context_profiles(
    os.environ.get("OLLAMA_UNIFY_MODEL_CONTEXT_PROFILES", "{}")
)
AUTO_MODEL_CONTEXT = env_bool("OLLAMA_UNIFY_AUTO_MODEL_CONTEXT", True)
HARD_MAX_CONTEXT = max(0, env_int("OLLAMA_UNIFY_CONTEXT_HARD_LIMIT", 0))
RESOLVED_MODEL_CONTEXT_PROFILES: dict[str, dict] = {}
MODEL_CONTEXT_RESOLUTIONS: dict[str, dict] = {}
MODEL_CONTEXT_LOCK = threading.RLock()
CONTEXT_PROFILE_UNSET = object()
ANON_POLL = env_float("OLLAMA_UNIFY_ANON_POLL", 0.5)
ANON_SETTLE = env_float("OLLAMA_UNIFY_ANON_SETTLE", 2.0)
ANON_MAX_DRAIN = env_float("OLLAMA_UNIFY_ANON_MAX_DRAIN", 15.0)
FOREIGN_RELEASE_TOLERANCE_MIB = max(
    0, env_int("OLLAMA_UNIFY_FOREIGN_RELEASE_TOLERANCE_MIB", 256)
)
BACKEND_TYPE = os.environ.get("OLLAMA_UNIFY_BACKEND_TYPE", "unknown")
SELECTED_GPUS = [value for value in os.environ.get("OLLAMA_UNIFY_SELECTED_GPUS", "").split(",") if value]
MODEL_GPU_PREFERENCES = env_model_gpu_preferences(
    "OLLAMA_UNIFY_MODEL_GPU_PREFERENCES", SELECTED_GPUS
)
OWNER_GPU_SCOPES = env_owner_gpu_scopes("OLLAMA_UNIFY_OWNER_GPU_SCOPES")
LEASE_STATE_PATH = Path(os.environ.get(
    "OLLAMA_UNIFY_LEASE_STATE", "/var/lib/ollama-unify/leases.json"
))
MODEL_POLICY_PATH = Path(os.environ.get(
    "OLLAMA_UNIFY_MODEL_POLICY_STATE",
    str(LEASE_STATE_PATH.with_name("model-gpu-policy.json")),
))
CACHE_STATE_PATH = Path(os.environ.get(
    "OLLAMA_UNIFY_CACHE_RESIDENCY_STATE", str(LEASE_STATE_PATH.with_name("cache-residency.json"))
))
CACHE_STARTUP_WRAPPER = """import os, sys
fd = int(sys.argv[1])
try:
    permit = os.read(fd, 1)
finally:
    os.close(fd)
if permit != b'\\x01':
    sys.exit(125)
os.execve(sys.argv[2], sys.argv[2:], os.environ)
"""
POOL_ENABLED = env_bool(
    "OLLAMA_UNIFY_POOL_ENABLED", BACKEND_TYPE == "cuda" and bool(SELECTED_GPUS)
)
POOL_MAX_SERVERS = env_int(
    "OLLAMA_UNIFY_POOL_MAX_SERVERS", max(1, len(SELECTED_GPUS) * 2)
)
POOL_MAX_QUEUE = max(1, env_int("OLLAMA_UNIFY_POOL_MAX_QUEUE", 64))
POOL_RESUME_TTL = max(
    0.1, env_float("OLLAMA_UNIFY_POOL_RESUME_TTL", 30.0)
)
RETAINED_REQUEST_MAX_BODY_BYTES = max(
    0, env_int("OLLAMA_UNIFY_RETAINED_REQUEST_MAX_BODY_BYTES", 16 * 1024 * 1024)
)
RETAINED_REQUEST_MAX_TOTAL_BYTES = max(
    0, env_int("OLLAMA_UNIFY_RETAINED_REQUEST_MAX_TOTAL_BYTES", 128 * 1024 * 1024)
)
COMPLETED_RESPONSE_TTL = max(
    0.1, env_float("OLLAMA_UNIFY_COMPLETED_RESPONSE_TTL", 120.0)
)
COMPLETED_RESPONSE_MAX_ENTRIES = max(
    1, env_int("OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_ENTRIES", 64)
)
COMPLETED_RESPONSE_MAX_BODY_BYTES = max(
    0, env_int("OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_BODY_BYTES", 8 * 1024 * 1024)
)
COMPLETED_RESPONSE_MAX_TOTAL_BYTES = max(
    0, env_int("OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_TOTAL_BYTES", 64 * 1024 * 1024)
)
POOL_PORT_START = env_int("OLLAMA_UNIFY_POOL_PORT_START", BACKEND_PORT + 1)
POOL_INSTANCE_PARALLEL = max(1, env_int("OLLAMA_UNIFY_POOL_INSTANCE_PARALLEL", 1))
POOL_IDLE_TIMEOUT = env_float("OLLAMA_UNIFY_POOL_IDLE_TIMEOUT", 300.0)
POOL_READY_TIMEOUT = env_float("OLLAMA_UNIFY_POOL_READY_TIMEOUT", 30.0)
POOL_LOAD_TIMEOUT = env_float("OLLAMA_UNIFY_POOL_LOAD_TIMEOUT", DRAIN_TIMEOUT)
# An admitted request is a renewable lease, not an unbounded counter. Backend
# headers and response chunks renew the activity deadline. If the caller goes
# away, non-replayable work is cancelled immediately; logical work gets a
# short absolute window in which to finish and enter the replay cache.
REQUEST_ACTIVITY_TTL = max(
    0.1, env_float("OLLAMA_UNIFY_REQUEST_ACTIVITY_TTL", DRAIN_TIMEOUT)
)
REQUEST_DETACHED_TTL = max(
    0.1, env_float("OLLAMA_UNIFY_REQUEST_DETACHED_TTL", 30.0)
)
REQUEST_CANCEL_GRACE = max(
    0.1, env_float("OLLAMA_UNIFY_REQUEST_CANCEL_GRACE", 5.0)
)
# _terminate_process has a 15s graceful wait, a 5s leader reap wait, and a
# final 5s process-group verification window. Give that worker one extra
# second, then renew/retry the stop without ever dropping its reservation.
LANE_STOP_ATTEMPT_TTL = max(REQUEST_CANCEL_GRACE, 26.0)
POOL_VRAM_RESERVE_MIB = env_int("OLLAMA_UNIFY_POOL_VRAM_RESERVE_MIB", 8192)
POOL_HOST_RESERVE_MIB = env_int("OLLAMA_UNIFY_POOL_HOST_RESERVE_MIB", 2048)
POOL_MODEL_OVERHEAD_PERCENT = max(
    100, env_int("OLLAMA_UNIFY_POOL_MODEL_OVERHEAD_PERCENT", 110)
)
OLLAMA_BINARY = os.environ.get("OLLAMA_UNIFY_OLLAMA_BINARY", "/usr/local/bin/ollama")
OLLAMA_MODELS = os.environ.get("OLLAMA_UNIFY_MODELS", "")
OLLAMA_CHILD_HOME = os.environ.get("OLLAMA_UNIFY_CHILD_HOME", "/var/lib/ollama-unify")
LOG = logging.getLogger("ollama-unify-negotiator")
CAPACITY_IO_DEADLINE = threading.local()

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
NATIVE_MODEL_PATHS = (
    "/api/generate", "/api/chat", "/api/embed", "/api/embeddings", "/api/rerank",
)
EMBEDDING_PATHS = ("/api/embed", "/api/embeddings", "/v1/embeddings")
INFERENCE_PATHS = NATIVE_MODEL_PATHS + (
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses",
)
SAFE_METADATA_PATHS = (
    "/api/ps", "/api/tags", "/api/version", "/api/show", "/v1/models",
)
CAPACITY_PATH = "/.well-known/ollama-unify-gpu-negotiator/capacity"
LOGICAL_REQUEST_HEADER = "X-Ollama-Unify-Logical-Request-Id"
RESUME_REQUEST_HEADER = "X-Ollama-Unify-Resume-Request"
GPU_UUIDS_HEADER = "X-Ollama-Unify-GPU-UUIDs"
CPU_OFFLOAD_HEADER = "X-Ollama-Unify-CPU-Offload"
WARM_ADMISSION_HEADER = "X-Ollama-Unify-Warm-Admission"
WARM_ADMISSION_SCHEMA = "io.ollama-unify.warm-admission.v1"


def canonical_model_tag(model: str) -> str:
    """Use Ollama's implicit latest tag without conflating distinct model aliases."""
    model = model.strip()
    if not model:
        return ""
    leaf = model.rsplit("/", 1)[-1]
    return model if ":" in leaf else f"{model}:latest"


def parse_gpu_uuid_constraint(value: Any, source: str, *, exact: bool = False) -> tuple[str, ...] | None:
    """Parse a presence-sensitive, ordered hard GPU allowlist.

    An allowlist constrains placement to matching members; it is not a demand
    that every listed device still exist unless exact=True. Hardware selection can legitimately
    shrink while a long-lived client still has the previous superset cached.
    Keep the live intersection in caller order, and fail closed only when no
    allowed GPU remains.
    """
    if exact and not isinstance(value, list):
        raise PermanentCapacityError("exclusive_group requires an explicit ordered GPU UUID array",
                                     400, "invalid_gpu_constraint")
    if value is None:
        return None
    if isinstance(value, str):
        values: list[Any] = [] if value.strip().lower() == "none" else value.split(",")
    elif isinstance(value, list):
        values = value
    else:
        raise PermanentCapacityError(
            f"{source} must be an array of GPU UUIDs",
            400,
            "invalid_gpu_constraint",
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in values:
        if not isinstance(candidate, str):
            raise PermanentCapacityError(
                f"{source} entries must be GPU UUID strings",
                400,
                "invalid_gpu_constraint",
            )
        gpu_uuid = candidate.strip()
        if (not gpu_uuid or "," in gpu_uuid
                or any(ord(character) < 32 or ord(character) == 127
                       for character in gpu_uuid)):
            raise PermanentCapacityError(
                f"{source} contains an invalid GPU UUID",
                400,
                "invalid_gpu_constraint",
            )
        if exact and gpu_uuid in seen:
            raise PermanentCapacityError("exclusive_group GPU UUIDs must be distinct",
                                         400, "invalid_gpu_constraint")
        if gpu_uuid not in seen:
            seen.add(gpu_uuid)
            normalized.append(gpu_uuid)
    if not normalized:
        raise PermanentCapacityError(
            "GPU constraint is present but contains no allowed GPU UUIDs",
            422,
            "gpu_constraint_empty",
        )
    selected = set(SELECTED_GPUS)
    if not selected:
        selected = {
            str(device.get("uuid") or "")
            for device in gpu_snapshot()
            if device.get("uuid")
        }
    available = [gpu_uuid for gpu_uuid in normalized if gpu_uuid in selected]
    unavailable = [gpu_uuid for gpu_uuid in normalized if gpu_uuid not in selected]
    if exact and unavailable:
        raise PermanentCapacityError("exclusive_group contains unselected GPU UUIDs: "
                                     + ", ".join(unavailable), 422, "gpu_constraint_unavailable")
    if not available:
        raise PermanentCapacityError(
            "GPU constraint has no UUIDs in the broker-selected set; requested: "
            + ", ".join(normalized),
            422,
            "gpu_constraint_unavailable",
        )
    if unavailable:
        LOG.warning(
            "%s pruned unavailable GPU UUIDs %s; effective allowlist is %s",
            source,
            unavailable,
            available,
        )
    return tuple(available)


def parse_capacity_placement(value: Any) -> str:
    if not isinstance(value, str) or value not in ("auto", "exclusive_group"):
        raise PermanentCapacityError("placement must be auto or exclusive_group",
                                     400, "invalid_capacity_request")
    return value


class BackendHTTPError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, data: bytes) -> None:
        self.method = method
        self.path = path
        self.status = status
        self.data = data
        super().__init__(
            f"Ollama backend {method} {path} returned {status}: {data[:300]!r}"
        )


def clamp_request(path: str, content_type: str, body: bytes) -> bytes:
    """Prevent clients from bypassing dynamic GPU fitting or the scanned context cap.

    Ollama decodes a JSON body on these paths whatever the request declares as
    its Content-Type, so a client that sends JSON under any other media type
    still reaches the model. Trusting the declared type here let exactly that
    client keep num_gpu, main_gpu, and an oversized num_ctx, which is the
    bypass this clamp exists to prevent. The body is parsed on path alone and
    left untouched when it does not decode.
    """
    del content_type
    is_native = path.startswith(NATIVE_MODEL_PATHS)
    is_openai = path.split("?", 1)[0] in ("/v1/chat/completions", "/v1/completions", "/v1/responses")
    if not body or not (is_native or is_openai):
        return body
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return body
    if not isinstance(payload, dict):
        return body
    options = payload.get("options")
    if options is None:
        options = {}
        payload["options"] = options
    if not isinstance(options, dict):
        raise PermanentCapacityError("model options must be a JSON object", 400,
                                     "invalid_model_options")
    if isinstance(options, dict):
        if is_native:
            options["num_gpu"] = -1
            options.pop("main_gpu", None)
        model = canonical_model_tag(str(payload.get("model") or ""))
        profile = (resolve_model_context_profile(model) if model and POOL_ENABLED
                   else effective_model_context_profile(model))
        if profile is not None:
            if not POOL_ENABLED:
                raise PermanentCapacityError(
                    "model-specific context requires managed GPU admission", 503,
                    "model_context_managed_pool_required",
                )
            requested = options.get("num_ctx")
            if requested is not None and (
                type(requested) is not int or requested > profile["context_length"]
            ):
                raise PermanentCapacityError(
                    "request exceeds the model-specific context policy", 400,
                    "model_context_policy_exceeded",
                )
            # The profiled lane always starts at the admitted context so a
            # smaller request cannot create a later, unaccounted VRAM growth.
            options["num_ctx"] = profile["context_length"]
            if is_openai and not MODEL_CONTEXT_RESOLUTIONS.get(model, {}).get("openai_context_compatible", True):
                raise PermanentCapacityError(
                    "OpenAI context cannot override the model's explicit num_ctx; "
                    "use the native API or align an exact model wrapper with the admitted context",
                    422, "model_context_openai_mismatch")
        elif MAX_CONTEXT > 0 and is_native:
            requested = options.get("num_ctx")
            if requested is None or (
                isinstance(requested, (int, float))
                and requested > MAX_CONTEXT
            ):
                options["num_ctx"] = MAX_CONTEXT
    payload.pop("num_gpu", None)
    payload.pop("main_gpu", None)
    return json.dumps(payload, separators=(",", ":")).encode()


def backend_json_at(host: str, port: int, method: str, path: str,
                    payload: dict[str, Any] | None = None,
                    timeout: float = 10.0) -> dict[str, Any]:
    deadline = getattr(CAPACITY_IO_DEADLINE, "deadline", None)
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BackgroundCapacityDeferred("speculative cold capacity time budget expired")
        timeout = min(timeout, remaining)
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BackgroundCapacityDeferred("speculative cold capacity time budget expired")
            if conn.sock is not None:
                conn.sock.settimeout(min(timeout, remaining))
        response = conn.getresponse()
        if deadline is None:
            data = response.read()
        else:
            chunks = []
            # HTTPConnection drops its socket on close-delimited responses.
            # The response still owns it, and must retain the absolute budget.
            sock = conn.sock or getattr(getattr(response.fp, "raw", None), "_sock", None)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackgroundCapacityDeferred("speculative cold capacity time budget expired")
                if sock is not None:
                    sock.settimeout(min(timeout, remaining))
                chunk = response.read1(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if response.isclosed():
                    break
            data = b"".join(chunks)
        if response.status >= 400:
            raise BackendHTTPError(method, path, response.status, data)
        return json.loads(data or b"{}")
    finally:
        conn.close()


def backend_json(method: str, path: str, payload: dict[str, Any] | None = None,
                 timeout: float = 10.0) -> dict[str, Any]:
    return backend_json_at(BACKEND_HOST, BACKEND_PORT, method, path, payload, timeout)


def model_gpu_weight_bytes(model: str, tag: dict, aggregate_bytes: int) -> int:
    """Subtract a known sidecar only from an exact native/local runtime graph.

    Stock Ollama GetModel loads model/projector/adapter layers and
    Model.String exposes their blob paths as FROM/ADAPTER. The omni exporter
    deliberately appends its bundle outside that graph (never a FROM).
    Unknown, remote, changed or unavailable evidence retains the original
    aggregate estimate; observed low VRAM is never used to shrink a promise.
    """
    if not OLLAMA_MODELS or type(tag.get("size")) is not int:
        return aggregate_bytes
    auxiliary_type = "application/vnd.robit.ollama.omni.bundle.v1+gguf"
    runtime_types = {"application/vnd.ollama.image." + kind
                     for kind in ("model", "projector", "adapter", "tensor")}
    metadata_types = {"application/vnd.ollama.image." + kind
                      for kind in ("template", "license", "params", "system", "messages")}
    try:
        name, separator, version = model.rpartition(":")
        parts = name.split("/")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
            return aggregate_bytes
        if len(parts) == 1:
            parts = ["registry.ollama.ai", "library", *parts]
        elif len(parts) == 2:
            parts = ["registry.ollama.ai", *parts]
        if (len(parts) != 3
                or not re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]+)?", parts[0])
                or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in parts[1:])):
            return aggregate_bytes
        root_alias = Path(OLLAMA_MODELS)
        root = root_alias.resolve(strict=True)
        manifests_alias = root / "manifests"
        manifests = manifests_alias.resolve(strict=True)
        path = manifests.joinpath(*parts, version)
        if path.resolve(strict=True) != path:
            return aggregate_bytes
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if (len(raw) > 1024 * 1024 or not re.fullmatch(r"[a-f0-9]{64}", str(tag.get("digest") or ""))
                or hashlib.sha256(raw).hexdigest() != tag["digest"]):
            return aggregate_bytes
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("ambiguous manifest")
                result[key] = value
            return result
        manifest = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(manifest, dict) or type(manifest.get("schemaVersion")) is not int or manifest["schemaVersion"] != 2:
            return aggregate_bytes
        config, layers = manifest.get("config"), manifest.get("layers")
        if (not isinstance(config, dict) or config.get("mediaType") != "application/vnd.docker.container.image.v1+json"
                or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(config.get("digest") or ""))
                or type(config.get("size")) is not int or not 0 < config["size"] <= 1024 * 1024
                or not isinstance(layers, list) or not 1 <= len(layers) <= 256):
            return aggregate_bytes
        if not any(isinstance(layer, dict) and layer.get("mediaType") == auxiliary_type for layer in layers):
            return aggregate_bytes
        blobs_alias = root / "blobs"
        blobs = blobs_alias.resolve(strict=True)
        runtime, identities, seen = set(), {}, set()
        excluded = total = 0
        for layer in layers:
            if (not isinstance(layer, dict) or layer.get("mediaType") not in runtime_types | metadata_types | {auxiliary_type}
                    or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(layer.get("digest") or ""))
                    or type(layer.get("size")) is not int or not 0 < layer["size"] <= 2 ** 63 - 1
                    or layer["digest"] in seen):
                return aggregate_bytes
            seen.add(layer["digest"])
            total += layer["size"]
            blob = blobs / layer["digest"].replace(":", "-")
            if blob.resolve(strict=True) != blob or not blob.is_file() or blob.stat().st_size != layer["size"]:
                return aggregate_bytes
            stat = blob.stat()
            identities[blob] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if layer["mediaType"] in runtime_types:
                runtime.add(blob)
            elif layer["mediaType"] == auxiliary_type:
                excluded += layer["size"]
        if (aggregate_bytes not in (total, total + config["size"])
                or not any(layer["mediaType"] == "application/vnd.ollama.image.model" for layer in layers)):
            return aggregate_bytes
        show = backend_json("POST", "/api/show", {"model": model}, timeout=10.0)
        if not isinstance(show, dict):
            return aggregate_bytes
        modelfile = show.get("modelfile")
        if not isinstance(modelfile, str) or len(modelfile) > 1024 * 1024:
            return aggregate_bytes
        native = set()
        for line in modelfile.splitlines():
            if re.match(r"^\s*DRAFT\s", line):
                return aggregate_bytes
            if not re.match(r"^\s*(?:FROM|ADAPTER)\s", line):
                continue
            fields = shlex.split(line)
            if len(fields) != 2 or not Path(fields[1]).is_absolute():
                return aggregate_bytes
            blob = Path(fields[1]).resolve(strict=True)
            if blob.parent != blobs or blob in native:
                return aggregate_bytes
            native.add(blob)
        if native != runtime or not native:
            return aggregate_bytes
        current_tags = backend_json("GET", "/api/tags", timeout=10.0)
        if not isinstance(current_tags, dict):
            return aggregate_bytes
        current = current_tags.get("models")
        matches = [item for item in current if isinstance(item, dict)
                   and canonical_model_tag(str(item.get("name") or item.get("model") or "")) == model] if isinstance(current, list) else []
        with path.open("rb") as stream:
            current_raw = stream.read(1024 * 1024 + 1)
        if (len(matches) != 1 or matches[0].get("digest") != tag["digest"]
                or matches[0].get("size") != tag.get("size") or current_raw != raw
                or path.resolve(strict=True) != path
                or root_alias.resolve(strict=True) != root
                or manifests_alias.resolve(strict=True) != manifests
                or blobs_alias.resolve(strict=True) != blobs):
            return aggregate_bytes
        for blob, identity in identities.items():
            stat = blob.stat()
            if (blob.resolve(strict=True) != blob
                    or identity != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)):
                return aggregate_bytes
        return aggregate_bytes - excluded
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, http.client.HTTPException):
        return aggregate_bytes


@dataclass(frozen=True)
class BackendProbe:
    available: bool
    models: list[dict[str, Any]]
    error: str | None
    checked_at: float


def probe_backend() -> BackendProbe:
    try:
        models = backend_json("GET", "/api/ps", timeout=3.0).get("models", [])
        return BackendProbe(True, models if isinstance(models, list) else [], None, time.time())
    except (OSError, RuntimeError, ValueError) as exc:
        return BackendProbe(False, [], str(exc), time.time())


def running_models(*, require_available: bool = False) -> list[dict[str, Any]]:
    probe = probe_backend()
    if require_available and not probe.available:
        raise RuntimeError(f"Ollama backend unavailable: {probe.error or 'unknown error'}")
    return probe.models


def unload_models_at(host: str, port: int, timeout: float = UNLOAD_TIMEOUT,
                     *, require_available: bool = True) -> list[str]:
    try:
        models = backend_json_at(host, port, "GET", "/api/ps", timeout=3.0).get("models", [])
    except (OSError, RuntimeError, ValueError) as exc:
        if require_available:
            raise RuntimeError(f"Ollama backend unavailable: {exc}") from exc
        return []
    if not isinstance(models, list):
        models = []
    names = [str(model.get("name") or model.get("model") or "") for model in models]
    names = [name for name in names if name]
    for name in names:
        attempts = (
            ("/api/generate", {"model": name, "keep_alive": 0, "stream": False}),
            ("/api/embed", {"model": name, "input": "", "keep_alive": 0}),
            ("/api/embeddings", {"model": name, "prompt": "", "keep_alive": 0}),
        )
        for attempt, (path, payload) in enumerate(attempts, 1):
            try:
                backend_json_at(
                    host, port, "POST", path, payload, timeout=min(timeout, 30.0)
                )
                break
            except BackendHTTPError as exc:
                if exc.status in (400, 404, 405) and attempt < len(attempts):
                    continue
                LOG.warning("unload request failed for %s: %s", name, exc)
                break
            except (OSError, RuntimeError, ValueError) as exc:
                LOG.warning("unload request failed for %s: %s", name, exc)
                break

    deadline = time.monotonic() + timeout
    while names and time.monotonic() < deadline:
        current = backend_json_at(host, port, "GET", "/api/ps", timeout=3.0).get("models", [])
        if not current:
            return names
        time.sleep(0.2)
    current = backend_json_at(host, port, "GET", "/api/ps", timeout=3.0).get("models", [])
    if names and current:
        raise TimeoutError(f"Ollama models did not unload within {timeout:.0f}s")
    return names


def unload_all_models(timeout: float = UNLOAD_TIMEOUT) -> list[str]:
    return unload_models_at(BACKEND_HOST, BACKEND_PORT, timeout)


def gpu_snapshot() -> list[dict[str, Any]]:
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    devices = []
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            continue
        try:
            devices.append({
                "uuid": fields[0], "total_mib": int(fields[1]),
                "used_mib": int(fields[2]), "free_mib": int(fields[3]),
            })
        except ValueError:
            continue
    return devices


def _query_gpu_health() -> dict[str, Any]:
    """Read driver recovery state, independently of apparently free VRAM."""
    health: dict[str, Any] = {
        "supported": BACKEND_TYPE == "cuda", "admission_blocked": False,
        "recovery_actions": {}, "missing_selected_gpu_ids": [], "error": None,
    }
    if BACKEND_TYPE != "cuda":
        return health
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=uuid,gpu_recovery_action",
            "--format=csv,noheader,nounits",
        ], check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        health.update(admission_blocked=True, error=str(exc))
        return health
    actions = {}
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) == 2 and fields[0].startswith("GPU-"):
            actions[fields[0]] = fields[1]
    diagnostic = (result.stdout + "\n" + result.stderr).strip()
    # Older drivers lack this query field. Preserve their existing admission
    # behavior, but explicitly expose the unavailable protection in discovery.
    if (not actions and result.returncode != 0
            and "gpu_recovery_action" in diagnostic
            and "not a valid field" in diagnostic.lower()):
        health.update(supported=False, error=diagnostic[:512])
        return health
    selected = set(SELECTED_GPUS)
    required = {
        gpu_uuid: action for gpu_uuid, action in actions.items()
        if action.strip("[]").lower() not in ("none", "n/a", "not supported")
        and (not selected or gpu_uuid in selected
             or "reboot" in action.lower())
    }
    missing = sorted(selected - actions.keys())
    health.update(recovery_actions=required, missing_selected_gpu_ids=missing)
    if result.returncode != 0 or not actions:
        health["error"] = diagnostic[:512] or "GPU recovery telemetry is unavailable"
    health["admission_blocked"] = bool(required or missing or health["error"])
    return health


_GPU_HEALTH_LOCK = threading.Lock()
_GPU_HEALTH_CACHE: tuple[float, tuple[Any, ...], dict[str, Any]] | None = None


def gpu_health_snapshot(*, refresh: bool = False) -> dict[str, Any]:
    """Coalesce queue polling; allocation transitions force fresh telemetry."""
    global _GPU_HEALTH_CACHE
    key = (BACKEND_TYPE, tuple(SELECTED_GPUS))
    with _GPU_HEALTH_LOCK:
        cached = _GPU_HEALTH_CACHE
        if (not refresh and cached is not None and cached[1] == key
                and time.monotonic() - cached[0] < 0.5):
            return dict(cached[2])
        health = _query_gpu_health()
        _GPU_HEALTH_CACHE = (time.monotonic(), key, health)
        return dict(health)


def inference_hold_status() -> dict[str, Any]:
    """A persistent operator hold blocks model execution, not lease control."""
    if not INFERENCE_HOLD_FILE:
        return {"active": False}
    try:
        with open(INFERENCE_HOLD_FILE, encoding="utf-8") as stream:
            reason = stream.read(512).strip()
    except FileNotFoundError:
        return {"active": False, "path": INFERENCE_HOLD_FILE}
    except (OSError, UnicodeError) as exc:
        reason = f"operator hold file cannot be read: {str(exc)[:160]}"
    return {"active": True, "path": INFERENCE_HOLD_FILE,
            "reason": reason or "operator manual inference control"}


def require_inference_enabled(*, request_id: str = "",
                              logical_request_id: str = "") -> None:
    hold = inference_hold_status()
    if hold["active"]:
        raise PermanentCapacityError(
            "Model inference is paused by the operator: " + hold["reason"],
            503, "operator_inference_hold", request_id=request_id,
            logical_request_id=logical_request_id,
        )


def require_gpu_health(*, request_id: str = "",
                       logical_request_id: str = "",
                       refresh: bool = False) -> None:
    health = gpu_health_snapshot(refresh=refresh)
    if not health["admission_blocked"]:
        return
    if health["recovery_actions"]:
        actions = ", ".join(
            f"{gpu_uuid}: {action}"
            for gpu_uuid, action in health["recovery_actions"].items()
        )
        raise PermanentCapacityError(
            "NVIDIA driver recovery is required before new GPU work: " + actions,
            503, "gpu_recovery_required", request_id=request_id,
            logical_request_id=logical_request_id,
        )
    raise CapacityError(
        "GPU health cannot be verified; refusing new GPU work: "
        + str(health["error"] or health["missing_selected_gpu_ids"]),
        503, "gpu_health_unavailable", request_id=request_id,
        logical_request_id=logical_request_id,
    )


def gpu_health_warnings(health: dict[str, Any]) -> list[str]:
    if health["admission_blocked"]:
        return ["New GPU work is blocked: NVIDIA recovery is required or "
                "GPU health cannot be verified. Inspect gpu_health before retrying."]
    if BACKEND_TYPE == "cuda" and not health["supported"]:
        return ["This NVIDIA driver does not support GPU recovery telemetry; "
                "recovery-state admission protection is unavailable."]
    return []


def managed_cgroup_pids() -> set[int]:
    pids: set[int] = set()
    for unit in ("ollama.service", "ollama-unify-negotiator.service"):
        try:
            result = subprocess.run(
                ["systemctl", "show", unit, "-p", "ControlGroup", "--value"],
                check=True, capture_output=True, text=True, timeout=3,
            )
            control_group = result.stdout.strip().lstrip("/")
            path = Path("/sys/fs/cgroup") / control_group / "cgroup.procs"
            pids.update(int(line) for line in path.read_text().splitlines() if line.isdigit())
        except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
            continue
    roots = set(pids)
    roots.add(os.getpid())
    descendants = set(roots)
    parent_by_pid: dict[int, int] = {}
    try:
        candidates = [item for item in Path("/proc").iterdir()
                      if item.name.isdigit()]
    except OSError:
        candidates = []
    for candidate in candidates:
        try:
            fields = candidate.joinpath("status").read_text().splitlines()
            parent_by_pid[int(candidate.name)] = next(
                int(line.split()[1]) for line in fields if line.startswith("PPid:")
            )
        except (OSError, StopIteration, ValueError, IndexError):
            continue
    frontier = set(roots)
    while frontier:
        children = {pid for pid, parent in parent_by_pid.items()
                    if parent in frontier and pid not in descendants}
        descendants.update(children)
        frontier = children
    return descendants


def foreign_gpu_usage(*, strict: bool = False) -> dict[str, int]:
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        if strict:
            raise RuntimeError("cannot verify foreign CUDA processes") from exc
        return {}
    if not result.stdout.strip():
        # There is nothing to classify. Avoid walking /proc on each scheduler
        # check before the first CUDA process exists.
        return {}
    ollama_pids = managed_cgroup_pids()
    usage: dict[str, int] = {}
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 3:
            if strict and raw.strip():
                raise RuntimeError("invalid CUDA process telemetry")
            continue
        try:
            pid = int(fields[0])
            used = int(fields[2]) if fields[2].isdigit() else 0
        except ValueError:
            if strict:
                raise RuntimeError("invalid CUDA process telemetry")
            continue
        if SELECTED_GPUS and fields[1] not in SELECTED_GPUS:
            continue
        if pid not in ollama_pids:
            usage[f"{pid}@{fields[1]}"] = used
    return usage


def process_gpu_usage(process: Any) -> dict[str, int]:
    """Attest every CUDA device used by one managed process group."""
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise CapacityError("cannot verify managed GPU placement", 503,
                            "gpu_placement_unverified") from exc
    usage: dict[str, int] = {}
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            raise CapacityError("invalid managed GPU placement telemetry", 503,
                                "gpu_placement_unverified")
        try:
            owned = os.getpgid(int(fields[0])) == process.pid
        except ProcessLookupError:
            continue
        except (OSError, AttributeError) as exc:
            raise CapacityError("cannot identify managed CUDA process group", 503,
                                "gpu_placement_unverified") from exc
        if owned:
            if not fields[2].isdigit():
                raise CapacityError("managed CUDA memory is unverifiable", 503,
                                    "gpu_placement_unverified")
            usage[fields[1]] = usage.get(fields[1], 0) + int(fields[2])
    return usage


def increased_foreign_gpu_usage(previous: dict[str, int],
                                current: dict[str, int]) -> dict[str, int]:
    """Return only new or larger foreign allocations on selected GPUs."""
    return {
        key: used for key, used in current.items()
        if used > previous.get(key, 0)
    }


def managed_cgroup_memory_headroom(
    membership_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, int | str]:
    """Inspect the managed parent and every cgroup-v2 ancestor, not base Ollama.

    The broker's children inherit this membership. Both high and max constrain
    additional pageable memory. An unlimited cgroup still has a finite physical
    host budget; missing or unreadable containment telemetry fails closed.
    """
    try:
        memberships = [line[3:] for line in membership_path.read_text().splitlines()
                       if line.startswith("0::")]
        if len(memberships) != 1:
            raise ValueError("no exact unified cgroup membership")
        root = cgroup_root.resolve()
        current = (root / memberships[0].lstrip("/")).resolve()
        if not current.is_relative_to(root):
            raise ValueError("invalid cgroup membership")
        headroom = None
        while True:
            # The host cgroup root lacks these files. A namespace-mounted
            # container root may expose real limits, which still constrain us.
            if current == root and not any((current / name).exists()
                    for name in ("memory.current", "memory.high", "memory.max")):
                break
            used = int((current / "memory.current").read_text().strip())
            if used < 0:
                raise ValueError("invalid cgroup memory usage")
            for name in ("memory.high", "memory.max"):
                value = (current / name).read_text().strip()
                if value == "max":
                    continue
                limit = int(value)
                if limit < 0:
                    raise ValueError("invalid cgroup memory limit")
                available = max(0, limit - used) // (1024 * 1024)
                headroom = available if headroom is None else min(headroom, available)
            if current == root:
                break
            current = current.parent
        return {"managed_cgroup_headroom_mib": "max" if headroom is None else headroom}
    except (OSError, ValueError):
        return {"managed_cgroup_headroom_mib": "unknown"}


def host_memory_snapshot() -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemAvailable:", "MemTotal:")):
                key, value, *_ = line.split()
                result[key.rstrip(":").lower() + "_mib"] = int(value) // 1024
    except (OSError, ValueError):
        pass
    try:
        control_group = subprocess.run(
            ["systemctl", "show", "ollama.service", "-p", "ControlGroup", "--value"],
            check=True, capture_output=True, text=True, timeout=3,
        ).stdout.strip().lstrip("/")
        base = Path("/sys/fs/cgroup") / control_group
        for source, target in (("memory.current", "ollama_current_bytes"),
                               ("memory.high", "ollama_high_bytes"),
                               ("memory.max", "ollama_max_bytes")):
            value = (base / source).read_text().strip()
            result[target] = int(value) if value.isdigit() else value
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        pass
    result.update(managed_cgroup_memory_headroom())
    return result


def discovery_document() -> dict[str, Any]:
    devices = gpu_snapshot()
    health = gpu_health_snapshot()
    backend = probe_backend()
    with MODEL_CONTEXT_LOCK:
        context_profiles = {**MODEL_CONTEXT_PROFILES, **RESOLVED_MODEL_CONTEXT_PROFILES}
        context_resolutions = {model: {key: value for key, value in resolution.items()
                                       if key != "checked_at"}
                               for model, resolution in MODEL_CONTEXT_RESOLUTIONS.items()}
    selected = set(SELECTED_GPUS)
    for device in devices:
        device["selected_for_ollama"] = not selected or device.get("uuid") in selected
    return {
        "schema": "io.ollama-unify.gpu-negotiator.discovery.v1",
        "protocol": "ollama-unify-gpu-lease/v1",
        "available": backend.available,
        "backend_available": backend.available,
        "backend_error": backend.error,
        "backend_checked_at": backend.checked_at,
        "backend": BACKEND_TYPE,
        "selected_gpu_ids": SELECTED_GPUS,
        "selected_gpu_count": len(SELECTED_GPUS),
        "gpus": devices,
        "gpu_health": health,
        "inference_hold": inference_hold_status(),
        "public_ollama_api": f"http://127.0.0.1:{LISTEN_PORT}",
        "ollama_backend": f"http://{BACKEND_HOST}:{BACKEND_PORT}",
        "control_socket": CONTROL_SOCKET,
        "discovery_file": "/usr/local/share/ollama-unify/gpu-negotiator.json",
        "agent_instructions": "/usr/local/share/ollama-unify/AGENTS.md",
        "well_known": f"http://127.0.0.1:{LISTEN_PORT}/.well-known/ollama-unify-gpu-negotiator",
        "capacity_endpoint": f"http://127.0.0.1:{LISTEN_PORT}{CAPACITY_PATH}",
        "lease_policy": lease_policy_document(),
        "active_leases": [],
        "lease_handoff_requests": [],
        "warnings": gpu_health_warnings(health),
        "pending_transition_timeout_seconds": PENDING_TIMEOUT,
        "heartbeat_reconnect_grace_seconds": HEARTBEAT_RECONNECT_GRACE,
        "client_history_policy": {
            "ttl_seconds": CLIENT_HISTORY_TTL,
            "max_clients": CLIENT_HISTORY_LIMIT,
            "max_lanes_per_client": CLIENT_LANE_HISTORY_LIMIT,
            "max_clients_per_lane": LANE_CLIENT_LIMIT,
        },
        "context_policy": {
            "default_max_context": 0 if POOL_ENABLED and AUTO_MODEL_CONTEXT else MAX_CONTEXT,
            "legacy_backend_max_context": MAX_CONTEXT,
            "managed_context_policy": "verified_model_maximum" if AUTO_MODEL_CONTEXT else "legacy_default",
            "hard_max_context": HARD_MAX_CONTEXT,
            "model_profiles": context_profiles,
            "model_context_resolutions": context_resolutions,
            "model_query_parameter": "model",
            "profile_context_is_fixed": not AUTO_MODEL_CONTEXT,
            "operator_profiles_are_memory_floors": AUTO_MODEL_CONTEXT,
        },
        "parallel_pool": {
            "enabled": POOL_ENABLED,
            "model_gpu_preferences": MODEL_GPU_PREFERENCES,
            "max_managed_servers": POOL_MAX_SERVERS,
            "max_queue": POOL_MAX_QUEUE,
            "resume_ttl_seconds": POOL_RESUME_TTL,
            "private_port_start": POOL_PORT_START,
            "private_port_end": POOL_PORT_START + max(32, POOL_MAX_SERVERS + 4) - 1,
            "instance_parallel": POOL_INSTANCE_PARALLEL,
            "idle_timeout_seconds": POOL_IDLE_TIMEOUT,
            "load_timeout_seconds": POOL_LOAD_TIMEOUT,
            "request_lifecycle": {
                "activity_ttl_seconds": REQUEST_ACTIVITY_TTL,
                "detached_ttl_seconds": REQUEST_DETACHED_TTL,
                "cancel_grace_seconds": REQUEST_CANCEL_GRACE,
                "lane_stop_attempt_ttl_seconds": LANE_STOP_ATTEMPT_TTL,
                "cancel_process_policy": (
                    "sigterm_only" if BACKEND_TYPE == "cuda" else "sigterm_then_sigkill"
                ),
                "renewal_events": ["backend_connected", "response_headers", "response_chunk"],
                "non_logical_disconnect": "cancel_immediately",
                "logical_disconnect": "renewable_completion_window",
            },
            "model_overhead_percent": POOL_MODEL_OVERHEAD_PERCENT,
            "vram_reserve_mib": POOL_VRAM_RESERVE_MIB,
            "host_reserve_mib_per_lane": POOL_HOST_RESERVE_MIB,
            "admission_protocol": {
                "logical_request_header": LOGICAL_REQUEST_HEADER,
                "resume_request_header": RESUME_REQUEST_HEADER,
                "resume_request_body": "omitted",
                "retained_request_max_body_bytes": (
                    RETAINED_REQUEST_MAX_BODY_BYTES
                ),
                "retained_request_max_total_bytes": (
                    RETAINED_REQUEST_MAX_TOTAL_BYTES
                ),
                "queue_position_header": "X-Ollama-Unify-Queue-Position",
                "queue_ticket_header": "X-Ollama-Unify-Queue-Ticket",
                "workload_class_header": "X-Ollama-Unify-Workload-Class",
                "workload_classes": [
                    "foreground", "interactive-control", "background",
                ],
                "admission_wait_header": "X-Ollama-Unify-Admission-Wait-Ms",
                "queue_policy_header": "X-Ollama-Unify-Queue-Policy",
                "queue_policies": ["wait", "yield"],
                "unlabelled_embedding_defaults": {
                    "paths": list(EMBEDDING_PATHS),
                    "workload_class": "background",
                    "queue_policy": "yield",
                    "explicit_headers_override": True,
                },
                "gpu_uuids_header": GPU_UUIDS_HEADER,
                "cpu_offload_header": CPU_OFFLOAD_HEADER,
                "cpu_offload_capacity_field": "cpu_offload",
                "cpu_offload_semantics": (
                    "explicit allow header or boolean capacity opt-in; full GPU preferred; "
                    "physical-scope overflow only, exclusive scoped GPUs and bounded pageable host memory; "
                    "no CPU-only fallback or warm certificate"
                ),
                "warm_admission_header": WARM_ADMISSION_HEADER,
                "warm_admission_schema": WARM_ADMISSION_SCHEMA,
                "warm_admission_proof_only_capacity_field": "warm_admission_proof",
                "capacity_placements": ["auto", "exclusive_group"],
                "exclusive_group_semantics": (
                    "explicit ordered group of at least two selected GPUs; every member "
                    "must satisfy model policy; exact group reuse and proof only; no pruning"
                ),
                "gpu_constraint_semantics": (
                    "ordered hard allowlist intersected with broker-selected GPUs; "
                    "rejected only when the intersection is empty"
                ),
                "retry_after_json_field": "retry_after_ms",
                "reason_codes": [
                    "client_gpu_disabled",
                    "client_gpu_policy_conflict",
                    "queue_admission_timeout",
                    "queue_full",
                    "lease_transition",
                    "lane_capacity_wait",
                    "reclaimable_placement_wait",
                    "background_capacity_deferred",
                    "warm_preflight_required",
                    "invalid_warm_admission",
                    "host_memory_unavailable",
                    "model_exceeds_gpu_capacity",
                    "model_context_memory_unverified",
                    "model_context_memory_exceeded",
                    "model_context_limit_unverified",
                    "model_context_identity_mismatch",
                    "model_context_identity_changed",
                    "model_context_runtime_mismatch",
                    "model_context_openai_mismatch",
                    "invalid_model_options",
                    "model_context_policy_exceeded",
                    "gpu_peer_group_wait",
                    "gpu_peer_group_reserved",
                    "gpu_placement_unverified",
                    "invalid_gpu_scope",
                    "model_not_installed",
                    "gpu_runtime_unavailable",
                    "gpu_recovery_required",
                    "operator_inference_hold",
                    "gpu_health_unavailable",
                "gpu_unregistered_workload",
                    "backend_start_failed",
                    "logical_request_conflict",
                    "logical_request_in_progress",
                    "logical_request_not_found",
                    "logical_request_cancelled",
                    "logical_request_expired",
                    "retained_request_unavailable",
                    "request_body_exceeds_resume_limit",
                    "request_retention_capacity",
                    "resume_body_forbidden",
                    "completed_response_unavailable",
                    "invalid_admission_header",
                    "invalid_gpu_constraint",
                    "gpu_constraint_empty",
                    "gpu_constraint_unavailable",
                ],
                "completed_response_replay": {
                    "schema": "io.ollama-unify.completed-response-cache.v1",
                    "scope": "logical inference requests",
                    "ttl_seconds": COMPLETED_RESPONSE_TTL,
                    "max_entries": COMPLETED_RESPONSE_MAX_ENTRIES,
                    "max_body_bytes": COMPLETED_RESPONSE_MAX_BODY_BYTES,
                    "max_total_bytes": COMPLETED_RESPONSE_MAX_TOTAL_BYTES,
                    "replay_header": "X-Ollama-Unify-Response-Replayed",
                    "response_hash_header": (
                        "X-Ollama-Unify-Completed-Response-Sha256"
                    ),
                    "content_exposed": False,
                },
            },
        },
        "commands": {
            "discover": "docker gpu discover",
            "status": "docker gpu status",
            "cooperative_run": (
                "docker gpu run --owner NAME --vram-mib MIB "
                "[--justification PURPOSE] [--expected-duration SECONDS] "
                "--gpu GPU_UUID "
                "--ready-command 'READINESS_CHECK' -- COMMAND"
            ),
            "manual": [
                "acquire", "scope", "ready", "prepare", "release", "heartbeat",
            ],
            "request_capacity": (
                f"POST {CAPACITY_PATH} with JSON "
                "{\"model\":\"TAG\",\"parallel\":N,\"gpu_uuids\":[\"GPU-...\"]}"
            ),
        },
        "requirements": {
            "cuda_deployments": (
                "Acquire a lease before loading CUDA models; signal ready only after GPU allocation is resident."
            ),
            "scoped_leases": (
                "Use repeated --gpu UUID options and give the child exactly those UUIDs in CUDA_VISIBLE_DEVICES. Pending or revoking scopes defer Ollama placement; bounded pending single-GPU budgets may coexist with other leases; active non-exclusive single-GPU scopes may share stable live VRAM with broker-owned Ollama lanes; explicit --exclusive single-GPU scopes and multi-GPU scopes are exclusive for their whole lifetime because Ollama lane churn during peer-to-peer (NVLink/NCCL) traffic is unsafe."
            ),
            "resize": "Call prepare before increasing VRAM use, then ready after the new allocation settles.",
            "release": "Free external CUDA allocations before releasing the lease.",
            "heartbeat_restart_safety": (
                "Heartbeat clients retain the exact lease token and GPU scope "
                "through a bounded negotiator restart; an explicit rejection "
                "or expired reconnect grace remains fail-closed."
            ),
            "pending_timeout": (
                "Pending transitions have an absolute, non-renewable deadline. "
                "After it expires, the broker revokes heartbeats and remains drained "
                "until the owner frees CUDA memory and releases the lease."
            ),
            "num_gpu_semantics": (
                "Ollama num_gpu is the number of GPU-offloaded layers, not the physical GPU count; -1 is automatic."
            ),
            "anonymous_cuda": "Unregistered activity quarantines affected GPUs and defers model load/unload; acquire before CUDA initialization. Initial anonymous allocation cannot be prevented.",
        },
    }


def agent_instructions_text() -> str:
    return """# Host CUDA negotiation\n\nThis host runs the ollama-unify GPU lease broker. Before creating, starting, or resizing any Docker/container/service deployment that uses CUDA:\n\n1. Run `docker gpu discover` and inspect the selected GPUs, active lease summaries, coordination warnings, and current policy.\n2. Launch long-running CUDA services with `docker gpu run --owner NAME --justification PURPOSE --expected-duration SECONDS --vram-mib MIB --gpu GPU_UUID --ready-command 'CHECK' -- COMMAND`. Repeat `--gpu` for each reserved device. The readiness check must pass only after CUDA models are resident.\n3. For an independently supervised service, use scoped `acquire --owner NAME --justification PURPOSE --expected-duration SECONDS --gpu GPU_UUID`, set the child's `CUDA_VISIBLE_DEVICES` to exactly the same UUIDs, load the CUDA models, call `ready`, call `prepare` before any VRAM growth, and call `release` only after freeing CUDA memory.\n4. Lease registration requires a specific owner. Provide --justification PURPOSE and --expected-duration SECONDS when setting up workloads to help other agents coordinate; these are recommendations, and missing metadata must not halt a basic deployment. Omitted purpose and release time remain unknown and do not affect heartbeat TTL. These fields and the GPU scope are visible to other local agents; tokens are not exposed in discovery. Review current lessees before acquiring.\n5. Positive --vram-mib single-GPU leases share memory when their combined promises and live usage fit; there is no lease-count cap. Zero-budget, explicit --exclusive and multi-GPU owners retain exclusive external scopes. Call prepare TOKEN --vram-mib NEW_TOTAL before exceeding the budget, and ready TOKEN --pid WORKLOAD_PID after allocation. The run wrapper registers its child automatically. Pending and revoking scoped leases defer Ollama placement. After `ready`, the broker can place Ollama lanes in measured free VRAM on single-GPU scopes. Explicit --exclusive single-GPU scopes and multi-GPU scopes stay exclusive until release: the broker retires Ollama lanes there at acquire and never places, loads, or reclaims lanes on them, because lane churn during peer-to-peer (NVLink/NCCL) traffic is unsafe. Explicit --exclusive requires fixed placement and cannot combine yield or cache evacuation; its reservation is not discarded while scoped CUDA contexts remain or telemetry is invalid. The active workload must call `prepare` before any VRAM growth. Unscoped leases retain a host-wide drain because placement cannot be proven. A non-exclusive revoked lease whose owner stops heartbeating can be abandoned once the revoke deadline passes; explicit exclusive leases additionally require verified absence of every scoped CUDA context. Always `release` rather than letting an owner exit.\n6. Never bypass the broker by assuming free VRAM from a static scan. Unregistered CUDA activity quarantines affected GPUs: broker model load/unload is deferred until it exits. This cannot prevent its initial allocation or replace cooperative leases.\n7. Ollama API `num_gpu` counts offloaded layers, not physical GPUs. Keep it automatic (`-1`).\n8. Clients that require exact Ollama placement must send an ordered hard allowlist as `gpu_uuids` on capacity requests and `X-Ollama-Unify-GPU-UUIDs` on inference requests. The broker uses the ordered intersection with its live selected GPUs, rejects an empty intersection, and never falls back outside the allowlist.\n9. Placement defaults to fixed: --gpu names the exact required set, including NVLinked pairs. Opt into --placement flexible only for a service whose supervisor can drain and restart; its --gpu options become ordered single-GPU candidates. Use --priority (default0, -1000..1000) to permit strictly higher-priority requests to ask flexible owners to relocate. The broker reserves a destination first and verifies source CUDA release; never change a live worker UUID. Generic run requires --drain-command; use --resume-command to reopen admission after cancelled handoffs. Independent owners read heartbeat.relocation, drain, call relocate TOKEN --handoff-id ID, start only on the returned UUID, then ready. Fixed/non-cooperative owners are never forcibly moved.\n\nMachine-readable discovery: `/usr/local/share/ollama-unify/gpu-negotiator.json` or `http://127.0.0.1:11434/.well-known/ollama-unify-gpu-negotiator`.\n"""


def foreign_usage_by_gpu(
    usage: dict[str, int], gpu_uuids: set[str] | None = None,
) -> dict[str, int]:
    """Aggregate foreign CUDA use by GPU instead of unstable process ID.

    CUDA clients can replace a worker process without changing their resident
    allocation. A PID-keyed release check treats that harmless process churn
    as new VRAM forever and can pin the negotiator in drain mode. Per-GPU
    totals preserve the safety property that matters for placement: a lease
    is releasable only after live foreign use on its scope returns to or below
    the pre-lease amount.
    """
    totals: dict[str, int] = {}
    for key, used in usage.items():
        _pid, separator, gpu_uuid = key.rpartition("@")
        if not separator or not gpu_uuid:
            continue
        if gpu_uuids is not None and gpu_uuid not in gpu_uuids:
            continue
        totals[gpu_uuid] = totals.get(gpu_uuid, 0) + max(0, int(used))
    return totals


def foreign_usage_at_or_below(
    baseline: dict[str, int], current: dict[str, int],
    gpu_uuids: set[str] | None = None,
    tolerance_mib: int = 0,
) -> bool:
    baseline_by_gpu = foreign_usage_by_gpu(baseline, gpu_uuids)
    current_by_gpu = foreign_usage_by_gpu(current, gpu_uuids)
    return all(
        used <= baseline_by_gpu.get(gpu_uuid, 0) + max(0, tolerance_mib)
        for gpu_uuid, used in current_by_gpu.items()
    )


def wait_for_foreign_settle(
    baseline: dict[str, int] | None = None,
    gpu_uuids: set[str] | None = None,
    *, ignored_processes: set[str] | None = None,
    strict_contexts: bool = False,
) -> None:
    if strict_contexts and baseline is None:
        baseline = {}
    baseline = ({key: value for key, value in baseline.items()
                 if key not in (ignored_processes or set())}
                if baseline is not None else None)
    def observed() -> dict[str, int]:
        return {key: value for key, value in foreign_gpu_usage(strict=BACKEND_TYPE == "cuda").items()
                if key not in (ignored_processes or set())}
    deadline = time.monotonic() + ANON_MAX_DRAIN
    last = foreign_usage_by_gpu(observed(), gpu_uuids)
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(ANON_POLL)
        current_raw = observed()
        current = foreign_usage_by_gpu(current_raw, gpu_uuids)
        if strict_contexts and baseline is not None:
            # Unknown ownership cannot borrow another process's freed bytes.
            # Even a zero-byte new CUDA context must disappear before release.
            clean = all(key in baseline and value <= baseline[key]
                        for key, value in current_raw.items()
                        if gpu_uuids is None or key.rsplit('@', 1)[-1] in gpu_uuids)
            if clean:
                return
            continue
        if current == last:
            if (time.monotonic() - stable_since >= ANON_SETTLE
                    and (baseline is None
                         or foreign_usage_at_or_below(
                             baseline, current_raw, gpu_uuids,
                             FOREIGN_RELEASE_TOLERANCE_MIB,
                         ))):
                return
        else:
            last = current
            stable_since = time.monotonic()
    if baseline is not None:
        raise TimeoutError(
            "foreign GPU allocation did not return to its pre-lease baseline"
            + ("; register the workload with ready --pid before release "
               "or wait for unverified CUDA contexts to exit" if strict_contexts else "")
        )


def workload_contains_pid(root: int, pid: int) -> bool:
    """Follow parent IDs, never command lines or process environments."""
    for _ in range(256):
        if pid == root:
            return True
        if pid <= 1:
            break
        try:
            pid = int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return False


def wait_for_owned_cuda_exit(keys: list[str]) -> None:
    deadline = time.monotonic() + ANON_MAX_DRAIN
    while True:
        live = foreign_gpu_usage(strict=BACKEND_TYPE == "cuda")
        if not set(keys).intersection(live):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("lease workload still has CUDA contexts; stop it before release")
        time.sleep(ANON_POLL)


CLIENT_HEADER = "X-Ollama-Unify-Client"
CLIENT_HISTORY_LIMIT = max(1, env_int("OLLAMA_UNIFY_CLIENT_HISTORY_LIMIT", 256))
CLIENT_HISTORY_TTL = max(
    0.1, env_float("OLLAMA_UNIFY_CLIENT_HISTORY_TTL", 3600.0)
)
CLIENT_LANE_HISTORY_LIMIT = max(
    1, env_int("OLLAMA_UNIFY_CLIENT_LANE_HISTORY_LIMIT", 32)
)
LANE_CLIENT_LIMIT = 8
DOCKER_SOCKET = "/var/run/docker.sock"
DOCKER_CACHE_SECONDS = 30.0
CGROUP_ROOT = "/sys/fs/cgroup"


def clean_client_text(value: Any, limit: int = 128) -> str:
    text = "".join(ch for ch in str(value or "") if ch.isprintable()).strip()
    return text[:limit]


def parse_socket_owner(output: str) -> dict[str, Any] | None:
    """Parse one `ss -tne` line into the owning uid, inode, and cgroup."""
    for line in output.splitlines():
        uid = re.search(r"\buid:(\d+)", line)
        inode = re.search(r"\bino:(\d+)", line)
        if not uid:
            continue
        cgroup = re.search(r"\bcgroup:(\S+)", line)
        return {
            "uid": int(uid.group(1)),
            "inode": int(inode.group(1)) if inode else 0,
            "cgroup": cgroup.group(1) if cgroup else "",
        }
    return None


def socket_owner(peer: tuple[str, int], local: tuple[str, int]
                 ) -> dict[str, Any] | None:
    """Find the host process side of a proxy connection without privileges.

    Socket diagnostics report the uid and cgroup of any socket in this
    network namespace. Clients in another namespace (bridge-network
    containers, remote hosts) are not found here.
    """
    def endpoint(host: str, port: int) -> str:
        return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"

    try:
        output = subprocess.run([
            "ss", "-tneH", "state", "established",
            "src", endpoint(*peer), "dst", endpoint(*local),
        ], capture_output=True, text=True, timeout=2, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_socket_owner(output)


def cgroup_unit(cgroup: str) -> tuple[str, str]:
    """Return the systemd unit and any Docker container id for a cgroup."""
    for part in reversed([part for part in cgroup.split("/") if part]):
        if part.startswith("docker-") and part.endswith(".scope"):
            return part, part[len("docker-"):-len(".scope")]
        if part.endswith((".service", ".scope")):
            return part, ""
    return "", ""


# Programs that host other programs in a login or terminal scope.
SESSION_PROGRAMS = frozenset({
    "bash", "sh", "dash", "zsh", "fish", "ksh", "tcsh", "tmux", "screen",
    "login", "su", "sudo", "sshd", "systemd", "(sd-pam)",
})
SCRIPT_SUFFIXES = (".py", ".js", ".mjs", ".cjs", ".ts", ".rb", ".pl", ".sh")


def process_name(pid: int) -> str:
    """Name a process by its program, plus its script for interpreters.

    Reads only argv[0] and a script-path argv[1] from the world-readable
    cmdline, never other arguments, which can carry secrets. Thread renames
    make /proc/PID/comm misleading, so it is only the fallback.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as stream:
            argv = [part.decode(errors="replace")
                    for part in stream.read().split(b"\0") if part]
    except OSError:
        argv = []
    if argv:
        name = os.path.basename(argv[0])
        if (len(argv) > 1 and not argv[1].startswith("-")
                and argv[1].endswith(SCRIPT_SUFFIXES)):
            name += " " + os.path.basename(argv[1])
        return name[:96]
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as stream:
            return stream.read().strip()
    except OSError:
        return ""


def socket_process(cgroup: str, inode: int) -> tuple[int | None, list[int]]:
    """Return the pid holding a socket inode, else the cgroup's pids.

    A socket's own process is found by its descriptors, which are readable
    for this broker's own user; other users' processes are narrowed to
    their cgroup members instead.
    """
    try:
        with open(f"{CGROUP_ROOT}{cgroup}/cgroup.procs", encoding="utf-8") as stream:
            pids = [int(value) for value in stream.read().split()]
    except (OSError, ValueError):
        return None, []
    wanted = f"socket:[{inode}]"
    for pid in pids:
        try:
            descriptors = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                if os.readlink(f"/proc/{pid}/fd/{descriptor}") == wanted:
                    return pid, pids
            except OSError:
                continue
    return (pids[0] if len(pids) == 1 else None), pids


class DockerDirectory:
    """Map container IPs and ids to names through the Docker API."""

    def __init__(self, socket_path: str = DOCKER_SOCKET) -> None:
        self.socket_path = socket_path
        self.lock = threading.Lock()
        self.loaded_at = 0.0
        self.by_ip: dict[str, str] = {}
        self.by_id: dict[str, str] = {}

    def _refresh(self) -> None:
        connection = http.client.HTTPConnection("localhost", timeout=2)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(2)
        try:
            client.connect(self.socket_path)
            connection.sock = client
            connection.request("GET", "/containers/json")
            containers = json.loads(connection.getresponse().read())
        finally:
            connection.close()
        by_ip, by_id = {}, {}
        for container in containers:
            name = str((container.get("Names") or ["?"])[0]).lstrip("/")
            by_id[str(container.get("Id") or "")] = name
            networks = (container.get("NetworkSettings") or {}).get("Networks") or {}
            for network in networks.values():
                for key in ("IPAddress", "GlobalIPv6Address"):
                    if network.get(key):
                        by_ip[str(network[key])] = name
        self.by_ip, self.by_id = by_ip, by_id

    def lookup(self, ip: str = "", container_id: str = "") -> str:
        with self.lock:
            def cached() -> str:
                if container_id:
                    return next((name for key, name in self.by_id.items()
                                 if key.startswith(container_id)), "")
                return self.by_ip.get(ip, "")

            found = cached()
            # Refresh on a miss so new containers resolve, at most once per
            # cache period to keep lookups cheap for unknown peers.
            if not found and time.monotonic() - self.loaded_at > DOCKER_CACHE_SECONDS:
                self.loaded_at = time.monotonic()
                try:
                    self._refresh()
                except (OSError, ValueError, http.client.HTTPException):
                    return ""
                found = cached()
            return found


DOCKER_DIRECTORY = DockerDirectory()


def identify_client(peer: tuple[str, int], local: tuple[str, int],
                    declared: str = "", user_agent: str = "") -> dict[str, Any]:
    """Describe which application is on the other end of a proxy connection.

    Records process names but never command lines, which can carry secrets.
    """
    host = peer[0].removeprefix("::ffff:")
    identity: dict[str, Any] = {
        "address": host,
        "declared": clean_client_text(declared),
        "user_agent": clean_client_text(user_agent, 160),
    }
    owner = socket_owner((host, peer[1]), (local[0].removeprefix("::ffff:"), local[1]))
    if owner is not None:
        identity["uid"] = owner["uid"]
        try:
            identity["user"] = pwd.getpwuid(owner["uid"]).pw_name
        except KeyError:
            identity["user"] = str(owner["uid"])
        unit, container_id = cgroup_unit(owner["cgroup"])
        identity["unit"] = unit
        if container_id:
            identity["container"] = (
                DOCKER_DIRECTORY.lookup(container_id=container_id)
                or container_id[:12]
            )
        pid, candidates = socket_process(owner["cgroup"], owner["inode"])
        if pid is not None:
            identity["pid"] = pid
            identity["process"] = process_name(pid)
        elif candidates:
            names = sorted({
                name for name in (process_name(value) for value in candidates[:16])
                if name
            })
            identity["candidate_processes"] = names
            # Descriptors of processes in another group are unreadable, so a
            # scope holding one program besides its shells names the caller.
            programs = [name for name in names
                        if name.split(" ", 1)[0] not in SESSION_PROGRAMS]
            if len(programs) == 1:
                identity["process"] = programs[0]
                identity["process_inferred"] = True
    else:
        container = DOCKER_DIRECTORY.lookup(ip=host)
        if container:
            identity["container"] = container
        else:
            try:
                identity["remote"] = not ipaddress.ip_address(host).is_loopback
            except ValueError:
                identity["remote"] = True
    identity["key"], identity["label"] = client_key_and_label(identity)
    return identity


def client_key_and_label(identity: dict[str, Any]) -> tuple[str, str]:
    user = identity.get("user")
    by = f" ({user})" if user else ""
    if identity.get("declared"):
        return f"app:{identity['declared']}", f"{identity['declared']}{by}"
    if identity.get("container"):
        return (f"container:{identity['container']}",
                f"container {identity['container']}")
    unit = str(identity.get("unit") or "")
    if unit.endswith(".service"):
        return f"unit:{user}:{unit}", f"{unit}{by}"
    if identity.get("process"):
        return (f"process:{user}:{identity['process']}",
                f"{identity['process']}{by}")
    if identity.get("candidate_processes"):
        names = identity["candidate_processes"]
        programs = [name for name in names
                    if name.split(" ", 1)[0] not in SESSION_PROGRAMS]
        return f"unit:{user}:{unit}", f"{' / '.join(programs or names)}{by}"
    if unit:
        return f"unit:{user}:{unit}", f"{unit}{by}"
    agent = identity.get("user_agent") or "unknown client"
    return f"remote:{identity['address']}", f"{identity['address']} · {agent}"


@dataclass
class Lease:
    token: str
    owner: str
    state: str
    requested_mib: int
    created_at: float
    heartbeat_at: float
    transition_started_at: float
    ttl: int
    foreign_baseline: dict[str, int] | None
    gpu_uuids: list[str]
    justification: str = ""
    expected_release_at: float = 0.0
    evacuation_id: str = ""
    yield_on_request: bool = False
    placement: str = "fixed"
    priority: int = 0
    eligible_gpu_uuids: list[str] = field(default_factory=list)
    exclusive: bool = False
    cuda_processes: list[str] | None = None
    capacity_mib: int = 0


def lease_requires_exclusive_gpus(gpu_uuids: list[str], exclusive: bool = False) -> bool:
    """Return whether a scope must never host broker-owned Ollama lanes.

    A multi-GPU owner (tensor parallel, NCCL, CUDA peer access) moves data
    across the NVLink/PCIe fabric between its GPUs. Starting, loading, or
    killing an Ollama lane on one of those GPUs while peer traffic is live
    preceded a fatal NVLink Xid 74 and a host lockup, so such scopes are
    exclusive for their whole lifetime, not only while pending.
    """
    return exclusive or len(gpu_uuids) > 1


def exclusive_cuda_contexts() -> dict[str, int]:
    """Enumerate ALL CUDA contexts, including canonical/base/orphan runners.

    Ordinary foreign accounting intentionally excludes managed processes. An
    explicit private scope may be granted or released only when even those
    contexts are absent, so this query must not use that classification.
    """
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot verify exclusive CUDA contexts") from exc
    contexts: dict[str, int] = {}
    for row in result.stdout.splitlines():
        if not row.strip():
            continue
        fields = [field.strip() for field in row.split(",")]
        if (len(fields) != 3 or not re.fullmatch(r"[0-9]{1,20}", fields[0])
                or int(fields[0]) <= 0 or not fields[1]
                or not re.fullmatch(r"[0-9]{1,20}", fields[2])):
            raise RuntimeError("invalid exclusive CUDA context telemetry")
        contexts[f"{int(fields[0])}@{fields[1]}"] = int(fields[2])
    return contexts



def require_exclusive_cuda_free(gpu_uuids: list[str]) -> None:
    """Require no CUDA context of any ownership before granting/releasing a scope."""
    scope = set(gpu_uuids)
    if any(key.rsplit("@", 1)[-1] in scope for key in exclusive_cuda_contexts()):
        raise RuntimeError("exclusive GPU scope still has CUDA contexts")


LEASE_COORDINATION_WARNING = (
    "GPU reservations coordinate running workloads; a reservation is not a disabled card. "
    "Inspect active_leases before acquiring and identify the owner. Purpose and "
    "expected duration are recommended, not required."
)


def lease_policy_document() -> dict[str, Any]:
    return {
        "sharing": {
            "single_gpu": "positive VRAM budgets share capacity; no lease-count limit",
            "capacity": "live free VRAM and all outstanding lease/lane promises",
            "unknown_budget": "zero remains exclusive against other external owners",
            "growth": "prepare --vram-mib NEW_TOTAL before growing beyond the budget",
            "readiness": "ready --pid WORKLOAD_PID disambiguates concurrent CUDA loaders",
        },
        "cooperative_handoff": {
            "schema": "io.ollama-unify.cooperative-handoff.v1",
            "opt_in_acquire_flag": "--yield-on-request",
            "single_gpu_only": True,
            "wait_timeout_seconds": HANDOFF_TIMEOUT,
            "release_required": True,
        },
        "relocation": {
            "schema": "io.ollama-unify.lease-relocation.v1",
            "placements": ["fixed", "flexible"],
            "default_placement": "fixed",
            "priority_range": [-1000, 1000],
            "default_priority": 0,
            "higher_priority_only": True,
            "owner_drains_and_restarts": True,
            "destination_reserved_before_drain": True,
            "source_cuda_release_required": True,
            "fixed_groups_remain_exclusive": True,
        },
        "warning": LEASE_COORDINATION_WARNING,
        "required_acquire_fields": ["owner"],
        "cli_required_options": ["--owner"],
        "recommended_acquire_fields": ["justification", "expected_duration_seconds"],
        "cli_recommended_options": ["--justification", "--expected-duration"],
        "metadata_advisory": (
            "Provide a meaningful purpose and expected duration when setting up "
            "a workload. Missing metadata is non-blocking and remains unknown; "
            "it does not change heartbeat TTL, GPU scope, or admission checks."
        ),
        "visibility": (
            "Owner, justification, GPU scope, and expected release are visible "
            "to all local broker clients; lease tokens are not exposed in discovery."
        ),
    }


def lease_public_summary(
    lease: Lease | dict[str, Any], now: float | None = None,
) -> dict[str, Any]:
    current = time.time() if now is None else now
    if isinstance(lease, Lease):
        raw = asdict(lease)
    else:
        raw = lease
    try:
        expected_release_at = float(raw.get("expected_release_at") or 0)
    except (TypeError, ValueError):
        expected_release_at = 0.0
    justification = str(raw.get("justification") or "").strip()
    if expected_release_at > 0:
        remaining = int(expected_release_at - current)
        expected_release_utc = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(expected_release_at)
        )
        horizon_status = "overdue" if remaining < 0 else "expected"
    else:
        remaining = None
        expected_release_utc = None
        horizon_status = "unknown"
    recommendations = []
    if not justification:
        recommendations.append(
            "Consider --justification PURPOSE (API: justification) to explain this reservation."
        )
    if expected_release_at <= 0:
        recommendations.append(
            "Consider --expected-duration SECONDS (API: expected_duration_seconds) "
            "to help other agents plan GPU access; this is separate from heartbeat TTL."
        )
    return {
        "owner": str(raw.get("owner") or "unknown"),
        "exclusive": lease_requires_exclusive_gpus(raw.get("gpu_uuids") or [], raw.get("exclusive") is True),
        "yield_on_request": raw.get("yield_on_request") is True,
        "placement": raw.get("placement", "fixed"),
        "priority": int(raw.get("priority") or 0),
        "eligible_gpu_uuids": list(raw.get("eligible_gpu_uuids") or []),
        "state": str(raw.get("state") or "unknown"),
        "gpu_uuids": [
            str(value) for value in (raw.get("gpu_uuids") or [])
            if isinstance(value, str)
        ],
        "requested_mib": max(0, int(raw.get("requested_mib") or 0)),
        "sharing": ("shared" if len(raw.get("gpu_uuids") or []) == 1
                    and not raw.get("exclusive") and int(raw.get("requested_mib") or 0) > 0
                    else "exclusive"),
        "justification": justification or "not provided",
        "created_at": float(raw.get("created_at") or 0),
        "expected_release_at": expected_release_at or None,
        "expected_release_utc": expected_release_utc,
        "seconds_until_expected_release": remaining,
        "horizon_status": horizon_status,
        "recommendations": recommendations,
    }


def lease_visibility_warnings(summaries: list[dict[str, Any]]) -> list[str]:
    if not summaries:
        # Coordination requirements remain in lease_policy. A confirmed empty
        # lease list is healthy state, not an operational warning.
        return []
    visible = []
    for lease in summaries:
        horizon = lease.get("expected_release_utc") or "unknown release time"
        policy = ("flexible" if lease.get("placement") == "flexible" else "fixed")
        visible.append(
            f"{lease.get('owner')} until {horizon} ({policy}, priority {lease.get('priority', 0)})"
        )
    return [(
        LEASE_COORDINATION_WARNING
        + " Active external leases: "
        + "; ".join(visible)
        + "."
    )]


def process_group_alive(process: Any) -> bool:
    if process is None:
        return False
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_group_identity(process: Any) -> list[dict[str, int]]:
    """Include native runner children, not just the long-lived Ollama server."""
    if process is None or process.poll() is not None:
        raise RuntimeError("managed server is not alive")
    members = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/stat", encoding="utf-8") as stream:
                fields = stream.read().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == process.pid and fields[0] not in {"Z", "X"}:
            members.append({"pid": int(entry.name), "start_time_ticks": int(fields[19])})
    if not any(item["pid"] == process.pid for item in members):
        raise RuntimeError("managed server process identity is unavailable")
    return sorted(members, key=lambda item: item["pid"])


def managed_server_identity(process: Any) -> tuple[int, int, int]:
    """Read the actual spawned server generation without exposing its env."""
    try:
        if process is None or process.poll() is not None:
            raise ValueError("server exited")
        with open(f"/proc/{process.pid}/stat", encoding="utf-8") as stream:
            fields = stream.read().rsplit(")", 1)[1].split()
        identity = (process.pid, int(fields[19]), int(fields[2]))
        if fields[0] in {"Z", "X"} or identity[1] <= 0 or identity[2] != process.pid:
            raise ValueError("server generation unavailable")
        return identity
    except (OSError, ValueError, IndexError, TypeError) as exc:
        raise CapacityError("managed server identity is unverified before model loading",
                            503, "gpu_placement_unverified", False, None) from exc


def assert_managed_server_scope(process: Any, scope: tuple[str, ...],
                                expected: tuple[int, int, int]) -> None:
    """Attest actual CVD before even a zero-token model load is permitted."""
    try:
        if managed_server_identity(process) != expected:
            raise ValueError("server generation changed")
        with open(f"/proc/{process.pid}/environ", "rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("environment exceeds inspection bound")
        values = [entry[len(b"CUDA_VISIBLE_DEVICES="):]
                  for entry in raw.split(b"\0") if entry.startswith(b"CUDA_VISIBLE_DEVICES=")]
        if len(values) != 1 or tuple(values[0].decode("ascii").split(",")) != scope:
            raise ValueError("actual CUDA scope differs")
        if managed_server_identity(process) != expected:
            raise ValueError("server generation changed during inspection")
    except (OSError, ValueError, IndexError, TypeError) as exc:
        # Never include raw values, full environment, or private exceptions.
        raise CapacityError("managed server CUDA scope is unverified before model loading",
                            503, "gpu_placement_unverified", False, None) from exc


def parse_warm_admission(raw: str) -> str:
    """Canonicalize a bounded, closed certificate; it grants no GPU authority."""
    try:
        if not raw or len(raw) > 16384:
            raise ValueError("invalid certificate length")
        proof = json.loads(raw)
        if not isinstance(proof, dict) or set(proof) != {
            "schema", "broker_instance_id", "model", "model_digest",
            "context_length", "gpu_uuids", "lanes",
        }:
            raise ValueError("invalid certificate fields")
        if proof["schema"] != WARM_ADMISSION_SCHEMA:
            raise ValueError("invalid certificate schema")
        for key in ("broker_instance_id", "model_digest"):
            if not isinstance(proof[key], str) or not re.fullmatch(r"[0-9a-f]{64}", proof[key]):
                raise ValueError("invalid certificate identity")
        if not isinstance(proof["model"], str) or not proof["model"] or canonical_model_tag(proof["model"]) != proof["model"]:
            raise ValueError("invalid certificate model")
        if type(proof["context_length"]) is not int or not 1 <= proof["context_length"] <= 2 ** 31 - 1:
            raise ValueError("invalid certificate context")
        scope = proof["gpu_uuids"]
        if not isinstance(scope, list) or not scope or any(not isinstance(gpu, str) or not gpu for gpu in scope) or len(set(scope)) != len(scope):
            raise ValueError("invalid certificate scope")
        lanes = proof["lanes"]
        if not isinstance(lanes, list) or not 1 <= len(lanes) <= 128:
            raise ValueError("invalid certificate lanes")
        for lane in lanes:
            if not isinstance(lane, dict) or set(lane) != {"id", "generation_id"} or not isinstance(lane["id"], str) or not re.fullmatch(r"lane-\d+", lane["id"]) or not isinstance(lane["generation_id"], str) or not re.fullmatch(r"[0-9a-f]{64}", lane["generation_id"]):
                raise ValueError("invalid certificate lane identity")
        if len({lane["id"] for lane in lanes}) != len(lanes):
            raise ValueError("duplicate certificate lane")
        return json.dumps(proof, sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise PermanentCapacityError("invalid warm admission certificate", 400,
                                     "invalid_warm_admission") from exc


@dataclass
class Lane:
    lane_id: str
    kind: str
    host: str
    port: int
    gpu_uuid: str | None
    model: str
    parallel: int
    reserved_mib: int
    created_at: float
    last_used: float
    process: Any = None
    in_flight: int = 0
    retiring: bool = False
    resolved_context_length: int | None = None
    # Which client's request started this lane, and who has used it since.
    triggered_by: dict[str, str] | None = None
    clients: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_clients: dict[str, int] = field(default_factory=dict)
    gpu_uuids: tuple[str, ...] = ()
    reserved_mib_by_gpu: dict[str, int] = field(default_factory=dict)
    observed_vram_mib_by_gpu: dict[str, int] = field(default_factory=dict)
    loading: bool = False
    context_profile: dict | None = None
    openai_context_compatible: bool = True
    warm_generation_id: str = ""
    warm_runtime_identity: dict[str, Any] | None = None
    cpu_offload: bool = False
    host_reserved_mib: int = 0

    def context_profile_matches(self) -> bool:
        return self.context_profile == effective_model_context_profile(self.model)

    @property
    def scope(self) -> tuple[str, ...]:
        return self.gpu_uuids or ((self.gpu_uuid,) if self.gpu_uuid else ())

    def allows(self, gpu_uuids: tuple[str, ...] | list[str] | None) -> bool:
        return gpu_uuids is None or set(self.scope).issubset(gpu_uuids)

    def can_serve(self, gpu_uuids: tuple[str, ...] | None, cpu_offload: bool) -> bool:
        return self.allows(gpu_uuids) and (not self.cpu_offload or (
            cpu_offload and set(self.scope) == set(gpu_uuids if gpu_uuids is not None else SELECTED_GPUS)))

    @property
    def protected_scope(self) -> tuple[str, ...]:
        # Failed placement must retain protection for observed devices too.
        return tuple(dict.fromkeys((*self.scope, *self.observed_vram_mib_by_gpu)))

    def public_summary(self) -> dict[str, Any]:
        alive = self.kind == "system" or process_group_alive(self.process)
        return {
            "id": self.lane_id,
            "kind": self.kind,
            "gpu_uuid": self.gpu_uuid,
            "gpu_uuids": list(self.scope),
            "reserved_mib_by_gpu": self.reserved_mib_by_gpu or {
                gpu: self.reserved_mib for gpu in self.scope
            },
            "observed_vram_mib_by_gpu": self.observed_vram_mib_by_gpu,
            "exclusive": len(self.scope) > 1 or self.cpu_offload,
            "cpu_offload": self.cpu_offload,
            "host_reserved_mib": self.host_reserved_mib,
            "state": "retiring" if alive and self.retiring else (
                "loading" if alive and self.loading else "ready" if alive else "stopped"
            ),
            "model": self.model or None,
            "parallel": self.parallel,
            "in_flight": self.in_flight,
            "reserved_mib": self.reserved_mib,
            "context_profile": self.context_profile,
            "context_resolution": MODEL_CONTEXT_RESOLUTIONS.get(self.model),
            "resolved_context_length": self.resolved_context_length,
            "triggered_by": self.triggered_by,
            "clients": sorted(
                ({
                    "key": key, **usage,
                    "expires_at": float(usage.get("last_seen") or 0)
                    + CLIENT_HISTORY_TTL,
                } for key, usage in self.clients.items()),
                key=lambda usage: -usage["last_seen"],
            ),
        }


class CapacityError(RuntimeError):
    def __init__(
        self,
        message: str,
        status: int = 503,
        reason_code: str = "lane_capacity_wait",
        retryable: bool = True,
        retry_after: int | None = 2,
        *,
        request_id: str = "",
        logical_request_id: str = "",
        admission_retained: bool = False,
        cause_reason_code: str = "",
        queue_position: int | None = None,
        queue_ticket: int | None = None,
    ) -> None:
        self.status = status
        self.reason_code = reason_code
        self.retryable = retryable
        self.retry_after = retry_after if retryable else None
        self.request_id = request_id
        self.logical_request_id = logical_request_id
        self.admission_retained = admission_retained
        self.cause_reason_code = cause_reason_code
        self.queue_position = queue_position
        self.queue_ticket = queue_ticket
        super().__init__(message)


class PermanentCapacityError(CapacityError):
    def __init__(
        self,
        message: str,
        status: int = 422,
        reason_code: str = "permanent_capacity_error",
        *,
        request_id: str = "",
        logical_request_id: str = "",
        queue_position: int | None = None,
        queue_ticket: int | None = None,
    ) -> None:
        super().__init__(
            message,
            status,
            reason_code,
            False,
            None,
            request_id=request_id,
            logical_request_id=logical_request_id,
            queue_position=queue_position,
            queue_ticket=queue_ticket,
        )


class BackgroundCapacityDeferred(CapacityError):
    """Yield optional work without reclaiming another resident model."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 503, "background_capacity_deferred", True, 2)


WARM_PREFLIGHT_PHASES = frozenset({
    "validation", "proof_issuance", "pre_admission", "queue_validation",
    "body_preparation", "post_admission_validation", "backend_dispatch_validation",
})
WARM_PREFLIGHT_CAUSES = frozenset({
    "unspecified", "broker_instance_mismatch", "model_mismatch", "gpu_scope_mismatch",
    "broker_draining", "lane_missing", "generation_mismatch", "lane_model_mismatch",
    "gpu_policy_mismatch", "gpu_blocked", "model_digest_mismatch", "context_mismatch",
    "lane_unavailable", "context_profile_mismatch", "runtime_identity_unavailable",
    "native_ps_unavailable", "native_ps_inventory_mismatch", "native_model_mismatch",
    "native_digest_mismatch", "native_context_mismatch", "native_gpu_residency_mismatch",
    "runtime_identity_changed", "selected_lane_not_certified", "embedding_metadata_unavailable",
    "embedding_identity_mismatch", "ownership_uncertain", "native_gpu_placement_mismatch",
})


class WarmPreflightRequired(CapacityError):
    def __init__(self, *, admission_retained: bool = False,
                 phase: str = "validation", causes: tuple[str, ...] = (), **kwargs: Any) -> None:
        self.warm_preflight_phase = phase if phase in WARM_PREFLIGHT_PHASES else "validation"
        self.warm_preflight_causes = tuple(sorted({cause for cause in causes
            if isinstance(cause, str) and cause in WARM_PREFLIGHT_CAUSES})) or ("unspecified",)
        kwargs["cause_reason_code"] = self.warm_preflight_causes[0]
        super().__init__("verified warm lanes changed; fresh hardware preflight is required",
                         503, "warm_preflight_required", True, 2,
                         admission_retained=admission_retained, **kwargs)


class WarmAdmissionOwnershipUncertain(PermanentCapacityError):
    """A stale or uncertain caller must not release or recertify another owner."""
    def __init__(self, admission: Any, failure: CapacityError | None = None) -> None:
        super().__init__("warm admission ownership cannot be safely released", 409,
                         "warm_admission_ownership_uncertain",
                         request_id=admission.request_id,
                         logical_request_id=admission.logical_request_id)
        self.cause_reason_code = "ownership_uncertain"
        self.warm_preflight_phase = getattr(failure, "warm_preflight_phase", "backend_dispatch_validation")
        self.warm_preflight_causes = ("ownership_uncertain",)


class AdmissionTimeoutError(CapacityError):
    def __init__(
        self,
        message: str,
        *,
        request_id: str = "",
        logical_request_id: str = "",
        admission_retained: bool = False,
        cause_reason_code: str = "",
        queue_position: int | None = None,
        queue_ticket: int | None = None,
    ) -> None:
        super().__init__(
            message,
            503,
            "queue_admission_timeout",
            True,
            2,
            request_id=request_id,
            logical_request_id=logical_request_id,
            admission_retained=admission_retained,
            cause_reason_code=cause_reason_code,
            queue_position=queue_position,
            queue_ticket=queue_ticket,
        )


class CompletedResponseUnavailableError(PermanentCapacityError):
    def __init__(self, response: "CompletedResponse") -> None:
        self.completed_status = response.status
        self.completed_body_bytes = response.body_bytes
        self.completed_body_sha256 = response.body_sha256
        self.unavailable_reason = (
            response.unavailable_reason or "response_body_not_retained"
        )
        super().__init__(
            "logical request already completed, but its response body cannot "
            "be replayed safely",
            409,
            "completed_response_unavailable",
            request_id=response.request_id,
            logical_request_id=response.logical_request_id,
        )


class ClientDisconnected(ConnectionError):
    pass


@dataclass(frozen=True)
class RetainedRequest:
    method: str
    path: str
    content_type: str
    body: bytes
    model: str
    fingerprint: str
    gpu_uuids: tuple[str, ...] | None = None
    warm_admission: str | None = None
    cpu_offload: bool = False


@dataclass(frozen=True)
class LogicalRequestTombstone:
    logical_request_id: str
    request_id: str
    reason_code: str
    expires_at: float
    warm_admission: str | None = None


@dataclass
class QueuedRequest:
    request_id: str
    logical_request_id: str
    request_fingerprint: str
    model: str
    enqueued_at: float
    deadline: float
    connected: Callable[[], bool]
    gpu_uuids: tuple[str, ...] | None = None
    request_path: str = ""
    retained_request: RetainedRequest | None = None
    queue_ticket: int = 0
    workload_class: str = "unspecified"
    queue_policy: str = "wait"
    phase: str = "queued"
    initial_position: int = 1
    last_error: str | None = None
    last_reason_code: str | None = None
    terminal_error: str | None = None
    terminal_status: int | None = None
    terminal_reason_code: str | None = None
    terminal_retryable: bool = False
    terminal_retry_after: int | None = None
    attached: bool = True
    resume_deadline: float | None = None
    client: dict[str, str] | None = None
    warm_admission: str | None = None
    cpu_offload: bool = False
    cold_capacity_retry_at: float = 0.0

    def public_summary(self, position: int, now: float) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "logical_request_id": self.logical_request_id or None,
            "model": self.model,
            "gpu_uuids": list(self.gpu_uuids) if self.gpu_uuids is not None else None,
            "request_path": self.request_path or None,
            "workload_class": self.workload_class,
            "queue_policy": self.queue_policy,
            "cpu_offload": self.cpu_offload,
            "position": position,
            "ticket": self.queue_ticket,
            "phase": self.phase,
            "wait_ms": max(0, int((now - self.enqueued_at) * 1000)),
            "last_error": self.last_error,
            "last_reason_code": self.last_reason_code,
        }


@dataclass
class ActiveRequest:
    """Exact ownership record for one admitted proxy request.

    Aggregate counters are presentation data. This record is the authority
    used for release, cancellation, and late-cleanup protection.
    """

    request_id: str
    lane: Lane
    logical_request_id: str
    admitted_at: float
    last_activity_at: float
    expires_at: float
    phase: str = "admitted"
    client_key: str = ""
    detached_at: float | None = None
    cancel_requested_at: float | None = None
    cancel_reason: str = ""
    backend: Any = None
    backend_started: bool = False
    warm_admission: str | None = None
    backend_completed: bool = False
    lane_stop_started: bool = False
    lane_stopped: bool = False
    # HTTPConnection drops .sock when a close-delimited response takes
    # ownership. Keep the exact transport so cancellation still reaches it.
    backend_socket: Any = field(default=None, repr=False)
    cpu_offload: bool = False
    # Serialize the last client write against EOF/reset observation. Without
    # this per-request guard, a client that closes immediately after reading a
    # complete fixed-length body can race the handler's completion commit and
    # incorrectly retire a healthy lane.
    terminal_lock: Any = field(default_factory=threading.RLock, repr=False)


@dataclass(frozen=True)
class Admission:
    lane: Lane
    request_id: str
    logical_request_id: str
    request_fingerprint: str
    queue_ms: int
    initial_position: int
    queue_ticket: int
    retained_request: RetainedRequest | None = None
    warm_admission: str | None = None
    active_request: ActiveRequest | None = field(default=None, repr=False, compare=False)
    cpu_offload: bool = False


@dataclass(frozen=True)
class CompletedResponse:
    """Bounded replay record for one completed logical inference request.

    Bodies never enter logs or discovery. A body-free record is an explicit
    tombstone: the backend completed the logical request, but replay is unsafe,
    so a duplicate must fail closed instead of generating a second response.
    """

    logical_request_id: str
    request_fingerprint: str
    request_id: str
    request_method: str
    request_path: str
    status: int
    reason: str
    headers: tuple[tuple[str, str], ...]
    body: bytes | None
    body_bytes: int
    body_sha256: str
    unavailable_reason: str | None
    completed_at: float
    expires_at: float
    warm_admission: str | None = None
    cpu_offload: bool = False


class Broker:
    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.transition = threading.RLock()
        self.instance_id = secrets.token_hex(32)
        self.leases = self._load_leases()
        # Ephemeral intent, not a CUDA lease. Restart/disconnect/timeout cancels
        # the request; incumbents keep their persisted lease and heartbeat.
        self.lease_handoff_requests: dict[str, dict[str, Any]] = {}
        # Operator allowlists of GPUs per model; absent means every GPU.
        self.model_gpu_policy: dict[str, list[str]] = self._load_model_policy()
        self.client_gpu_policy, self.client_policy_labels = self._load_client_policy()
        self.cache_policies, self.evacuations = self._load_cache_state()
        self.draining = any(
            lease.state in ("pending", "active", "revoking") and not lease.gpu_uuids
            for lease in self.leases.values()
        )
        self.active_requests = 0
        self.last_reason = "restored lease transition" if self.draining else "startup"
        self.stopping = threading.Event()
        self.anonymous_running = False
        self._foreign_usage_checked_at = 0.0
        self._foreign_usage_cache: dict[str, int] | None = None
        self.waiters: list[QueuedRequest] = []
        self.active_request_records: dict[str, ActiveRequest] = {}
        self.logical_in_flight: dict[str, tuple[str, str, int]] = {}
        self.logical_tombstones: dict[str, LogicalRequestTombstone] = {}
        self.retained_request_bytes = 0
        self.next_queue_ticket = 1
        self.completed_responses: dict[str, CompletedResponse] = {}
        self.completed_response_bytes = 0
        self.completed_response_recorded_total = 0
        self.completed_response_replayed_total = 0
        self.completed_response_unavailable_total = 0
        self.completed_response_conflict_total = 0
        self.completed_response_expired_total = 0
        self.completed_response_evicted_total = 0
        self.reconcile_retry_at = 0.0
        self.reconcile_last_error = ""
        self.reconciling_model: str | None = None
        self.queue_enqueued_total = 0
        self.queue_admitted_total = 0
        self.queue_cancelled_total = 0
        self.queue_timed_out_total = 0
        self.queue_admission_timeout_total = 0
        self.queue_resumed_total = 0
        self.queue_stale_total = 0
        self.queue_duplicate_total = 0
        self.queue_rejected_total = 0
        self.queue_wait_ms_total = 0
        self.queue_wait_ms_max = 0
        self.queue_peak = 0
        self.request_activity_renewed_total = 0
        self.request_disconnected_total = 0
        self.request_expired_total = 0
        self.request_cancelled_total = 0
        self.request_forced_lane_stop_total = 0
        self.request_forced_release_total = 0
        self.request_terminal_release_total = 0
        now = time.time()
        self.lanes: dict[str, Lane] = {
            "base": Lane(
                "base", "system", BACKEND_HOST, BACKEND_PORT, None, "",
                POOL_INSTANCE_PARALLEL, 0, now, now,
            )
        }
        self.next_lane_id = 1
        self.clients: dict[str, dict[str, Any]] = {}
        if any(lease.state == "revoking" for lease in self.leases.values()):
            with self.cv:
                self._persist_leases_locked()

    def _load_leases(self) -> dict[str, Lease]:
        try:
            payload = json.loads(LEASE_STATE_PATH.read_text())
            raw_leases = payload.get("leases", []) if isinstance(payload, dict) else []
        except FileNotFoundError:
            return {}
        except (OSError, TypeError, ValueError) as exc:
            LOG.error("cannot load lease state %s: %s", LEASE_STATE_PATH, exc)
            return {}
        now = time.time()
        leases: dict[str, Lease] = {}
        for raw in raw_leases:
            if isinstance(raw, dict) and type(raw.get("exclusive", False)) is not bool:
                raise RuntimeError("invalid persisted exclusive lease flag; refusing to discard reservations")
            try:
                created_at = float(raw["created_at"])
                state = str(raw["state"])
                transition_started_at = float(
                    raw.get("transition_started_at", created_at)
                )
                if (state == "pending" and PENDING_TIMEOUT > 0
                        and now - transition_started_at >= PENDING_TIMEOUT):
                    state = "revoking"
                baseline_raw = raw.get("foreign_baseline")
                baseline = (
                    {str(key): max(0, int(value))
                     for key, value in baseline_raw.items()}
                    if isinstance(baseline_raw, dict) else None
                )
                lease = Lease(
                    exclusive=raw.get("exclusive", False),
                    cuda_processes=([str(key) for key in raw["cuda_processes"]]
                                    if isinstance(raw.get("cuda_processes"), list) else None),
                    capacity_mib=max(0, int(raw.get("capacity_mib") or 0)),
                    token=str(raw["token"]), owner=str(raw["owner"]),
                    state=state, requested_mib=max(0, int(raw["requested_mib"])),
                    created_at=created_at, heartbeat_at=float(raw["heartbeat_at"]),
                    transition_started_at=transition_started_at,
                    ttl=max(0, int(raw["ttl"])), foreign_baseline=baseline,
                    gpu_uuids=[
                        str(value) for value in raw.get("gpu_uuids", [])
                        if isinstance(value, str)
                    ] if isinstance(raw.get("gpu_uuids", []), list) else [],
                    justification=str(raw.get("justification") or ""),
                    expected_release_at=float(
                        raw.get("expected_release_at") or 0
                    ),
                    evacuation_id=str(raw.get("evacuation_id") or ""),
                    yield_on_request=(raw.get("yield_on_request") is True
                                      and len(raw.get("gpu_uuids") or []) == 1),
                    placement=("flexible" if raw.get("placement") == "flexible"
                               and len(raw.get("gpu_uuids") or []) == 1 else "fixed"),
                    priority=(raw.get("priority", 0) if type(raw.get("priority", 0)) is int
                              and -1000 <= raw.get("priority", 0) <= 1000 else 0),
                    eligible_gpu_uuids=[str(value) for value in (raw.get("eligible_gpu_uuids")
                                        if isinstance(raw.get("eligible_gpu_uuids"), list) else [])
                                        if isinstance(value, str)],
                )
            except (KeyError, TypeError, ValueError):
                continue
            if lease.state not in ("pending", "active", "revoking"):
                continue
            if (lease.ttl > 0 and now - lease.heartbeat_at > lease.ttl
                    and lease.state != "revoking"):
                lease.state = "revoking"
            leases[lease.token] = lease
        if leases:
            LOG.warning("restored %s persisted GPU lease(s)", len(leases))
        return leases

    def _load_model_policy(self) -> dict[str, list[str]]:
        try:
            raw = json.loads(MODEL_POLICY_PATH.read_text()).get("models", {})
        except (OSError, ValueError, AttributeError):
            return {}
        return {
            canonical_model_tag(str(model)): [str(value) for value in gpus]
            for model, gpus in raw.items()
            if isinstance(gpus, list) and gpus
        }

    def _persist_model_policy_locked(self) -> None:
        MODEL_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = MODEL_POLICY_PATH.with_name(
            f".{MODEL_POLICY_PATH.name}.{os.getpid()}.tmp"
        )
        payload = {
            "schema": "io.ollama-unify.gpu-negotiator.model-gpu-policy.v1",
            "models": self.model_gpu_policy,
            "clients": self.client_gpu_policy,
            "client_labels": self.client_policy_labels,
        }
        with temp_path.open("w", encoding="utf-8") as stream:
            os.chmod(temp_path, 0o600)
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, MODEL_POLICY_PATH)

    def _load_client_policy(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Keep operator policies independent of expiring attribution history."""
        try:
            payload = json.loads(MODEL_POLICY_PATH.read_text())
        except FileNotFoundError:
            return {}, {}
        # Invalid durable policy must not silently re-enable a paused client.
        if not isinstance(payload, dict):
            raise ValueError("invalid GPU policy document")
        policies, labels = payload.get("clients", {}), payload.get("client_labels", {})
        if not isinstance(policies, dict) or not isinstance(labels, dict):
            raise ValueError("invalid client GPU policies")
        if len(policies) > CLIENT_HISTORY_LIMIT:
            raise ValueError("too many saved client GPU policies")
        for key, gpus in policies.items():
            if (not isinstance(key, str) or not key or len(key) > 512
                    or any(ord(char) < 32 for char in key)
                    or not isinstance(gpus, list)
                    or any(not isinstance(gpu, str) or not gpu for gpu in gpus)):
                raise ValueError("invalid saved client GPU policy")
        return ({key: list(dict.fromkeys(gpus)) for key, gpus in policies.items()},
                {key: clean_client_text(labels.get(key) or key)
                 for key in policies})

    def _client_gpu_constraint_locked(
        self, client_key: str, gpu_uuids: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        allowed = self.client_gpu_policy.get(client_key)
        if allowed is None:
            return gpu_uuids
        if not allowed:
            raise PermanentCapacityError(
                "GPU requests are paused for this client; select an allowed GPU to resume",
                403, "client_gpu_disabled")
        narrowed = (tuple(allowed) if gpu_uuids is None else
                    tuple(gpu for gpu in gpu_uuids if gpu in allowed))
        if not narrowed:
            raise PermanentCapacityError(
                "the request GPU scope does not overlap this client's allowed GPUs",
                409, "client_gpu_policy_conflict")
        return narrowed

    def _reject_client_waiter_locked(self, waiter: QueuedRequest,
                                     failure: PermanentCapacityError) -> None:
        waiter.phase = "failed"
        waiter.terminal_error = str(failure)
        waiter.terminal_status = failure.status
        waiter.terminal_reason_code = failure.reason_code
        self._record_logical_tombstone_locked(
            waiter.logical_request_id, waiter.request_id, failure.reason_code)
        self.queue_rejected_total += 1
        self._remove_waiter_locked(waiter)

    def _load_cache_state(self) -> tuple[dict, dict]:
        try:
            raw = json.loads(CACHE_STATE_PATH.read_text())
        except FileNotFoundError:
            return {}, {}
        if (not isinstance(raw, dict) or raw.get("schema") != "io.ollama-unify.cache-residency.v1"
                or not isinstance(raw.get("policies"), dict) or not isinstance(raw.get("operations"), dict)):
            raise ValueError("invalid cache residency journal; refusing to discard recovery ownership")
        operations = raw["operations"]
        for operation in operations.values():
            if not isinstance(operation, dict) or not isinstance(operation.get("moves"), list):
                raise ValueError("invalid evacuation operation")
            if operation.get("state") not in ("lease_pending", "rolled_back", "failed", "cancelled"):
                operation["state"] = "recovery_pending"
        return raw["policies"], operations

    def _persist_cache_state_locked(self) -> None:
        CACHE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_STATE_PATH.with_name(f".{CACHE_STATE_PATH.name}.{os.getpid()}.tmp")
        data = {"schema": "io.ollama-unify.cache-residency.v1",
                "policies": self.cache_policies, "operations": self.evacuations}
        with temporary.open("w", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            stream.write(json.dumps(data, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CACHE_STATE_PATH)
        directory = os.open(CACHE_STATE_PATH.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def _evacuation_public(operation: dict) -> dict:
        return json.loads(json.dumps({key: value for key, value in operation.items()
            if key not in ("lease_token", "request")}))

    def _evacuation_recovery_scopes_locked(self, owner: str = "") -> set[str]:
        protected = set()
        for operation in self.evacuations.values():
            if operation["id"] == owner or operation.get("state") not in ("recovery_pending", "restoration_pending"):
                continue
            protected.update(operation["gpu_uuids"])
            for move in operation["moves"]:
                protected.update(move["source"]["gpu_uuids"])
                protected.update(move["destination"]["gpu_uuids"])
            protected.update(operation.get("destination_reservation", {}))
            for attempt in operation.get("spawn_attempts", []):
                protected.update(attempt["gpu_uuids"])
        return protected

    @staticmethod
    def _recorded_runtime_exited(identity: dict) -> bool:
        group_id = identity.get("process_group_id")
        if group_id:
            try:
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    except FileNotFoundError:
                        continue
                    if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
                        return False
            except (OSError, ValueError, IndexError):
                return False
        for member in identity.get("runtime_processes", []):
            try:
                fields = Path(f"/proc/{member['pid']}/stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[19]) == member["start_time_ticks"] and fields[0] not in ("Z", "X"):
                    return False
            except FileNotFoundError:
                continue
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                return False
        return True

    def recover_evacuation(self, operation_id: str) -> dict:
        """Explicit recovery, never adoption or termination of an orphan PID.

        A restart cannot reconstruct old active requests. Until every recorded
        process generation exits, all affected scopes remain protected. After
        exit, restore exact archived cache identities under current hard policy;
        this operation creates no external grant or inference replay.
        """
        with self.transition:
            with self.cv:
                operation = self.evacuations.get(operation_id)
                if operation is None or operation.get("state") not in ("recovery_pending", "restoration_pending"):
                    raise ValueError("evacuation does not require recovery")
                identities = list(operation.get("source_identities", {}).values())
                identities += [move["destination"] for move in operation["moves"]]
                identities += operation.get("spawn_attempts", [])
                if operation.get("destination_attempt"):
                    identities.append(operation["destination_attempt"])
                if any(not self._recorded_runtime_exited(identity) for identity in identities):
                    raise CapacityError("recorded process groups have not exited; recovery retains scopes",
                                        reason_code="evacuation_recovery_wait")
                if self._cache_policy_identity_locked() != operation["policy_sha256"]:
                    raise CapacityError("cache policy changed; operator must cancel the obsolete recovery",
                                        reason_code="evacuation_changed")
            self.begin_drain(f"recover cache evacuation {operation_id}")
            restored = []
            try:
                for source in operation.get("source_identities", {}).values():
                    completed = next((move for move in operation["moves"]
                                      if move["source"]["lane_id"] == source["lane_id"] and move["state"] == "moved"), None)
                    scope = tuple(completed["destination"]["gpu_uuids"] if completed else source["gpu_uuids"])
                    with self.cv:
                        if (operation.get("cancel_requested") or self.stopping.is_set()
                                or self._cache_policy_identity_locked() != operation["policy_sha256"]
                                or len([lane for lane in self.lanes.values() if lane.kind == "managed"]) >= POOL_MAX_SERVERS):
                            raise CapacityError("recovery owner, policy or lane capacity changed", reason_code="evacuation_changed")
                        allowed = self._policy_constraint_locked(source["model"], scope)
                        if allowed != scope:
                            raise CapacityError("recovery cannot widen model hard scope", reason_code="evacuation_changed")
                    self._require_safe_gpu_transition(scope)
                    required, capabilities = self._model_profile(source["model"])
                    if effective_model_context_profile(source["model"]) != source["context_profile"]:
                        raise CapacityError("recovery context identity changed", reason_code="evacuation_changed")
                    live = {d["uuid"]: d for d in self._placement_devices(self._unregistered_gpus_locked()
                        | self._lease_blocked_gpus_locked() | self._evacuation_recovery_scopes_locked(operation_id), operation_id)}
                    if any(live.get(gpu, {}).get("free_mib", 0) < required + POOL_VRAM_RESERVE_MIB for gpu in scope):
                        raise CapacityError("recovery destination is unavailable", reason_code="evacuation_no_destination")
                    lane = self._spawn_lane(source["model"], scope, required, capabilities, "",
                        {"key": "cache-recovery", "label": "cache evacuation recovery"},
                        expected_profile=source["context_profile"], reservation_operation=operation_id)
                    restored.append(lane)
                    if self._cache_lane_identity(lane)["native"] != source["native"]:
                        raise CapacityError("recovery artifact identity changed", reason_code="evacuation_changed")
                    with self.cv:
                        if (operation.get("cancel_requested") or self.stopping.is_set()
                                or self._cache_policy_identity_locked() != operation["policy_sha256"]):
                            raise CapacityError("recovery changed across warm-up", reason_code="evacuation_changed")
                with self.cv:
                    operation["state"] = "lease_pending" if any(lease.evacuation_id == operation_id for lease in self.leases.values()) else "rolled_back"
                    operation["recovered_lanes"] = [self._cache_lane_identity(lane) for lane in restored]
                    operation.pop("destination_reservation", None)
                    self._persist_cache_state_locked()
                return {"ok": True, "evacuation": self._evacuation_public(operation)}
            except Exception:
                self._stop_lanes(restored, "incomplete cache recovery")
                raise
            finally:
                self.end_drain()

    def set_cache_policy(self, model: str, movable: bool, priority: int,
                         gpu_uuids: list[str]) -> dict:
        model = canonical_model_tag(model)
        if (not model or type(movable) is not bool or type(priority) is not int
                or not -1000 <= priority <= 1000 or not isinstance(gpu_uuids, list)
                or len(set(gpu_uuids)) != len(gpu_uuids)
                or any(gpu not in SELECTED_GPUS for gpu in gpu_uuids)
                or (movable and not gpu_uuids)):
            raise ValueError("cache policy needs explicit movable/priority and selected destination UUIDs")
        with self.cv:
            previous = self.cache_policies.get(model)
            policy = {"movable": movable, "priority": priority, "gpu_uuids": list(gpu_uuids)}
            self.cache_policies[model] = policy
            try:
                self._persist_cache_state_locked()
            except Exception:
                if previous is None:
                    self.cache_policies.pop(model, None)
                else:
                    self.cache_policies[model] = previous
                raise
            self.cv.notify_all()
        return {"ok": True, "model": model, "cache_policy": policy}

    def cancel_evacuation(self, operation_id: str) -> dict:
        with self.cv:
            operation = self.evacuations.get(operation_id)
            if operation is None:
                raise KeyError("unknown evacuation")
            if operation["state"] == "lease_pending":
                raise ValueError("evacuation already granted a lease; release that lease normally")
            operation["cancel_requested"] = True
            if operation["state"] in ("recovery_pending", "restoration_pending"):
                identities = list(operation.get("source_identities", {}).values())
                identities += [move["destination"] for move in operation["moves"]]
                identities += operation.get("spawn_attempts", [])
                if operation.get("destination_attempt"):
                    identities.append(operation["destination_attempt"])
                if all(self._recorded_runtime_exited(identity) for identity in identities):
                    operation["state"] = "cancelled"
                    operation.pop("destination_reservation", None)
            self._persist_cache_state_locked()
            self.cv.notify_all()
            return {"ok": True, "evacuation": self._evacuation_public(operation)}

    def archive_evacuation(self, operation_id: str) -> dict:
        with self.transition, self.cv:
            operation = self.evacuations.get(operation_id)
            if operation is None:
                raise KeyError("unknown evacuation")
            if (operation["state"] not in ("lease_pending", "rolled_back", "failed", "cancelled")
                    or any(lease.evacuation_id == operation_id for lease in self.leases.values())):
                raise ValueError("only settled evacuation without a linked lease can be archived")
            del self.evacuations[operation_id]
            try:
                self._persist_cache_state_locked()
            except Exception:
                self.evacuations[operation_id] = operation
                raise
            return {"ok": True, "archived_evacuation": self._evacuation_public(operation)}

    def _cache_policy_identity_locked(self) -> str:
        return hashlib.sha256(json.dumps([self.cache_policies, self.model_gpu_policy],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _cache_lane_identity(self, lane: Lane) -> dict:
        before = process_group_identity(lane.process)
        rows = backend_json_at(lane.host, lane.port, "GET", "/api/ps", timeout=3).get("models")
        if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
                or canonical_model_tag(str(rows[0].get("name") or rows[0].get("model") or "")) != lane.model
                or not re.fullmatch(r"[a-f0-9]{64}", str(rows[0].get("digest") or ""))
                or type(rows[0].get("size")) is not int or rows[0]["size"] <= 0
                or rows[0].get("size_vram") != rows[0]["size"]
                or type(rows[0].get("context_length")) is not int or rows[0]["context_length"] <= 0
                or not lane.context_profile_matches()
                or (lane.context_profile and (
                    rows[0].get("digest") != lane.context_profile["model_digest"]
                    or rows[0].get("context_length") != lane.context_profile["context_length"]))):
            raise CapacityError("cache source identity is not fully GPU verified", reason_code="gpu_placement_unverified")
        usage = process_gpu_usage(lane.process)
        if (set(usage) != set(lane.scope) or any(value <= 0 for value in usage.values())
                or any(value > lane.reserved_mib_by_gpu.get(gpu, lane.reserved_mib) for gpu, value in usage.items())):
            raise CapacityError("cache physical placement differs from its exact scope", reason_code="gpu_placement_unverified")
        if before != process_group_identity(lane.process):
            raise CapacityError("cache runtime changed during inspection", reason_code="gpu_placement_unverified")
        installed = [row for row in backend_json("GET", "/api/tags", timeout=3).get("models", [])
                     if canonical_model_tag(str(row.get("name") or row.get("model") or "")) == lane.model]
        if len(installed) != 1 or installed[0].get("digest") != rows[0]["digest"]:
            raise CapacityError("installed cache artifact changed", reason_code="evacuation_changed")
        native = {key: rows[0].get(key) for key in ("digest", "context_length", "size", "size_vram")}
        return {"lane_id": lane.lane_id, "model": lane.model, "gpu_uuids": list(lane.scope),
                "reserved_mib": lane.reserved_mib, "reserved_mib_by_gpu": lane.reserved_mib_by_gpu,
                "context_profile": lane.context_profile, "native": native,
                "process_group_id": lane.process.pid, "runtime_processes": before}

    def _assert_evacuation_current(self, operation: dict) -> None:
        with self.cv:
            if (self.evacuations.get(operation["id"]) is not operation
                    or operation.get("cancel_requested") or self.stopping.is_set()
                    or self._cache_policy_identity_locked() != operation["policy_sha256"]):
                raise CapacityError("evacuation ownership or policy changed", reason_code="evacuation_changed")
            if operation["kind"] == "prepare":
                lease = self.leases.get(operation["request"]["token"])
                if lease is None or lease.state != "active" or lease.owner != operation["owner"]:
                    raise CapacityError("external resize owner changed", reason_code="evacuation_changed")

    def _commit_cache_evacuation_locked(self, operation_id: str, lease: Lease) -> None:
        # The lease link was persisted while this CV lock was held. Publish
        # the grant phase before a concurrent cancellation can report success.
        operation = self.evacuations[operation_id]
        operation["lease_token"] = lease.token
        operation["state"] = "lease_pending"
        operation["completed_at"] = time.time()
        self._persist_cache_state_locked()

    def _plan_cache_evacuation(self, operation: dict) -> list[tuple[Lane, tuple[str, ...]]]:
        target = set(operation["gpu_uuids"])
        with self.cv:
            lanes = [lane for lane in self.lanes.values() if lane.kind == "managed"]
            victims = [lane for lane in lanes if set(lane.protected_scope).intersection(target)]
            queued = {waiter.model for waiter in self.waiters}
            eligible = []
            for lane in lanes:
                policy = self.cache_policies.get(lane.model, {})
                if (policy.get("movable") is True and policy.get("priority", 1001) < operation["priority"]
                        and not lane.in_flight and not lane.loading and not lane.retiring and lane.model not in queued):
                    eligible.append(lane)
            if any(lane not in eligible for lane in victims):
                raise CapacityError("target has protected cache or queued certified demand", reason_code="evacuation_protected")
            if not victims:
                return []
            if len(lanes) >= POOL_MAX_SERVERS:
                raise CapacityError("copy-first evacuation needs one spare managed lane", reason_code="evacuation_no_destination")
            blocked = self._ollama_blocked_gpus_locked() | self._reserved_gpus_locked() | target
        devices = self._placement_devices(blocked)
        free = {device["uuid"]: max(0, int(device["free_mib"]) - POOL_VRAM_RESERVE_MIB) for device in devices}
        # At most eight explicitly movable caches, one move per source, and a
        # fixed search bound. Copy-first ordering can consolidate an off-target
        # cache before placing a target cache; no unconstrained task planner.
        eligible.sort(key=lambda lane: (lane not in victims, -lane.reserved_mib, lane.lane_id))
        eligible = eligible[:8]
        if any(lane not in eligible for lane in victims):
            raise CapacityError("evacuation exceeds bounded cache plan", reason_code="evacuation_no_destination")
        visits = 0
        def search(remaining: list[Lane], available: dict[str, int], path: list) -> list | None:
            nonlocal visits
            visits += 1
            if visits > 2000:
                return None
            if not any(lane in victims for lane in remaining):
                return path
            for lane in remaining:
                # Existing peer groups are indivisible and cannot be packed
                # into a singleton. No automatic repartition in this feature.
                if len(lane.scope) != 1:
                    continue
                policy = self.cache_policies[lane.model]
                with self.cv:
                    allowed = self._policy_constraint_locked(lane.model, tuple(policy["gpu_uuids"])) or ()
                choices = [gpu for gpu in allowed if gpu in available and gpu not in lane.scope
                           and available[gpu] >= lane.reserved_mib]
                choices.sort(key=lambda gpu: (available[gpu] - lane.reserved_mib, gpu))
                for gpu in choices:
                    next_free = dict(available)
                    next_free[gpu] -= lane.reserved_mib
                    for old in lane.scope:
                        if old not in blocked:
                            next_free[old] = next_free.get(old, 0) + lane.reserved_mib_by_gpu.get(old, lane.reserved_mib)
                    found = search([item for item in remaining if item is not lane], next_free, [*path, (lane, (gpu,))])
                    if found is not None:
                        return found
            return None
        plan = search(eligible, free, [])
        if plan is None:
            raise CapacityError("no safe copy-first destination plan", reason_code="evacuation_no_destination")
        return plan

    def _rollback_cache_evacuation(self, operation: dict, moves: list[tuple[Lane, Lane]]) -> None:
        # Restore exact model/profile/source scopes; never stop the verified
        # destination until its replacement is warm. Failed rollback keeps a
        # truthful restoration obligation and all surviving process promises.
        incomplete = False
        for source, destination in reversed(moves):
            try:
                if process_group_alive(source.process) and source.retiring:
                    incomplete = True
                    continue
                if not process_group_alive(source.process):
                    with self.cv:
                        allowed = self._policy_constraint_locked(source.model, source.scope)
                    if not source.allows(allowed) or not source.context_profile_matches():
                        raise RuntimeError("original cache policy changed")
                    required, capabilities = self._model_profile(source.model)
                    live = {d["uuid"]: d for d in self._placement_devices(self._ollama_blocked_gpus_locked())}
                    if any(live.get(gpu, {}).get("free_mib", 0) < required + POOL_VRAM_RESERVE_MIB for gpu in source.scope):
                        raise RuntimeError("rollback capacity is unavailable")
                    restored = self._spawn_lane(source.model, source.scope, required, capabilities, "",
                        {"key": "cache-rollback", "label": "cache evacuation rollback"},
                        dict(source.reserved_mib_by_gpu), expected_profile=source.context_profile,
                        reservation_operation=operation["id"])
                    if self._cache_lane_identity(restored)["native"] != operation["source_identities"][source.lane_id]["native"]:
                        self._stop_lanes([restored], "rollback identity changed")
                        raise RuntimeError("original cache artifact changed")
                if self._stop_lanes([destination], "cache evacuation rollback"):
                    incomplete = True
            except Exception:
                incomplete = True
        with self.cv:
            operation["state"] = "restoration_pending" if incomplete else (
                "cancelled" if operation.get("cancel_requested") else "rolled_back")
            if not incomplete:
                operation.pop("destination_reservation", None)
            self._persist_cache_state_locked()

    def _lease_evacuation(self, kind: str, request: dict, priority: int, operation_id: str) -> dict:
        if (type(priority) is not int or not -1000 <= priority <= 1000
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,127}", operation_id)):
            raise ValueError("evacuation requires bounded priority and stable operation ID")
        fingerprint = hashlib.sha256(json.dumps([kind, request, priority], sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
        with self.transition:
            with self.cv:
                old = self.evacuations.get(operation_id)
                if old:
                    if old["request_sha256"] != fingerprint:
                        raise ValueError("evacuation ID already binds a different request")
                    linked = next((lease for lease in self.leases.values() if lease.evacuation_id == operation_id), None)
                    if linked:
                        return {"ok": True, "lease": asdict(linked), "evacuation": self._evacuation_public(old)}
                    raise CapacityError("evacuation has a durable terminal or recovery state; inspect status", 409,
                                        "evacuation_requires_recovery", False, None)
                if len(self.evacuations) >= 256:
                    raise ValueError("evacuation journal is full; archive completed operations before accepting more")
                if kind == "prepare":
                    lease = self.leases.get(request["token"])
                    if lease is None or lease.state != "active":
                        raise ValueError("only an active exact lease can prepare evacuation")
                    scope, owner = list(lease.gpu_uuids), lease.owner
                else:
                    scope, owner = list(request["requested_gpu_uuids"]), request["owner"]
                if not scope or len(set(scope)) != len(scope) or set(scope) - set(SELECTED_GPUS):
                    raise ValueError("evacuation requires an exact selected GPU scope")
                if self.pending_lease():
                    raise CapacityError("another external transition is pending", reason_code="lease_transition")
                if set(scope).intersection(self._handoff_scopes_locked()):
                    raise CapacityError("scope reserved for cooperative handoff", reason_code="lease_transition")
                if any((not lease.gpu_uuids or set(scope).intersection(lease.gpu_uuids))
                       and lease.token != request.get("token") for lease in self.leases.values()):
                    raise CapacityError("requested scope has another external owner", reason_code="lease_transition")
                operation = {"id": operation_id, "kind": kind, "owner": owner, "priority": priority,
                    "gpu_uuids": scope, "request_sha256": fingerprint, "request": request,
                    "policy_sha256": self._cache_policy_identity_locked(), "state": "draining",
                    "created_at": time.time(), "moves": [], "source_identities": {}}
                self.evacuations[operation_id] = operation
                try:
                    self._persist_cache_state_locked()
                except Exception:
                    # No drain, spawn or grant has started. Remove only this
                    # uncommitted owner, so its failed journal cannot wedge ID.
                    if self.evacuations.get(operation_id) is operation:
                        del self.evacuations[operation_id]
                    raise
            moved, plan = [], []
            try:
                self.begin_drain(f"cache evacuation {operation_id}")
                self._assert_evacuation_current(operation)
                plan = self._plan_cache_evacuation(operation)
                for source, destination_scope in plan:
                    identity = self._cache_lane_identity(source)
                    self._assert_evacuation_current(operation)
                    with self.cv:
                        if self.lanes.get(source.lane_id) is not source or source.in_flight or source.retiring:
                            raise CapacityError("cache source generation changed", reason_code="evacuation_changed")
                        operation["source_identities"][source.lane_id] = identity
                        operation["current_source"] = source.lane_id
                        operation.pop("destination_attempt", None)
                        operation["state"] = "destination_reserved"
                        operation["destination_reservation"] = {gpu: source.reserved_mib for gpu in destination_scope}
                        self._persist_cache_state_locked()
                    allowed = {d["uuid"]: d for d in self._placement_devices(set(scope), operation_id)}
                    if any(allowed.get(gpu, {}).get("free_mib", 0) < source.reserved_mib + POOL_VRAM_RESERVE_MIB for gpu in destination_scope):
                        raise CapacityError("destination live capacity changed", reason_code="evacuation_no_destination")
                    required, capabilities = self._model_profile(source.model)
                    if required > source.reserved_mib or not source.context_profile_matches():
                        raise CapacityError("source memory identity changed", reason_code="evacuation_changed")
                    destination = self._spawn_lane(source.model, destination_scope, required, capabilities, "",
                        {"key": "cache-evacuation", "label": f"cache evacuation {operation_id}"},
                        expected_profile=source.context_profile, reservation_operation=operation_id)
                    moved.append((source, destination))
                    destination_identity = self._cache_lane_identity(destination)
                    self._assert_evacuation_current(operation)
                    if identity != self._cache_lane_identity(source) or destination_identity["native"] != identity["native"]:
                        raise CapacityError("cache artifact or runtime changed", reason_code="evacuation_changed")
                    with self.cv:
                        if any(waiter.model == source.model for waiter in self.waiters):
                            raise CapacityError("new queued demand protects original cache", reason_code="evacuation_protected")
                        operation["moves"].append({"source": identity, "destination": destination_identity, "state": "source_retiring"})
                        operation.pop("destination_reservation", None)
                        source.retiring = True
                        operation["state"] = "source_retiring"
                        self._persist_cache_state_locked()
                    if self._stop_lanes([source], "lower priority cache evacuation"):
                        raise CapacityError("source process group has not exited", reason_code="lane_stop_failed")
                    with self.cv:
                        operation["moves"][-1]["state"] = "moved"
                        self._persist_cache_state_locked()
                self._assert_evacuation_current(operation)
                with self.cv:
                    if any(lane.kind == "managed" and set(lane.protected_scope).intersection(scope)
                           for lane in self.lanes.values()):
                        raise CapacityError("requested scope is not yet free", reason_code="evacuation_changed")
                result = self.acquire(**request, priority=priority, _evacuation_commit_id=operation_id) if kind == "acquire" else self.prepare(request["token"], _evacuation_commit_id=operation_id)
                with self.cv:
                    lease = self.leases[result["lease"]["token"]]
                    lease.evacuation_id = operation_id
                    self._persist_leases_locked()
                    operation["lease_token"] = lease.token
                    operation["state"] = "lease_pending"
                    operation["completed_at"] = time.time()
                    self._persist_cache_state_locked()
                    result["lease"] = asdict(lease)
                    result["evacuation"] = self._evacuation_public(operation)
                return result
            except Exception as exc:
                with self.cv:
                    operation["failure_code"] = exc.reason_code if isinstance(exc, CapacityError) else "evacuation_failed"
                    if any(lease.evacuation_id == operation_id for lease in self.leases.values()):
                        # A persisted pending grant is irrevocable here. A lost
                        # journal write must not rollback through its scope.
                        operation["state"] = "lease_pending"
                        raise
                    # A failed warm-up can leave an owned process group whose
                    # exit is not yet verified. Preserve it in the operation.
                    attempt = operation.get("destination_attempt", {})
                    attempt_lane = self.lanes.get(attempt.get("lane_id"))
                    if attempt_lane is not None and not any(destination is attempt_lane for _, destination in moved):
                        source = next((lane for lane, _ in plan if lane.lane_id == operation.get("current_source")), None)
                        if source is not None:
                            moved.append((source, attempt_lane))
                self._rollback_cache_evacuation(operation, moved)
                raise
            finally:
                self.end_drain()

    def _policy_constraint_locked(
        self, model: str, gpu_uuids: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        """Narrow a request's GPU constraint by the model's operator policy.

        Returns an empty tuple when the request and the policy share no GPU.
        """
        allowed = self.model_gpu_policy.get(canonical_model_tag(model))
        if allowed is None:
            return gpu_uuids
        if gpu_uuids is None:
            return tuple(allowed)
        return tuple(gpu_uuid for gpu_uuid in gpu_uuids if gpu_uuid in allowed)

    def _capacity_scope_locked(self, model: str, gpu_uuids: tuple[str, ...] | None,
                               placement: str, client_key: str = "") -> tuple[str, ...] | None:
        parse_capacity_placement(placement)
        if placement == "exclusive_group":
            if (gpu_uuids is None or len(gpu_uuids) < 2
                    or len(set(gpu_uuids)) != len(gpu_uuids)):
                raise PermanentCapacityError("exclusive_group requires at least two distinct GPUs",
                                             422, "invalid_gpu_constraint")
            if any(gpu not in SELECTED_GPUS for gpu in gpu_uuids):
                raise PermanentCapacityError("exclusive_group contains an unselected GPU",
                                             422, "gpu_constraint_unavailable")
        client_scope = self._client_gpu_constraint_locked(client_key, gpu_uuids)
        allowed = self._policy_constraint_locked(model, client_scope)
        if placement == "exclusive_group" and allowed != gpu_uuids:
            raise PermanentCapacityError("model or client policy excludes a requested group member",
                                         409, "gpu_policy_conflict")
        return allowed

    def _persist_leases_locked(self) -> None:
        LEASE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = LEASE_STATE_PATH.with_name(
            f".{LEASE_STATE_PATH.name}.{os.getpid()}.tmp"
        )
        payload = {
            "schema": "io.ollama-unify.gpu-negotiator.leases.v2",
            "leases": [asdict(lease) for lease in self.leases.values()],
        }
        temp_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, LEASE_STATE_PATH)

    def _prune_dead_lanes_locked(self) -> None:
        dead = [lane_id for lane_id, lane in self.lanes.items()
                if lane.kind == "managed"
                and not process_group_alive(lane.process)]
        for lane_id in dead:
            lane = self.lanes.pop(lane_id)
            LOG.warning("managed Ollama lane stopped unexpectedly: %s", lane.lane_id)

    def _reserved_gpus_locked(self) -> set[str]:
        return {
            gpu_uuid
            for lease in self.leases.values()
            for gpu_uuid in lease.gpu_uuids
            if lease.state in ("pending", "active", "revoking")
        }

    def _public_lease_summaries_locked(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            lease_public_summary(lease, now)
            for lease in sorted(
                self.leases.values(),
                key=lambda item: (item.expected_release_at or float("inf"), item.owner),
            )
        ]

    def _unregistered_gpus_locked(self, *, refresh: bool = False) -> set[str]:
        if BACKEND_TYPE != "cuda":
            return set()
        # Coalesce read-only scheduler/status polls. Destructive transitions
        # always force a fresh process query immediately before acting.
        if refresh or time.monotonic() - self._foreign_usage_checked_at >= min(ANON_POLL, 0.25):
            try:
                self._foreign_usage_cache = foreign_gpu_usage(strict=True)
            except RuntimeError:
                self._foreign_usage_cache = None
            self._foreign_usage_checked_at = time.monotonic()
        usage = self._foreign_usage_cache
        if usage is None:
            return set(SELECTED_GPUS)
        if any(not lease.gpu_uuids and lease.state in ("pending", "active", "revoking")
               for lease in self.leases.values()):
            return set()
        return {key.rsplit("@", 1)[-1] for key in usage} - self._reserved_gpus_locked()

    def _peer_reserved_gpus_locked(
        self, ignore_lane_id: str | None = None, *,
        ignore_lane_ids: set[str] | None = None,
    ) -> set[str]:
        return {gpu for lane in self.lanes.values()
                if lane.kind == "managed" and (len(lane.scope) > 1 or lane.cpu_offload)
                and lane.lane_id != ignore_lane_id
                and lane.lane_id not in (ignore_lane_ids or ())
                for gpu in lane.protected_scope}

    def _require_safe_gpu_transition(
        self, gpu_uuids: str | tuple[str, ...], ignore_lane_id: str | None = None,
    ) -> None:
        scope = {gpu_uuids} if isinstance(gpu_uuids, str) else set(gpu_uuids)
        if scope - set(SELECTED_GPUS):
            raise CapacityError("CUDA transition touches an unmonitored GPU; retaining the process reservation",
                                503, "gpu_placement_unverified", False, None)
        with self.cv:
            blocked = self._unregistered_gpus_locked(refresh=True)
            peers = self._peer_reserved_gpus_locked(ignore_lane_id)
            leases = self._lease_blocked_gpus_locked()
        affected = scope.intersection(blocked | peers | leases)
        if affected:
            raise CapacityError(
                "CUDA activity, unavailable telemetry, or a managed peer group on "
                f"{sorted(affected)}; model load/unload is deferred until it clears",
                503, "gpu_unregistered_workload" if scope.intersection(blocked)
                else "lease_transition" if scope.intersection(leases)
                else "gpu_peer_group_reserved", True, 1,
            )

    def _unload_base_models(self) -> list[str]:
        # The base backend has no provable GPU scope. An empty metadata/CPU
        # backend needs no unload. Never evict a resident base model around
        # an unknown CUDA owner on any selected GPU.
        models = running_models(require_available=True)
        if not models:
            return []
        with self.cv:
            if (self._unregistered_gpus_locked(refresh=True)
                    or self._peer_reserved_gpus_locked()
                    or self._lease_blocked_gpus_locked()):
                raise CapacityError(
                    "Base model unload deferred around CUDA activity or a managed GPU group",
                    503, "gpu_unregistered_workload", True, 1,
                )
        return unload_all_models()

    def _ollama_blocked_gpus_locked(self) -> set[str]:
        """Return scoped GPUs whose external allocation is not stable.

        A pending or revoking lease can still change its CUDA footprint, so an
        Ollama lane must not use those GPUs. An active lease has completed its
        readiness contract: its allocation is resident and it must call
        prepare before any later VRAM growth. Active single-GPU scopes can
        therefore host managed Ollama lanes when the live free-VRAM admission
        check fits. Active multi-GPU scopes stay blocked: lane churn there
        runs concurrently with the owner's peer-to-peer traffic.

        Bounded single-GPU external owners can share the GPU's VRAM budget.
        Pending transitions still defer Ollama lane churn until ready.
        """
        return (self._unregistered_gpus_locked() | self._lease_blocked_gpus_locked()
                | self._evacuation_recovery_scopes_locked()
                | self._handoff_scopes_locked())

    def _lease_blocked_gpus_locked(self) -> set[str]:
        if any(not lease.gpu_uuids and lease.state in ("pending", "active", "revoking")
               for lease in self.leases.values()):
            return set(SELECTED_GPUS)
        return {
            gpu_uuid
            for lease in self.leases.values()
            for gpu_uuid in lease.gpu_uuids
            if lease.state in ("pending", "revoking")
            or (lease.state == "active"
                and lease_requires_exclusive_gpus(lease.gpu_uuids, lease.exclusive))
        }

    def _global_transition_lease_locked(self) -> Lease | None:
        return next(
            (
                lease for lease in self.leases.values()
                if lease.state in ("pending", "active", "revoking")
                and not lease.gpu_uuids
            ),
            None,
        )

    def _plan_lease_gpus(
        self, requested_mib: int, requested_gpu_uuids: list[str],
        handoff_id: str = "", *, exclusive: bool = False,
        ignore_token: str = "", check_capacity: bool = True,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Check scope compatibility and promises, never just a free-VRAM scan."""
        inventory = [
            device for device in gpu_snapshot()
            if not SELECTED_GPUS or device.get("uuid") in SELECTED_GPUS
        ]
        by_uuid = {str(device.get("uuid") or ""): device for device in inventory}
        requested = list(dict.fromkeys(requested_gpu_uuids))
        if not requested:
            return [], inventory
        unknown = [gpu_uuid for gpu_uuid in requested if gpu_uuid not in by_uuid]
        if unknown:
            raise RuntimeError(
                f"requested GPU UUIDs are not selected and available: {unknown}"
            )
        with self.cv:
            reserved = (self._evacuation_recovery_scopes_locked()
                        | self._handoff_scopes_locked(handoff_id))
            shareable = len(requested) == 1 and not exclusive and requested_mib > 0
            conflicting_leases = [lease_public_summary(lease)
                for token, lease in self.leases.items()
                if token != ignore_token and lease.state in ("pending", "active", "revoking")
                and (not lease.gpu_uuids or set(lease.gpu_uuids).intersection(requested))
                and (not shareable or len(lease.gpu_uuids) != 1 or lease.exclusive
                     or lease.requested_mib <= 0 or lease.state == "revoking")]
        conflicts = set(requested).intersection(reserved)
        if conflicts or conflicting_leases:
            holders = "; ".join(
                f"{lease['owner']} ({lease['justification']}; expected release "
                f"{lease['expected_release_utc'] or 'unknown'})"
                for lease in conflicting_leases
            )
            raise RuntimeError(
                f"requested GPU UUIDs are already leased exclusively or reserved: {sorted(conflicts) or requested}; "
                f"current lessee(s): {holders or 'unknown'}"
            )
        devices = self._lease_capacity_devices([by_uuid[gpu_uuid] for gpu_uuid in requested],
                                               ignore_token=ignore_token)
        aggregate_free = sum(int(device.get("free_mib") or 0) for device in devices)
        if check_capacity and requested_mib > 0 and requested_mib > aggregate_free:
            raise RuntimeError(
                f"requested {requested_mib} MiB but scoped GPUs have only "
                f"{aggregate_free} MiB available after existing lease/lane reservations"
            )
        return requested, devices

    def _external_promises_locked(self, ignore_token: str = "") -> dict[str, int]:
        promises: dict[str, int] = {}
        for token, lease in self.leases.items():
            if token == ignore_token or lease.state not in ("pending", "active", "revoking"):
                continue
            # Multi-GPU reservations cannot share, so no per-shard assumption
            # grants admission there. Single-device budgets are exact totals.
            if len(lease.gpu_uuids) == 1:
                gpu = lease.gpu_uuids[0]
                promises[gpu] = promises.get(gpu, 0) + lease.requested_mib
        return promises

    def _external_unrealized_locked(self, usage: dict[str, int],
                                    ignore_token: str = "") -> dict[str, int]:
        gaps: dict[str, int] = {}
        for token, lease in self.leases.items():
            if (token == ignore_token or len(lease.gpu_uuids) != 1
                    or lease.state not in ("pending", "active", "revoking")):
                continue
            gpu = lease.gpu_uuids[0]
            resident = (sum(usage.get(key, 0) for key in lease.cuda_processes or [])
                        if lease.state == "active" else 0)
            gaps[gpu] = gaps.get(gpu, 0) + max(0, lease.requested_mib-resident)
        return gaps

    def _lease_capacity_devices(self, devices: list[dict[str, Any]], *,
                                ignore_token: str = "", reclaim_lanes: bool = False,
                                reservation_owner: str = "") -> list[dict[str, Any]]:
        usage = foreign_gpu_usage(strict=True)
        managed_resident: dict[str, int] = {}
        if BACKEND_TYPE == "cuda":
            for key, used in exclusive_cuda_contexts().items():
                if key not in usage:
                    gpu = key.rsplit("@", 1)[-1]
                    managed_resident[gpu] = managed_resident.get(gpu, 0)+used
        with self.cv:
            promised = self._external_promises_locked(ignore_token)
            gaps = self._external_unrealized_locked(usage, ignore_token)
            ceilings: dict[str, int] = {}
            owned: dict[str, int] = {}
            for token, lease in self.leases.items():
                if token == ignore_token or len(lease.gpu_uuids) != 1:
                    continue
                gpu = lease.gpu_uuids[0]
                if lease.capacity_mib > 0:
                    ceilings[gpu] = min(ceilings.get(gpu, lease.capacity_mib), lease.capacity_mib)
                owned[gpu] = owned.get(gpu, 0)+sum(usage.get(key, 0) for key in lease.cuda_processes or [])
            lanes: dict[str, int] = {}
            for lane in self.lanes.values():
                if lane.kind != "managed":
                    continue
                for gpu in lane.protected_scope:
                    lanes[gpu] = lanes.get(gpu, 0) + int(
                        lane.reserved_mib_by_gpu.get(gpu, lane.reserved_mib))
            for operation in self.evacuations.values():
                if operation["id"] == reservation_owner:
                    continue
                if operation.get("destination_attempt", {}).get("lane_id") in self.lanes:
                    continue
                for gpu, amount in operation.get("destination_reservation", {}).items():
                    lanes[gpu] = lanes.get(gpu, 0) + amount
        result = []
        for device in devices:
            gpu = device["uuid"]
            physical = max(0, int(device.get("physical_free_mib", device.get("free_mib")) or 0))
            # Physical free already excludes resident allocations, driver and
            # display overhead. Subtract only promises not yet resident.
            lane_gap = max(0, lanes.get(gpu, 0)-managed_resident.get(gpu, 0))
            available = max(0, physical - gaps.get(gpu, 0) - lane_gap)
            if reclaim_lanes:
                available = max(0, physical+managed_resident.get(gpu, 0)-gaps.get(gpu, 0))
            # Keep verified usable capacity fixed while leases hold it:
            # a concurrent free must not widen outstanding promises.
            ceiling = ceilings.get(gpu, min(int(device["total_mib"]),
                physical+owned.get(gpu, 0)+managed_resident.get(gpu, 0)))
            available = min(available, max(0, ceiling-promised.get(gpu, 0)
                                          -(0 if reclaim_lanes else lanes.get(gpu, 0))))
            result.append({**device, "physical_free_mib": physical,
                "external_reserved_mib": promised.get(gpu, 0),
                "lease_capacity_mib": ceiling,
                "free_mib": available if reclaim_lanes else min(physical, available)})
        return result

    def _wait_scope_idle(self, scope: set[str]) -> None:
        deadline = time.monotonic()+DRAIN_TIMEOUT
        with self.cv:
            lanes = [lane for lane in self.lanes.values() if lane.kind == "managed"
                     and scope.intersection(lane.protected_scope)]
            previous = {lane.lane_id: lane.retiring for lane in lanes}
            for lane in lanes:
                lane.retiring = True
            self.cv.notify_all()
            while any(lane.in_flight for lane in lanes):
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    for lane in lanes:
                        lane.retiring = previous[lane.lane_id]
                    self.cv.notify_all()
                    raise TimeoutError("requests on the requested GPU scope did not drain")
                self.cv.wait(min(remaining, 0.5))

    def _remove_completed_response_locked(
        self, logical_request_id: str, *, expired: bool = False,
        evicted: bool = False,
    ) -> None:
        response = self.completed_responses.pop(logical_request_id, None)
        if response is None:
            return
        if response.body is not None:
            self.completed_response_bytes = max(
                0, self.completed_response_bytes - len(response.body)
            )
        if expired or (evicted and response.warm_admission is not None):
            self._record_logical_tombstone_locked(
                logical_request_id,
                response.request_id,
                "logical_request_expired" if expired else "completed_response_evicted",
                warm_admission=response.warm_admission,
            )
        if expired:
            self.completed_response_expired_total += 1
        if evicted:
            self.completed_response_evicted_total += 1

    def _prune_completed_responses_locked(
        self, now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else now
        expired = [
            logical_request_id
            for logical_request_id, response in self.completed_responses.items()
            if current >= response.expires_at
        ]
        for logical_request_id in expired:
            self._remove_completed_response_locked(
                logical_request_id, expired=True
            )

    def _evict_oldest_completed_response_locked(self) -> bool:
        if not self.completed_responses:
            return False
        oldest = min(
            self.completed_responses,
            key=lambda logical_request_id: self.completed_responses[
                logical_request_id
            ].completed_at,
        )
        self._remove_completed_response_locked(oldest, evicted=True)
        return True

    def completed_response_lookup(
        self, logical_request_id: str, request_fingerprint: str,
    ) -> CompletedResponse | None:
        if not logical_request_id:
            return None
        with self.cv:
            self._prune_completed_responses_locked()
            response = self.completed_responses.get(logical_request_id)
            if response is None:
                return None
            self.queue_duplicate_total += 1
            if response.request_fingerprint != request_fingerprint:
                self.completed_response_conflict_total += 1
                raise PermanentCapacityError(
                    "logical request ID was reused with different request content",
                    409,
                    "logical_request_conflict",
                    request_id=response.request_id,
                    logical_request_id=logical_request_id,
                )
            if response.body is None:
                self.completed_response_unavailable_total += 1
                raise CompletedResponseUnavailableError(response)
            self.completed_response_replayed_total += 1
            return response

    def _record_logical_tombstone_locked(
        self, logical_request_id: str, request_id: str, reason_code: str,
        *, warm_admission: str | None = None,
    ) -> None:
        if not logical_request_id:
            return
        owner = next((item for item in self.waiters if item.request_id == request_id), None)
        owner = owner or self.active_request_records.get(request_id)
        self.logical_tombstones[logical_request_id] = LogicalRequestTombstone(
            logical_request_id=logical_request_id,
            request_id=request_id,
            reason_code=reason_code,
            expires_at=time.monotonic() + COMPLETED_RESPONSE_TTL,
            warm_admission=warm_admission or (owner.warm_admission if owner is not None else None),
        )

    def _prune_logical_tombstones_locked(
        self, now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else now
        for logical_request_id in [
            key for key, tombstone in self.logical_tombstones.items()
            if current >= tombstone.expires_at
        ]:
            self.logical_tombstones.pop(logical_request_id, None)

    def _raise_logical_tombstone_locked(self, logical_request_id: str) -> None:
        self._prune_logical_tombstones_locked()
        tombstone = self.logical_tombstones.get(logical_request_id)
        if tombstone is None:
            return
        raise PermanentCapacityError(
            "logical request can no longer be resumed",
            409,
            tombstone.reason_code,
            request_id=tombstone.request_id,
            logical_request_id=logical_request_id,
        )

    def resume_request_lookup(
        self, logical_request_id: str, method: str, path: str,
        warm_admission: str | None = None,
        gpu_uuids: tuple[str, ...] | None = None,
        cpu_offload: bool | None = None,
    ) -> tuple[RetainedRequest | None, CompletedResponse | None]:
        if not logical_request_id:
            raise PermanentCapacityError(
                "resume requires a logical request ID",
                400,
                "invalid_admission_header",
            )
        with self.cv:
            now = time.monotonic()
            self._prune_completed_responses_locked(now)
            self._prune_stale_waiters_locked(now)
            self._raise_logical_tombstone_locked(logical_request_id)
            response = self.completed_responses.get(logical_request_id)
            if response is not None:
                if cpu_offload is not None and cpu_offload != response.cpu_offload:
                    raise PermanentCapacityError("resume CPU offload permission differs from original request", 409,
                                                 "logical_request_conflict")
                if warm_admission is not None and warm_admission != response.warm_admission:
                    raise PermanentCapacityError("resume warm certificate differs from original request", 409,
                                                 "logical_request_conflict")
                if (response.warm_admission is not None and gpu_uuids is not None
                        and list(gpu_uuids) != json.loads(response.warm_admission)["gpu_uuids"]):
                    raise PermanentCapacityError("resume GPU constraint differs from original request", 409,
                                                 "logical_request_conflict")
                if response.request_method != method or response.request_path != path:
                    raise PermanentCapacityError(
                        "logical request ID was resumed on a different method or path",
                        409,
                        "logical_request_conflict",
                        request_id=response.request_id,
                        logical_request_id=logical_request_id,
                    )
                self.queue_duplicate_total += 1
                if response.body is None:
                    self.completed_response_unavailable_total += 1
                    raise CompletedResponseUnavailableError(response)
                self.completed_response_replayed_total += 1
                return None, response
            active = self.logical_in_flight.get(logical_request_id)
            if active is not None:
                _, request_id, queue_ticket = active
                active_record = self.active_request_records.get(request_id)
                if cpu_offload is not None and (active_record is None or cpu_offload != active_record.cpu_offload):
                    raise PermanentCapacityError("resume CPU offload permission differs from original request", 409,
                                                 "logical_request_conflict")
                if warm_admission is not None and (active_record is None or warm_admission != active_record.warm_admission):
                    raise PermanentCapacityError("resume warm certificate differs from original request", 409,
                                                 "logical_request_conflict")
                self.queue_duplicate_total += 1
                raise CapacityError(
                    "logical request is already admitted and in progress",
                    409,
                    "logical_request_in_progress",
                    True,
                    1,
                    request_id=request_id,
                    logical_request_id=logical_request_id,
                    queue_ticket=queue_ticket,
                )
            waiter = next(
                (
                    item for item in self.waiters
                    if item.logical_request_id == logical_request_id
                ),
                None,
            )
            if waiter is None:
                raise PermanentCapacityError(
                    "logical request is not retained by this broker",
                    409,
                    "logical_request_not_found",
                    logical_request_id=logical_request_id,
                )
            retained = waiter.retained_request
            if retained is None:
                raise PermanentCapacityError(
                    "logical request body was not retained",
                    409,
                    "retained_request_unavailable",
                    request_id=waiter.request_id,
                    logical_request_id=logical_request_id,
                )
            if cpu_offload is not None and cpu_offload != retained.cpu_offload:
                raise PermanentCapacityError("resume CPU offload permission differs from retained request", 409,
                                             "logical_request_conflict")
            if warm_admission is not None and warm_admission != retained.warm_admission:
                raise PermanentCapacityError("resume warm certificate differs from retained request", 409,
                                             "logical_request_conflict")
            if retained.method != method or retained.path != path:
                raise PermanentCapacityError(
                    "logical request ID was resumed on a different method or path",
                    409,
                    "logical_request_conflict",
                    request_id=waiter.request_id,
                    logical_request_id=logical_request_id,
                )
            return retained, None

    def record_completed_response(
        self,
        *,
        logical_request_id: str,
        request_fingerprint: str,
        request_id: str,
        request_method: str,
        request_path: str,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
        body: bytes | None,
        body_bytes: int,
        body_sha256: str,
        unavailable_reason: str | None = None,
        warm_admission: str | None = None,
        cpu_offload: bool = False,
    ) -> None:
        if not logical_request_id:
            return
        retained = body
        unavailable = unavailable_reason
        if (
            retained is not None
            and len(retained) > COMPLETED_RESPONSE_MAX_BODY_BYTES
        ):
            retained = None
            unavailable = "response_body_exceeds_per_entry_limit"
        if (
            retained is not None
            and len(retained) > COMPLETED_RESPONSE_MAX_TOTAL_BYTES
        ):
            retained = None
            unavailable = "response_body_exceeds_total_cache_limit"
        now = time.monotonic()
        with self.cv:
            tombstone = self.logical_tombstones.get(logical_request_id)
            if tombstone is not None and tombstone.request_id == request_id:
                self.logical_tombstones.pop(logical_request_id, None)
            self._prune_completed_responses_locked(now)
            self._remove_completed_response_locked(logical_request_id)
            while (
                len(self.completed_responses) >= COMPLETED_RESPONSE_MAX_ENTRIES
            ):
                if not self._evict_oldest_completed_response_locked():
                    break
            retained_bytes = len(retained) if retained is not None else 0
            while (
                retained is not None
                and self.completed_response_bytes + retained_bytes
                    > COMPLETED_RESPONSE_MAX_TOTAL_BYTES
            ):
                if not self._evict_oldest_completed_response_locked():
                    retained = None
                    retained_bytes = 0
                    unavailable = "response_body_exceeds_total_cache_limit"
                    break
            response = CompletedResponse(
                logical_request_id=logical_request_id,
                request_fingerprint=request_fingerprint,
                request_id=request_id,
                request_method=request_method,
                request_path=request_path,
                status=status,
                reason=reason,
                headers=tuple(headers),
                body=retained,
                body_bytes=body_bytes,
                body_sha256=body_sha256,
                unavailable_reason=(
                    unavailable if retained is None else None
                ),
                completed_at=now,
                expires_at=now + COMPLETED_RESPONSE_TTL,
                warm_admission=warm_admission,
                cpu_offload=cpu_offload,
            )
            self.completed_responses[logical_request_id] = response
            self.completed_response_bytes += retained_bytes
            self.completed_response_recorded_total += 1
            self.cv.notify_all()

    def _completed_response_summary_locked(self) -> dict[str, Any]:
        self._prune_completed_responses_locked()
        retained_entries = sum(
            response.body is not None
            for response in self.completed_responses.values()
        )
        return {
            "schema": "io.ollama-unify.completed-response-cache.v1",
            "ttl_seconds": COMPLETED_RESPONSE_TTL,
            "max_entries": COMPLETED_RESPONSE_MAX_ENTRIES,
            "max_body_bytes": COMPLETED_RESPONSE_MAX_BODY_BYTES,
            "max_total_bytes": COMPLETED_RESPONSE_MAX_TOTAL_BYTES,
            "entries": len(self.completed_responses),
            "retained_entries": retained_entries,
            "unavailable_entries": (
                len(self.completed_responses) - retained_entries
            ),
            "retained_bytes": self.completed_response_bytes,
            "recorded_total": self.completed_response_recorded_total,
            "replayed_total": self.completed_response_replayed_total,
            "unavailable_total": self.completed_response_unavailable_total,
            "conflict_total": self.completed_response_conflict_total,
            "expired_total": self.completed_response_expired_total,
            "evicted_total": self.completed_response_evicted_total,
        }

    def _lane_summaries_locked(self) -> list[dict[str, Any]]:
        self._prune_dead_lanes_locked()
        return [lane.public_summary() for lane in self.lanes.values()]

    def _queue_summary_locked(self, include_requests: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        self._prune_stale_waiters_locked(now)
        by_model: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        requests = []
        for position, waiter in enumerate(self.waiters, 1):
            by_model[waiter.model] = by_model.get(waiter.model, 0) + 1
            phase_counts[waiter.phase] = phase_counts.get(waiter.phase, 0) + 1
            if include_requests:
                requests.append(waiter.public_summary(position, now))
        result: dict[str, Any] = {
            "depth": len(self.waiters),
            "limit": POOL_MAX_QUEUE,
            "peak": self.queue_peak,
            "oldest_wait_ms": max(
                (max(0, int((now - item.enqueued_at) * 1000))
                 for item in self.waiters), default=0,
            ),
            "by_model": by_model,
            "phase_counts": phase_counts,
            "enqueued_total": self.queue_enqueued_total,
            "admitted_total": self.queue_admitted_total,
            "cancelled_total": self.queue_cancelled_total,
            "timed_out_total": self.queue_timed_out_total,
            "admission_timeout_total": self.queue_admission_timeout_total,
            "resumed_total": self.queue_resumed_total,
            "stale_total": self.queue_stale_total,
            "duplicate_total": self.queue_duplicate_total,
            "rejected_total": self.queue_rejected_total,
            "wait_ms_total": self.queue_wait_ms_total,
            "wait_ms_max": self.queue_wait_ms_max,
            "wait_ms_mean": (
                self.queue_wait_ms_total // self.queue_admitted_total
                if self.queue_admitted_total else 0
            ),
            "reconciling_model": self.reconciling_model,
            "retained_request_bytes": self.retained_request_bytes,
            "retained_request_max_total_bytes": RETAINED_REQUEST_MAX_TOTAL_BYTES,
        }
        if include_requests:
            result["requests"] = requests
        return result

    def _select_lane_locked(
        self,
        model: str,
        routable: bool,
        gpu_uuids: tuple[str, ...] | None = None,
        certified_lane_ids: set[str] | None = None,
        cpu_offload: bool = False,
    ) -> Lane | None:
        if routable and model:
            require_inference_enabled()
        self._prune_dead_lanes_locked()
        base = self.lanes["base"]
        blocked_gpus = self._ollama_blocked_gpus_locked()
        if not routable:
            return base if base.in_flight < base.parallel else None
        gpu_uuids = self._policy_constraint_locked(model, gpu_uuids)
        matching = [lane for lane in self.lanes.values()
                    if lane.kind == "managed" and lane.model == model
                    and (certified_lane_ids is None or lane.lane_id in certified_lane_ids)
                    and lane.context_profile_matches()
                    and lane.can_serve(gpu_uuids, cpu_offload)
                    and not set(lane.scope).intersection(blocked_gpus)
                    and not lane.loading
                    and not lane.retiring
                    and lane.in_flight < lane.parallel]
        managed_model_exists = any(
            lane.kind == "managed" and lane.model == model
            for lane in self.lanes.values()
        )
        if matching:
            preference_rank = {
                gpu_uuid: index for index, gpu_uuid in enumerate(gpu_uuids or ())
            }
            return min(matching, key=lambda lane: (
                lane.cpu_offload,
                lane.in_flight,
                preference_rank.get(lane.gpu_uuid, len(preference_rank)),
                lane.created_at,
            ))
        if managed_model_exists:
            return None
        if POOL_ENABLED:
            return None
        if blocked_gpus:
            # A legacy system backend has no exact UUID scope. It must not
            # bypass scoped quarantine or an exclusive peer lease.
            return None
        if any(lane.kind == "managed" for lane in self.lanes.values()):
            return None
        return base if base.in_flight < base.parallel else None

    def _warm_runtime_identity_locked(self, lane: Lane) -> dict[str, Any]:
        if lane.cpu_offload:
            raise WarmPreflightRequired(causes=("native_gpu_residency_mismatch",))
        if lane.kind != "managed" or lane.retiring or lane.loading:
            raise WarmPreflightRequired(causes=("lane_unavailable",))
        if not lane.context_profile_matches():
            raise WarmPreflightRequired(causes=("context_profile_mismatch",))
        if not process_group_alive(lane.process):
            raise WarmPreflightRequired(causes=("runtime_identity_unavailable",))
        try:
            before = process_group_identity(lane.process)
            try:
                rows = backend_json_at(lane.host, lane.port, "GET", "/api/ps", timeout=2.0).get("models")
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                raise WarmPreflightRequired(causes=("native_ps_unavailable",)) from exc
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                raise WarmPreflightRequired(causes=("native_ps_inventory_mismatch",))
            row = rows[0]
            native = {
                "name": canonical_model_tag(str(row.get("name") or row.get("model") or "")),
                "digest": row.get("digest"), "context_length": row.get("context_length"),
                "size": row.get("size"), "size_vram": row.get("size_vram"),
            }
            if lane.context_profile is None:
                # Embedding runners have no completion/KV context profile.
                # Certify their observed native context only after verifying
                # the current installed artifact and embedding-only capability.
                expected_digest = self._warm_embedding_digest(lane.model)
                expected_context = native["context_length"]
            else:
                expected_digest = lane.context_profile["model_digest"]
                expected_context = lane.context_profile["context_length"]
            if native["name"] != lane.model:
                raise WarmPreflightRequired(causes=("native_model_mismatch",))
            if native["digest"] != expected_digest:
                raise WarmPreflightRequired(causes=("native_digest_mismatch",))
            if (type(native["context_length"]) is not int
                    or not 1 <= native["context_length"] <= 2 ** 31 - 1
                    or native["context_length"] != expected_context
                    or (HARD_MAX_CONTEXT > 0 and native["context_length"] > HARD_MAX_CONTEXT)):
                raise WarmPreflightRequired(causes=("native_context_mismatch",))
            if (type(native["size"]) is not int or native["size"] <= 0
                    or type(native["size_vram"]) is not int
                    or native["size_vram"] != native["size"]):
                raise WarmPreflightRequired(causes=("native_gpu_residency_mismatch",))
            if len(lane.scope) > 1:
                try:
                    observed = process_gpu_usage(lane.process)
                except CapacityError as exc:
                    raise WarmPreflightRequired(causes=("native_gpu_placement_mismatch",)) from exc
                lane.observed_vram_mib_by_gpu = observed
                if set(observed) != set(lane.scope) or any(used <= 0 for used in observed.values()):
                    raise WarmPreflightRequired(causes=("native_gpu_placement_mismatch",))
            after = process_group_identity(lane.process)
            if before != after:
                raise WarmPreflightRequired(causes=("runtime_identity_changed",))
            return {
                "server_process": next(item for item in before if item["pid"] == lane.process.pid),
                "runtime_processes": before, "native_ps": native,
            }
        except WarmPreflightRequired:
            raise
        except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise WarmPreflightRequired(causes=("runtime_identity_unavailable",)) from exc

    @staticmethod
    def _warm_embedding_digest(model: str) -> str:
        def installed() -> dict[str, Any]:
            rows = backend_json("GET", "/api/tags", timeout=2.0).get("models", [])
            matching = [item for item in rows if isinstance(item, dict)
                        and canonical_model_tag(str(item.get("name") or item.get("model") or "")) == model]
            if len(matching) != 1:
                raise ValueError("exact embedding artifact is not installed")
            return matching[0]

        before = installed()
        digest = before.get("digest")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("embedding artifact digest is not verified")
        capabilities = before.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            capabilities = backend_json("POST", "/api/show", {"model": model}, timeout=2.0).get("capabilities")
        after = installed()
        if (after.get("digest") != digest or not isinstance(capabilities, list)
                or "embedding" not in capabilities or "completion" in capabilities
                or (isinstance(after.get("capabilities"), list) and after["capabilities"]
                    and set(after["capabilities"]) != set(capabilities))):
            raise ValueError("current artifact is not embedding-only")
        return digest

    def warm_admission_proof(self, model: str, parallel: int,
                             gpu_uuids: tuple[str, ...] | None,
                             placement: str = "auto", *, client_key: str = "") -> dict[str, Any]:
        """Inspect existing warm capacity only. Never load, grow, or refresh idle time."""
        model = canonical_model_tag(model)
        if not POOL_ENABLED or not model or gpu_uuids is None or parallel < 1:
            raise PermanentCapacityError("warm proof requires a model, positive parallel and explicit GPU scope",
                                         400, "invalid_warm_admission")
        with self.cv:
            blocked = self._ollama_blocked_gpus_locked()
            allowed = self._capacity_scope_locked(model, gpu_uuids, placement, client_key)
            if placement == "exclusive_group" and parallel > POOL_INSTANCE_PARALLEL:
                raise PermanentCapacityError("one exact exclusive group supports only its configured parallel slots",
                                             422, "parallel_exceeds_capacity")
            lanes = [lane for lane in self.lanes.values()
                     if lane.kind == "managed" and lane.model == model
                     and lane.allows(allowed)
                     and (placement != "exclusive_group" or lane.scope == gpu_uuids)
                     and not set(lane.scope).intersection(blocked)
                     and not lane.retiring and not lane.loading]
            if self.draining or not lanes or sum(lane.parallel for lane in lanes) < parallel:
                raise WarmPreflightRequired(phase="proof_issuance",
                    causes=("broker_draining" if self.draining else "lane_unavailable",))
            descriptors = []
            for lane in lanes:
                try:
                    runtime = self._warm_runtime_identity_locked(lane)
                except WarmPreflightRequired as exc:
                    raise WarmPreflightRequired(phase="proof_issuance",
                        causes=exc.warm_preflight_causes) from exc
                if runtime != lane.warm_runtime_identity:
                    lane.warm_generation_id = secrets.token_hex(32)
                    lane.warm_runtime_identity = runtime
                descriptors.append({
                    "id": lane.lane_id, "generation_id": lane.warm_generation_id,
                    "model": model, "model_digest": runtime["native_ps"]["digest"],
                    "context_length": runtime["native_ps"]["context_length"],
                    "gpu_uuids": list(lane.scope), **runtime,
                })
            identities = {(item["model_digest"], item["context_length"]) for item in descriptors}
            if len(identities) != 1:
                causes = tuple(cause for field, cause in (
                    (0, "model_digest_mismatch"), (1, "context_mismatch"))
                    if len({identity[field] for identity in identities}) > 1)
                raise WarmPreflightRequired(phase="proof_issuance",
                    causes=causes)
            digest, context = next(iter(identities))
            proof = {"schema": WARM_ADMISSION_SCHEMA, "broker_instance_id": self.instance_id,
                     "model": model, "model_digest": digest, "context_length": context,
                     "gpu_uuids": list(gpu_uuids),
                     "lanes": [{"id": item["id"], "generation_id": item["generation_id"]}
                               for item in descriptors]}
            return {
                "ok": True, "schema": "io.ollama-unify.gpu-negotiator.capacity.v1",
                "requested_model": model, "canonical_model": model,
                "requested_parallel": parallel, "requested_gpu_uuids": list(gpu_uuids),
                "requested_placement": placement,
                "admitted_parallel": sum(lane.parallel for lane in lanes),
                "public_ollama_api": f"http://127.0.0.1:{LISTEN_PORT}",
                "lanes": [lane.public_summary() for lane in lanes],
                "warm_admission": proof, "warm_admission_lanes": descriptors,
            }

    def _validate_warm_admission_locked(self, certificate: str, model: str,
                                        gpu_uuids: tuple[str, ...] | None, *,
                                        phase: str = "validation") -> set[str]:
        proof = json.loads(certificate)
        for failed, cause in (
            (proof["broker_instance_id"] != self.instance_id, "broker_instance_mismatch"),
            (proof["model"] != model, "model_mismatch"),
            (gpu_uuids is None or proof["gpu_uuids"] != list(gpu_uuids), "gpu_scope_mismatch"),
            (self.draining, "broker_draining"),
        ):
            if failed:
                raise WarmPreflightRequired(phase=phase, causes=(cause,))
        allowed = self._policy_constraint_locked(model, gpu_uuids)
        blocked = self._ollama_blocked_gpus_locked()
        valid = set()
        causes = []
        for identity in proof["lanes"]:
            lane = self.lanes.get(identity["id"])
            # A retired member may disappear normally. Only current certified
            # generations can be selected; at least one must still be ready.
            if lane is None:
                causes.append("lane_missing")
                continue
            failed = next((cause for condition, cause in (
                (lane.warm_generation_id != identity["generation_id"], "generation_mismatch"),
                (lane.model != model, "lane_model_mismatch"),
                (not lane.allows(allowed), "gpu_policy_mismatch"),
                (bool(set(lane.scope).intersection(blocked)), "gpu_blocked"),
            ) if condition), None)
            if failed:
                causes.append(failed)
                continue
            expected = lane.context_profile or ((lane.warm_runtime_identity or {}).get("native_ps"))
            if not expected or expected.get("model_digest", expected.get("digest")) != proof["model_digest"]:
                causes.append("model_digest_mismatch")
                continue
            if expected.get("context_length") != proof["context_length"]:
                causes.append("context_mismatch")
                continue
            try:
                runtime = self._warm_runtime_identity_locked(lane)
            except WarmPreflightRequired as exc:
                causes.extend(exc.warm_preflight_causes)
                continue
            if runtime == lane.warm_runtime_identity:
                valid.add(lane.lane_id)
            else:
                causes.append("runtime_identity_changed")
        if not valid:
            raise WarmPreflightRequired(phase=phase, causes=tuple(causes))
        return valid

    @staticmethod
    def validate_warm_body(certificate: str, model: str, path: str, body: bytes,
                           *, prepared: bool = False) -> None:
        proof = json.loads(certificate)
        if model != proof["model"] or path not in INFERENCE_PATHS:
            raise PermanentCapacityError("warm certificate does not bind the request model/path", 400,
                                         "invalid_warm_admission")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise PermanentCapacityError("warm inference requires a JSON object", 400, "invalid_warm_admission")
        if path in NATIVE_MODEL_PATHS:
            options = payload.get("options", {})
            # Only prediction options are variable in a certified request.
            # Unknown/new loader options fail closed until explicitly reviewed.
            prediction_options = {
                "num_keep", "seed", "num_predict", "top_k", "top_p", "min_p", "typical_p",
                "repeat_last_n", "temperature", "repeat_penalty", "presence_penalty",
                "frequency_penalty", "stop", "mirostat", "mirostat_tau", "mirostat_eta",
                "penalize_newline", "tfs_z", "num_ctx", "num_gpu",
            }
            if (not isinstance(options, dict)
                    or set(options).difference(prediction_options)
                    or ("num_ctx" in options and (type(options["num_ctx"]) is not int
                        or options["num_ctx"] != proof["context_length"]))
                    or ("num_gpu" in options and (type(options["num_gpu"]) is not int or options["num_gpu"] != -1))
                    or payload.get("runner") not in (None, "")
                    or (payload.get("keep_alive") is not None
                        and not (prepared and type(payload["keep_alive"]) is int
                                 and payload["keep_alive"] == -1))):
                raise PermanentCapacityError("warm inference cannot change the certified runtime allocation or unload it",
                                             400, "invalid_warm_admission")

    def _model_profile(self, model: str) -> tuple[int, set[str]]:
        model = canonical_model_tag(model)
        tags = backend_json("GET", "/api/tags", timeout=10.0).get("models", [])
        if not isinstance(tags, list):
            tags = []
        wanted = model.removesuffix(":latest")
        match = None
        for item in tags:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            if canonical_model_tag(name) == model or name.removesuffix(":latest") == wanted:
                match = item
                break
        if match is None:
            raise PermanentCapacityError(
                f"model {model!r} is not installed on the managed Ollama store",
                404,
                "model_not_installed",
            )
        try:
            size_bytes = int(match.get("size") or 0)
        except (TypeError, ValueError):
            size_bytes = 0
        if size_bytes <= 0:
            raise PermanentCapacityError(
                f"model {model!r} has no local size metadata",
                422,
                "model_metadata_invalid",
            )
        model_mib = math.ceil(model_gpu_weight_bytes(model, match, size_bytes) / (1024 * 1024))
        profile = resolve_model_context_profile(model, match)
        extra_context_mib = 0
        if profile is not None:
            extra_context_mib = profile["extra_vram_mib"] * POOL_INSTANCE_PARALLEL
        capabilities = match.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            # Standard Ollama tags may omit capabilities; embedding-only
            # models still need the correct native warm-up endpoint.
            capabilities = backend_json("POST", "/api/show", {"model": model},
                                        timeout=10.0).get("capabilities", [])
        if not isinstance(capabilities, list):
            capabilities = []
        return (
            math.ceil(model_mib * POOL_MODEL_OVERHEAD_PERCENT / 100)
            + POOL_VRAM_RESERVE_MIB + extra_context_mib,
            {str(capability).lower() for capability in capabilities},
        )

    def _placement_devices(
        self, blocked: set[str], reservation_owner: str = "", *,
        reclaiming: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return capacity after honoring every live lane's promised VRAM.

        Physical free VRAM can rise while a live lane retains its reservation.
        Treat that promise as committed until the complete process group exits
        so another lane cannot consume the same capacity. With `reclaiming`,
        return an optimistic post-reclamation bound, never an admission grant:
        only those lanes' promises and physical footprint may be released.
        """
        with self.cv:
            peer_reserved = self._peer_reserved_gpus_locked(ignore_lane_ids=reclaiming)
            reclaimable_gpus = {gpu for lane in self.lanes.values()
                                if lane.lane_id in (reclaiming or ())
                                for gpu in lane.protected_scope}
        devices = [
            device for device in gpu_snapshot()
            if device.get("uuid") in SELECTED_GPUS
            and device.get("uuid") not in blocked
            and device.get("uuid") not in peer_reserved
        ]
        foreign_by_gpu: dict[str, int] = {}
        foreign_usage = foreign_gpu_usage(strict=True)
        for key, used_mib in foreign_usage.items():
            _pid, separator, gpu_uuid = key.rpartition("@")
            if separator:
                foreign_by_gpu[gpu_uuid] = (
                    foreign_by_gpu.get(gpu_uuid, 0) + int(used_mib)
                )
        with self.cv:
            self._prune_dead_lanes_locked()
            external_gaps = self._external_unrealized_locked(foreign_usage)
            reserved_by_gpu: dict[str, int] = {}
            for lane in self.lanes.values():
                if (lane.kind != "managed" or not lane.scope
                        or lane.lane_id in (reclaiming or ())):
                    continue
                for gpu in lane.scope:
                    reserved_by_gpu[gpu] = reserved_by_gpu.get(gpu, 0) + max(
                        0, int(lane.reserved_mib_by_gpu.get(gpu, lane.reserved_mib)))
            for operation in self.evacuations.values():
                if operation["id"] == reservation_owner:
                    continue
                attempt = operation.get("destination_attempt", {})
                if attempt.get("lane_id") in self.lanes:
                    continue  # atomic transfer to the published lane promise
                for gpu, promised in operation.get("destination_reservation", {}).items():
                    reserved_by_gpu[gpu] = reserved_by_gpu.get(gpu, 0) + promised
        available = []
        for device in devices:
            gpu_uuid = str(device.get("uuid") or "")
            physical_free = max(0, int(device.get("free_mib") or 0))
            if gpu_uuid in reclaimable_gpus:
                # Its exact physical release is unknown until exit. An upper
                # bound avoids rejecting feasible reclaim on stale telemetry.
                physical_free = max(0, int(device.get("total_mib") or 0))
            promised_free = max(
                0,
                min(int(device.get("total_mib") or 0),
                    max(0, int(device.get("free_mib") or 0))
                    + int(device.get("used_mib", int(device.get("total_mib") or 0)
                              - int(device.get("free_mib") or 0))))
                - foreign_by_gpu.get(gpu_uuid, 0)
                - external_gaps.get(gpu_uuid, 0)
                - reserved_by_gpu.get(gpu_uuid, 0),
            )
            available.append({
                **device,
                "physical_free_mib": physical_free,
                "reserved_mib": reserved_by_gpu.get(gpu_uuid, 0),
                "free_mib": min(physical_free, promised_free),
            })
        # Ordinary warm inference must not pay additional ownership queries
        # when no external reservation exists.
        if not reclaiming and self.leases:
            lease_capacity = {d["uuid"]: d["free_mib"] for d in self._lease_capacity_devices(
                devices, reservation_owner=reservation_owner)}
            for device in available:
                device["free_mib"] = min(device["free_mib"], lease_capacity[device["uuid"]])
        return available

    def _require_reclamation_feasible(
        self, gpu_uuids: tuple[str, ...] | None, required_mib: int,
        missing: int, replaceable: list[Lane],
    ) -> None:
        # Called under cv, before marking or stopping any victim. Even freeing
        # every eligible lane must allow all requested single-GPU placements.
        devices = self._placement_devices(
            self._ollama_blocked_gpus_locked(),
            reclaiming={lane.lane_id for lane in replaceable},
        )
        capacity = sum(int(device.get("free_mib") or 0) // required_mib
                       for device in devices
                       if gpu_uuids is None or device.get("uuid") in gpu_uuids)
        if capacity < missing:
            raise CapacityError(
                f"reclaiming all eligible idle lanes cannot fit {missing} "
                f"new lane(s) requiring {required_mib} MiB within the GPU scope "
                f"and live reservations (at most {capacity} fit)",
                reason_code="lane_capacity_wait",
            )

    @staticmethod
    def _warm_request(model: str, capabilities: set[str], request_path: str
                      ) -> tuple[str, dict[str, Any]]:
        # The broker owns eviction timing; native idle expiry bypasses guards.
        keep_alive = -1
        if request_path in EMBEDDING_PATHS or (
            not request_path and "embedding" in capabilities
            and "completion" not in capabilities
        ):
            return "/api/embed", {
                # Empty input loads the runner without computing an embedding.
                "model": model, "input": [], "keep_alive": keep_alive,
            }
        if request_path == "/api/rerank" or (
            not request_path and "reranking" in capabilities
            and "completion" not in capabilities
        ):
            return "/api/rerank", {
                "model": model,
                "query": "warmup",
                "documents": ["warmup"],
                "keep_alive": keep_alive,
            }
        options = {"num_predict": 0}
        profile = effective_model_context_profile(model)
        warm_context = profile["context_length"] if profile else MAX_CONTEXT
        if warm_context > 0:
            options["num_ctx"] = warm_context
        return "/api/generate", {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        }

    def _available_port_locked(self) -> int:
        used = {lane.port for lane in self.lanes.values()}
        stop = POOL_PORT_START + max(32, POOL_MAX_SERVERS + 4)
        for port in range(POOL_PORT_START, stop):
            if port in used or port in (LISTEN_PORT, BACKEND_PORT):
                continue
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                candidate.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
            finally:
                candidate.close()
        raise CapacityError(
            "no loopback port is available for another managed Ollama lane",
            reason_code="backend_start_failed",
        )

    @staticmethod
    def _terminate_process(process: Any, *, allow_force: bool = True) -> bool:
        if process is None:
            return True
        process_group = process.pid

        try:
            os.killpg(process_group, signal.SIGTERM)
        except OSError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if not allow_force:
                return False
            try:
                os.killpg(process_group, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        # The lane leader can exit before runner grandchildren in the same
        # process group. Reservation release requires the complete group to be
        # gone, not merely a reaped parent Popen.
        if process_group_alive(process):
            if not allow_force:
                return False
            try:
                os.killpg(process_group, signal.SIGKILL)
            except OSError:
                pass
            deadline = time.monotonic() + 5.0
            while (
                process_group_alive(process)
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
        return process.poll() is not None and not process_group_alive(process)

    def _spawn_lane(self, model: str, gpu_uuid: str | tuple[str, ...], required_mib: int,
                    capabilities: set[str], request_path: str,
                    triggered_by: dict[str, str] | None = None,
                    reserved_mib_by_gpu: dict[str, int] | None = None,
                    expected_profile: Any = CONTEXT_PROFILE_UNSET,
                    reservation_operation: str = "", *,
                    cpu_offload: bool = False, host_reserved_mib: int = 0) -> Lane:
        require_inference_enabled()
        scope = (gpu_uuid,) if isinstance(gpu_uuid, str) else tuple(gpu_uuid)
        if not scope or len(set(scope)) != len(scope) or any(gpu not in SELECTED_GPUS for gpu in scope):
            raise PermanentCapacityError("invalid managed GPU scope", 422, "invalid_gpu_scope")
        require_gpu_health(refresh=True)
        self._require_safe_gpu_transition(scope)
        model = canonical_model_tag(model)
        if not os.access(OLLAMA_BINARY, os.X_OK):
            raise PermanentCapacityError(
                f"managed Ollama binary is not executable: {OLLAMA_BINARY}",
                500,
                "backend_start_failed",
            )
        with self.cv:
            port = self._available_port_locked()
            lane_id = f"lane-{self.next_lane_id}"
            self.next_lane_id += 1
        env = os.environ.copy()
        if cpu_offload:
            # Pageable CPU model layers only. Presence enables these GGML
            # flags even when the inherited value is the string "0".
            for key in ("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "GGML_CUDA_REGISTER_HOST",
                        "LLAMA_ARG_N_GPU_LAYERS", "LLAMA_ARG_FIT_TARGET"):
                env.pop(key, None)
        if len(scope) > 1:
            # These GGML flags are enabled by their presence, including a
            # value of "0". Keep layer splitting on the host-staged path;
            # never inherit peer access or unified-memory spill into a group.
            env.pop("GGML_CUDA_P2P", None)
            env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
        env.update({
            "HOME": OLLAMA_CHILD_HOME,
            "OLLAMA_HOST": f"127.0.0.1:{port}",
            "CUDA_VISIBLE_DEVICES": ",".join(scope),
            "HIP_VISIBLE_DEVICES": "-1",
            "ROCR_VISIBLE_DEVICES": "-1",
            "GPU_DEVICE_ORDINAL": "-1",
            "GGML_VK_VISIBLE_DEVICES": "-1",
            "OLLAMA_VULKAN": "0",
            "OLLAMA_IGPU_ENABLE": "0",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": str(POOL_INSTANCE_PARALLEL),
            "OLLAMA_SCHED_SPREAD": "1" if len(scope) > 1 else "0",
            "OLLAMA_KEEP_ALIVE": "-1",
            "OLLAMA_MAX_QUEUE": "64",
            "OLLAMA_GPU_OVERHEAD": str(POOL_VRAM_RESERVE_MIB * 1024 * 1024),
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
            "GGML_CUDA_NO_PINNED": "1",
            "LLAMA_ARG_FIT": "on",
        })
        profile = effective_model_context_profile(model)
        if expected_profile is not CONTEXT_PROFILE_UNSET and profile != expected_profile:
            raise CapacityError("model context identity changed after memory admission", 503,
                                "model_context_identity_changed")
        profile = dict(profile) if profile else None
        lane_context = profile["context_length"] if profile else MAX_CONTEXT
        if lane_context > 0:
            env["OLLAMA_CONTEXT_LENGTH"] = str(lane_context)
        if OLLAMA_MODELS:
            env["OLLAMA_MODELS"] = OLLAMA_MODELS
        LOG.info("managed lane starting id=%s gpu=%s model=%s port=%s",
                 lane_id, gpu_uuid, model, port)
        capacity_deadline = getattr(CAPACITY_IO_DEADLINE, "deadline", None)
        if capacity_deadline is not None and time.monotonic() >= capacity_deadline:
            raise BackgroundCapacityDeferred("speculative cold capacity time budget expired before startup")
        startup_read = startup_write = None
        command = [OLLAMA_BINARY, "serve"]
        if reservation_operation:
            # This wrapper cannot execute CUDA until its exact process
            # generation is fsynced. Only the parent owns the CLOEXEC write
            # end: death before publication sends EOF, never an orphan load.
            startup_read, startup_write = os.pipe2(os.O_CLOEXEC)
            command = [sys.executable, "-c", CACHE_STARTUP_WRAPPER,
                       str(startup_read), OLLAMA_BINARY, "serve"]
        try:
            process = subprocess.Popen(
                command, env=env, stdin=subprocess.DEVNULL,
                pass_fds=() if startup_read is None else (startup_read,),
                # Inherit systemd's bounded journal instead of discarding
                # CUDA/GSP failures. The wrapper exec preserves PID/start/PGRP.
                start_new_session=True,
            )
        except Exception:
            if startup_write is not None:
                os.close(startup_write)
            raise
        finally:
            if startup_read is not None:
                os.close(startup_read)
        LOG.info("managed lane spawned id=%s gpu=%s model=%s pid=%s",
                 lane_id, gpu_uuid, model, process.pid)
        now = time.time()
        lane = Lane(lane_id, "managed", "127.0.0.1", port, scope[0], model,
                    POOL_INSTANCE_PARALLEL, required_mib, now, now, process,
                    gpu_uuids=scope,
                    reserved_mib_by_gpu=dict(reserved_mib_by_gpu or {scope[0]: required_mib}),
                    loading=True, context_profile=profile,
                    openai_context_compatible=MODEL_CONTEXT_RESOLUTIONS.get(model, {}).get(
                        "openai_context_compatible", True),
                    cpu_offload=cpu_offload, host_reserved_mib=host_reserved_mib)
        lane.triggered_by = triggered_by
        with self.cv:
            # Publish the entire reservation before warm-up can start peer
            # traffic. Loading lanes are never eligible for inference.
            self.lanes[lane_id] = lane
            if reservation_operation:
                try:
                    operation = self.evacuations[reservation_operation]
                    attempt = {"lane_id": lane_id, "gpu_uuids": list(scope),
                        "process_group_id": process.pid, "runtime_processes": process_group_identity(process),
                        "startup_protocol": "registered_before_exec.v1"}
                    operation["destination_attempt"] = attempt
                    operation.setdefault("spawn_attempts", []).append(attempt)
                    self._persist_cache_state_locked()
                    os.write(startup_write, b"\x01")
                except Exception:
                    lane.retiring = True
                    lane.loading = False
                    self._stop_lanes([lane], "cache journal publication failed")
                    raise
                finally:
                    os.close(startup_write)
            self.cv.notify_all()
        deadline = min(time.monotonic() + POOL_READY_TIMEOUT,
                       capacity_deadline if capacity_deadline is not None else float("inf"))
        try:
            server_identity = managed_server_identity(process)
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise CapacityError(
                        f"managed Ollama lane exited with status {process.returncode}"
                    )
                try:
                    backend_json_at("127.0.0.1", port, "GET", "/api/version", timeout=0.5)
                    break
                except (OSError, RuntimeError, ValueError):
                    time.sleep(0.1)
            else:
                raise CapacityError(
                    f"managed Ollama lane did not become ready within {POOL_READY_TIMEOUT:.0f}s"
                )
            with self.cv:
                for waiter in self.waiters:
                    if waiter.model == model:
                        waiter.phase = "loading-model"
                        break
                self.cv.notify_all()
            warm_path, warm_payload = self._warm_request(
                model, capabilities, request_path
            )
            if cpu_offload:
                # Match inference's automatic fitting even when an installed
                # Modelfile carries a historical fixed-layer override.
                warm_payload.setdefault("options", {})["num_gpu"] = -1
            if profile:
                warm_payload["options"]["num_ctx"] = profile["context_length"]
            require_gpu_health(refresh=True)
            self._require_safe_gpu_transition(scope, lane_id)
            LOG.info("managed lane loading id=%s gpu=%s model=%s pid=%s",
                     lane_id, gpu_uuid, model, process.pid)
            assert_managed_server_scope(process, scope, server_identity)
            backend_json_at(
                "127.0.0.1", port, "POST", warm_path, warm_payload,
                timeout=POOL_LOAD_TIMEOUT,
            )
            resident = backend_json_at(
                "127.0.0.1", port, "GET", "/api/ps", timeout=3.0
            ).get("models", [])
            resident_model = next((
                item
                for item in resident if isinstance(resident, list)
                and isinstance(item, dict)
                and canonical_model_tag(str(
                    item.get("name") or item.get("model") or ""
                )) == model
            ), None)
            if resident_model is None:
                raise CapacityError(
                    f"managed Ollama lane did not make model {model!r} resident"
                )
            size_vram = resident_model.get("size_vram")
            if (
                isinstance(size_vram, bool)
                or not isinstance(size_vram, (int, float))
                or size_vram <= 0
            ):
                raise PermanentCapacityError(
                    f"managed Ollama lane for model {model!r} loaded with "
                    f"size_vram={size_vram!r} under GPU {gpu_uuid}; CUDA did "
                    "not make the model GPU-resident",
                    503,
                    "gpu_runtime_unavailable",
                )
            actual_context = verified_model_context(model, resident, profile)
            if profile and math.ceil(size_vram / (1024 * 1024)) > required_mib:
                raise PermanentCapacityError(
                    f"resolved context allocation exceeds its reserved budget: observed={math.ceil(size_vram / (1024 * 1024))} MiB reserved={required_mib} MiB",
                    503, "model_context_memory_exceeded")
            if len(scope) > 1 or cpu_offload:
                total_size = resident_model.get("size")
                if (isinstance(total_size, bool) or not isinstance(total_size, (int, float))
                        or total_size <= 0 or (not cpu_offload and size_vram < total_size)):
                    raise PermanentCapacityError(
                        "split model did not become fully GPU-resident", 503,
                        "gpu_runtime_unavailable")
                if cpu_offload and math.ceil(max(0, total_size - size_vram) / (1024 * 1024)) > host_reserved_mib:
                    raise PermanentCapacityError(
                        "CPU model allocation exceeds its reserved host budget", 503,
                        "model_context_memory_exceeded")
                observed = process_gpu_usage(process)
                lane.observed_vram_mib_by_gpu = observed
                if set(observed) != set(scope) or any(used <= 0 for used in observed.values()):
                    raise CapacityError(
                        f"managed GPU placement differs from reserved group: expected={list(scope)} observed={observed}",
                        503, "gpu_placement_unverified")
                if cpu_offload and any(used > lane.reserved_mib_by_gpu[gpu]
                                       for gpu, used in observed.items()):
                    raise PermanentCapacityError("GPU allocation exceeds its reservation", 503,
                                                 "model_context_memory_exceeded")
                require_gpu_health(refresh=True)
                self._require_safe_gpu_transition(scope, lane_id)
            with self.cv:
                allowed = self._policy_constraint_locked(model, scope)
                if (lane.retiring or process.poll() is not None
                        or not lane.allows(allowed) or not lane.context_profile_matches()):
                    raise CapacityError("managed lane retired during warm-up",
                                        reason_code="lane_capacity_wait")
                lane.resolved_context_length = actual_context
                lane.loading = False
                self.cv.notify_all()
        except Exception as exc:
            LOG.error("managed lane load failed id=%s gpu=%s model=%s pid=%s: %s",
                      lane_id, gpu_uuid, model, process.pid, exc)
            lane.retiring = True
            lane.loading = False
            self._stop_lanes([lane], "warm-up failed")
            if (isinstance(exc, BackendHTTPError)
                    and 400 <= exc.status < 500
                    and exc.status not in (408, 409, 425, 429)):
                raise PermanentCapacityError(
                    str(exc), exc.status, "backend_rejected_model"
                ) from exc
            raise
        LOG.info(
            "managed Ollama lane warm and ready id=%s gpu=%s model=%s "
            "reserved_mib=%s triggered_by=%s",
            lane_id, gpu_uuid, model, required_mib,
            (triggered_by or {}).get("label", "unknown"),
        )
        return lane

    def _stop_lanes(self, lanes: list[Lane], reason: str) -> list[Lane]:
        # Hold the same transition lock through the final safety check and
        # teardown, so lease acquisition cannot grant a scope in between.
        with self.transition:
            return self._stop_lanes_under_transition(lanes, reason)

    def _stop_lanes_under_transition(self, lanes: list[Lane], reason: str) -> list[Lane]:
        failed: list[Lane] = []
        for lane in lanes:
            if not process_group_alive(lane.process):
                # A repeated stop must not address an old private port that
                # may already belong to a replacement backend.
                with self.cv:
                    if self.lanes.get(lane.lane_id) is lane:
                        self.lanes.pop(lane.lane_id, None)
                    self.cv.notify_all()
                continue
            try:
                self._require_safe_gpu_transition(lane.protected_scope, lane.lane_id)
            except CapacityError:
                # Keep both process and reservation: CUDA teardown itself is
                # unsafe while an unknown owner could be doing peer traffic.
                lane.retiring = True
                with self.cv:
                    self.lanes[lane.lane_id] = lane
                    self.cv.notify_all()
                failed.append(lane)
                LOG.warning("managed lane stop deferred id=%s gpu=%s model=%s reason=%s",
                            lane.lane_id, lane.gpu_uuid, lane.model, reason)
                continue
            LOG.info("managed lane unloading id=%s gpu=%s model=%s pid=%s reason=%s",
                     lane.lane_id, lane.gpu_uuid, lane.model,
                     getattr(lane.process, "pid", None), reason)
            try:
                unload_models_at(lane.host, lane.port, min(UNLOAD_TIMEOUT, 15.0),
                                 require_available=False)
            except Exception as exc:
                LOG.warning("managed lane unload failed id=%s: %s", lane.lane_id, exc)
            if self._terminate_process(lane.process):
                with self.cv:
                    if self.lanes.get(lane.lane_id) is lane:
                        self.lanes.pop(lane.lane_id, None)
                    self.cv.notify_all()
                LOG.info(
                    "managed Ollama lane stopped id=%s gpu=%s model=%s exit=%s reason=%s",
                    lane.lane_id, lane.gpu_uuid, lane.model,
                    getattr(lane.process, "returncode", None), reason,
                )
                continue
            lane.retiring = True
            lane.last_used = 0
            with self.cv:
                self.lanes[lane.lane_id] = lane
                self.cv.notify_all()
            failed.append(lane)
            LOG.error(
                "managed Ollama lane process group survived stop; retaining "
                "its reservation id=%s reason=%s",
                lane.lane_id,
                reason,
            )
        return failed

    def stop_pool_lanes(
        self, reason: str, gpu_uuids: set[str] | None = None,
    ) -> list[str]:
        with self.cv:
            lanes = [
                lane for lane in self.lanes.values()
                if lane.kind == "managed"
                and (gpu_uuids is None or set(lane.protected_scope).intersection(gpu_uuids))
            ]
            for lane in lanes:
                lane.retiring = True
            self.cv.notify_all()
        failed = self._stop_lanes(lanes, reason)
        if failed:
            raise RuntimeError(
                "managed lane process group did not stop: "
                + ", ".join(lane.lane_id for lane in failed)
            )
        return [lane.lane_id for lane in lanes]

    def _require_cpu_host_budget(self, requested_mib: int) -> None:
        """Conservatively reserve the whole estimate, not a guessed layer split.

        Existing promises are subtracted even when already resident, so delayed
        paging cannot double spend host capacity. This deliberately favors safe
        deferral over optimistic admission near memory/cgroup limits.
        """
        snapshot = host_memory_snapshot()
        available = snapshot.get("memavailable_mib")
        cgroup = snapshot.get("managed_cgroup_headroom_mib", "unknown")
        if type(available) is not int or available <= 0 or cgroup == "unknown":
            raise CapacityError("CPU offload requires verified host and managed cgroup memory budgets",
                                reason_code="host_memory_unavailable")
        if cgroup != "max":
            if type(cgroup) is not int or cgroup < 0:
                raise CapacityError("managed cgroup memory budget is unavailable",
                                    reason_code="host_memory_unavailable")
            available = min(available, cgroup)
        with self.cv:
            promised = sum(max(POOL_HOST_RESERVE_MIB, lane.host_reserved_mib)
                           for lane in self.lanes.values() if lane.kind == "managed")
        if requested_mib + promised > available:
            raise CapacityError(
                f"CPU offload reserves {requested_mib} MiB host memory plus {promised} MiB "
                f"existing promises, but only {available} MiB verified headroom is available",
                reason_code="host_memory_unavailable")

    def _ensure_cpu_offload_capacity(
        self, model: str, parallel: int, required_mib: int,
        capabilities: set[str], request_path: str,
        gpu_uuids: tuple[str, ...] | None, triggered_by: dict[str, str] | None,
        expected_profile: Any, *, allow_reclaim: bool,
    ) -> dict[str, Any]:
        """Own an exact exclusive GPU scope and conservative pageable RAM.

        Called only after proving that the model cannot fit the allowed physical
        aggregate. CPU-only inference and spilling around busy GPU owners are
        never an alternate admission path.
        """
        if parallel > POOL_INSTANCE_PARALLEL:
            raise PermanentCapacityError("one CPU-offload scope supports only its configured parallel slots",
                                         422, "parallel_exceeds_capacity")
        scope = tuple(gpu_uuids) if gpu_uuids is not None else tuple(
            gpu for gpu in dict.fromkeys(MODEL_GPU_PREFERENCES.get(model, []) + list(SELECTED_GPUS))
            if gpu in SELECTED_GPUS)
        if not scope:
            raise PermanentCapacityError("CPU offload requires at least one allowed GPU", 422,
                                         "invalid_gpu_scope")
        with self.cv:
            self._prune_dead_lanes_locked()
            blocked = self._ollama_blocked_gpus_locked() | self._reserved_gpus_locked()
            existing = [lane for lane in self.lanes.values()
                        if lane.kind == "managed" and lane.model == model and lane.cpu_offload
                        and lane.scope == scope and lane.context_profile_matches()
                        and not lane.loading and not lane.retiring
                        and not set(lane.scope).intersection(blocked)]
            if existing:
                return {"ok": True, "schema": "io.ollama-unify.gpu-negotiator.capacity.v1",
                        "requested_model": model, "canonical_model": model,
                        "requested_parallel": parallel, "requested_cpu_offload": True,
                        "requested_gpu_uuids": list(scope),
                        "admitted_parallel": sum(lane.parallel for lane in existing),
                        "lanes": [lane.public_summary() for lane in existing]}
            if set(scope).intersection(blocked):
                raise CapacityError("CPU offload GPU scope is reserved or unavailable",
                                    reason_code="gpu_peer_group_wait")
            queued_models = {waiter.model for waiter in self.waiters if waiter.model != model}
            victims = [lane for lane in self.lanes.values() if lane.kind == "managed"
                       and set(lane.protected_scope).intersection(scope)]
            if any(lane.in_flight or lane.loading or lane.model in queued_models for lane in victims):
                raise CapacityError("CPU offload GPU scope still has active owners",
                                    reason_code="gpu_peer_group_wait")
            if victims and not allow_reclaim:
                raise BackgroundCapacityDeferred("background CPU offload yields before lane reclamation")
            if len(self.lanes) - 1 - len(victims) + 1 > POOL_MAX_SERVERS:
                raise CapacityError("managed lane limit reached for CPU offload")
        host_reservation = required_mib + POOL_HOST_RESERVE_MIB
        # Prove feasibility before disrupting any idle GPU owner.
        self._require_cpu_host_budget(host_reservation)
        inventory = {str(device["uuid"]): device for device in gpu_snapshot()}
        if any(gpu not in inventory for gpu in scope):
            raise CapacityError("CPU offload GPU telemetry is unavailable",
                                reason_code="gpu_peer_group_wait")
        with self.cv:
            for lane in victims:
                lane.retiring = True
            self.cv.notify_all()
        if self._stop_lanes(victims, "exclusive CPU offload placement"):
            raise CapacityError("CPU offload scope retirement is incomplete", reason_code="lane_stop_failed")
        self._require_safe_gpu_transition(scope)
        inventory = {str(device["uuid"]): device for device in gpu_snapshot()}
        if any(gpu not in inventory
               or int(inventory[gpu].get("total_mib") or 0) <= POOL_VRAM_RESERVE_MIB
               or int(inventory[gpu].get("free_mib") or 0)
                   < int(inventory[gpu]["total_mib"]) - POOL_VRAM_RESERVE_MIB for gpu in scope):
            raise CapacityError("CPU offload requires its complete GPU reservation to be free",
                                reason_code="gpu_peer_group_wait")
        self._require_cpu_host_budget(host_reservation)
        lane = self._spawn_lane(model, scope, required_mib, capabilities, request_path,
                                triggered_by, {gpu: int(inventory[gpu]["total_mib"]) for gpu in scope},
                                expected_profile=expected_profile, cpu_offload=True,
                                host_reserved_mib=host_reservation)
        return {"ok": True, "schema": "io.ollama-unify.gpu-negotiator.capacity.v1",
                "requested_model": model, "canonical_model": model,
                "requested_parallel": parallel, "requested_cpu_offload": True,
                "requested_gpu_uuids": list(scope), "admitted_parallel": lane.parallel,
                "lanes": [lane.public_summary()]}

    def _ensure_group_capacity(
        self, model: str, parallel: int, required_mib: int,
        capabilities: set[str], request_path: str,
        gpu_uuids: tuple[str, ...] | None,
        triggered_by: dict[str, str] | None,
        expected_profile: Any = CONTEXT_PROFILE_UNSET,
        *, allow_reclaim: bool = True, exact_group: bool = False,
    ) -> dict[str, Any]:
        """Create exclusive whole-process peer groups under the transition lock.

        Ollama chooses the tensor partition from actual device memory. We do
        not guess equal shards: every member's entire capacity stays reserved
        until the backend process group has completely exited.
        """
        wanted = math.ceil(parallel / POOL_INSTANCE_PARALLEL)
        while True:
            with self.cv:
                self._prune_dead_lanes_locked()
                if any(not lease.gpu_uuids and lease.state in ("pending", "active", "revoking")
                       for lease in self.leases.values()):
                    raise CapacityError("unscoped external lease prevents exclusive GPU group placement",
                                        reason_code="lease_transition")
                blocked = self._ollama_blocked_gpus_locked() | self._reserved_gpus_locked()
                existing = [lane for lane in self.lanes.values()
                            if lane.kind == "managed" and lane.model == model
                            and not lane.cpu_offload
                            and lane.context_profile_matches()
                            and len(lane.scope) > 1 and lane.allows(gpu_uuids)
                            and (not exact_group or lane.scope == gpu_uuids)
                            and not lane.retiring and not lane.loading
                            and not set(lane.scope).intersection(blocked)]
                if len(existing) >= wanted:
                    if exact_group:
                        for lane in existing:
                            self._warm_runtime_identity_locked(lane)
                    break
                queued_models = {waiter.model for waiter in self.waiters if waiter.model != model}
                # An idle different-model group may be retired as one whole
                # owner. Any demand, activity, or blocked member protects all
                # its members from reclamation.
                blocked |= {gpu for lane in self.lanes.values()
                            if lane.kind == "managed" and len(lane.scope) > 1
                            and ((lane.model == model and not lane.cpu_offload) or lane.in_flight or lane.loading
                                 or lane.model in queued_models
                                 or set(lane.protected_scope).intersection(blocked))
                            for gpu in lane.protected_scope}
                blocked |= {gpu for lane in self.lanes.values()
                            if lane.kind == "managed"
                            and (lane.in_flight or lane.loading or lane.model in queued_models)
                            for gpu in lane.scope}
            inventory = {str(device["uuid"]): device for device in gpu_snapshot()
                         if device.get("uuid") in SELECTED_GPUS}
            order = list(gpu_uuids) if gpu_uuids is not None else list(dict.fromkeys(
                MODEL_GPU_PREFERENCES.get(model, []) + list(SELECTED_GPUS)))
            if (exact_group and all(gpu in inventory for gpu in order)
                    and required_mib > sum(max(0, int(inventory[gpu]["total_mib"])
                        - POOL_VRAM_RESERVE_MIB) for gpu in order)):
                raise PermanentCapacityError("model exceeds the exact GPU group's physical capacity and headroom",
                                             422, "model_exceeds_gpu_capacity")
            scope: list[str] = []
            capacity = 0
            for gpu in order:
                if gpu in blocked or gpu not in inventory:
                    if exact_group:
                        raise CapacityError("requested exclusive GPU group is not completely available",
                                            reason_code="gpu_peer_group_wait")
                    continue
                scope.append(gpu)
                capacity += max(0, int(inventory[gpu]["total_mib"]) - POOL_VRAM_RESERVE_MIB)
                if not exact_group and len(scope) > 1 and capacity >= required_mib:
                    break
            if len(scope) < 2 or capacity < required_mib:
                raise CapacityError("no complete exclusive GPU group is currently available",
                                    reason_code="gpu_peer_group_wait")
            members = set(scope)
            with self.cv:
                victims = [lane for lane in self.lanes.values()
                           if lane.kind == "managed" and set(lane.scope).intersection(members)]
                if victims and not allow_reclaim:
                    raise BackgroundCapacityDeferred(
                        "background work yields because GPU group placement would retire resident lanes"
                    )
                if any(lane.in_flight or lane.loading or lane.model in queued_models
                       or set(lane.protected_scope).intersection(blocked) for lane in victims):
                    raise CapacityError("GPU group still has active owners",
                                        reason_code="gpu_peer_group_wait")
                if len(self.lanes) - 1 - len(victims) + 1 > POOL_MAX_SERVERS:
                    raise CapacityError("managed lane limit reached for GPU group")
                for lane in victims:
                    lane.retiring = True
                self.cv.notify_all()
            if self._stop_lanes(victims, "exclusive GPU group placement"):
                raise CapacityError("GPU group retirement is not complete",
                                    reason_code="lane_stop_failed")
            self._require_safe_gpu_transition(tuple(scope))
            inventory = {str(device["uuid"]): device for device in gpu_snapshot()}
            usable = sum(max(0, int(inventory.get(gpu, {}).get("free_mib") or 0)
                             - POOL_VRAM_RESERVE_MIB) for gpu in scope)
            if usable < required_mib:
                raise CapacityError("GPU group lacks live free VRAM after retirement",
                                    reason_code="gpu_peer_group_wait")
            available_host = int(host_memory_snapshot().get("memavailable_mib") or 0)
            with self.cv:
                cpu_promises = any(lane.host_reserved_mib for lane in self.lanes.values())
            if cpu_promises:
                self._require_cpu_host_budget(POOL_HOST_RESERVE_MIB)
            if available_host and available_host < POOL_HOST_RESERVE_MIB:
                raise CapacityError("host memory unavailable for GPU group",
                                    reason_code="host_memory_unavailable")
            self._spawn_lane(model, tuple(scope), required_mib, capabilities,
                             request_path, triggered_by,
                             {gpu: int(inventory[gpu]["total_mib"]) for gpu in scope},
                             expected_profile=expected_profile)
        with self.cv:
            lanes = [lane.public_summary() for lane in existing]
        return {
            "ok": True, "schema": "io.ollama-unify.gpu-negotiator.capacity.v1",
            "requested_model": model, "canonical_model": model,
            "requested_parallel": parallel,
            "requested_placement": "exclusive_group" if exact_group else "auto",
            "requested_gpu_uuids": list(gpu_uuids) if gpu_uuids is not None else None,
            "admitted_parallel": sum(lane["parallel"] for lane in lanes),
            "public_ollama_api": f"http://127.0.0.1:{LISTEN_PORT}", "lanes": lanes,
        }

    def ensure_capacity(
        self,
        model: str,
        parallel: int,
        request_path: str = "",
        gpu_uuids: tuple[str, ...] | None = None,
        triggered_by: dict[str, str] | None = None,
        *, allow_reclaim: bool = True, placement: str = "auto", cpu_offload: bool = False,
    ) -> dict[str, Any]:
        if type(cpu_offload) is not bool:
            raise PermanentCapacityError("cpu_offload must be boolean", 400, "invalid_capacity_request")
        if cpu_offload and placement != "auto":
            raise PermanentCapacityError("CPU offload does not support exclusive_group certification", 400,
                                         "invalid_capacity_request")
        require_inference_enabled()
        model = canonical_model_tag(model)
        if not model:
            raise PermanentCapacityError(
                "capacity request requires a model tag", 400, "invalid_capacity_request"
            )
        require_gpu_health(refresh=True)
        with self.cv:
            requested_gpu_uuids = gpu_uuids
            gpu_uuids = self._capacity_scope_locked(
                model, gpu_uuids, placement, str((triggered_by or {}).get("key") or ""))
            cancelling_request = self._model_cancellation_in_progress_locked(
                model
            )
        if gpu_uuids == ():
            raise PermanentCapacityError(
                f"model {model!r} is restricted to GPUs "
                f"{self.model_gpu_policy.get(model)}, none of which the request "
                f"allows ({list(requested_gpu_uuids or ())})",
                409,
                "gpu_policy_conflict",
            )
        if parallel < 1:
            raise PermanentCapacityError(
                "parallel must be at least 1", 400, "invalid_capacity_request"
            )
        if placement == "exclusive_group" and parallel > POOL_INSTANCE_PARALLEL:
            raise PermanentCapacityError("one exact exclusive group supports only its configured parallel slots",
                                         422, "parallel_exceeds_capacity")
        if cancelling_request:
            raise CapacityError(
                f"model {model!r} is cancelling an expired request; retry "
                "after its managed lane stops",
                503,
                "request_cancellation_in_progress",
                True,
                1,
            )
        maximum = POOL_MAX_SERVERS * POOL_INSTANCE_PARALLEL
        if parallel > maximum:
            raise PermanentCapacityError(
                f"requested parallel={parallel}, but configured maximum is {maximum}",
                422,
                "parallel_exceeds_capacity",
            )
        if not POOL_ENABLED:
            raise PermanentCapacityError(
                "managed Ollama parallel pool is disabled",
                409,
                "pool_disabled",
            )
        with self.transition:
            # A policy can change while this capacity request waits behind a
            # lease transition. Recompute from the caller's original scope.
            with self.cv:
                gpu_uuids = self._capacity_scope_locked(
                    model, requested_gpu_uuids, placement,
                    str((triggered_by or {}).get("key") or ""))
                if gpu_uuids == ():
                    raise PermanentCapacityError("model policy excludes the request GPUs",
                                                 409, "gpu_policy_conflict")
            profile_data = None
            with MODEL_CONTEXT_LOCK:
                if AUTO_MODEL_CONTEXT or effective_model_context_profile(model):
                    profile_data = self._model_profile(model)
                expected_profile = effective_model_context_profile(model)
                expected_profile = dict(expected_profile) if expected_profile else None
            with self.cv:
                if self.draining or self._global_transition_lease_locked():
                    raise CapacityError(
                        "GPU lease transition is in progress",
                        reason_code="lease_transition",
                    )
                self._prune_dead_lanes_locked()
                stale = [lane for lane in self.lanes.values()
                         if lane.kind == "managed" and lane.model == model
                         and not lane.context_profile_matches()]
                if stale and not allow_reclaim:
                    raise BackgroundCapacityDeferred(
                        "background work yields because the resident model context would need replacement"
                    )
                if any(lane.in_flight or lane.loading for lane in stale):
                    raise CapacityError("previous model context identity is still in use",
                                        503, "model_context_identity_changed")
                for lane in stale:
                    lane.retiring = True
            if stale and self._stop_lanes(stale, "model context identity changed"):
                raise CapacityError("previous model context lane has not completely stopped",
                                    reason_code="lane_stop_failed")
            if placement == "exclusive_group":
                if profile_data is None:
                    with MODEL_CONTEXT_LOCK:
                        profile_data = self._model_profile(model)
                        expected_profile = effective_model_context_profile(model)
                        expected_profile = dict(expected_profile) if expected_profile else None
                required_mib, capabilities = profile_data
                return self._ensure_group_capacity(
                    model, parallel, required_mib, capabilities, request_path,
                    gpu_uuids, triggered_by, expected_profile,
                    allow_reclaim=allow_reclaim, exact_group=True)
            with self.cv:
                blocked = self._ollama_blocked_gpus_locked()
                existing = [lane for lane in self.lanes.values()
                            if lane.kind == "managed" and lane.model == model
                            and lane.context_profile_matches()
                            and not set(lane.scope).intersection(blocked) and not lane.retiring
                            and not lane.loading and lane.can_serve(gpu_uuids, cpu_offload)]
            desired_servers = math.ceil(parallel / POOL_INSTANCE_PARALLEL)
            if len(existing) < desired_servers:
                with MODEL_CONTEXT_LOCK:
                    if profile_data is None:
                        profile_data = self._model_profile(model)
                        expected_profile = effective_model_context_profile(model)
                        expected_profile = dict(expected_profile) if expected_profile else None
                    required_mib, capabilities = profile_data
                selected_devices = [
                    device for device in gpu_snapshot()
                    if device.get("uuid") in SELECTED_GPUS
                    and (gpu_uuids is None or device.get("uuid") in gpu_uuids)
                ]
                if selected_devices and required_mib > max(
                    int(device.get("total_mib") or 0)
                    for device in selected_devices
                ):
                    capacity = sum(max(0, int(device.get("total_mib") or 0))
                                   for device in selected_devices)
                    if required_mib > capacity or len(selected_devices) < 2:
                        if cpu_offload:
                            return self._ensure_cpu_offload_capacity(
                                model, parallel, required_mib, capabilities, request_path,
                                gpu_uuids, triggered_by, expected_profile,
                                allow_reclaim=allow_reclaim)
                        raise PermanentCapacityError(
                            f"model {model!r} requires {required_mib} MiB per lane, "
                            f"but selected GPUs have {capacity} MiB combined physical capacity",
                            422, "model_exceeds_gpu_capacity")
                    return self._ensure_group_capacity(
                        model, parallel, required_mib, capabilities, request_path,
                        gpu_uuids, triggered_by, expected_profile,
                        allow_reclaim=allow_reclaim)
                missing = desired_servers - len(existing)
                host = host_memory_snapshot()
                available_host = int(host.get("memavailable_mib") or 0)
                required_host = missing * POOL_HOST_RESERVE_MIB
                with self.cv:
                    cpu_promises = any(lane.host_reserved_mib for lane in self.lanes.values())
                if cpu_promises:
                    self._require_cpu_host_budget(required_host)
                if available_host and available_host < required_host:
                    raise CapacityError(
                        f"{missing} new lane(s) reserve {required_host} MiB host memory, "
                        f"but only {available_host} MiB is available",
                        reason_code="host_memory_unavailable",
                    )
                with self.cv:
                    self._prune_dead_lanes_locked()
                    managed = [lane for lane in self.lanes.values()
                               if lane.kind == "managed"]
                    overflow = max(
                        0, len(managed) + missing - POOL_MAX_SERVERS,
                    )
                    if overflow and not allow_reclaim:
                        raise BackgroundCapacityDeferred(
                            "background work yields because the managed lane limit requires replacement"
                        )
                    queued_models = {
                        waiter.model for waiter in self.waiters
                        if waiter.model != model
                    }
                    replaceable = sorted(
                        (lane for lane in managed
                         if (lane.model != model or lane.cpu_offload
                             or (gpu_uuids is not None
                                 and not lane.allows(gpu_uuids)))
                         and lane.in_flight == 0
                         and not lane.loading
                         and not set(lane.protected_scope).intersection(blocked)
                         and (len(lane.scope) == 1 or not set(lane.protected_scope).intersection(
                             self._reserved_gpus_locked()))
                         and lane.model not in queued_models),
                        key=lambda lane: (lane.last_used, lane.created_at),
                    )
                    if len(replaceable) < overflow:
                        raise CapacityError(
                            "managed Ollama lane capacity reached; no idle lane can be replaced",
                            reason_code="reclaimable_placement_wait",
                        )
                    if overflow:
                        self._require_reclamation_feasible(
                            gpu_uuids, required_mib, missing, replaceable)
                    retired = replaceable[:overflow]
                    for lane in retired:
                        lane.retiring = True
                    if retired:
                        self.cv.notify_all()
                if retired:
                    failed = self._stop_lanes(
                        retired, f"idle lane replacement for {model}"
                    )
                    if failed:
                        raise CapacityError(
                            "retired managed lane process group did not stop",
                            reason_code="lane_stop_failed",
                        )
                with self.cv:
                    self._prune_dead_lanes_locked()
                    blocked = self._ollama_blocked_gpus_locked()
                placements: list[str] = []
                devices: list[dict[str, Any]] = []
                preferred_gpus = (
                    list(gpu_uuids)
                    if gpu_uuids is not None
                    else MODEL_GPU_PREFERENCES.get(model, [])
                )
                preference_rank = {
                    gpu_uuid: index
                    for index, gpu_uuid in enumerate(preferred_gpus)
                }
                while True:
                    devices = self._placement_devices(blocked)
                    if gpu_uuids is not None:
                        devices = [
                            device for device in devices
                            if device.get("uuid") in gpu_uuids
                        ]
                    virtual_free = {
                        str(device.get("uuid") or ""):
                            int(device.get("free_mib") or 0)
                        for device in devices
                    }
                    placements = []
                    for _ in range(missing):
                        candidates = [
                            (gpu_uuid, free_mib)
                            for gpu_uuid, free_mib in virtual_free.items()
                            if free_mib >= required_mib
                        ]
                        if not candidates:
                            break
                        preferred = [candidate for candidate in candidates
                                     if candidate[0] in preference_rank]
                        if preferred:
                            chosen_uuid, _ = min(
                                preferred,
                                key=lambda candidate: (
                                    preference_rank[candidate[0]],
                                    -candidate[1],
                                    candidate[0],
                                ),
                            )
                        else:
                            chosen_uuid, _ = max(
                                candidates,
                                key=lambda candidate: (
                                    candidate[1], candidate[0]
                                ),
                            )
                        placements.append(chosen_uuid)
                        virtual_free[chosen_uuid] -= required_mib
                    if len(placements) == missing:
                        break

                    if not allow_reclaim:
                        raise BackgroundCapacityDeferred(
                            "background work yields because no unreserved GPU placement fits without reclamation"
                        )

                    device_uuids = set(virtual_free)
                    with self.cv:
                        self._prune_dead_lanes_locked()
                        group_candidate_gpus = set(gpu_uuids or SELECTED_GPUS)
                        group_blocked = self._ollama_blocked_gpus_locked() | self._reserved_gpus_locked()
                        queued_models = {
                            waiter.model for waiter in self.waiters
                            if waiter.model != model
                        }
                        replaceable = sorted(
                            (
                                lane for lane in self.lanes.values()
                                if lane.kind == "managed"
                                and (lane.model != model or lane.cpu_offload
                                     or (gpu_uuids is not None
                                         and not lane.allows(gpu_uuids)))
                                and lane.in_flight == 0
                                and not lane.loading
                                and (bool(set(lane.scope).intersection(device_uuids))
                                     or ((len(lane.scope) > 1 or lane.cpu_offload)
                                         and bool(set(lane.scope).intersection(group_candidate_gpus))
                                         and not set(lane.protected_scope).intersection(group_blocked)))
                                and lane.model not in queued_models
                            ),
                            key=lambda lane: (lane.last_used, lane.created_at),
                        )
                        if replaceable:
                            self._require_reclamation_feasible(
                                gpu_uuids, required_mib, missing, replaceable)
                        victim = replaceable[0] if replaceable else None
                        if victim is not None:
                            victim.retiring = True
                            self.cv.notify_all()
                    if victim is None:
                        free = sorted(
                            (int(device.get("free_mib") or 0),
                             str(device.get("uuid") or ""))
                            for device in devices
                        )
                        with self.cv:
                            reclaimable_later = any(
                                lane.kind == "managed"
                                and lane.model != model
                                and bool(set(lane.scope).intersection(device_uuids))
                                for lane in self.lanes.values()
                            )
                        raise CapacityError(
                            f"model {model!r} requires {required_mib} MiB per lane; "
                            f"only {len(placements)} of {missing} required lane "
                            f"placements fit and no idle lane can be reclaimed "
                            f"(free={free})",
                            reason_code=(
                                "reclaimable_placement_wait"
                                if reclaimable_later
                                else "lane_capacity_wait"
                            ),
                        )
                    failed = self._stop_lanes(
                        [victim], f"live VRAM reclamation for {model}",
                    )
                    if failed:
                        raise CapacityError(
                            "reclaimed managed lane process group did not stop",
                            reason_code="lane_stop_failed",
                        )
                created: list[Lane] = []
                try:
                    for chosen_uuid in placements:
                        created.append(self._spawn_lane(
                            model, chosen_uuid, required_mib,
                            capabilities, request_path, triggered_by,
                            expected_profile=expected_profile,
                        ))
                except Exception:
                    with self.cv:
                        for lane in created:
                            lane.retiring = True
                        self.cv.notify_all()
                    self._stop_lanes(created, "capacity rollback")
                    raise
            with self.cv:
                lanes = [lane for lane in self._lane_summaries_locked()
                         if lane["kind"] == "managed" and lane["model"] == model
                         and (cpu_offload or not lane.get("cpu_offload", False))
                         and (gpu_uuids is None
                              or set(lane["gpu_uuids"]).issubset(gpu_uuids))]
                admitted = sum(
                    int(lane["parallel"]) for lane in lanes
                    if lane["state"] == "ready"
                )
            return {
                "ok": True,
                "schema": "io.ollama-unify.gpu-negotiator.capacity.v1",
                "requested_model": model,
                "canonical_model": model,
                "requested_parallel": parallel,
                "requested_gpu_uuids": (
                    list(gpu_uuids) if gpu_uuids is not None else None
                ),
                "admitted_parallel": admitted,
                "public_ollama_api": f"http://127.0.0.1:{LISTEN_PORT}",
                "lanes": lanes,
            }

    def _remove_waiter_locked(self, waiter: QueuedRequest) -> None:
        try:
            self.waiters.remove(waiter)
        except ValueError:
            pass
        self._release_waiter_retention_locked(waiter)
        self.cv.notify_all()

    def _release_waiter_retention_locked(self, waiter: QueuedRequest) -> None:
        retained = waiter.retained_request
        if retained is None:
            return
        self.retained_request_bytes = max(
            0, self.retained_request_bytes - len(retained.body)
        )
        waiter.retained_request = None

    def _prune_stale_waiters_locked(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        stale = [
            waiter for waiter in self.waiters
            if not waiter.attached
            and waiter.resume_deadline is not None
            and current >= waiter.resume_deadline
        ]
        for waiter in stale:
            waiter.phase = "stale"
            self._record_logical_tombstone_locked(
                waiter.logical_request_id,
                waiter.request_id,
                "logical_request_expired",
            )
            self._remove_waiter_locked(waiter)
            self.queue_stale_total += 1

    def _register_active_request_locked(
        self,
        lane: Lane,
        request_id: str,
        logical_request_id: str,
    ) -> ActiveRequest:
        if request_id in self.active_request_records:
            raise RuntimeError(f"request {request_id} is already admitted")
        now = time.monotonic()
        self.active_request_records[request_id] = ActiveRequest(
            request_id=request_id,
            lane=lane,
            logical_request_id=logical_request_id,
            admitted_at=now,
            last_activity_at=now,
            expires_at=now + REQUEST_ACTIVITY_TTL,
        )
        lane.in_flight += 1
        lane.last_used = time.time()
        self.active_requests = len(self.active_request_records)
        return self.active_request_records[request_id]

    def _owned_active_request_locked(self, admission: Admission) -> ActiveRequest | None:
        active = self.active_request_records.get(admission.request_id)
        if admission.warm_admission is not None and (
                active is not admission.active_request or active is None
                or active.request_id != admission.request_id
                or active.lane is not admission.lane
                or active.logical_request_id != admission.logical_request_id
                or active.warm_admission != admission.warm_admission
                or (admission.logical_request_id and self.logical_in_flight.get(
                    admission.logical_request_id) != (admission.request_fingerprint,
                        admission.request_id, admission.queue_ticket))):
            return None
        return active

    def reject_warm_before_backend(self, admission: Admission, failure: CapacityError) -> CapacityError:
        """Release only a proven, exact never-dispatched owner, atomically.

        Unknown/cancelled/started state is terminal for this handler and remains
        owned by its existing lifecycle. It must never authorize fresh inference.
        """
        with self.cv:
            active = self._owned_active_request_locked(admission)
            logical = admission.logical_request_id
            if (admission.warm_admission is None or active is None
                    or active.backend_started or active.backend is not None
                    or active.backend_completed or active.cancel_requested_at is not None
                    or active.detached_at is not None
                    or (logical and self.logical_in_flight.get(logical) != (
                        admission.request_fingerprint, admission.request_id, admission.queue_ticket))
                    or any(item.request_id == admission.request_id
                           or (logical and item.logical_request_id == logical) for item in self.waiters)
                    or (logical and (logical in self.completed_responses or logical in self.logical_tombstones))):
                return WarmAdmissionOwnershipUncertain(admission, failure)
            self.proxy_exit(admission, active.lane.model, False)
            if isinstance(failure, WarmPreflightRequired):
                failure.admission_retained = False
            return failure

    def bind_active_request_client(
        self, admission: Admission, client_key: str,
    ) -> None:
        with self.cv:
            active = self._owned_active_request_locked(admission)
            if active is not None:
                active.client_key = client_key

    def request_backend_started(
        self, admission: Admission, backend: Any,
    ) -> bool:
        with self.cv:
            active = self._owned_active_request_locked(admission)
            if admission.warm_admission is not None and (active is None
                    or active.cancel_requested_at is not None or active.detached_at is not None
                    or active.backend_started or active.backend is not None or active.backend_completed):
                raise WarmAdmissionOwnershipUncertain(admission)
            if active is None or active.cancel_requested_at is not None:
                return False
            if admission.warm_admission is not None:
                proof = json.loads(admission.warm_admission)
                try:
                    eligible = self._validate_warm_admission_locked(
                        admission.warm_admission, active.lane.model,
                        tuple(proof["gpu_uuids"]), phase="backend_dispatch_validation")
                    if active.lane.lane_id not in eligible:
                        raise WarmPreflightRequired(phase="backend_dispatch_validation",
                                                   causes=("selected_lane_not_certified",))
                except WarmPreflightRequired as exc:
                    failure = self.reject_warm_before_backend(admission, exc)
                    if failure is exc:
                        raise
                    raise failure from exc
            active.backend = backend
            active.backend_started = True
            active.phase = "backend_connecting"
            now = time.monotonic()
            active.last_activity_at = now
            active.expires_at = (
                active.detached_at + REQUEST_DETACHED_TTL
                if active.detached_at is not None
                else now + REQUEST_ACTIVITY_TTL
            )
            if (
                active.detached_at is not None
                and not active.logical_request_id
            ):
                active.expires_at = now
            self.cv.notify_all()
            return True

    def renew_request_activity(
        self, admission: Admission, phase: str,
    ) -> bool:
        with self.cv:
            active = self._owned_active_request_locked(admission)
            if active is None or active.cancel_requested_at is not None:
                return False
            now = time.monotonic()
            active.phase = phase
            active.last_activity_at = now
            active.expires_at = (
                active.detached_at + REQUEST_DETACHED_TTL
                if active.detached_at is not None
                else now + REQUEST_ACTIVITY_TTL
            )
            self.request_activity_renewed_total += 1
            self.cv.notify_all()
            return True

    def request_backend_connected(self, admission: Admission, sock: Any) -> bool:
        """Bind the exact socket before dispatch, including cancellation races."""
        with self.cv:
            active = self._owned_active_request_locked(admission)
            if active is None or active.cancel_requested_at is not None:
                return False
            active.backend_socket = sock
            return True

    def _mark_active_cancelling_locked(
        self,
        active: ActiveRequest,
        reason: str,
        *,
        expired: bool,
    ) -> bool:
        if active.cancel_requested_at is not None:
            return False
        now = time.monotonic()
        active.cancel_requested_at = now
        active.cancel_reason = reason
        active.phase = "cancelling"
        active.expires_at = now + REQUEST_CANCEL_GRACE
        if active.lane.kind == "managed":
            # Stop admitting fresh work to a lane whose backend termination is
            # not yet proven, even when that lane allows parallel requests.
            active.lane.retiring = True
        self.request_cancelled_total += 1
        if expired:
            self.request_expired_total += 1
        if active.logical_request_id:
            self._record_logical_tombstone_locked(
                active.logical_request_id,
                active.request_id,
                reason,
            )
        self.cv.notify_all()
        return True

    def request_client_detached(self, admission: Admission) -> None:
        cancel = None
        with self.cv:
            active = self._owned_active_request_locked(admission)
        if active is None:
            return
        with active.terminal_lock:
            with self.cv:
                current = self.active_request_records.get(
                    admission.request_id
                )
                if (
                    current is not active
                    or active.detached_at is not None
                    or active.backend_completed
                ):
                    return
                now = time.monotonic()
                active.detached_at = now
                active.phase = "detached"
                self.request_disconnected_total += 1
                if active.logical_request_id:
                    active.expires_at = now + REQUEST_DETACHED_TTL
                else:
                    if self._mark_active_cancelling_locked(
                        active, "client_disconnected", expired=False
                    ):
                        cancel = active
                self.cv.notify_all()
        if cancel is not None:
            self._cancel_backend_transport(cancel)

    def active_request_terminal_lock(
        self, admission: Admission,
    ) -> Any | None:
        with self.cv:
            active = self._owned_active_request_locked(admission)
            return active.terminal_lock if active is not None else None

    def request_backend_complete(self, admission: Admission) -> bool:
        with self.cv:
            active = self._owned_active_request_locked(admission)
            if active is None:
                return False
            if active.cancel_requested_at is not None:
                # Cancellation and completion are competing terminal events.
                # Once cancellation commits, EOF caused by our own socket
                # shutdown must never become a successful/replayable response.
                return False
            if active.backend_completed:
                return True
            active.backend_completed = True
            active.phase = "backend_complete"
            now = time.monotonic()
            active.last_activity_at = now
            active.expires_at = now + REQUEST_CANCEL_GRACE
            tombstone = self.logical_tombstones.get(
                active.logical_request_id
            )
            if (
                tombstone is not None
                and tombstone.request_id == active.request_id
            ):
                self.logical_tombstones.pop(active.logical_request_id, None)
            self.cv.notify_all()
            return True

    def active_request_cancel_reason(self, admission: Admission) -> str:
        with self.cv:
            active = self._owned_active_request_locked(admission)
            return active.cancel_reason if active is not None else ""

    def note_backend_failure(
        self, admission: Admission, reason: str,
    ) -> None:
        with self.cv:
            active = self._owned_active_request_locked(admission)
            if active is None or active.backend_completed:
                return
            self._mark_active_cancelling_locked(
                active,
                reason,
                expired=reason in {
                    "backend_activity_timeout", "detached_request_expired",
                },
            )

    @staticmethod
    def _cancel_backend_transport(active: ActiveRequest) -> None:
        backend = active.backend
        if backend is None:
            return
        sock = active.backend_socket
        if sock is None:
            sock = getattr(backend, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            backend.close()
        except OSError:
            pass

    def _active_request_summary_locked(self) -> dict[str, Any]:
        now = time.monotonic()
        records = list(self.active_request_records.values())
        phases: dict[str, int] = {}
        for active in records:
            phases[active.phase] = phases.get(active.phase, 0) + 1
        finite_expiries = [
            active.expires_at for active in records
            if math.isfinite(active.expires_at)
        ]
        return {
            "tracked": len(records),
            "phase_counts": phases,
            "detached": sum(
                active.detached_at is not None for active in records
            ),
            "cancelling": sum(
                active.cancel_requested_at is not None for active in records
            ),
            "oldest_age_ms": max(
                (int((now - active.admitted_at) * 1000) for active in records),
                default=0,
            ),
            "nearest_expiry_ms": max(
                0,
                int((min(finite_expiries) - now) * 1000)
                if finite_expiries else 0,
            ),
            "activity_ttl_seconds": REQUEST_ACTIVITY_TTL,
            "detached_ttl_seconds": REQUEST_DETACHED_TTL,
            "cancel_grace_seconds": REQUEST_CANCEL_GRACE,
            "lane_stop_attempt_ttl_seconds": LANE_STOP_ATTEMPT_TTL,
            "cancel_process_policy": (
                "sigterm_only" if BACKEND_TYPE == "cuda" else "sigterm_then_sigkill"
            ),
            "activity_renewed_total": self.request_activity_renewed_total,
            "disconnected_total": self.request_disconnected_total,
            "expired_total": self.request_expired_total,
            "cancelled_total": self.request_cancelled_total,
            "forced_lane_stop_total": self.request_forced_lane_stop_total,
            "forced_release_total": self.request_forced_release_total,
            "terminal_release_total": self.request_terminal_release_total,
        }

    def _model_cancellation_in_progress_locked(self, model: str) -> bool:
        return any(
            active.lane.kind == "managed"
            and active.lane.model == model
            and active.cancel_requested_at is not None
            and not active.lane_stopped
            for active in self.active_request_records.values()
        )

    def _release_active_request_locked(
        self,
        request_id: str,
        model: str,
        succeeded: bool,
        client_key: str = "",
    ) -> Lane | None:
        active = self.active_request_records.pop(request_id, None)
        if active is None:
            return None
        lane = active.lane
        lane.in_flight = max(0, lane.in_flight - 1)
        key = client_key or active.client_key
        if key:
            remaining = lane.active_clients.get(key, 0) - 1
            if remaining > 0:
                lane.active_clients[key] = remaining
            else:
                lane.active_clients.pop(key, None)
        lane.last_used = time.time()
        if succeeded and model:
            lane.model = model
        self.active_requests = len(self.active_request_records)
        if active.logical_request_id:
            logical = self.logical_in_flight.get(active.logical_request_id)
            if logical is not None and logical[1] == active.request_id:
                self.logical_in_flight.pop(active.logical_request_id, None)
        retired = None
        if (
            lane.kind == "managed"
            and lane.retiring
            and not lane.in_flight
            and self.lanes.get(lane.lane_id) is lane
        ):
            retired = lane
        self.cv.notify_all()
        return retired

    def _schedule_cancelled_lane_stop_locked(self, lane: Lane) -> bool:
        lane_requests = [
            active for active in self.active_request_records.values()
            if active.lane is lane
        ]
        if (
            not lane_requests
            or lane.kind != "managed"
            or lane.process is None
            or any(active.cancel_requested_at is None for active in lane_requests)
            or any(active.lane_stop_started for active in lane_requests)
        ):
            return False
        for active in lane_requests:
            active.lane_stop_started = True
            active.phase = "stopping_lane"
            active.expires_at = time.monotonic() + LANE_STOP_ATTEMPT_TTL
        lane.retiring = True
        self.request_forced_lane_stop_total += 1
        self.cv.notify_all()
        return True

    def _stop_expired_lane(self, lane: Lane) -> None:
        group_stopped = False
        allow_force = BACKEND_TYPE != "cuda"
        stop_reason = "transition_busy"
        # A drain may own transition while waiting for this exact active
        # request. Never wait for that lock: retain ownership and retry after
        # the bounded drain finishes or aborts instead of deadlocking it.
        acquired = self.transition.acquire(blocking=False)
        if acquired:
            try:
                self._require_safe_gpu_transition(lane.protected_scope, lane.lane_id)
                group_stopped = self._terminate_process(lane.process, allow_force=allow_force)
                stop_reason = "process_group_alive"
            except CapacityError as exc:
                stop_reason = exc.reason_code
            finally:
                self.transition.release()
        if not group_stopped:
            LOG.error(
                "managed Ollama lane did not stop after request cancellation "
                "id=%s gpu=%s reason=%s hard_kill_allowed=%s; retaining its admission accounting",
                lane.lane_id,
                lane.gpu_uuid,
                stop_reason,
                allow_force,
            )
            with self.cv:
                now = time.monotonic()
                for active in self.active_request_records.values():
                    if active.lane is lane:
                        active.lane_stop_started = False
                        active.phase = "lane_stop_failed"
                        active.expires_at = now + REQUEST_CANCEL_GRACE
                self.cv.notify_all()
            return
        LOG.warning(
            "managed Ollama lane stopped after request cancellation "
            "id=%s gpu=%s hard_kill_allowed=%s",
            lane.lane_id,
            lane.gpu_uuid,
            allow_force,
        )
        with self.cv:
            self.lanes.pop(lane.lane_id, None)
            stopped = [
                active for active in self.active_request_records.values()
                if active.lane is lane and active.cancel_requested_at is not None
            ]
            for active in stopped:
                active.lane_stopped = True
                self._release_active_request_locked(
                    active.request_id, "", False
                )
                self.request_forced_release_total += 1
            self.cv.notify_all()

    def active_request_watchdog(self) -> None:
        """Wait on exact request deadlines and cancel only their ownership.

        This is deadline/event driven: admissions, backend activity, client
        detach, and terminal release all notify the condition. There is no
        fixed polling cadence.
        """
        while True:
            cancel: list[ActiveRequest] = []
            stop_lanes: list[Lane] = []
            with self.cv:
                while not self.stopping.is_set():
                    now = time.monotonic()
                    due = [
                        active for active in self.active_request_records.values()
                        if active.expires_at <= now
                    ]
                    if due:
                        break
                    deadline = min(
                        (active.expires_at
                         for active in self.active_request_records.values()),
                        default=None,
                    )
                    self.cv.wait(
                        None if deadline is None else max(0.01, deadline - now)
                    )
                if self.stopping.is_set():
                    return
                now = time.monotonic()
                force_release: list[ActiveRequest] = []
                terminal_release: list[ActiveRequest] = []
                for active in due:
                    current = self.active_request_records.get(active.request_id)
                    if current is not active:
                        continue
                    if (
                        active.backend_completed
                        and active.cancel_requested_at is None
                    ):
                        # The normal handler finalizer releases this admission
                        # immediately after caching any logical response. Keep
                        # that terminal phase leased too, so an interrupted
                        # finalizer cannot pin lane accounting forever. A
                        # tombstone closes the narrow replay/regeneration race;
                        # record_completed_response clears it for this exact
                        # request if finalization subsequently completes.
                        if active.logical_request_id:
                            self._record_logical_tombstone_locked(
                                active.logical_request_id,
                                active.request_id,
                                "completion_finalization_timeout",
                            )
                        terminal_release.append(active)
                        continue
                    if active.cancel_requested_at is None:
                        if active.detached_at is not None:
                            reason = (
                                "detached_request_expired"
                                if active.logical_request_id
                                else "client_disconnected"
                            )
                        else:
                            reason = "backend_activity_timeout"
                        if self._mark_active_cancelling_locked(
                            active,
                            reason,
                            expired=reason != "client_disconnected",
                        ):
                            cancel.append(active)
                        continue
                    if active.lane_stopped:
                        force_release.append(active)
                        continue
                    lane = active.lane
                    if active.lane_stop_started:
                        # The bounded stop worker failed to report a terminal
                        # result. Renew ownership and retry; never release the
                        # lane's reservation merely because its worker died.
                        active.lane_stop_started = False
                        active.phase = "lane_stop_attempt_expired"
                    if self._schedule_cancelled_lane_stop_locked(lane):
                        stop_lanes.append(lane)
                    else:
                        active.expires_at = now + REQUEST_CANCEL_GRACE
                        cancel.append(active)
                for active in force_release:
                    retired = self._release_active_request_locked(
                        active.request_id, "", False
                    )
                    if retired is not None:
                        stop_lanes.append(retired)
                    self.request_forced_release_total += 1
                for active in terminal_release:
                    retired = self._release_active_request_locked(
                        active.request_id, "", False
                    )
                    if retired is not None:
                        stop_lanes.append(retired)
                    self.request_terminal_release_total += 1
            for active in cancel:
                self._cancel_backend_transport(active)
            seen_lane_ids: set[str] = set()
            for lane in stop_lanes:
                if lane.lane_id in seen_lane_ids:
                    continue
                seen_lane_ids.add(lane.lane_id)
                threading.Thread(
                    target=self._stop_expired_lane,
                    args=(lane,),
                    daemon=True,
                ).start()

    @staticmethod
    def _waiter_terminal_failure(waiter: QueuedRequest) -> CapacityError:
        if waiter.terminal_retryable:
            return CapacityError(
                waiter.terminal_error or "broker admission failed",
                waiter.terminal_status or 503,
                waiter.terminal_reason_code or "lane_capacity_wait",
                True,
                waiter.terminal_retry_after or 2,
                request_id=waiter.request_id,
                logical_request_id=waiter.logical_request_id,
            )
        return PermanentCapacityError(
            waiter.terminal_error or "broker admission failed",
            waiter.terminal_status or 422,
            waiter.terminal_reason_code or "permanent_capacity_error",
            request_id=waiter.request_id,
            logical_request_id=waiter.logical_request_id,
        )

    def proxy_enter(self, model: str, routable: bool,
                    connected: Callable[[], bool] = lambda: True,
                    request_id: str = "",
                    allow_during_drain: bool = False,
                    request_path: str = "",
                    admission_wait: float | None = None,
                    logical_request_id: str = "",
                    request_fingerprint: str = "",
                    retained_request: RetainedRequest | None = None,
                    resume_request: bool = False,
                    workload_class: str = "unspecified",
                    queue_policy: str = "wait",
                    gpu_uuids: tuple[str, ...] | None = None,
                    warm_admission: str | None = None,
                    client: dict[str, str] | None = None,
                    cpu_offload: bool = False) -> Admission:
        if type(cpu_offload) is not bool:
            raise PermanentCapacityError("cpu_offload must be boolean", 400, "invalid_admission_header")
        if cpu_offload and warm_admission is not None:
            raise PermanentCapacityError("CPU offload cannot use a full-GPU warm certificate", 400,
                                         "invalid_warm_admission")
        if cpu_offload and routable and not POOL_ENABLED:
            raise PermanentCapacityError("CPU offload requires the managed Ollama pool", 409, "pool_disabled")
        model = canonical_model_tag(model)
        request_id = request_id or secrets.token_hex(8)
        if routable and model:
            require_inference_enabled(request_id=request_id,
                                      logical_request_id=logical_request_id)
            with self.cv:
                client_scope = self._client_gpu_constraint_locked(
                    str((client or {}).get("key") or ""), gpu_uuids)
            if client_scope is not None and not POOL_ENABLED:
                raise PermanentCapacityError(
                    "client GPU policies require the managed Ollama lane pool",
                    409, "pool_disabled", request_id=request_id,
                    logical_request_id=logical_request_id)
        if warm_admission is not None and (not routable or not model or not POOL_ENABLED):
            raise PermanentCapacityError("warm admission requires managed inference", 400, "invalid_warm_admission")
        if routable and model and not allow_during_drain:
            require_gpu_health(request_id=request_id,
                               logical_request_id=logical_request_id)
        if routable and model and gpu_uuids is not None and not POOL_ENABLED:
            raise PermanentCapacityError(
                "hard GPU constraints require the managed Ollama lane pool",
                409,
                "pool_disabled",
                request_id=request_id,
                logical_request_id=logical_request_id,
            )
        if routable and model and POOL_ENABLED:
            with self.cv:
                narrowed = self._policy_constraint_locked(model, client_scope)
            if narrowed == ():
                raise PermanentCapacityError(
                    f"model {model!r} is restricted to GPUs "
                    f"{self.model_gpu_policy.get(model)}, none of which the "
                    "request allows",
                    409,
                    "gpu_policy_conflict",
                    request_id=request_id,
                    logical_request_id=logical_request_id,
                )
        enqueued_at = time.monotonic()
        wait_seconds = DRAIN_TIMEOUT
        if admission_wait is not None:
            wait_seconds = max(0.1, min(DRAIN_TIMEOUT, admission_wait))
        deadline = enqueued_at + wait_seconds

        if not routable or not model or not POOL_ENABLED:
            while True:
                if routable and model and not allow_during_drain:
                    require_gpu_health(request_id=request_id,
                                       logical_request_id=logical_request_id)
                if not connected():
                    raise ClientDisconnected("client disconnected before broker admission")
                with self.cv:
                    revoked = self.blocking_revoking_lease()
                    if revoked is not None and not allow_during_drain:
                        raise CapacityError(
                            f"GPU lease transition for {revoked.owner} was revoked; "
                            "waiting for verified CUDA release",
                            503,
                            "lease_transition",
                            True,
                            2,
                            request_id=request_id,
                            logical_request_id=logical_request_id,
                        )
                    while self.draining and not allow_during_drain:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise AdmissionTimeoutError(
                                "GPU negotiation is still draining Ollama",
                                request_id=request_id,
                                logical_request_id=logical_request_id,
                            )
                        self.cv.wait(min(remaining, 0.25))
                    lane = self._select_lane_locked(model, routable, gpu_uuids)
                    if lane is not None:
                        active = self._register_active_request_locked(
                            lane, request_id, logical_request_id
                        )
                        return Admission(
                            lane, request_id, logical_request_id,
                            request_fingerprint,
                            max(0, int((time.monotonic() - enqueued_at) * 1000)),
                            1, 0, retained_request, active_request=active,
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AdmissionTimeoutError(
                            "Ollama system lane is busy",
                            request_id=request_id,
                            logical_request_id=logical_request_id,
                        )
                    self.cv.wait(min(remaining, 0.25))

        with self.cv:
            self._prune_stale_waiters_locked(enqueued_at)
            if logical_request_id:
                if resume_request:
                    self._raise_logical_tombstone_locked(logical_request_id)
                active = self.logical_in_flight.get(logical_request_id)
                if active is not None:
                    active_fingerprint, active_request_id, queue_ticket = active
                    self.queue_duplicate_total += 1
                    if active_fingerprint != request_fingerprint:
                        raise PermanentCapacityError(
                            "logical request ID was reused with different request content",
                            409,
                            "logical_request_conflict",
                            request_id=active_request_id,
                            logical_request_id=logical_request_id,
                        )
                    raise CapacityError(
                        "logical request is already admitted and in progress",
                        409,
                        "logical_request_in_progress",
                        True,
                        1,
                        request_id=active_request_id,
                        logical_request_id=logical_request_id,
                        queue_ticket=queue_ticket,
                    )
                waiter = next(
                    (
                        item for item in self.waiters
                        if item.logical_request_id == logical_request_id
                    ),
                    None,
                )
                if waiter is not None:
                    if (
                        waiter.request_fingerprint != request_fingerprint
                        or waiter.model != model
                        or waiter.request_path != request_path
                        or waiter.gpu_uuids != gpu_uuids
                        or waiter.cpu_offload != cpu_offload
                    ):
                        self.queue_duplicate_total += 1
                        raise PermanentCapacityError(
                            "logical request ID was reused with different request content",
                            409,
                            "logical_request_conflict",
                            request_id=waiter.request_id,
                            logical_request_id=logical_request_id,
                            queue_position=self.waiters.index(waiter) + 1,
                            queue_ticket=waiter.queue_ticket,
                        )
                    if waiter.attached:
                        self.queue_duplicate_total += 1
                        raise CapacityError(
                            "logical request already has an attached admission attempt",
                            409,
                            "logical_request_in_progress",
                            True,
                            1,
                            request_id=waiter.request_id,
                            logical_request_id=logical_request_id,
                        )
                    waiter.connected = connected
                    waiter.deadline = deadline
                    waiter.attached = True
                    waiter.resume_deadline = None
                    waiter.workload_class = workload_class
                    waiter.queue_policy = queue_policy
                    if waiter.phase == "detached":
                        waiter.phase = "queued"
                    request_id = waiter.request_id
                    enqueued_at = waiter.enqueued_at
                    self.queue_resumed_total += 1
                    self.cv.notify_all()
                else:
                    waiter = None
            else:
                waiter = None
            revoked = self.blocking_revoking_lease()
            if revoked is not None and not allow_during_drain:
                raise CapacityError(
                    f"GPU lease transition for {revoked.owner} was revoked; "
                    "waiting for verified CUDA release",
                    503,
                    "lease_transition",
                    True,
                    2,
                    request_id=request_id,
                    logical_request_id=logical_request_id,
                )
            if waiter is None and len(self.waiters) >= POOL_MAX_QUEUE:
                self.queue_rejected_total += 1
                raise CapacityError(
                    f"broker queue is full ({len(self.waiters)}/{POOL_MAX_QUEUE}); retry after current admissions settle",
                    503,
                    "queue_full",
                    True,
                    2,
                    request_id=request_id,
                    logical_request_id=logical_request_id,
                )
            if waiter is None:
                # This precedes logical ownership and body retention. A caller
                # can recertify and reuse this ID only when no work was admitted.
                self._prune_logical_tombstones_locked()
                tombstone = self.logical_tombstones.get(logical_request_id)
                if tombstone is not None and tombstone.warm_admission != warm_admission:
                    raise PermanentCapacityError("logical request warm certificate changed", 409,
                                                 "logical_request_conflict")
                if warm_admission is not None:
                    try:
                        self._validate_warm_admission_locked(warm_admission, model, gpu_uuids, phase="pre_admission")
                    except WarmPreflightRequired as exc:
                        if tombstone is not None:
                            raise WarmPreflightRequired(admission_retained=True,
                                request_id=tombstone.request_id,
                                logical_request_id=logical_request_id,
                                phase=exc.warm_preflight_phase, causes=exc.warm_preflight_causes) from exc
                        raise
                if resume_request:
                    raise PermanentCapacityError(
                        "logical request is not retained by this broker",
                        409,
                        "logical_request_not_found",
                        logical_request_id=logical_request_id,
                    )
                if logical_request_id:
                    # A full-body submission is an explicit new attempt after
                    # a prior cancel/expiry. Body-free resume remains closed
                    # over the tombstone in resume_request_lookup().
                    self.logical_tombstones.pop(logical_request_id, None)
                if retained_request is not None:
                    retained_bytes = len(retained_request.body)
                    if retained_bytes > RETAINED_REQUEST_MAX_BODY_BYTES:
                        raise PermanentCapacityError(
                            "request body exceeds retained-resume per-entry limit",
                            413,
                            "request_body_exceeds_resume_limit",
                            request_id=request_id,
                            logical_request_id=logical_request_id,
                        )
                    if (
                        self.retained_request_bytes + retained_bytes
                        > RETAINED_REQUEST_MAX_TOTAL_BYTES
                    ):
                        raise CapacityError(
                            "retained-request capacity is temporarily full",
                            503,
                            "request_retention_capacity",
                            True,
                            2,
                            request_id=request_id,
                            logical_request_id=logical_request_id,
                        )
                queue_ticket = self.next_queue_ticket
                self.next_queue_ticket += 1
                waiter = QueuedRequest(
                    request_id=request_id,
                    logical_request_id=logical_request_id,
                    request_fingerprint=request_fingerprint,
                    model=model,
                    enqueued_at=enqueued_at,
                    deadline=deadline,
                    connected=connected,
                    gpu_uuids=gpu_uuids,
                    request_path=request_path,
                    retained_request=retained_request,
                    queue_ticket=queue_ticket,
                    workload_class=workload_class,
                    queue_policy=queue_policy,
                    initial_position=len(self.waiters) + 1,
                    client=client,
                    warm_admission=warm_admission,
                    cpu_offload=cpu_offload,
                )
                self.waiters.append(waiter)
                if retained_request is not None:
                    self.retained_request_bytes += len(retained_request.body)
                self.queue_enqueued_total += 1
                self.queue_peak = max(self.queue_peak, len(self.waiters))
                self.cv.notify_all()
        while True:
            if not connected():
                with self.cv:
                    if waiter.terminal_error:
                        raise self._waiter_terminal_failure(waiter)
                    if waiter not in self.waiters:
                        continue
                    waiter.phase = "cancelled"
                    self.queue_cancelled_total += 1
                    self._record_logical_tombstone_locked(
                        waiter.logical_request_id,
                        waiter.request_id,
                        "logical_request_cancelled",
                    )
                    self._remove_waiter_locked(waiter)
                raise ClientDisconnected("client disconnected while queued")
            try:
                require_gpu_health(request_id=waiter.request_id,
                                   logical_request_id=waiter.logical_request_id)
            except CapacityError:
                with self.cv:
                    if waiter in self.waiters:
                        if waiter.warm_admission is not None:
                            self._record_logical_tombstone_locked(waiter.logical_request_id,
                                waiter.request_id, "gpu_health_unavailable")
                        self._remove_waiter_locked(waiter)
                raise
            with self.cv:
                if waiter.terminal_error:
                    raise self._waiter_terminal_failure(waiter)
                try:
                    client_scope = self._client_gpu_constraint_locked(
                        str((waiter.client or {}).get("key") or ""), waiter.gpu_uuids)
                except PermanentCapacityError as exc:
                    self._reject_client_waiter_locked(waiter, exc)
                    raise self._waiter_terminal_failure(waiter) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {waiter.last_error}" if waiter.last_error else ""
                    if (
                        waiter.logical_request_id
                        and waiter.queue_policy == "wait"
                    ):
                        waiter.phase = "detached"
                        waiter.attached = False
                        waiter.resume_deadline = time.monotonic() + POOL_RESUME_TTL
                        self.queue_admission_timeout_total += 1
                        self.cv.notify_all()
                        raise AdmissionTimeoutError(
                            f"broker queue admission deadline expired{detail}",
                            request_id=waiter.request_id,
                            logical_request_id=waiter.logical_request_id,
                            admission_retained=True,
                            cause_reason_code=waiter.last_reason_code or "",
                            queue_position=self.waiters.index(waiter) + 1,
                            queue_ticket=waiter.queue_ticket,
                        )
                    waiter.phase = "timed-out"
                    queue_position = self.waiters.index(waiter) + 1
                    self.queue_timed_out_total += 1
                    if waiter.warm_admission is not None:
                        self._record_logical_tombstone_locked(waiter.logical_request_id,
                            waiter.request_id, "queue_admission_timeout")
                    self._remove_waiter_locked(waiter)
                    raise AdmissionTimeoutError(
                        f"broker queue admission deadline expired{detail}",
                        request_id=waiter.request_id,
                        cause_reason_code=waiter.last_reason_code or "",
                        queue_position=queue_position,
                        queue_ticket=waiter.queue_ticket,
                    )
                self._prune_dead_lanes_locked()
                certified_ids = None
                if waiter.warm_admission is not None:
                    try:
                        certified_ids = self._validate_warm_admission_locked(
                            waiter.warm_admission, model, waiter.gpu_uuids, phase="queue_validation")
                        certified_ids = {lane_id for lane_id in certified_ids
                                         if self.lanes[lane_id].allows(client_scope)}
                        if not certified_ids:
                            raise WarmPreflightRequired(phase="queue_validation",
                                causes=("gpu_policy_mismatch",))
                    except WarmPreflightRequired as exc:
                        # Keep exact retained ownership until its ordinary TTL.
                        # A different certificate cannot widen an admitted job.
                        if waiter.logical_request_id:
                            waiter.attached = False
                            waiter.phase = "detached"
                            waiter.resume_deadline = time.monotonic() + POOL_RESUME_TTL
                            raise WarmPreflightRequired(admission_retained=True,
                                request_id=waiter.request_id,
                                logical_request_id=waiter.logical_request_id,
                                phase=exc.warm_preflight_phase, causes=exc.warm_preflight_causes) from exc
                        self._remove_waiter_locked(waiter)
                        raise
                waiter_index = next(
                    (index for index, item in enumerate(self.waiters)
                     if item is waiter),
                    -1,
                )
                next_for_model = waiter_index >= 0 and not any(
                    item.model == model for item in self.waiters[:waiter_index]
                )
                if not self.draining and next_for_model:
                    lane = self._select_lane_locked(model, True, client_scope, certified_ids, waiter.cpu_offload)
                    if lane is not None:
                        waiter.phase = "generating"
                        retained_for_admission = waiter.retained_request
                        self.waiters.pop(waiter_index)
                        self._release_waiter_retention_locked(waiter)
                        queue_ms = max(
                            0, int((time.monotonic() - enqueued_at) * 1000)
                        )
                        self.queue_admitted_total += 1
                        self.queue_wait_ms_total += queue_ms
                        self.queue_wait_ms_max = max(
                            self.queue_wait_ms_max, queue_ms
                        )
                        active = self._register_active_request_locked(
                            lane,
                            waiter.request_id,
                            waiter.logical_request_id,
                        )
                        self.active_request_records[waiter.request_id].warm_admission = waiter.warm_admission
                        self.active_request_records[waiter.request_id].cpu_offload = waiter.cpu_offload
                        if waiter.logical_request_id:
                            self.logical_in_flight[waiter.logical_request_id] = (
                                waiter.request_fingerprint,
                                waiter.request_id,
                                waiter.queue_ticket,
                            )
                        self.cv.notify_all()
                        return Admission(
                            lane, waiter.request_id,
                            waiter.logical_request_id,
                            waiter.request_fingerprint,
                            queue_ms,
                            waiter.initial_position,
                            waiter.queue_ticket,
                            retained_for_admission,
                            waiter.warm_admission, active_request=active, cpu_offload=waiter.cpu_offload,
                        )
                self.cv.wait(min(remaining, 0.25))

    def _disjoint_cold_waiter_locked(self, now: float) -> tuple[QueuedRequest, tuple[str, ...]] | None:
        self._prune_stale_waiters_locked(now)
        if (self.draining or self._global_transition_lease_locked()
                or now >= self.reconcile_retry_at):
            return None
        head = next((item for item in self.waiters if item.warm_admission is None), None)
        if head is None or head.last_reason_code not in (
            "gpu_peer_group_wait", "gpu_peer_group_reserved", "lease_transition",
            "lane_capacity_wait", "reclaimable_placement_wait",
        ):
            return None
        self._prune_dead_lanes_locked()
        managed_count = sum(lane.kind == "managed" for lane in self.lanes.values())
        protected: set[str] = set()
        seen_models: set[str] = set()
        for item in self.waiters:
            try:
                allowed = self._capacity_scope_locked(item.model, item.gpu_uuids, "auto",
                    str((item.client or {}).get("key") or ""))
            except PermanentCapacityError:
                return None  # The ordinary owner resolves changed authority.
            order = (list(allowed) if allowed is not None else list(dict.fromkeys(
                MODEL_GPU_PREFERENCES.get(item.model, []) + list(SELECTED_GPUS))))
            effective = tuple(gpu for gpu in order if gpu in SELECTED_GPUS)
            disjoint = tuple(gpu for gpu in effective if gpu not in protected)
            first_for_model = item.model not in seen_models
            if (item is not head and first_for_model and disjoint
                    and item.warm_admission is None and not item.cpu_offload
                    and now >= item.cold_capacity_retry_at
                    and (not item.attached or item.connected())
                    and managed_count + len(seen_models) + 1 <= POOL_MAX_SERVERS):
                return item, disjoint
            seen_models.add(item.model)
            protected.update(effective)
        return None

    def _reconcile_disjoint_cold_waiter(self) -> None:
        """Use spare GPUs while a different model's admission is blocked.

        Every older request retains its effective GPU scope and one server
        slot. A speculative attempt cannot evict a lane, widen an allowlist,
        enable CPU offload, or pass an older request for the same model.
        """
        with self.cv:
            candidate = self._disjoint_cold_waiter_locked(time.monotonic())
            if candidate is None:
                return
            selected, scope = candidate
            selected.cold_capacity_retry_at = time.monotonic() + 2.0
            selected.phase = "starting-server"
            self.cv.notify_all()
        # Never wait behind a lease transition just to do speculative work.
        if not self.transition.acquire(blocking=False):
            with self.cv:
                if selected in self.waiters:
                    selected.phase = "queued"
            return
        previous_deadline = getattr(CAPACITY_IO_DEADLINE, "deadline", None)
        CAPACITY_IO_DEADLINE.deadline = time.monotonic() + 5.0
        try:
            # Original scope, logical fingerprint and ticket stay unchanged.
            # This narrower scope applies only to non-reclaiming cold placement.
            self.ensure_capacity(selected.model, 1, selected.request_path, scope,
                                 triggered_by=selected.client, allow_reclaim=False,
                                 cpu_offload=False)
        except (CapacityError, OSError, RuntimeError, TimeoutError, ValueError, TypeError) as exc:
            with self.cv:
                if selected in self.waiters:
                    selected.phase = "queued"
                    selected.cold_capacity_retry_at = time.monotonic() + 30.0
                self.cv.notify_all()
            LOG.debug("speculative cold placement deferred model=%s: %s", selected.model, exc)
            # The ordinary turn owns any terminal failure. A failed speculative
            # load keeps existing process/VRAM cleanup guards and backs off.
        else:
            with self.cv:
                if selected in self.waiters:
                    selected.phase = "ready"
                self.cv.notify_all()
        finally:
            if previous_deadline is None:
                del CAPACITY_IO_DEADLINE.deadline
            else:
                CAPACITY_IO_DEADLINE.deadline = previous_deadline
            self.transition.release()

    def capacity_reconciler(self) -> None:
        """One scheduler owns managed-lane creation and replacement."""
        while not self.stopping.is_set():
            self._reconcile_disjoint_cold_waiter()
            with self.cv:
                self._prune_stale_waiters_locked()
                if (self.draining or self._global_transition_lease_locked()
                        or not self.waiters):
                    self.cv.wait(0.5)
                    continue
                # Conditional warm requests never authorize capacity creation.
                # Unbound requests still use the ordinary scheduler unchanged.
                waiter = next((item for item in self.waiters if item.warm_admission is None), None)
                if waiter is None:
                    self.cv.wait(0.25)
                    continue
                try:
                    client_scope = self._client_gpu_constraint_locked(
                        str((waiter.client or {}).get("key") or ""), waiter.gpu_uuids)
                except PermanentCapacityError as exc:
                    self._reject_client_waiter_locked(waiter, exc)
                    continue
                blocked_gpus = self._ollama_blocked_gpus_locked()
                selected_gpus = set(client_scope or SELECTED_GPUS)
                if selected_gpus and selected_gpus.issubset(blocked_gpus):
                    # This state changes only when the lease changes. Repeated
                    # ensure_capacity calls cannot create a lane and previously
                    # retried every two seconds, adding noise and CPU churn
                    # while clients accumulated. Park on the lease-state CV;
                    # waiter deadlines/disconnects and lease transitions notify.
                    waiter.phase = "waiting-lease"
                    waiter.last_error = (
                        "all selected GPUs are reserved by pending or revoking leases"
                    )
                    waiter.last_reason_code = "lease_transition"
                    self.reconcile_retry_at = time.monotonic() + 5.0
                    self.reconcile_last_error = waiter.last_error
                    self.reconciling_model = None
                    # Already queued cold work must get a scheduling pass even
                    # when no new request/lease notification arrives.
                    bypass_ready = self._disjoint_cold_waiter_locked(time.monotonic()) is not None
                    self.cv.wait(0.01 if bypass_ready else 5.0)
                    continue
                if waiter.phase == "waiting-lease":
                    # A lease notification can make the head runnable before
                    # its polling deadline; give its original owner priority.
                    self.reconcile_retry_at = 0.0
                model = waiter.model
                placement_scope = self._policy_constraint_locked(model, client_scope)
                if self._model_cancellation_in_progress_locked(model):
                    waiter.phase = "waiting-request-cancellation"
                    waiter.last_error = (
                        "an expired request is stopping its managed lane"
                    )
                    waiter.last_reason_code = (
                        "request_cancellation_in_progress"
                    )
                    self.reconcile_retry_at = 0.0
                    self.reconcile_last_error = waiter.last_error
                    self.reconciling_model = None
                    self.cv.wait()
                    continue
                matching = [lane for lane in self.lanes.values()
                            if lane.kind == "managed" and lane.model == model
                            and lane.context_profile_matches()
                            and lane.can_serve(placement_scope, waiter.cpu_offload)
                            and not set(lane.scope).intersection(blocked_gpus)
                            and not lane.retiring and not lane.loading]
                if any(lane.in_flight < lane.parallel for lane in matching):
                    waiter.phase = "ready"
                    self.cv.notify_all()
                    self.cv.wait(0.05)
                    continue
                if matching and all(lane.cpu_offload for lane in matching):
                    # One exclusive scope has a fixed number of slots. More
                    # callers queue for its release instead of requesting an
                    # impossible second copy and permanently failing admission.
                    waiter.phase = "queued"
                    self.cv.wait(0.25)
                    continue
                desired_parallel = sum(lane.parallel for lane in matching) + 1
                maximum = POOL_MAX_SERVERS * POOL_INSTANCE_PARALLEL
                if desired_parallel > maximum:
                    waiter.phase = "queued"
                    self.cv.wait(0.25)
                    continue
                now = time.monotonic()
                if now < self.reconcile_retry_at:
                    self.cv.wait(min(0.5, self.reconcile_retry_at - now))
                    continue
                waiter.phase = "starting-server"
                self.reconciling_model = model
                self.cv.notify_all()
            try:
                self.ensure_capacity(
                    model,
                    desired_parallel,
                    waiter.request_path,
                    waiter.gpu_uuids,
                    triggered_by=waiter.client,
                    allow_reclaim=not (
                        waiter.workload_class == "background"
                        and waiter.queue_policy == "yield"
                    ),
                    cpu_offload=waiter.cpu_offload,
                )
                with self.cv:
                    self.reconcile_retry_at = 0.0
                    self.reconcile_last_error = ""
                    self.reconciling_model = None
                    if waiter in self.waiters:
                        waiter.phase = "ready"
                    self.cv.notify_all()
            except (PermanentCapacityError, BackgroundCapacityDeferred) as exc:
                message = str(exc)
                with self.cv:
                    if waiter in self.waiters:
                        waiter.phase = "failed"
                        waiter.last_error = message
                        waiter.last_reason_code = exc.reason_code
                        waiter.terminal_error = message
                        waiter.terminal_status = exc.status
                        waiter.terminal_reason_code = exc.reason_code
                        waiter.terminal_retryable = exc.retryable
                        waiter.terminal_retry_after = exc.retry_after
                        self._record_logical_tombstone_locked(
                            waiter.logical_request_id,
                            waiter.request_id,
                            exc.reason_code,
                        )
                        self.queue_rejected_total += 1
                        self._remove_waiter_locked(waiter)
                    self.reconcile_retry_at = 0.0
                    self.reconcile_last_error = ""
                    self.reconciling_model = None
                    self.cv.notify_all()
                if isinstance(exc, BackgroundCapacityDeferred):
                    LOG.info("background capacity deferred model=%s: %s", model, message)
                else:
                    LOG.error("managed capacity rejected model=%s: %s", model, message)
            except (CapacityError, OSError, RuntimeError, TimeoutError) as exc:
                message = str(exc)
                reason_code = (
                    exc.reason_code
                    if isinstance(exc, CapacityError)
                    else "backend_start_failed"
                )
                with self.cv:
                    if waiter in self.waiters:
                        waiter.phase = "queued"
                        waiter.last_error = message
                        waiter.last_reason_code = reason_code
                    changed = message != self.reconcile_last_error
                    self.reconcile_last_error = message
                    self.reconcile_retry_at = time.monotonic() + 2.0
                    self.reconciling_model = None
                    self.cv.notify_all()
                if changed:
                    LOG.warning("managed capacity queued model=%s: %s", model, message)

    def prepare_managed_body(self, lane: Lane, path: str, body: bytes,
                             warm_admission: str | None = None) -> bytes:
        if lane.kind != "managed" or not body:
            return body
        if not lane.context_profile_matches():
            raise CapacityError("model context identity changed before backend dispatch", 503,
                                "model_context_identity_changed")
        if (path in ("/v1/chat/completions", "/v1/completions", "/v1/responses")
                and lane.context_profile and not lane.openai_context_compatible):
            raise PermanentCapacityError(
                "OpenAI context cannot override the model's explicit num_ctx; "
                "use the native API or align an exact model wrapper with the admitted context",
                422, "model_context_openai_mismatch")
        if path not in NATIVE_MODEL_PATHS:
            return body
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            return body
        if not isinstance(payload, dict):
            return body
        prepared_context = lane.context_profile["context_length"] if lane.context_profile else None
        if warm_admission is not None and lane.context_profile is None:
            # The original body was checked before clamping/registration.
            # Replace only the legacy unprofiled MAX_CONTEXT projection with
            # this verified embedding runner's exact native allocation.
            proof = json.loads(warm_admission)
            native = (lane.warm_runtime_identity or {}).get("native_ps", {})
            try:
                installed_digest = self._warm_embedding_digest(lane.model)
            except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
                raise WarmPreflightRequired(admission_retained=True, phase="body_preparation",
                                           causes=("embedding_metadata_unavailable",)) from exc
            if (native.get("digest") != proof["model_digest"]
                    or native.get("context_length") != proof["context_length"]
                    or installed_digest != proof["model_digest"]
                    or (HARD_MAX_CONTEXT > 0 and proof["context_length"] > HARD_MAX_CONTEXT)):
                raise WarmPreflightRequired(admission_retained=True, phase="body_preparation",
                                           causes=("embedding_identity_mismatch",))
            prepared_context = proof["context_length"]
        if prepared_context is not None:
            # A queued/retained request may have been prepared before a tag's
            # context identity changed. The selected lane owns the final
            # allocation contract; never reload it from a stale request cap.
            options = payload.setdefault("options", {})
            if not isinstance(options, dict):
                raise PermanentCapacityError("model options must be a JSON object", 400,
                                             "invalid_model_options")
            options["num_ctx"] = prepared_context
        if payload.get("keep_alive") == 0:
            self._require_safe_gpu_transition(lane.protected_scope, lane.lane_id)
            # Do not leave an empty backend that can reload implicitly on its
            # next request. Retire it through the guarded process lifecycle.
            with self.cv:
                lane.retiring = True
                self.cv.notify_all()
            LOG.info("managed explicit unload id=%s gpu=%s model=%s",
                     lane.lane_id, lane.gpu_uuid, lane.model)
        else:
            payload["keep_alive"] = -1
        return json.dumps(payload, separators=(",", ":")).encode()

    def proxy_exit(
        self,
        admission: Admission,
        model: str,
        succeeded: bool,
        client_key: str = "",
    ) -> None:
        retired = None
        expired_lane = None
        with self.cv:
            active = self._owned_active_request_locked(admission)
            if admission.warm_admission is not None and active is None:
                return
            lane = active.lane if active is not None else admission.lane
            if (
                active is not None
                and active.cancel_requested_at is not None
                and active.backend_started
                and not active.backend_completed
                and active.lane.kind == "managed"
                and not active.lane_stopped
            ):
                active.phase = "backend_terminal"
                active.expires_at = time.monotonic() + REQUEST_CANCEL_GRACE
                if self._schedule_cancelled_lane_stop_locked(active.lane):
                    expired_lane = active.lane
                else:
                    self.cv.notify_all()
            else:
                retired = self._release_active_request_locked(
                    admission.request_id, model, succeeded, client_key
                )
                if self._schedule_cancelled_lane_stop_locked(lane):
                    expired_lane = lane
        if expired_lane is not None:
            threading.Thread(
                target=self._stop_expired_lane,
                args=(expired_lane,),
                daemon=True,
            ).start()
        if retired is not None:
            threading.Thread(
                target=self._stop_lanes, args=([retired], "operator GPU policy"),
                daemon=True,
            ).start()

    def begin_drain(self, reason: str) -> None:
        deadline = time.monotonic() + DRAIN_TIMEOUT
        with self.cv:
            self.draining = True
            self.last_reason = reason
            self.cv.notify_all()
            while self.active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.draining = False
                    self.cv.notify_all()
                    raise TimeoutError(f"{self.active_requests} Ollama request(s) did not drain")
                self.cv.wait(min(remaining, 1.0))

    def end_drain(self) -> None:
        with self.cv:
            self.draining = False
            self.cv.notify_all()

    def pending_lease(self) -> bool:
        return any(
            lease.state in ("pending", "revoking")
            for lease in self.leases.values()
        )

    def revoking_lease(self) -> Lease | None:
        return next(
            (lease for lease in self.leases.values()
             if lease.state == "revoking"),
            None,
        )

    def blocking_revoking_lease(self) -> Lease | None:
        return next(
            (lease for lease in self.leases.values()
             if lease.state == "revoking" and not lease.gpu_uuids),
            None,
        )

    def _revoke_locked(self, lease: Lease, reason: str) -> None:
        if lease.state == "revoking":
            return
        lease.state = "revoking"
        # Restart the transition clock on entry to "revoking". It previously
        # kept the pending timestamp, so the revoke deadline below measured
        # from lease creation and a lease revoked late never aged correctly.
        lease.transition_started_at = time.time()
        self.last_reason = reason
        message = (
            f"GPU lease transition revoked for {lease.owner}: {reason}; "
            "reserved GPUs remain unavailable until the owner frees CUDA memory and releases"
        )
        if not lease.gpu_uuids:
            self.draining = True
            message = (
                f"GPU lease transition revoked for {lease.owner}: {reason}; "
                "the broker remains drained until the owner frees CUDA memory and releases"
            )
            for waiter in list(self.waiters):
                waiter.phase = "failed"
                waiter.last_error = message
                waiter.last_reason_code = "lease_transition"
                waiter.terminal_error = message
                waiter.terminal_status = 503
                waiter.terminal_reason_code = "lease_transition"
                waiter.terminal_retryable = True
                waiter.terminal_retry_after = 2
                self.queue_rejected_total += 1
                self._remove_waiter_locked(waiter)
        self._persist_leases_locked()
        self.cv.notify_all()
        LOG.error(message)

    def _abandon_locked(self, lease: Lease, reason: str) -> None:
        """Drop a revoked lease whose owner never came back to release it.

        A reservation is a scheduling hint layered on live GPU telemetry. It
        cannot protect memory that a dead owner has already freed, and while
        it survives it blocks every Ollama admission on its scope forever.
        Dropping it restores admission only when the owner has exited.
        Remaining foreign CUDA becomes unregistered and quarantines the GPU,
        so an abandoned lease cannot authorize churn around a live owner.
        """
        self.leases.pop(lease.token, None)
        self.last_reason = reason
        message = (
            f"GPU lease abandoned for {lease.owner}: {reason}; "
            "scope released; remaining unregistered CUDA still blocks model transitions"
        )
        if not lease.gpu_uuids and self._global_transition_lease_locked() is None:
            # This lease held the host-wide drain. No other global transition
            # remains, so admission must resume with it.
            self.draining = False
        self._persist_leases_locked()
        self.cv.notify_all()
        LOG.error(message)

    def _abandon_if_revoke_deadline_expired_locked(
        self, lease: Lease, now: float | None = None,
    ) -> bool:
        if lease.state != "revoking" or REVOKE_TIMEOUT <= 0:
            return False
        current = time.time() if now is None else now
        if lease.ttl > 0 and current - lease.heartbeat_at <= lease.ttl:
            # The owner is still heartbeating, so it is alive and working
            # through its release. Revocation waits for it.
            return False
        if current - lease.transition_started_at < REVOKE_TIMEOUT:
            return False
        if lease.exclusive:
            try:
                require_exclusive_cuda_free(lease.gpu_uuids)
            except (OSError, RuntimeError):
                return False
        self._abandon_locked(
            lease,
            f"revoke deadline of {REVOKE_TIMEOUT:g}s expired "
            "with no owner heartbeat",
        )
        return True

    def _revoke_if_pending_deadline_expired_locked(
        self, lease: Lease, now: float | None = None,
    ) -> bool:
        current = time.time() if now is None else now
        if (lease.state == "pending" and PENDING_TIMEOUT > 0
                and current - lease.transition_started_at >= PENDING_TIMEOUT):
            self._revoke_locked(
                lease,
                f"pending deadline of {PENDING_TIMEOUT:g}s expired",
            )
            return True
        return lease.state == "revoking"

    def _handoff_summaries_locked(self) -> list[dict[str, Any]]:
        now = time.time()
        for key, request in list(self.lease_handoff_requests.items()):
            if request["state"] == "waiting" and request["expires_at"] <= now:
                self.lease_handoff_requests.pop(key, None)
        return [dict(request) for request in self.lease_handoff_requests.values()]

    def _handoff_scopes_locked(self, ignore_id: str = "") -> set[str]:
        return {gpu for request in self._handoff_summaries_locked()
                if request["id"] != ignore_id
                for gpu in request["gpu_uuids"] + [
                    move["to_gpu_uuid"] for move in request.get("relocations", [])]}

    @staticmethod
    def _lease_ref(lease: Lease) -> str:
        # Public correlation must not disclose a lease's bearer token.
        return hashlib.sha256(lease.token.encode()).hexdigest()[:24]

    def _flexible_gpu_locked(self, candidates: list[str], requested_mib: int,
                             unavailable: set[str] | None = None) -> str:
        blocked = (self._handoff_scopes_locked()
                   | self._evacuation_recovery_scopes_locked()
                   | self._peer_reserved_gpus_locked()
                   | self._unregistered_gpus_locked(refresh=True)
                   | (unavailable or set()))
        for gpu in candidates:
            if gpu in SELECTED_GPUS and gpu not in blocked:
                try:
                    self._plan_lease_gpus(requested_mib, [gpu])
                    return gpu
                except RuntimeError:
                    continue
        raise RuntimeError("no eligible destination has verified free capacity; existing owners remain in place")

    def _wait_for_cooperative_handoff(
        self, scope: list[str], owner: str, justification: str,
        expected_duration_seconds: int, requested_mib: int, request_alive: Any = None,
        priority: int = 0, exclusive: bool = False,
    ) -> str:
        with self.cv:
            scope_set = set(scope)
            if not scope or any(gpu not in SELECTED_GPUS for gpu in scope):
                return ""  # Ordinary admission owns invalid/unscoped rejection.
            if len(scope) == 1 and requested_mib > 0 and not exclusive:
                try:
                    self._plan_lease_gpus(requested_mib, scope)
                    return ""  # Fitting leases share; no relocation is needed.
                except RuntimeError:
                    pass
            if scope_set.intersection(self._handoff_scopes_locked()):
                raise RuntimeError("another cooperative handoff already targets this GPU scope")
            live = list(self.leases.values())
            conflicts = [lease for lease in live
                         if not lease.gpu_uuids or scope_set.intersection(lease.gpu_uuids)]
            if self.pending_lease() or not conflicts or any(
                    lease.state != "active" or len(lease.gpu_uuids) != 1
                    or (lease.placement == "flexible" and lease.priority >= priority)
                    or (lease.placement != "flexible" and not lease.yield_on_request)
                    for lease in conflicts):
                return ""  # Non-opted-in owners retain their existing contract.
            if any(lease.owner == owner for lease in live):
                raise RuntimeError("an existing owner cannot wait for a cooperative handoff")
            inventory = {device["uuid"]: device for device in gpu_snapshot()}
            if (any(gpu not in inventory for gpu in scope)
                    or sum(int(inventory[gpu].get("total_mib") or 0) for gpu in scope) < requested_mib):
                raise ValueError("handoff request exceeds verified GPU hardware capacity")
            if len(self.lease_handoff_requests) >= 32:
                raise RuntimeError("cooperative handoff request capacity reached")
            moves = []
            destinations = set(scope)
            for lease in conflicts:
                if lease.placement != "flexible":
                    continue
                target = self._flexible_gpu_locked(
                    lease.eligible_gpu_uuids, lease.requested_mib, destinations)
                destinations.add(target)
                moves.append({"lease_ref": self._lease_ref(lease), "owner": lease.owner,
                              "from_gpu_uuids": list(lease.gpu_uuids),
                              "to_gpu_uuid": target, "priority": lease.priority})
            request_id = "handoff_" + secrets.token_urlsafe(18)
            now = time.time()
            self.lease_handoff_requests[request_id] = {
                "id": request_id, "state": "waiting", "owner": owner,
                "justification": justification, "gpu_uuids": list(scope),
                "requested_mib": requested_mib,
                "priority": priority, "relocations": moves,
                "expected_duration_seconds": expected_duration_seconds,
                "created_at": now, "expires_at": now + HANDOFF_TIMEOUT,
                "incumbents": [{"owner": lease.owner, "gpu_uuids": list(lease.gpu_uuids)}
                               for lease in conflicts],
            }
            self.cv.notify_all()
            deadline = time.monotonic() + HANDOFF_TIMEOUT
            try:
                # Never hold self.transition here: release needs that lock.
                moving_refs = {move["lease_ref"] for move in moves}
                while any(not lease.gpu_uuids or scope_set.intersection(lease.gpu_uuids)
                          or (self._lease_ref(lease) in moving_refs and lease.state != "active")
                          for lease in self.leases.values()):
                    if self.stopping.is_set() or (request_alive is not None and not request_alive()):
                        raise RuntimeError("cooperative handoff requester disconnected or broker stopping")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or request_id not in self.lease_handoff_requests:
                        raise TimeoutError("cooperative handoff timed out; incumbent ownership is retained")
                    self.cv.wait(min(remaining, 0.5))
                if (self.stopping.is_set() or time.monotonic() >= deadline
                        or request_id not in self.lease_handoff_requests
                        or (request_alive is not None and not request_alive())):
                    raise TimeoutError("cooperative handoff expired before successor admission")
                return request_id
            except Exception:
                self.lease_handoff_requests.pop(request_id, None)
                self.cv.notify_all()
                raise

    def relocate(self, token: str, handoff_id: str) -> dict[str, Any]:
        """Commit a cooperative owner's drained single-GPU lease to its reserved destination."""
        require_gpu_health(refresh=True)
        with self.transition:
            with self.cv:
                lease = self.leases.get(token)
                request = self.lease_handoff_requests.get(handoff_id)
                if (lease is None or lease.state != "active" or lease.placement != "flexible"
                        or request is None or request["expires_at"] <= time.time()):
                    raise RuntimeError("relocation requires an active flexible lease and a live handoff")
                move = next((m for m in request.get("relocations", [])
                             if m["lease_ref"] == self._lease_ref(lease)
                             and m["from_gpu_uuids"] == lease.gpu_uuids), None)
                if move is None or move["to_gpu_uuid"] not in lease.eligible_gpu_uuids:
                    raise RuntimeError("handoff has no matching reserved destination for this owner")
                source = list(lease.gpu_uuids)
                target = move["to_gpu_uuid"]
            # The owner must stop its CUDA workers first. This never kills them.
            ignored = {key for other in self.leases.values() if other.token != token
                       for key in other.cuda_processes or []}
            wait_for_foreign_settle(lease.foreign_baseline, set(source), ignored_processes=ignored)
            def verify_source_exit() -> None:
                # Aggregate release tolerance is useful for ordinary accounting
                # drift, but cannot authorize a live small CUDA context to move.
                current = foreign_gpu_usage(strict=True)
                if any(key not in ignored and key.rsplit("@", 1)[-1] in source
                       and (key not in lease.foreign_baseline
                            or used > lease.foreign_baseline[key])
                       for key, used in current.items()):
                    raise RuntimeError("source CUDA processes remain above their pre-lease baseline")
            verify_source_exit()
            self.begin_drain(f"cooperative relocation by {lease.owner}")
            try:
                _, devices = self._plan_lease_gpus(lease.requested_mib, [target], handoff_id)
                baseline = foreign_gpu_usage()
                require_gpu_health(refresh=True)
                verify_source_exit()
                with self.cv:
                    if target in (self._unregistered_gpus_locked(refresh=True)
                                  | self._peer_reserved_gpus_locked()):
                        raise RuntimeError("relocation destination acquired unregistered or peer CUDA work")
                    if (self.leases.get(token) is not lease
                            or self.lease_handoff_requests.get(handoff_id) is not request
                            or request["expires_at"] <= time.time()
                            or self._revoke_if_pending_deadline_expired_locked(lease)):
                        raise RuntimeError("relocation ownership or handoff changed while draining")
                    previous = (lease.gpu_uuids, lease.state, lease.foreign_baseline,
                                lease.transition_started_at, lease.heartbeat_at, lease.cuda_processes,
                                lease.capacity_mib)
                    lease.gpu_uuids = [target]
                    lease.state = "pending"
                    lease.foreign_baseline = baseline
                    lease.cuda_processes = None
                    lease.capacity_mib = int(devices[0].get("lease_capacity_mib") or 0)
                    lease.transition_started_at = lease.heartbeat_at = time.time()
                    try:
                        self._persist_leases_locked()
                    except Exception:
                        (lease.gpu_uuids, lease.state, lease.foreign_baseline,
                         lease.transition_started_at, lease.heartbeat_at, lease.cuda_processes,
                         lease.capacity_mib) = previous
                        raise
                    self.cv.notify_all()
                return {"ok": True, "lease": asdict(lease), "previous_gpu_uuids": source,
                        "gpus": devices, "public_lease": lease_public_summary(lease)}
            finally:
                self.end_drain()

    def acquire(
        self, owner: str, requested_mib: int, ttl: int,
        requested_gpu_uuids: list[str] | None = None,
        justification: str = "",
        expected_duration_seconds: int = 0,
        *, priority: int | None = None, evacuation_id: str = "",
        _evacuation_commit_id: str = "",
        yield_on_request: bool = False, request_alive: Any = None,
        placement: str = "fixed", exclusive: bool = False,
    ) -> dict[str, Any]:
        owner = owner.strip()
        justification = justification.strip()
        if not owner or owner == "unknown":
            raise ValueError("lease acquisition requires a specific non-empty owner")
        if len(owner) > 128:
            raise ValueError("lease owner must be at most 128 characters")
        if len(justification) > 512:
            raise ValueError("lease justification must be at most 512 characters")
        if expected_duration_seconds < 0:
            raise ValueError(
                "expected_duration_seconds must be non-negative (zero means unknown)"
            )
        if type(exclusive) is not bool:
            raise ValueError("exclusive must be a Boolean")
        if type(yield_on_request) is not bool:
            raise ValueError("yield_on_request must be a Boolean")
        if placement not in ("fixed", "flexible"):
            raise ValueError("placement must be fixed or flexible")
        lease_priority = 0 if priority is None else priority
        if type(lease_priority) is not int or not -1000 <= lease_priority <= 1000:
            raise ValueError("priority must be an integer between -1000 and 1000")
        requested_scope = list(dict.fromkeys(requested_gpu_uuids or OWNER_GPU_SCOPES.get(owner, [])))
        eligible = []
        if exclusive and (not requested_scope or placement != "fixed" or yield_on_request or evacuation_id):
            raise ValueError("exclusive requires an explicit fixed GPU scope without yield or evacuation")
        if placement == "flexible":
            if evacuation_id or yield_on_request:
                raise ValueError("flexible placement cannot combine cache evacuation or legacy yield")
            eligible = requested_scope or list(SELECTED_GPUS)
            if any(gpu not in SELECTED_GPUS for gpu in eligible):
                raise ValueError("flexible candidates must be selected GPU UUIDs")
            require_gpu_health(refresh=True)
            with self.cv:
                requested_scope = [self._flexible_gpu_locked(eligible, requested_mib)]
        if yield_on_request and (len(requested_scope) != 1 or evacuation_id):
            raise ValueError("cooperative yield requires exactly one GPU and no cache evacuation")
        if evacuation_id:
            return self._lease_evacuation("acquire", {
                "owner": owner, "requested_mib": requested_mib, "ttl": ttl,
                "requested_gpu_uuids": requested_gpu_uuids or OWNER_GPU_SCOPES.get(owner, []),
                "justification": justification, "expected_duration_seconds": expected_duration_seconds,
            }, priority, evacuation_id)
        require_gpu_health(refresh=True)
        handoff_id = self._wait_for_cooperative_handoff(
            requested_scope, owner, justification, expected_duration_seconds, requested_mib, request_alive,
            lease_priority, exclusive,
        )
        try:
            if handoff_id:
                require_gpu_health(refresh=True)
                if request_alive is not None and not request_alive():
                    raise RuntimeError("cooperative requester disconnected before admission")
            return self._acquire_once(owner, requested_mib, ttl, requested_scope,
                                      justification, expected_duration_seconds,
                                      _evacuation_commit_id, yield_on_request, handoff_id, request_alive,
                                      placement, lease_priority, eligible, exclusive)
        finally:
            if handoff_id:
                with self.cv:
                    self.lease_handoff_requests.pop(handoff_id, None)
                    self.cv.notify_all()

    def _acquire_once(
        self, owner: str, requested_mib: int, ttl: int, requested_scope: list[str],
        justification: str, expected_duration_seconds: int,
        _evacuation_commit_id: str, yield_on_request: bool, handoff_id: str, request_alive: Any,
        placement: str = "fixed", priority: int = 0, eligible_gpu_uuids: list[str] | None = None,
        exclusive: bool = False,
    ) -> dict[str, Any]:
        with self.transition:
            with self.cv:
                reserved = self._handoff_scopes_locked(handoff_id)
                if reserved and (not requested_scope or reserved.intersection(requested_scope)):
                    raise RuntimeError("GPU scope is reserved for a cooperative handoff requester")
                if handoff_id:
                    request = self.lease_handoff_requests.get(handoff_id)
                    if request is None or request["expires_at"] <= time.time():
                        raise TimeoutError("cooperative handoff expired before admission")
                    request["state"] = "admitting"
                live = [lease for lease in self.leases.values()
                        if lease.state in ("pending", "active", "revoking")]
                if any(not lease.gpu_uuids for lease in live) or (not requested_scope and live):
                    raise RuntimeError("unscoped leases require exclusive host-wide ownership")
            drain_host = not requested_scope or bool(running_models(require_available=True))
            if drain_host:
                self.begin_drain(f"lease acquire by {owner}")
            try:
                stopped: list[str] = []
                # The system lane is not GPU-scoped, so unload any model it
                # holds before measuring a scoped reservation. Managed lanes
                # have exact GPU UUIDs and may remain resident when the new
                # external allocation already fits around them.
                unloaded = self._unload_base_models()
                if requested_scope:
                    gpu_uuids, devices = self._plan_lease_gpus(
                        requested_mib, requested_scope, handoff_id,
                        exclusive=exclusive, check_capacity=False,
                    )
                    aggregate_free = sum(
                        int(device["free_mib"]) for device in devices
                    )
                    with self.cv:
                        overlaps_group = bool(set(gpu_uuids).intersection(
                            self._peer_reserved_gpus_locked()))
                    if lease_requires_exclusive_gpus(gpu_uuids, exclusive) or overlaps_group:
                        # Retire lanes before the owner starts peer traffic;
                        # the scope stays blocked for the lease's lifetime.
                        self._wait_scope_idle(set(gpu_uuids))
                        stopped = self.stop_pool_lanes(
                            "exclusive GPU lease acquire",
                            set(gpu_uuids),
                        )
                    elif requested_mib > 0 and requested_mib > aggregate_free:
                        upper_bound = self._lease_capacity_devices(devices, reclaim_lanes=True)
                        if requested_mib > sum(int(d["free_mib"]) for d in upper_bound):
                            raise RuntimeError("insufficient VRAM after existing external lease reservations")
                        self._wait_scope_idle(set(gpu_uuids))
                        stopped = self.stop_pool_lanes(
                            "lease acquire capacity reclamation",
                            set(gpu_uuids),
                        )
                    # Refresh telemetry after any base unload or managed-lane
                    # retirement and enforce the reservation against the
                    # resulting live capacity.
                    gpu_uuids, devices = self._plan_lease_gpus(
                        requested_mib, requested_scope, handoff_id, exclusive=exclusive,
                    )
                else:
                    stopped = self.stop_pool_lanes("lease acquire")
                    gpu_uuids, devices = self._plan_lease_gpus(
                        requested_mib, [],
                    )
                aggregate_free = sum(int(device["free_mib"]) for device in devices)
                if requested_mib > 0 and devices and requested_mib > aggregate_free:
                    raise RuntimeError(
                        f"requested {requested_mib} MiB but only {aggregate_free} MiB is free after Ollama unload"
                    )
                require_gpu_health(refresh=True)
                if handoff_id and request_alive is not None and not request_alive():
                    raise RuntimeError("cooperative requester disconnected before lease grant")
                now = time.time()
                if exclusive:
                    require_exclusive_cuda_free(gpu_uuids)
                token = "lease_" + secrets.token_urlsafe(24)
                lease = Lease(
                    token, owner, "pending", requested_mib, now, now, now, ttl,
                    foreign_gpu_usage(), gpu_uuids, justification,
                    now + expected_duration_seconds if expected_duration_seconds else 0.0,
                    _evacuation_commit_id,
                    yield_on_request,
                    placement, priority, list(eligible_gpu_uuids or []), exclusive,
                    capacity_mib=(int(devices[0].get("lease_capacity_mib") or 0)
                                  if len(gpu_uuids) == 1 else 0),
                )
                with self.cv:
                    if _evacuation_commit_id:
                        self._assert_evacuation_current(self.evacuations[_evacuation_commit_id])
                    self.leases[token] = lease
                    try:
                        self._persist_leases_locked()
                    except Exception:
                        self.leases.pop(token, None)
                        raise
                    if _evacuation_commit_id:
                        self._commit_cache_evacuation_locked(_evacuation_commit_id, lease)
                LOG.info("lease acquired owner=%s requested_mib=%s unloaded=%s", owner, requested_mib, unloaded)
                if gpu_uuids and drain_host:
                    self.end_drain()
                return {"ok": True, "lease": asdict(lease), "unloaded": unloaded,
                        "stopped_lanes": stopped,
                        "gpus": devices, "host_memory": host_memory_snapshot(),
                        "coordination_warning": LEASE_COORDINATION_WARNING,
                        "public_lease": lease_public_summary(lease)}
            except Exception:
                if drain_host:
                    self.end_drain()
                raise

    def ready(self, token: str, workload_pid: int | None = None,
              workload_pids: list[int] | None = None) -> dict[str, Any]:
        require_gpu_health(refresh=True)
        if workload_pids is not None and (not isinstance(workload_pids, list)
                                          or not workload_pids or len(workload_pids) > 256):
            raise ValueError("workload_pids must list live workload parents")
        roots = ([workload_pid] if workload_pid is not None else []) + (workload_pids or [])
        if any(type(pid) is not int or pid <= 0 or not Path(f"/proc/{pid}").exists() for pid in roots):
            raise ValueError("workload_pid must identify the live workload parent")
        with self.transition:
            usage = foreign_gpu_usage(strict=BACKEND_TYPE == "cuda")
            with self.cv:
                lease = self.leases.get(token)
                if lease is None:
                    raise KeyError("unknown lease")
                if self._revoke_if_pending_deadline_expired_locked(lease):
                    raise RuntimeError(
                        "lease transition was revoked; free CUDA memory and release the lease"
                    )
                if lease.state != "pending" and not (lease.state == "active" and roots):
                    raise RuntimeError("lease is not pending")
                peers = [other for other in self.leases.values() if other.token != token
                         and set(other.gpu_uuids).intersection(lease.gpu_uuids)]
                claimed = {key for other in peers for key in other.cuda_processes or []}
                candidates = [key for key, used in usage.items()
                    if key.rsplit("@", 1)[-1] in lease.gpu_uuids and key not in claimed
                    and (roots and any(workload_contains_pid(root, int(key.split('@')[0])) for root in roots)
                         or not roots and (key not in (lease.foreign_baseline or {})
                             or used > (lease.foreign_baseline or {}).get(key, 0)))]
                ambiguous = not roots and candidates and any(
                    other.state == "pending" or other.cuda_processes is None for other in peers)
                previous = lease.state, lease.heartbeat_at, lease.cuda_processes
                lease.cuda_processes = None if ambiguous else candidates
                lease.state = "active"
                lease.heartbeat_at = time.time()
                try:
                    self._persist_leases_locked()
                except Exception:
                    lease.state, lease.heartbeat_at, lease.cuda_processes = previous
                    raise
                self.cv.notify_all()
            devices = gpu_snapshot()
            if lease.gpu_uuids:
                self.end_drain()
            LOG.info("lease ready owner=%s", lease.owner)
            return {"ok": True, "lease": asdict(lease), "gpus": devices,
                    "host_memory": host_memory_snapshot()}

    def scope(self, token: str, requested_gpu_uuids: list[str]) -> dict[str, Any]:
        """Convert a live lease to a compatible, budgeted GPU scope.

        This transition does not stop the external workload. It is safe only
        when every foreign allocation that grew after acquire is contained in
        the requested scope. The anonymous watcher continues to protect every
        unreserved GPU from later growth.
        """
        require_gpu_health(refresh=True)
        with self.transition:
            requested = list(dict.fromkeys(requested_gpu_uuids))
            if not requested:
                raise RuntimeError("scope requires at least one GPU UUID")
            inventory = [
                device for device in gpu_snapshot()
                if not SELECTED_GPUS or device.get("uuid") in SELECTED_GPUS
            ]
            by_uuid = {
                str(device.get("uuid") or ""): device for device in inventory
            }
            unknown = [gpu_uuid for gpu_uuid in requested if gpu_uuid not in by_uuid]
            if unknown:
                raise RuntimeError(
                    f"requested GPU UUIDs are not selected and available: {unknown}"
                )
            current_foreign = foreign_gpu_usage()
            with self.cv:
                lease = self.leases.get(token)
                if lease is None:
                    raise KeyError("unknown lease")
                if lease.state not in ("pending", "active"):
                    raise RuntimeError("only a pending or active lease can change scope")
                if (lease.exclusive or lease.yield_on_request or lease.placement == "flexible") and requested != lease.gpu_uuids:
                    raise RuntimeError("cooperative owners must release before changing GPU scope")
                if set(requested).intersection(self._handoff_scopes_locked()):
                    raise RuntimeError("scope reserved for cooperative handoff")
                _, planned_devices = self._plan_lease_gpus(lease.requested_mib, requested,
                    exclusive=lease.exclusive, ignore_token=token,
                    check_capacity=set(requested) != set(lease.gpu_uuids))
                if any(lane.kind == "managed"
                       and set(requested).intersection(lane.protected_scope)
                       and (len(lane.scope) > 1 or lease_requires_exclusive_gpus(requested, lease.exclusive))
                       for lane in self.lanes.values()):
                    raise RuntimeError(
                        "live scope change would overlap managed peer traffic; "
                        "stop the external owner and acquire a new scoped lease"
                    )
                baseline = lease.foreign_baseline or {}
                outside_growth = {
                    key: used - baseline.get(key, 0)
                    for key, used in current_foreign.items()
                    if used > baseline.get(key, 0)
                    and key.rsplit("@", 1)[-1] not in requested
                }
                if outside_growth:
                    raise RuntimeError(
                        "cannot narrow live lease scope; foreign CUDA allocation "
                        f"grew outside the requested GPUs: {outside_growth}"
                    )
                aggregate_total = sum(
                    int(by_uuid[gpu_uuid].get("total_mib") or 0)
                    for gpu_uuid in requested
                )
                if lease.requested_mib > aggregate_total:
                    raise RuntimeError(
                        f"lease requests {lease.requested_mib} MiB but scoped GPUs "
                        f"provide only {aggregate_total} MiB"
                    )
                previous_scope = list(lease.gpu_uuids)
                previous = lease.gpu_uuids, lease.heartbeat_at, lease.capacity_mib
                lease.gpu_uuids = requested
                if set(requested) != set(previous_scope):
                    lease.capacity_mib = (int(planned_devices[0].get("lease_capacity_mib") or 0)
                                          if len(requested) == 1 else 0)
                lease.heartbeat_at = time.time()
                try:
                    self._persist_leases_locked()
                except Exception:
                    lease.gpu_uuids, lease.heartbeat_at, lease.capacity_mib = previous
                    raise
                state = lease.state
                result = asdict(lease)
            with self.cv:
                if self._global_transition_lease_locked() is None:
                    self.draining = False
                    self.cv.notify_all()
            stopped = []
            if lease_requires_exclusive_gpus(requested, lease.exclusive):
                # The scope is now blocked, so no new lane lands here. Retire
                # lanes already resident rather than leave them to idle out
                # at an unpredictable moment during peer traffic.
                stopped = self.stop_pool_lanes(
                    "exclusive multi-GPU lease scope", set(requested),
                )
            LOG.warning(
                "lease scope changed live owner=%s previous=%s current=%s",
                lease.owner, previous_scope, requested,
            )
            return {
                "ok": True,
                "lease": result,
                "stopped_lanes": stopped,
                "previous_gpu_uuids": previous_scope,
                "verified_foreign_growth": {
                    key: used - (lease.foreign_baseline or {}).get(key, 0)
                    for key, used in current_foreign.items()
                    if used > (lease.foreign_baseline or {}).get(key, 0)
                },
                "gpus": [by_uuid[gpu_uuid] for gpu_uuid in requested],
            }

    def prepare(self, token: str, *, priority: int | None = None,
                evacuation_id: str = "", _evacuation_commit_id: str = "",
                requested_mib: int | None = None) -> dict[str, Any]:
        if priority is not None and not evacuation_id:
            raise ValueError("priority requires an explicit evacuation ID")
        if requested_mib is not None and (type(requested_mib) is not int or requested_mib <= 0):
            raise ValueError("requested_mib must be a positive total VRAM budget")
        if evacuation_id:
            if requested_mib is not None:
                raise ValueError("resize the budget separately from cache evacuation")
            return self._lease_evacuation("prepare", {"token": token}, priority, evacuation_id)
        require_gpu_health(refresh=True)
        with self.transition:
            with self.cv:
                lease = self.leases.get(token)
                if lease is None:
                    raise KeyError("unknown lease")
                if lease.state != "active":
                    raise RuntimeError("only an active lease can prepare a resize")
                target = lease.requested_mib if requested_mib is None else requested_mib
                if target < lease.requested_mib:
                    raise ValueError("prepare grows the total budget; release before reducing it")
                scope = list(lease.gpu_uuids)
                if not scope and self.pending_lease():
                    raise RuntimeError("a lease transition is already pending")
            drain_host = not scope or bool(running_models(require_available=True))
            if drain_host:
                self.begin_drain(f"lease resize by {lease.owner}")
            try:
                stopped = []
                if scope:
                    _, compatible = self._plan_lease_gpus(target, scope, exclusive=lease.exclusive,
                        ignore_token=token, check_capacity=False)
                    current = self._lease_capacity_devices(compatible)
                    additional = target-lease.requested_mib
                    reclaim = (lease_requires_exclusive_gpus(scope, lease.exclusive)
                               or target == 0 or additional > sum(d["free_mib"] for d in current))
                    if reclaim:
                        upper = self._lease_capacity_devices(compatible, reclaim_lanes=True)
                        if additional > sum(d["free_mib"] for d in upper):
                            raise RuntimeError("insufficient VRAM after other lease reservations")
                        self._wait_scope_idle(set(scope))
                        stopped = self.stop_pool_lanes("lease prepare", set(scope))
                    if additional > sum(d["free_mib"] for d in self._lease_capacity_devices(gpu_snapshot())
                                        if d["uuid"] in scope):
                        raise RuntimeError("requested growth does not fit live VRAM and reservations")
                else:
                    stopped = self.stop_pool_lanes("lease prepare")
                unloaded = self._unload_base_models()
                with self.cv:
                    if _evacuation_commit_id:
                        self._assert_evacuation_current(self.evacuations[_evacuation_commit_id])
                    now = time.time()
                    previous = (lease.state, lease.heartbeat_at, lease.transition_started_at,
                                lease.evacuation_id, lease.requested_mib)
                    lease.state, lease.heartbeat_at, lease.transition_started_at = "pending", now, now
                    lease.evacuation_id = _evacuation_commit_id or lease.evacuation_id
                    lease.requested_mib = target
                    try:
                        self._persist_leases_locked()
                    except Exception:
                        (lease.state, lease.heartbeat_at, lease.transition_started_at,
                         lease.evacuation_id, lease.requested_mib) = previous
                        raise
                    if _evacuation_commit_id:
                        self._commit_cache_evacuation_locked(_evacuation_commit_id, lease)
                    self.cv.notify_all()
                if scope and drain_host:
                    self.end_drain()
                return {"ok": True, "lease": asdict(lease), "unloaded": unloaded,
                        "stopped_lanes": stopped, "gpus": gpu_snapshot()}
            except Exception:
                if drain_host:
                    self.end_drain()
                raise

    def release(self, token: str, reason: str = "lease release",
                force: bool = False) -> dict[str, Any]:
        with self.transition:
            with self.cv:
                lease = self.leases.get(token)
                if lease is None:
                    raise KeyError("unknown lease")
                scoped_gpus = set(lease.gpu_uuids)
                scoped_release = bool(scoped_gpus)
                already_drained = (
                    lease.state in ("pending", "revoking") and self.draining
                )
            # A scoped lease has never owned the host. Its release must not
            # stop unrelated Ollama lanes or turn a failed owner-cleanup check
            # into a host-wide drain. Live GPU telemetry remains authoritative
            # for later lane placement on the released scope.
            if not scoped_release and not already_drained:
                self.begin_drain(f"{reason} by {lease.owner}")
            completed = False
            try:
                # Acquisition already drained an unscoped owner. Release
                # must never tear down CUDA alongside that owner's remaining
                # allocations, including after a broker restart.
                stopped = []
                unloaded = []
                if force and not lease.exclusive:
                    # Operator override for a lease whose owner is gone. The
                    # settle check can never pass once the owner's allocation
                    # is unverifiable, so refusing here would leave the scope
                    # pinned with no supported way to reclaim it.
                    LOG.error(
                        "lease release forced owner=%s; foreign CUDA release "
                        "is not verified", lease.owner,
                    )
                else:
                    if scoped_release and lease.cuda_processes is not None:
                        wait_for_owned_cuda_exit(lease.cuda_processes)
                    ignored = {key for other in self.leases.values() if other.token != token
                               for key in other.cuda_processes or []}
                    sharing = scoped_release and any(
                        other.token != token and other.state in ("pending", "active", "revoking")
                        and (not other.gpu_uuids or scoped_gpus.intersection(other.gpu_uuids))
                        for other in self.leases.values())
                    # A ready owner can acquire additional child contexts.
                    # Shared cleanup must not hide them behind another lease.
                    # The last owner retains legacy accounting tolerance;
                    # any leftover unregistered context quarantines its GPU.
                    wait_for_foreign_settle(lease.foreign_baseline,
                        scoped_gpus if scoped_release else None, ignored_processes=ignored,
                        strict_contexts=sharing)
                    if lease.exclusive:
                        require_exclusive_cuda_free(lease.gpu_uuids)
                with self.cv:
                    self.leases.pop(token, None)
                    self._persist_leases_locked()
                completed = True
                LOG.info("lease released owner=%s reason=%s", lease.owner, reason)
                return {"ok": True, "released": token, "unloaded": unloaded,
                        "stopped_lanes": stopped,
                        "gpus": gpu_snapshot(), "host_memory": host_memory_snapshot()}
            finally:
                if completed:
                    if not scoped_release:
                        self.end_drain()
                    else:
                        # Repair a stale drain left by an older broker version
                        # after this same scoped release failed. Do not clear a
                        # real unscoped pending/revoking transition.
                        with self.cv:
                            if self._global_transition_lease_locked() is None:
                                self.draining = False
                                self.cv.notify_all()
                else:
                    with self.cv:
                        if not scoped_release:
                            self.draining = True
                        self.last_reason = (
                            f"{reason} incomplete for {lease.owner}; "
                            "foreign CUDA release is not verified"
                        )
                        self.cv.notify_all()

    def revoke(self, token: str, reason: str = "operator revoke") -> dict[str, Any]:
        """Revoke a lease on operator request.

        The owner's next heartbeat fails, telling it to free CUDA memory and
        release. The scope stays blocked until then; an owner that stops
        heartbeating is abandoned after the revoke deadline.
        """
        with self.cv:
            lease = self.leases.get(token)
            if lease is None:
                raise KeyError("unknown lease")
            if lease.state not in ("pending", "active", "revoking"):
                raise RuntimeError(f"cannot revoke a {lease.state} lease")
            self._revoke_locked(lease, reason)
            return {"ok": True, "lease": asdict(lease)}

    def set_model_gpus(self, model: str,
                       gpu_uuids: list[str] | None) -> dict[str, Any]:
        """Set which GPUs may host a model's lanes, moving lanes that break it.

        Idle lanes on a now-disallowed GPU stop at once, busy ones after
        their in-flight requests. Replacements start on allowed GPUs in the
        background under normal capacity rules; if none fits, the next
        request for the model queues for capacity as usual.
        """
        model = canonical_model_tag(model)
        if not model:
            raise ValueError("model GPU policy requires a model tag")
        known = SELECTED_GPUS or [
            str(device.get("uuid")) for device in gpu_snapshot()
        ]
        allowed = None
        if gpu_uuids is not None:
            allowed = list(dict.fromkeys(str(value) for value in gpu_uuids))
            if not allowed:
                raise ValueError(
                    "at least one GPU must stay allowed; clear the policy "
                    "to allow every GPU"
                )
            unknown = [value for value in allowed if value not in known]
            if unknown:
                raise ValueError(f"GPUs are not brokered here: {unknown}")
            if set(allowed) >= set(known):
                allowed = None
        with self.cv:
            if allowed is None:
                self.model_gpu_policy.pop(model, None)
            else:
                self.model_gpu_policy[model] = allowed
            self._persist_model_policy_locked()
            moved, retiring = [], []
            for lane in list(self.lanes.values()):
                if (lane.kind != "managed" or lane.model != model
                        or allowed is None or lane.allows(allowed)
                        or lane.retiring):
                    continue
                lane.retiring = True
                if lane.in_flight:
                    retiring.append(lane.lane_id)
                else:
                    moved.append(lane)
            staying = sum(
                lane.parallel for lane in self.lanes.values()
                if lane.kind == "managed" and lane.model == model
                and not lane.retiring
            )
            wanted = staying + sum(lane.parallel for lane in moved) + (
                len(retiring) * POOL_INSTANCE_PARALLEL
            )
            self.cv.notify_all()
        LOG.warning("model GPU policy model=%s allowed=%s moving=%s",
                    model, allowed or "all", [lane.lane_id for lane in moved] + retiring)

        def migrate() -> None:
            failed = self._stop_lanes(moved, "operator GPU policy")
            if failed:
                LOG.error(
                    "model %s cannot move until lane process groups stop: %s",
                    model,
                    [lane.lane_id for lane in failed],
                )
                return
            if not moved and not retiring:
                return
            try:
                self.ensure_capacity(
                    model, max(1, wanted),
                    triggered_by={"key": "operator", "label": "operator GPU policy"},
                )
            except (CapacityError, OSError, RuntimeError, TimeoutError) as exc:
                LOG.warning("model %s could not move to allowed GPUs yet: %s",
                            model, exc)

        threading.Thread(target=migrate, daemon=True).start()
        return {"ok": True, "model": model, "allowed_gpu_uuids": allowed,
                "stopping_lanes": [lane.lane_id for lane in moved],
                "retiring_lanes": retiring}

    def set_client_gpus(self, client_key: str,
                        gpu_uuids: list[str] | None) -> dict[str, Any]:
        """Change future admission without interrupting active or shared lanes."""
        if (not isinstance(client_key, str) or not client_key or len(client_key) > 512
                or any(ord(char) < 32 or ord(char) == 127 for char in client_key)):
            raise ValueError("client GPU policy requires a valid client key")
        if gpu_uuids is not None and (
                not isinstance(gpu_uuids, list)
                or any(not isinstance(gpu, str) or not gpu for gpu in gpu_uuids)):
            raise ValueError("gpu_uuids must be an array of UUIDs or null")
        known = SELECTED_GPUS or [str(device.get("uuid")) for device in gpu_snapshot()]
        allowed = None if gpu_uuids is None else list(dict.fromkeys(gpu_uuids))
        if allowed is not None:
            unknown = [gpu for gpu in allowed if gpu not in known]
            if unknown:
                raise ValueError(f"GPUs are not brokered here: {unknown}")
            if allowed and set(allowed) == set(known):
                allowed = None
        with self.cv:
            if (allowed is not None and client_key not in self.client_gpu_policy
                    and len(self.client_gpu_policy) >= CLIENT_HISTORY_LIMIT):
                raise ValueError("saved client GPU policy limit reached; clear an unused policy")
            old_policy, old_labels = dict(self.client_gpu_policy), dict(self.client_policy_labels)
            if allowed is None:
                self.client_gpu_policy.pop(client_key, None)
                self.client_policy_labels.pop(client_key, None)
            else:
                self.client_gpu_policy[client_key] = allowed
                identity = self.clients.get(client_key, {}).get("identity", {})
                self.client_policy_labels[client_key] = clean_client_text(
                    identity.get("label") or self.client_policy_labels.get(client_key) or client_key)
            try:
                self._persist_model_policy_locked()
            except OSError:
                self.client_gpu_policy, self.client_policy_labels = old_policy, old_labels
                raise
            self.reconcile_retry_at = 0.0
            self.cv.notify_all()
        LOG.info("client GPU policy client=%s allowed=%s", client_key, allowed)
        return {"ok": True, "client_key": client_key, "allowed_gpu_uuids": allowed,
                "active_requests_unchanged": True}

    def set_client_gpu(self, client_key: str, gpu_uuid: str,
                       allowed: bool) -> dict[str, Any]:
        """Atomically edit one checkbox without overwriting other pending clicks."""
        known = SELECTED_GPUS or [str(device.get("uuid")) for device in gpu_snapshot()]
        if not isinstance(gpu_uuid, str) or gpu_uuid not in known:
            raise ValueError("GPU is not brokered here")
        if not isinstance(allowed, bool):
            raise ValueError("allowed must be a boolean")
        if not isinstance(client_key, str):
            raise ValueError("client GPU policy requires a valid client key")
        with self.cv:
            effective = set(self.client_gpu_policy.get(client_key, known))
            if allowed:
                effective.add(gpu_uuid)
            else:
                effective.discard(gpu_uuid)
            return self.set_client_gpus(client_key, [gpu for gpu in known if gpu in effective])

    def stop_lane(self, lane_id: str, force: bool = False) -> dict[str, Any]:
        with self.cv:
            self._prune_dead_lanes_locked()
            lane = self.lanes.get(lane_id)
            if lane is None:
                raise KeyError("unknown lane")
            if lane.kind != "managed":
                raise RuntimeError("only broker-managed lanes can be stopped")
            if lane.in_flight and not force:
                raise RuntimeError(
                    f"lane {lane_id} is serving {lane.in_flight} request(s); "
                    "retry when idle or force the stop"
                )
            lane.retiring = True
            self.cv.notify_all()
        failed = self._stop_lanes([lane], "operator stop")
        if failed:
            raise RuntimeError(
                f"lane {lane_id} process group survived stop and remains tracked"
            )
        return {"ok": True, "stopped_lanes": [lane_id], "gpus": gpu_snapshot()}

    def record_client_use(self, client: dict[str, Any], lane: Lane,
                          model: str) -> str:
        """Account an admitted request to its client, model, and lane."""
        now = time.time()
        key = str(client.get("key") or "unknown")
        with self.cv:
            record = self.clients.get(key)
            if record is None:
                record = self.clients[key] = {
                    "first_seen": now, "requests": 0, "models": {}, "lanes": {},
                }
            record["identity"] = client
            record["last_seen"] = now
            record["requests"] += 1
            if model:
                record["models"][model] = record["models"].get(model, 0) + 1
            usage = record["lanes"].setdefault(lane.lane_id, {
                "model": model or lane.model, "gpu_uuid": lane.gpu_uuid,
                "gpu_uuids": list(lane.scope),
                "kind": lane.kind, "requests": 0,
            })
            usage["requests"] += 1
            usage["last_seen"] = now
            lane_usage = lane.clients.setdefault(key, {
                "label": client.get("label", key), "requests": 0,
            })
            lane_usage["requests"] += 1
            lane_usage["last_seen"] = now
            lane.active_clients[key] = lane.active_clients.get(key, 0) + 1
            if len(lane.clients) > LANE_CLIENT_LIMIT:
                removable = [
                    item for item in lane.clients
                    if item not in lane.active_clients
                ]
                if removable:
                    oldest = min(
                        removable,
                        key=lambda item: lane.clients[item]["last_seen"],
                    )
                    lane.clients.pop(oldest)

            self._prune_client_history_locked(now)
            return key

    def _prune_client_history_locked(self, now: float | None = None) -> None:
        """Expire and bound every client-attribution history surface.

        An inference that outlives the history TTL must keep its attribution
        until it exits. All other history is an observability cache, not
        durable state, and is therefore bounded by both time and cardinality.
        """
        current = time.time() if now is None else now
        cutoff = current - CLIENT_HISTORY_TTL
        active_pairs = set()
        for lane in self.lanes.values():
            active_keys = set(lane.active_clients)
            active_pairs.update((lane.lane_id, key) for key in active_keys)
            for key, usage in list(lane.clients.items()):
                if (key not in active_keys
                        and float(usage.get("last_seen") or 0) < cutoff):
                    lane.clients.pop(key, None)
        active_clients = {key for _lane_id, key in active_pairs}
        for key, record in list(self.clients.items()):
            lanes = record.get("lanes") or {}
            for lane_id, usage in list(lanes.items()):
                if (lane_id, key) not in active_pairs and float(
                    usage.get("last_seen") or 0
                ) < cutoff:
                    lanes.pop(lane_id, None)
            if len(lanes) > CLIENT_LANE_HISTORY_LIMIT:
                protected = {
                    lane_id for lane_id in lanes
                    if (lane_id, key) in active_pairs
                }
                removable = sorted(
                    (lane_id for lane_id in lanes if lane_id not in protected),
                    key=lambda lane_id: float(
                        lanes[lane_id].get("last_seen") or 0
                    ),
                )
                excess = len(lanes) - CLIENT_LANE_HISTORY_LIMIT
                for lane_id in removable[:excess]:
                    lanes.pop(lane_id, None)
            if (key not in active_clients
                    and float(record.get("last_seen") or 0) < cutoff):
                self.clients.pop(key, None)

        excess = len(self.clients) - CLIENT_HISTORY_LIMIT
        if excess > 0:
            removable_clients = sorted(
                (key for key in self.clients if key not in active_clients),
                key=lambda key: float(
                    self.clients[key].get("last_seen") or 0
                ),
            )
            for key in removable_clients[:excess]:
                self.clients.pop(key, None)

    def _client_summaries_locked(self) -> list[dict[str, Any]]:
        live = set(self.lanes)
        summaries = []
        for key, record in self.clients.items():
            lanes = [
                {
                    "id": lane_id, "live": lane_id in live, **usage,
                    "expires_at": float(usage.get("last_seen") or 0)
                    + CLIENT_HISTORY_TTL,
                }
                for lane_id, usage in record["lanes"].items()
            ]
            summaries.append({
                "key": key, "identity": record["identity"],
                "first_seen": record["first_seen"],
                "last_seen": record["last_seen"],
                "expires_at": record["last_seen"] + CLIENT_HISTORY_TTL,
                "requests": record["requests"], "models": dict(record["models"]),
                "lanes": sorted(lanes, key=lambda usage: -usage["last_seen"]),
            })
        for key in sorted(self.client_gpu_policy.keys() - self.clients.keys()):
            summaries.append({
                "key": key, "identity": {"key": key, "label": self.client_policy_labels[key]},
                "first_seen": 0, "last_seen": 0, "expires_at": None,
                "requests": 0, "models": {}, "lanes": [], "policy_only": True,
            })
        return sorted(summaries, key=lambda summary: -summary["last_seen"])

    def heartbeat(self, token: str) -> dict[str, Any]:
        with self.cv:
            lease = self.leases.get(token)
            if lease is None:
                raise KeyError("unknown lease")
            if self._revoke_if_pending_deadline_expired_locked(lease):
                raise RuntimeError(
                    "lease transition was revoked; free CUDA memory and release the lease"
                )
            lease.heartbeat_at = time.time()
            self._persist_leases_locked()
            relocation = next((dict(move, handoff_id=request["id"],
                                    requested_by=request["owner"],
                                    requested_priority=request.get("priority", 0),
                                    expires_at=request["expires_at"])
                               for request in self._handoff_summaries_locked()
                               for move in request.get("relocations", [])
                               if move["lease_ref"] == self._lease_ref(lease)
                               and move["from_gpu_uuids"] == lease.gpu_uuids), None)
            return {"ok": True, "lease": asdict(lease), "relocation": relocation}

    def aggregate_running_models(self) -> dict[str, Any]:
        with self.cv:
            self._prune_dead_lanes_locked()
            lanes = list(self.lanes.values())
        models: list[dict[str, Any]] = []
        for lane in lanes:
            try:
                payload = backend_json_at(
                    lane.host, lane.port, "GET", "/api/ps", timeout=3.0
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if lane.kind == "system":
                    raise RuntimeError(f"Ollama backend unavailable: {exc}") from exc
                LOG.warning("cannot inspect managed lane %s: %s", lane.lane_id, exc)
                continue
            lane_models = payload.get("models", [])
            if not isinstance(lane_models, list):
                continue
            for raw in lane_models:
                if not isinstance(raw, dict):
                    continue
                model = dict(raw)
                model["ollama_unify_lane"] = lane.lane_id
                if lane.gpu_uuid:
                    model["ollama_unify_gpu_uuid"] = lane.gpu_uuid
                    model["ollama_unify_gpu_uuids"] = list(lane.scope)
                models.append(model)
        return {"models": models}

    def status(self) -> dict[str, Any]:
        with self.cv:
            self._prune_client_history_locked()
            leases = [asdict(lease) for lease in self.leases.values()]
            handoffs = self._handoff_summaries_locked()
            lease_summaries = self._public_lease_summaries_locked()
            draining = self.draining
            active = self.active_requests
            logical_in_flight = len(self.logical_in_flight)
            reason = self.last_reason
            lanes = self._lane_summaries_locked()
            queue = self._queue_summary_locked(include_requests=True)
            request_lifecycle = self._active_request_summary_locked()
            completed_responses = self._completed_response_summary_locked()
            clients = self._client_summaries_locked()
            model_gpu_policy = {
                model: list(gpus) for model, gpus in self.model_gpu_policy.items()
            }
            client_gpu_policy = {key: list(gpus) for key, gpus in self.client_gpu_policy.items()}
            cache_residency = {"policies": json.loads(json.dumps(self.cache_policies)),
                "operations": [self._evacuation_public(value) for value in self.evacuations.values()]}
            unregistered_gpus = sorted(self._unregistered_gpus_locked())
        backend = probe_backend()
        health = gpu_health_snapshot()
        return {"ok": True, "backend_available": backend.available,
                "backend_error": backend.error, "backend_checked_at": backend.checked_at,
                "selected_gpu_ids": SELECTED_GPUS,
                "selected_gpu_count": len(SELECTED_GPUS),
                "draining": draining, "active_requests": active,
                "last_reason": reason, "leases": leases,
                "lease_handoff_requests": handoffs,
                "lease_policy": lease_policy_document(),
                "lease_summaries": lease_summaries,
                "warnings": lease_visibility_warnings(lease_summaries)
                + gpu_health_warnings(health)
                + (["Unregistered CUDA activity or unavailable process telemetry: "
                    "model load/unload deferred on " + ", ".join(unregistered_gpus)]
                   if unregistered_gpus else []),
                "unregistered_gpu_quarantine": unregistered_gpus,
                "pending_transition_timeout_seconds": PENDING_TIMEOUT,
                "gpus": gpu_snapshot(),
                "gpu_health": health,
                "inference_hold": inference_hold_status(),
                "parallel_pool": {
                    "enabled": POOL_ENABLED,
                    "max_managed_servers": POOL_MAX_SERVERS,
                    "max_queue": POOL_MAX_QUEUE,
                    "resume_ttl_seconds": POOL_RESUME_TTL,
                    "logical_in_flight": logical_in_flight,
                    "instance_parallel": POOL_INSTANCE_PARALLEL,
                    "lanes": lanes,
                    "queue": queue,
                    "request_lifecycle": request_lifecycle,
                    "completed_responses": completed_responses,
                },
                "clients": clients,
                "client_history_policy": {
                    "ttl_seconds": CLIENT_HISTORY_TTL,
                    "max_clients": CLIENT_HISTORY_LIMIT,
                    "max_lanes_per_client": CLIENT_LANE_HISTORY_LIMIT,
                    "max_clients_per_lane": LANE_CLIENT_LIMIT,
                },
                "model_gpu_policy": model_gpu_policy,
                "client_gpu_policy": client_gpu_policy,
                "cache_residency": cache_residency,
                "foreign_gpu_processes": foreign_gpu_usage(), "models": backend.models,
                "host_memory": host_memory_snapshot()}

    def anonymous_watcher(self) -> None:
        previous: set[str] = set()
        while not self.stopping.wait(ANON_POLL):
            with self.cv:
                blocked = self._unregistered_gpus_locked(refresh=True)
                if blocked != previous:
                    self.cv.notify_all()
            if blocked != previous:
                LOG.warning("unregistered CUDA quarantine changed previous=%s current=%s; "
                            "load/unload deferred, resident lanes preserved",
                            sorted(previous), sorted(blocked))
                previous = blocked

    def lease_reaper(self) -> None:
        while not self.stopping.wait(5.0):
            now = time.time()
            with self.cv:
                for lease in list(self.leases.values()):
                    if self._revoke_if_pending_deadline_expired_locked(lease, now):
                        # The lease is revoking. Its TTL branch below can only
                        # re-revoke it, which is a no-op, so revocation itself
                        # needs the deadline that reclaims a dead owner.
                        self._abandon_if_revoke_deadline_expired_locked(
                            lease, now,
                        )
                        continue
                    if (lease.ttl > 0
                            and now - lease.heartbeat_at > lease.ttl):
                        self._revoke_locked(
                            lease,
                            f"heartbeat lease TTL of {lease.ttl}s expired",
                        )

    def pool_reaper(self) -> None:
        interval = max(1.0, min(30.0, POOL_IDLE_TIMEOUT / 3))
        while not self.stopping.wait(interval):
            if not self.transition.acquire(blocking=False):
                continue
            cutoff = time.time() - POOL_IDLE_TIMEOUT
            try:
                with self.cv:
                    blocked = self._unregistered_gpus_locked()
                    lanes = [lane for lane in self.lanes.values()
                             if lane.kind == "managed" and lane.in_flight == 0
                             and not lane.loading
                             and not set(lane.scope).intersection(blocked)
                             and (lane.retiring or lane.last_used < cutoff)]
                    for lane in lanes:
                        lane.retiring = True
                    if lanes:
                        self.cv.notify_all()
                self._stop_lanes(lanes, "idle timeout")
            finally:
                self.transition.release()

    def shutdown(self) -> None:
        with self.cv:
            self.stopping.set()
            self.cv.notify_all()
        try:
            self.stop_pool_lanes("broker shutdown")
        except RuntimeError as exc:
            LOG.error("broker shutdown retained live lane reservations: %s", exc)


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    broker: Broker

    def _send_json(self, status: int, payload: dict[str, Any],
                   headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _client_identity(self) -> dict[str, Any]:
        """Resolve the calling application once per connection.

        A declared name may change between requests on one connection, so
        it is refreshed without repeating the socket and process lookups.
        """
        declared = clean_client_text(self.headers.get(CLIENT_HEADER))
        agent = clean_client_text(self.headers.get("User-Agent"), 160)
        cached = getattr(self, "_identity", None)
        if (cached is None or cached.get("declared") != declared
                or cached.get("user_agent") != agent):
            try:
                cached = identify_client(
                    self.client_address[:2], self.connection.getsockname()[:2],
                    declared, agent,
                )
            except Exception as exc:  # noqa: BLE001 - never fail a request
                LOG.warning("client identification failed: %s", exc)
                cached = {"address": str(self.client_address[0]),
                          "declared": declared, "user_agent": agent}
                cached["key"], cached["label"] = client_key_and_label(cached)
            self._identity = cached
        return cached

    def _client_reference(self) -> dict[str, str]:
        identity = self._client_identity()
        return {"key": identity["key"], "label": identity["label"]}

    def _client_connected(self) -> bool:
        try:
            readable, _, exceptional = select.select(
                [self.connection], [], [self.connection], 0
            )
            if exceptional:
                return False
            if not readable:
                return True
            return bool(self.connection.recv(
                1, socket.MSG_PEEK | socket.MSG_DONTWAIT
            ))
        except BlockingIOError:
            return True
        except OSError:
            return False

    def _watch_active_client(
        self, admission: Admission, stop_socket: socket.socket,
    ) -> None:
        """Wait for client EOF/reset without polling the inference cadence."""
        try:
            readable, _, exceptional = select.select(
                [self.connection, stop_socket], [], [self.connection], None
            )
            if stop_socket in readable:
                return
            if exceptional:
                self.broker.request_client_detached(admission)
                return
            if self.connection in readable:
                try:
                    data = self.connection.recv(
                        1, socket.MSG_PEEK | socket.MSG_DONTWAIT
                    )
                except BlockingIOError:
                    return
                except OSError:
                    data = b""
                if not data:
                    self.broker.request_client_detached(admission)
                # Non-empty bytes belong to an uncommon pipelined next
                # request. Leave them untouched for BaseHTTPRequestHandler.
        except (OSError, ValueError):
            self.broker.request_client_detached(admission)

    @staticmethod
    def _stop_client_watcher(
        stop_writer: socket.socket | None,
        stop_reader: socket.socket | None,
        watcher: threading.Thread | None,
    ) -> None:
        if stop_writer is not None:
            try:
                stop_writer.send(b"x")
            except OSError:
                pass
        if watcher is not None:
            watcher.join(timeout=0.25)
        for endpoint in (stop_writer, stop_reader):
            if endpoint is not None:
                try:
                    endpoint.close()
                except OSError:
                    pass

    def _requested_admission_wait(self) -> float | None:
        """Return a bounded broker-admission wait requested by the client.

        Inference clients need a short, observable admission cycle when a GPU
        lease is pending. Without this hint, the HTTP request remains silent
        for the broker's full drain timeout and the client cannot distinguish
        queueing from a dead model server. Direct Ollama ignores this private
        header, so clients can send it without endpoint-specific branching.
        """
        raw = self.headers.get("X-Ollama-Unify-Admission-Wait-Ms", "").strip()
        if not raw:
            return None
        try:
            milliseconds = int(raw)
        except ValueError:
            return None
        if milliseconds <= 0:
            return None
        return max(0.1, min(DRAIN_TIMEOUT, milliseconds / 1000.0))

    def _requested_admission_controls(self, path: str) -> tuple[str, str, str]:
        logical_request_id = self.headers.get(
            LOGICAL_REQUEST_HEADER, ""
        ).strip()
        if logical_request_id and (
            len(logical_request_id) > 128
            or any(
                not (
                    character.isascii()
                    and (character.isalnum() or character in "._:-")
                )
                for character in logical_request_id
            )
        ):
            raise PermanentCapacityError(
                "logical request ID must be 1-128 letters, digits, '.', '_', ':', or '-'",
                400,
                "invalid_admission_header",
            )
        declared_workload_class = self.headers.get(
            "X-Ollama-Unify-Workload-Class", ""
        ).strip().lower()
        # Older embedding clients omit scheduling labels. Treat that endpoint
        # family as optional by default so it cannot evict warm foreground
        # weights. Explicit scheduling headers retain their existing meaning.
        unlabelled_embedding = path in EMBEDDING_PATHS and not declared_workload_class
        workload_class = declared_workload_class or (
            "background" if unlabelled_embedding else "unspecified"
        )
        if workload_class not in {
            "foreground", "interactive-control", "background", "unspecified",
        }:
            raise PermanentCapacityError(
                "workload class must be foreground, interactive-control, background, or unspecified",
                400,
                "invalid_admission_header",
                logical_request_id=logical_request_id,
            )
        queue_policy = self.headers.get(
            "X-Ollama-Unify-Queue-Policy", ""
        ).strip().lower() or ("yield" if unlabelled_embedding else "wait")
        if queue_policy not in {"wait", "yield"}:
            raise PermanentCapacityError(
                "queue policy must be wait or yield",
                400,
                "invalid_admission_header",
                logical_request_id=logical_request_id,
            )
        return logical_request_id, workload_class, queue_policy

    def _requested_gpu_uuids(self) -> tuple[str, ...] | None:
        if GPU_UUIDS_HEADER not in self.headers:
            return None
        return parse_gpu_uuid_constraint(
            self.headers.get(GPU_UUIDS_HEADER, ""),
            GPU_UUIDS_HEADER,
        )

    def _requested_cpu_offload(self) -> bool:
        if CPU_OFFLOAD_HEADER not in self.headers:
            return False
        values = self.headers.get_all(CPU_OFFLOAD_HEADER, [])
        if len(values) != 1 or values[0].strip().lower() not in ("allow", "deny"):
            raise PermanentCapacityError(f"{CPU_OFFLOAD_HEADER} must be allow or deny", 400,
                                         "invalid_admission_header")
        return values[0].strip().lower() == "allow"

    def _requested_warm_admission(self) -> str | None:
        if WARM_ADMISSION_HEADER not in self.headers:
            return None
        if len(self.headers.get_all(WARM_ADMISSION_HEADER, [])) != 1:
            raise PermanentCapacityError("duplicate warm admission header", 400, "invalid_warm_admission")
        return parse_warm_admission(self.headers.get(WARM_ADMISSION_HEADER, ""))

    def _send_capacity_failure(
        self,
        failure: CapacityError,
        *,
        request_id: str = "",
        logical_request_id: str = "",
        workload_class: str = "unspecified",
        queue_policy: str = "wait",
        queue: dict[str, Any] | None = None,
    ) -> None:
        broker_request_id = failure.request_id or request_id
        logical_id = failure.logical_request_id or logical_request_id
        payload: dict[str, Any] = {
            "ok": False,
            "error": str(failure),
            "retryable": failure.retryable,
            "reason_code": failure.reason_code,
        }
        if broker_request_id:
            payload["request_id"] = broker_request_id
        if logical_id:
            payload["logical_request_id"] = logical_id
        if queue is not None:
            payload["queue"] = queue
        if failure.retry_after is not None:
            payload["retry_after_ms"] = failure.retry_after * 1000
        if failure.cause_reason_code:
            payload["cause_reason_code"] = failure.cause_reason_code
        phase = getattr(failure, "warm_preflight_phase", None)
        causes = getattr(failure, "warm_preflight_causes", ())
        if phase in WARM_PREFLIGHT_PHASES:
            payload["warm_preflight_phase"] = phase
            payload["warm_preflight_causes"] = sorted(set(causes).intersection(WARM_PREFLIGHT_CAUSES))
        if failure.admission_retained:
            payload["admission_retained"] = True
            payload["resume_ttl_ms"] = int(POOL_RESUME_TTL * 1000)
        if isinstance(failure, WarmPreflightRequired):
            payload["admission_retained"] = failure.admission_retained
            payload["backend_started"] = False
        if failure.queue_position is not None:
            payload["queue_position"] = failure.queue_position
        if failure.queue_ticket is not None:
            payload["queue_ticket"] = failure.queue_ticket
        if isinstance(failure, CompletedResponseUnavailableError):
            payload["completed_response"] = {
                "schema": "io.ollama-unify.completed-response-receipt.v1",
                "status": failure.completed_status,
                "body_bytes": failure.completed_body_bytes,
                "body_sha256": failure.completed_body_sha256,
                "retained": False,
                "unavailable_reason": failure.unavailable_reason,
            }
        headers = {
            "X-Ollama-Unify-Retryable": str(failure.retryable).lower(),
            "X-Ollama-Unify-Reason-Code": failure.reason_code,
            "X-Ollama-Unify-Workload-Class": workload_class,
            "X-Ollama-Unify-Queue-Policy": queue_policy,
        }
        if failure.retry_after is not None:
            headers["Retry-After"] = str(failure.retry_after)
        if broker_request_id:
            headers["X-Ollama-Unify-Request-Id"] = broker_request_id
        if logical_id:
            headers[LOGICAL_REQUEST_HEADER] = logical_id
        if failure.admission_retained:
            headers["X-Ollama-Unify-Admission-Retained"] = "true"
        if failure.queue_position is not None:
            headers["X-Ollama-Unify-Queue-Position"] = str(
                failure.queue_position
            )
        if failure.queue_ticket is not None:
            headers["X-Ollama-Unify-Queue-Ticket"] = str(failure.queue_ticket)
        if isinstance(failure, CompletedResponseUnavailableError):
            headers["X-Ollama-Unify-Completed-Response-Sha256"] = (
                failure.completed_body_sha256
            )
        self._send_json(failure.status, payload, headers)

    def _send_completed_response(self, response: CompletedResponse) -> None:
        """Replay retained bytes without contacting or admitting a backend."""
        if response.body is None:
            raise CompletedResponseUnavailableError(response)
        try:
            self.send_response(response.status, response.reason)
            for key, value in response.headers:
                if key.lower() not in HOP_HEADERS and key.lower() != "content-length":
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("X-Ollama-Unify-Response-Replayed", "true")
            self.send_header(
                "X-Ollama-Unify-Completed-Response-Sha256",
                response.body_sha256,
            )
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response.body)
                self.wfile.flush()
        except OSError:
            pass

    def _handle(self) -> None:
        path = self.path.split("?", 1)[0]
        if self.command == "GET" and path == "/.well-known/ollama-unify-gpu-negotiator":
            requested_model = parse_qs(self.path.partition("?")[2]).get("model", [""])[0]
            if requested_model:
                try:
                    resolve_model_context_profile(requested_model)
                except CapacityError as exc:
                    self._send_capacity_failure(exc)
                    return
                except (OSError, RuntimeError, ValueError) as exc:
                    self._send_capacity_failure(CapacityError(str(exc), reason_code="backend_start_failed"))
                    return
            document = discovery_document()
            with self.broker.cv:
                lease_summaries = self.broker._public_lease_summaries_locked()
                document["lease_handoff_requests"] = self.broker._handoff_summaries_locked()
                unregistered_gpus = sorted(self.broker._unregistered_gpus_locked())
                document["unregistered_gpu_quarantine"] = unregistered_gpus
                if unregistered_gpus:
                    document["warnings"].append(
                        "Unregistered CUDA activity or unavailable process telemetry: "
                        "model load/unload deferred on " + ", ".join(unregistered_gpus)
                    )
                document["active_leases"] = lease_summaries
                document["warnings"] = lease_visibility_warnings(lease_summaries) + document["warnings"]
                document["parallel_pool"]["lanes"] = self.broker._lane_summaries_locked()
                document["parallel_pool"]["queue"] = self.broker._queue_summary_locked()
                document["parallel_pool"]["request_lifecycle"] = (
                    self.broker._active_request_summary_locked()
                )
                document["parallel_pool"]["completed_responses"] = (
                    self.broker._completed_response_summary_locked()
                )
            self._send_json(200, document)
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > 16 * 1024 * 1024:
            self._send_json(413, {"error": "request body exceeds broker limit"})
            return
        body = self.rfile.read(length) if length else b""
        content_type = self.headers.get("Content-Type", "")
        if self.command == "POST" and (path in INFERENCE_PATHS or path == CAPACITY_PATH):
            try:
                require_inference_enabled()
            except PermanentCapacityError as exc:
                self._send_capacity_failure(exc)
                return
        if self.command == "POST" and path == CAPACITY_PATH:
            try:
                payload = json.loads(body or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("capacity body must be a JSON object")
                parallel = payload.get("parallel", 1)
                if isinstance(parallel, bool) or not isinstance(parallel, int):
                    raise ValueError("parallel must be an integer")
                endpoint = str(payload.get("endpoint") or "")
                if endpoint and endpoint not in INFERENCE_PATHS:
                    raise ValueError(
                        "endpoint must be a supported Ollama inference path"
                    )
                placement = parse_capacity_placement(payload.get("placement", "auto"))
                gpu_uuids = parse_gpu_uuid_constraint(
                    payload.get("gpu_uuids")
                    if "gpu_uuids" in payload else None,
                    "gpu_uuids", exact=placement == "exclusive_group",
                )
                proof_only = payload.get("warm_admission_proof", False)
                if type(proof_only) is not bool:
                    raise ValueError("warm_admission_proof must be boolean")
                cpu_offload = payload.get("cpu_offload", False)
                if type(cpu_offload) is not bool:
                    raise ValueError("cpu_offload must be boolean")
                if proof_only and cpu_offload:
                    raise ValueError("CPU offload cannot use a full-GPU warm certificate")
                if proof_only:
                    result = self.broker.warm_admission_proof(
                        str(payload.get("model") or ""), parallel, gpu_uuids, placement,
                        client_key=self._client_reference()["key"])
                else:
                    result = self.broker.ensure_capacity(
                        str(payload.get("model") or ""), parallel, endpoint, gpu_uuids,
                        triggered_by=self._client_reference(),
                        placement=placement,
                        cpu_offload=cpu_offload,
                    )
                self._send_json(200, result)
            except PermanentCapacityError as exc:
                self._send_capacity_failure(exc)
            except CapacityError as exc:
                self._send_capacity_failure(exc)
            except (OSError, RuntimeError, TimeoutError) as exc:
                self._send_capacity_failure(CapacityError(
                    str(exc), reason_code="backend_start_failed"
                ))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            return
        if self.command == "GET" and path == "/api/ps":
            admission = None
            try:
                admission = self.broker.proxy_enter(
                    "", False, self._client_connected,
                    allow_during_drain=True,
                )
                self._send_json(200, self.broker.aggregate_running_models())
            except ClientDisconnected:
                return
            except (OSError, RuntimeError, TimeoutError) as exc:
                self._send_json(503, {"error": str(exc)})
            finally:
                if admission is not None:
                    self.broker.proxy_exit(admission, "", True)
            return
        model = ""
        admission = None
        request_id = secrets.token_hex(8)
        logical_request_id = ""
        fingerprint = ""
        workload_class = "unspecified"
        queue_policy = "wait"
        retained_request = None
        gpu_uuids = None
        cpu_offload = False
        warm_admission = None
        is_resume = False
        client_key = ""
        try:
            (
                logical_request_id,
                workload_class,
                queue_policy,
            ) = self._requested_admission_controls(path)
            gpu_uuids = self._requested_gpu_uuids()
            cpu_offload = self._requested_cpu_offload()
            warm_admission = self._requested_warm_admission()
            resume_header = self.headers.get(RESUME_REQUEST_HEADER, "").strip()
            if resume_header and resume_header.lower() != "true":
                raise PermanentCapacityError(
                    f"{RESUME_REQUEST_HEADER} must be true when present",
                    400,
                    "invalid_admission_header",
                )
            is_resume = resume_header.lower() == "true"
            if is_resume:
                if body:
                    raise PermanentCapacityError(
                        "resume request must not resend the inference body",
                        400,
                        "resume_body_forbidden",
                        logical_request_id=logical_request_id,
                    )
                if path not in INFERENCE_PATHS:
                    raise PermanentCapacityError(
                        "resume is supported only for inference paths",
                        400,
                        "invalid_admission_header",
                        logical_request_id=logical_request_id,
                    )
                retained_request, completed = self.broker.resume_request_lookup(
                    logical_request_id, self.command, path, warm_admission, gpu_uuids,
                    cpu_offload=cpu_offload if CPU_OFFLOAD_HEADER in self.headers else None,
                )
                if completed is not None:
                    self._send_completed_response(completed)
                    return
                if retained_request is None:
                    raise PermanentCapacityError(
                        "logical request body was not retained",
                        409,
                        "retained_request_unavailable",
                        logical_request_id=logical_request_id,
                    )
                body = retained_request.body
                content_type = retained_request.content_type
                model = retained_request.model
                fingerprint = retained_request.fingerprint
                if (
                    gpu_uuids is not None
                    and gpu_uuids != retained_request.gpu_uuids
                ):
                    raise PermanentCapacityError(
                        "resume GPU constraint differs from the retained request",
                        409,
                        "logical_request_conflict",
                        logical_request_id=logical_request_id,
                    )
                gpu_uuids = retained_request.gpu_uuids
                cpu_offload = retained_request.cpu_offload
                warm_admission = retained_request.warm_admission
            else:
                if warm_admission is not None:
                    try:
                        original_payload = json.loads(body)
                        original_model = canonical_model_tag(str(original_payload.get("model") or "")) if isinstance(original_payload, dict) else ""
                        self.broker.validate_warm_body(warm_admission, original_model, path, body)
                    except (ValueError, TypeError) as exc:
                        raise PermanentCapacityError("warm inference requires valid JSON", 400,
                                                     "invalid_warm_admission") from exc
                body = clamp_request(self.path, content_type, body)
                # The routed model is read from the body on path alone, for the
                # same reason the clamp above is. Keying this on the declared
                # Content-Type left an inference request with a JSON body sent
                # under any other media type with no model to place, and the
                # broker then waited for capacity for a model named "" that no
                # lane could ever serve, so the request hung until the client
                # gave up while Ollama would have answered it.
                if body and path in INFERENCE_PATHS:
                    try:
                        payload = json.loads(body)
                        if isinstance(payload, dict):
                            model = canonical_model_tag(
                                str(payload.get("model") or "")
                            )
                    except (TypeError, ValueError):
                        pass
                fingerprint = hashlib.sha256(
                    self.command.encode()
                    + b"\0"
                    + self.path.encode()
                    + b"\0"
                    + content_type.lower().encode()
                    + b"\0"
                    + ",".join(gpu_uuids or ()).encode()
                    + b"\0"
                    + (b"cpu-offload-allow\0" if cpu_offload else b"")
                    + ((b"warm-admission\0" + warm_admission.encode() + b"\0")
                       if warm_admission is not None else b"")
                    + body
                ).hexdigest()
                if logical_request_id and path in INFERENCE_PATHS:
                    completed = self.broker.completed_response_lookup(
                        logical_request_id, fingerprint
                    )
                    if completed is not None:
                        self._send_completed_response(completed)
                        return
                    retained_request = RetainedRequest(
                        method=self.command,
                        path=path,
                        content_type=content_type,
                        body=body,
                        model=model,
                        fingerprint=fingerprint,
                        gpu_uuids=gpu_uuids,
                        warm_admission=warm_admission,
                        cpu_offload=cpu_offload,
                    )
            admission = self.broker.proxy_enter(
                model, path in INFERENCE_PATHS, self._client_connected, request_id,
                allow_during_drain=path in SAFE_METADATA_PATHS,
                request_path=path,
                admission_wait=self._requested_admission_wait(),
                logical_request_id=logical_request_id,
                request_fingerprint=fingerprint,
                retained_request=retained_request,
                resume_request=is_resume,
                workload_class=workload_class,
                queue_policy=queue_policy,
                gpu_uuids=gpu_uuids,
                warm_admission=warm_admission,
                client=self._client_reference(),
                cpu_offload=cpu_offload,
            )
        except ClientDisconnected:
            return
        except CapacityError as exc:
            with self.broker.cv:
                queue = self.broker._queue_summary_locked()
            self._send_capacity_failure(
                exc,
                request_id=request_id,
                logical_request_id=logical_request_id,
                workload_class=workload_class,
                queue_policy=queue_policy,
                queue=queue,
            )
            return
        except (OSError, RuntimeError, TimeoutError) as exc:
            with self.broker.cv:
                queue = self.broker._queue_summary_locked()
            self._send_capacity_failure(
                CapacityError(str(exc), reason_code="backend_start_failed"),
                request_id=request_id,
                logical_request_id=logical_request_id,
                workload_class=workload_class,
                queue_policy=queue_policy,
                queue=queue,
            )
            return
        lane = admission.lane
        try:
            body = self.broker.prepare_managed_body(lane, path, body, admission.warm_admission)
            if admission.warm_admission is not None:
                self.broker.validate_warm_body(admission.warm_admission, model, path, body, prepared=True)
                with self.broker.cv:
                    eligible = self.broker._validate_warm_admission_locked(
                        admission.warm_admission, model, gpu_uuids, phase="post_admission_validation")
                    if lane.lane_id not in eligible:
                        raise WarmPreflightRequired(phase="post_admission_validation",
                                                   causes=("selected_lane_not_certified",))
        except CapacityError as exc:
            if admission.warm_admission is not None:
                exc = self.broker.reject_warm_before_backend(admission, exc)
            else:
                self.broker.proxy_exit(admission, model, False)
            with self.broker.cv:
                queue = self.broker._queue_summary_locked()
            self._send_capacity_failure(
                exc, request_id=request_id, logical_request_id=logical_request_id,
                workload_class=workload_class, queue_policy=queue_policy, queue=queue,
            )
            return
        client_key = self.broker.record_client_use(
            self._client_identity(), lane, model
        )
        self.broker.bind_active_request_client(admission, client_key)
        if not self._client_connected():
            self.broker.request_client_detached(admission)
            self.broker.proxy_exit(
                admission, model, False, client_key
            )
            admission = None
            return
        response_started = False
        succeeded = False
        backend = None
        response = None
        watcher_stop_reader = None
        watcher_stop_writer = None
        client_watcher = None
        client_socket_timeout = self.connection.gettimeout()
        try:
            watcher_stop_reader, watcher_stop_writer = socket.socketpair()
            client_watcher = threading.Thread(
                target=self._watch_active_client,
                args=(admission, watcher_stop_reader),
                daemon=True,
            )
            client_watcher.start()
            self.connection.settimeout(REQUEST_ACTIVITY_TTL)
            broker_control_headers = {
                "x-ollama-unify-admission-wait-ms",
                "x-ollama-unify-logical-request-id",
                "x-ollama-unify-resume-request",
                "x-ollama-unify-workload-class",
                "x-ollama-unify-queue-policy",
                "x-ollama-unify-gpu-uuids",
                "x-ollama-unify-cpu-offload",
                "x-ollama-unify-warm-admission",
            }
            headers = {key: value for key, value in self.headers.items()
                       if key.lower() not in HOP_HEADERS
                       and key.lower() not in {"host", "content-length"}
                       and key.lower() not in broker_control_headers}
            if content_type and not any(
                key.lower() == "content-type" for key in headers
            ):
                headers["Content-Type"] = content_type
            backend = http.client.HTTPConnection(
                lane.host, lane.port, timeout=REQUEST_ACTIVITY_TTL
            )
            if not self.broker.request_backend_started(admission, backend):
                raise ConnectionAbortedError(
                    "request was cancelled before backend connection"
                )
            backend.connect()
            if not self.broker.request_backend_connected(admission, backend.sock):
                raise ConnectionAbortedError(
                    "request was cancelled before backend dispatch"
                )
            backend.request(self.command, self.path, body=body if body else None, headers=headers)
            if not self.broker.renew_request_activity(
                admission, "request_sent"
            ):
                backend.close()
                raise ConnectionAbortedError(
                    "request was cancelled while contacting backend"
                )
            response = backend.getresponse()
            if not self.broker.renew_request_activity(
                admission, "response_headers"
            ):
                raise ConnectionAbortedError(
                    "request was cancelled before response headers"
                )
            forwarded_headers = [
                (key, value) for key, value in response.getheaders()
                if key.lower() not in HOP_HEADERS
                and key.lower() != "content-length"
            ]
            broker_headers = [
                ("X-Ollama-Unify-Lane", lane.lane_id),
                ("X-Ollama-Unify-Request-Id", admission.request_id),
                ("X-Ollama-Unify-Workload-Class", workload_class),
                ("X-Ollama-Unify-Queue-Policy", queue_policy),
                ("X-Ollama-Unify-Queue-Ms", str(admission.queue_ms)),
                (
                    "X-Ollama-Unify-Queue-Position",
                    str(admission.initial_position),
                ),
                (
                    "X-Ollama-Unify-Queue-Ticket",
                    str(admission.queue_ticket),
                ),
            ]
            if admission.logical_request_id:
                broker_headers.append((
                    LOGICAL_REQUEST_HEADER, admission.logical_request_id,
                ))
            content_length = response.getheader("Content-Length")
            client_writable = True
            try:
                self.send_response(response.status, response.reason)
                for key, value in forwarded_headers + broker_headers:
                    self.send_header(key, value)
                if content_length is not None:
                    self.send_header("Content-Length", content_length)
                else:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                response_started = True
            except OSError:
                client_writable = False
                self.broker.request_client_detached(admission)
                if not admission.logical_request_id:
                    raise ClientDisconnected(
                        "client disconnected before response headers"
                    )

            retain_body = bool(
                admission.logical_request_id and path in INFERENCE_PATHS
            )
            unavailable_reason = None
            declared_length = -1
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    declared_length = -1
                if declared_length > COMPLETED_RESPONSE_MAX_BODY_BYTES:
                    retain_body = False
                    unavailable_reason = "declared_body_exceeds_per_entry_limit"
            retained_chunks: list[bytes] = []
            response_body_bytes = 0
            response_digest = hashlib.sha256()
            body_expected = (
                self.command != "HEAD"
                and response.status not in (204, 304)
                and not 100 <= response.status < 200
            )
            reader = getattr(response, "read1", response.read)
            while True:
                chunk = reader(65536)
                if not chunk:
                    if (
                        body_expected
                        and declared_length >= 0
                        and response_body_bytes != declared_length
                    ):
                        self.broker.note_backend_failure(
                            admission, "backend_incomplete_response"
                        )
                        raise http.client.IncompleteRead(
                            b"", declared_length - response_body_bytes
                        )
                    if not self.broker.request_backend_complete(admission):
                        raise ConnectionAbortedError(
                            "backend response ended after request cancellation"
                        )
                    break
                if not self.broker.renew_request_activity(
                    admission, "response_body"
                ):
                    raise ConnectionAbortedError(
                        "request was cancelled during response"
                    )
                response_body_bytes += len(chunk)
                response_digest.update(chunk)
                if declared_length >= 0 and response_body_bytes > declared_length:
                    self.broker.note_backend_failure(
                        admission, "backend_incomplete_response"
                    )
                    raise http.client.IncompleteRead(b"", 0)
                if retain_body:
                    if (
                        response_body_bytes
                        <= COMPLETED_RESPONSE_MAX_BODY_BYTES
                    ):
                        retained_chunks.append(chunk)
                    else:
                        retained_chunks.clear()
                        retain_body = False
                        unavailable_reason = (
                            "observed_body_exceeds_per_entry_limit"
                        )
                terminal_chunk = (
                    declared_length >= 0
                    and response_body_bytes == declared_length
                )
                terminal_lock = (
                    self.broker.active_request_terminal_lock(admission)
                    if terminal_chunk else None
                )
                if terminal_lock is not None:
                    terminal_lock.acquire()
                try:
                    if client_writable:
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except OSError:
                            # Continue draining the completed backend response.
                            # A retry with the logical ID can then replay it
                            # instead of launching duplicate generation.
                            client_writable = False
                            self.broker.request_client_detached(admission)
                            if not admission.logical_request_id:
                                raise ClientDisconnected(
                                    "client disconnected during response"
                                )
                    if terminal_chunk:
                        if not self.broker.request_backend_complete(admission):
                            raise ConnectionAbortedError(
                                "backend completed after request cancellation"
                            )
                finally:
                    if terminal_lock is not None:
                        terminal_lock.release()
            succeeded = response.status < 500
            if admission.logical_request_id and path in INFERENCE_PATHS:
                self.broker.record_completed_response(
                    logical_request_id=admission.logical_request_id,
                    request_fingerprint=admission.request_fingerprint,
                    request_id=admission.request_id,
                    request_method=self.command,
                    request_path=path,
                    status=response.status,
                    reason=response.reason or "",
                    headers=forwarded_headers + broker_headers,
                    body=(b"".join(retained_chunks) if retain_body else None),
                    body_bytes=response_body_bytes,
                    body_sha256=response_digest.hexdigest(),
                    unavailable_reason=unavailable_reason,
                    warm_admission=admission.warm_admission,
                    cpu_offload=admission.cpu_offload,
                )
        except CapacityError as exc:
            self._send_capacity_failure(exc, request_id=admission.request_id,
                                        logical_request_id=admission.logical_request_id)
            if isinstance(exc, WarmAdmissionOwnershipUncertain):
                # Its owner (possibly a replacement) must keep its lifecycle.
                admission = None
        except ClientDisconnected:
            pass
        except Exception as exc:
            failure_reason = (
                "backend_activity_timeout"
                if isinstance(exc, (TimeoutError, socket.timeout))
                else "backend_transport_failed"
            )
            self.broker.note_backend_failure(admission, failure_reason)
            cancel_reason = self.broker.active_request_cancel_reason(admission)
            if cancel_reason == "client_disconnected":
                LOG.info(
                    "proxy client disconnected %s %s request=%s",
                    self.command, self.path, admission.request_id,
                )
            else:
                LOG.error(
                    "proxy error %s %s request=%s reason=%s: %s",
                    self.command,
                    self.path,
                    admission.request_id,
                    cancel_reason or "backend_unavailable",
                    exc,
                )
            if not response_started and cancel_reason != "client_disconnected":
                status = 504 if cancel_reason == "backend_activity_timeout" else 503
                self._send_json(status, {
                    "error": f"Ollama backend unavailable: {exc}",
                    "reason_code": cancel_reason or "backend_unavailable",
                    "retryable": True,
                    "request_id": admission.request_id,
                })
            elif response_started:
                # A fixed-length response that ends early otherwise leaves
                # the client waiting forever for bytes the backend will never
                # produce. EOF is the only valid terminal signal at this point.
                self.close_connection = True
                try:
                    self.connection.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
        finally:
            try:
                self.connection.settimeout(client_socket_timeout)
            except OSError:
                pass
            if response is not None:
                response.close()
            if backend is not None:
                backend.close()
            if admission is not None:
                self.broker.proxy_exit(
                    admission, model, succeeded, client_key,
                )
            self._stop_client_watcher(
                watcher_stop_writer,
                watcher_stop_reader,
                client_watcher,
            )

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_HEAD = _handle
    do_OPTIONS = _handle

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("proxy %s - %s", self.address_string(), fmt % args)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ControlHandler(socketserver.StreamRequestHandler):
    broker: Broker

    def request_alive(self) -> bool:
        try:
            return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b""
        except BlockingIOError:
            return True
        except OSError:
            return False

    def handle(self) -> None:
        while raw_request := self.rfile.readline(1024 * 1024):
            persistent = False
            try:
                request = json.loads(raw_request)
                persistent = request.get("persistent") is True
                action = request.get("action")
                if action == "acquire":
                    requested_gpu_uuids = request.get("gpu_uuids")
                    if not isinstance(requested_gpu_uuids, list):
                        requested_gpu_uuids = []
                    result = self.broker.acquire(
                        str(request.get("owner") or "unknown"),
                        max(0, int(request.get("requested_mib") or 0)),
                        max(0, int(request.get("ttl", DEFAULT_LEASE_TTL))),
                        [str(value) for value in requested_gpu_uuids],
                        str(request.get("justification") or ""),
                        int(request.get("expected_duration_seconds") or 0),
                        priority=request.get("priority"),
                        evacuation_id=str(request.get("evacuation_id") or ""),
                        yield_on_request=request.get("yield_on_request", False),
                        request_alive=self.request_alive,
                        placement=request.get("placement", "fixed"),
                        exclusive=request.get("exclusive", False),
                    )
                elif action == "relocate":
                    result = self.broker.relocate(str(request.get("token") or ""),
                                                  str(request.get("handoff_id") or ""))
                elif action == "ready":
                    result = self.broker.ready(str(request.get("token") or ""),
                                               workload_pid=request.get("workload_pid"),
                                               workload_pids=request.get("workload_pids"))
                elif action == "scope":
                    requested_gpu_uuids = request.get("gpu_uuids")
                    if not isinstance(requested_gpu_uuids, list):
                        requested_gpu_uuids = []
                    result = self.broker.scope(
                        str(request.get("token") or ""),
                        [str(value) for value in requested_gpu_uuids],
                    )
                elif action == "prepare":
                    result = self.broker.prepare(str(request.get("token") or ""),
                        priority=request.get("priority"), evacuation_id=str(request.get("evacuation_id") or ""),
                        requested_mib=request.get("requested_mib"))
                elif action == "set_cache_policy":
                    result = self.broker.set_cache_policy(str(request.get("model") or ""),
                        request.get("movable"), request.get("priority"), request.get("gpu_uuids"))
                elif action == "cancel_evacuation":
                    result = self.broker.cancel_evacuation(str(request.get("evacuation_id") or ""))
                elif action == "recover_evacuation":
                    result = self.broker.recover_evacuation(str(request.get("evacuation_id") or ""))
                elif action == "archive_evacuation":
                    result = self.broker.archive_evacuation(str(request.get("evacuation_id") or ""))
                elif action == "release":
                    result = self.broker.release(
                        str(request.get("token") or ""),
                        force=bool(request.get("force")),
                    )
                elif action == "heartbeat":
                    result = self.broker.heartbeat(str(request.get("token") or ""))
                elif action == "revoke":
                    result = self.broker.revoke(
                        str(request.get("token") or ""),
                        str(request.get("reason") or "operator revoke"),
                    )
                elif action == "set_model_gpus":
                    requested = request.get("gpu_uuids")
                    result = self.broker.set_model_gpus(
                        str(request.get("model") or ""),
                        None if requested is None
                        else [str(value) for value in requested],
                    )
                elif action == "set_client_gpus":
                    if "gpu_uuids" not in request:
                        raise ValueError("gpu_uuids is required; use null for all or [] for none")
                    result = self.broker.set_client_gpus(
                        request.get("client_key"), request["gpu_uuids"])
                elif action == "set_client_gpu":
                    result = self.broker.set_client_gpu(
                        request.get("client_key"), request.get("gpu_uuid"),
                        request.get("allowed"))
                elif action == "stop_lane":
                    result = self.broker.stop_lane(
                        str(request.get("lane_id") or ""),
                        force=bool(request.get("force")),
                    )
                elif action == "status":
                    result = self.broker.status()
                else:
                    raise ValueError(f"unknown action: {action}")
            except Exception as exc:
                result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            try:
                self.wfile.write(json.dumps(result, separators=(",", ":")).encode() + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if not persistent:
                return


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def send_control(
    payload: dict[str, Any], timeout_override: float | None = None
) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    response_file = None
    try:
        timeout = (
            timeout_override
            if timeout_override is not None
            else (
                HEARTBEAT_TIMEOUT
                if payload.get("action") == "heartbeat"
                else DRAIN_TIMEOUT + UNLOAD_TIMEOUT + ANON_MAX_DRAIN + 10
                # CLI does not inherit systemd's timeout overrides. Include
                # the protocol maximum, not the CLI's local default.
                + (600.0 if payload.get("action") == "acquire" else 0)
            )
        )
        client.settimeout(timeout)
        client.connect(CONTROL_SOCKET)
        client.sendall(json.dumps(payload).encode() + b"\n")
        response_file = client.makefile("rb")
        raw_response = response_file.readline(1024 * 1024)
        if not raw_response:
            raise ConnectionError("negotiator closed the control connection without a response")
        response = json.loads(raw_response)
    finally:
        if response_file is not None:
            response_file.close()
        client.close()
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "negotiator request failed"))
    return response


def heartbeat_reconnect_grace(ttl: int) -> float:
    """Bound reconnect time below the persisted lease's expiry window."""
    if ttl <= 0:
        return HEARTBEAT_RECONNECT_GRACE
    return max(0.1, min(HEARTBEAT_RECONNECT_GRACE, ttl / 2.0))


def heartbeat_with_reconnect(
    token: str,
    grace: float,
    stopped: threading.Event | None = None,
) -> dict[str, Any] | None:
    """Renew one exact lease across a bounded broker restart.

    Connection failures can mean that systemd is replacing the negotiator.
    Retry only that transport condition, using the same persisted token. An
    explicit broker rejection remains terminal so a revoked or invalid lease
    still fails closed immediately.
    """
    deadline = time.monotonic() + max(0.1, grace)
    retry_delay = 0.1
    last_error: Exception | None = None
    while True:
        if stopped is not None and stopped.is_set():
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConnectionError(
                f"lease heartbeat could not reconnect within {grace:.1f}s: "
                f"{last_error or 'broker unavailable'}"
            )
        try:
            return send_control(
                {"action": "heartbeat", "token": token},
                timeout_override=min(HEARTBEAT_TIMEOUT, max(0.1, remaining)),
            )
        except RuntimeError:
            # The broker answered and rejected this exact lease. Retrying
            # cannot repair an invalid, expired, or revoking lease.
            raise
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            delay = min(retry_delay, max(0.0, deadline - time.monotonic()))
            if delay <= 0:
                continue
            if stopped is not None:
                stopped.wait(delay)
            else:
                time.sleep(delay)
            retry_delay = min(2.0, retry_delay * 2.0)


def watch_heartbeat(token: str, requested_interval: float) -> int:
    """Renew a lease and reconnect through bounded negotiator restarts."""
    stopped = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def stop(_signum: int, _frame: Any) -> None:
        stopped.set()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, stop)
        response = heartbeat_with_reconnect(
            token, HEARTBEAT_RECONNECT_GRACE, stopped
        )
        if response is None:
            return 0
        lease = response.get("lease") or {}
        ttl = max(0, int(lease.get("ttl") or 0))
        interval = requested_interval
        if interval <= 0:
            interval = max(2.0, min(30.0, ttl / 3 if ttl else 30.0))
        reconnect_grace = heartbeat_reconnect_grace(ttl)
        print(json.dumps({
            "ok": True,
            "watching": True,
            "interval": interval,
            "reconnect_grace": reconnect_grace,
        }), flush=True)
        while not stopped.wait(interval):
            if heartbeat_with_reconnect(token, reconnect_grace, stopped) is None:
                return 0
        return 0
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def serve() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    broker = Broker()
    control_path = Path(CONTROL_SOCKET)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    if control_path.exists() or control_path.is_socket():
        control_path.unlink()

    ControlHandler.broker = broker
    ProxyHandler.broker = broker
    control = ThreadingUnixServer(CONTROL_SOCKET, ControlHandler)
    os.chmod(CONTROL_SOCKET, 0o660)
    proxy = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)

    def shutdown(_signum: int, _frame: Any) -> None:
        with broker.cv:
            broker.stopping.set()
            broker.cv.notify_all()
        threading.Thread(target=control.shutdown, daemon=True).start()
        threading.Thread(target=proxy.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=control.serve_forever, daemon=True).start()
    threading.Thread(target=broker.capacity_reconciler, daemon=True).start()
    threading.Thread(target=broker.anonymous_watcher, daemon=True).start()
    threading.Thread(target=broker.lease_reaper, daemon=True).start()
    threading.Thread(target=broker.pool_reaper, daemon=True).start()
    threading.Thread(target=broker.active_request_watchdog, daemon=True).start()
    LOG.info(
        "proxy listening on %s:%s; backend %s:%s; control %s; pool enabled=%s max=%s",
        LISTEN_HOST, LISTEN_PORT, BACKEND_HOST, BACKEND_PORT, CONTROL_SOCKET,
        POOL_ENABLED, POOL_MAX_SERVERS,
    )
    try:
        proxy.serve_forever()
    finally:
        broker.shutdown()
        control.shutdown()
        control.server_close()
        proxy.server_close()
        try:
            control_path.unlink()
        except FileNotFoundError:
            pass
    return 0


def print_lease_recommendations(acquired: dict[str, Any]) -> None:
    for suggestion in acquired.get("public_lease", {}).get("recommendations", []):
        print(f"ollama-unify recommendation (non-blocking): {suggestion}", file=sys.stderr)


def lease_run_flexible(args: argparse.Namespace) -> int:
    """Supervise one restartable owner; never retarget or forcibly kill a CUDA context."""
    if not args.drain_command or args.stop_timeout <= 0:
        raise ValueError("flexible run requires --drain-command and a positive --stop-timeout")
    acquired = send_control({"action": "acquire", "owner": args.owner,
                             "requested_mib": args.vram_mib, "ttl": args.ttl,
                             "gpu_uuids": args.gpu, "placement": "flexible",
                             "priority": args.priority, "evacuation_id": args.evacuation_id,
                             "justification": args.justification,
                             "expected_duration_seconds": args.expected_duration})
    print_lease_recommendations(acquired)
    lease = acquired["lease"]
    token = lease["token"]
    child = None
    stop = threading.Event()
    heartbeat_stop = threading.Event()
    relocation: list[dict[str, Any]] = []
    relocation_lock = threading.Lock()
    heartbeat_errors: list[str] = []
    handlers = {}

    def signal_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    def heartbeat() -> None:
        interval = max(1.0, min(5.0, args.ttl / 3 if args.ttl else 5.0))
        while not heartbeat_stop.wait(interval):
            try:
                response = heartbeat_with_reconnect(token, heartbeat_reconnect_grace(args.ttl), heartbeat_stop)
                if response is None:
                    return
                move = response.get("relocation")
                with relocation_lock:
                    relocation[:] = [move] if move else []
            except Exception as exc:
                heartbeat_errors.append(str(exc))
                stop.set()
                return

    def stop_child() -> bool:
        if child is None:
            return True
        if process_group_alive(child):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + args.stop_timeout
        while process_group_alive(child) and time.monotonic() < deadline:
            child.poll()
            time.sleep(0.1)
        child.poll()
        return not process_group_alive(child)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, signal_stop)
        threading.Thread(target=heartbeat, daemon=True).start()
        while not stop.is_set():
            env = os.environ.copy()
            env.update({"OLLAMA_UNIFY_GPU_LEASE": token,
                        "OLLAMA_UNIFY_GPU_LEASE_SUPERVISED": "1",
                        "CUDA_VISIBLE_DEVICES": ",".join(lease["gpu_uuids"]),
                        "HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"})
            print(f"ollama-unify: starting {args.owner} on {env['CUDA_VISIBLE_DEVICES']}", file=sys.stderr)
            child = subprocess.Popen(args.command, env=env, start_new_session=True)
            env["OLLAMA_UNIFY_SUPERVISED_CHILD_PID"] = str(child.pid)
            deadline = time.monotonic() + args.ready_timeout
            while not stop.is_set() and child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("relocated workload did not become ready before timeout")
                try:
                    check = subprocess.run(args.ready_command, shell=True, env=env, timeout=10,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.TimeoutExpired:
                    continue
                if check.returncode == 0:
                    break
                stop.wait(0.5)
            if stop.is_set():
                break
            if child.poll() is not None:
                raise RuntimeError("flexible workload exited before readiness")
            send_control({"action": "ready", "token": token, "workload_pid": child.pid})
            move = None
            draining_owner = False
            while not stop.wait(0.2):
                if child.poll() is not None:
                    return child.returncode
                with relocation_lock:
                    candidate = dict(relocation[0]) if relocation else None
                if candidate and candidate["expires_at"] > time.time():
                    draining_owner = True
                    try:
                        check = subprocess.run(args.drain_command, shell=True, env=env,
                                               timeout=max(1.0, args.stop_timeout),
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except subprocess.TimeoutExpired:
                        continue
                    if check.returncode != 0:
                        # Busy owners keep their lease; the requester remains bounded.
                        continue
                    current = send_control({"action": "heartbeat", "token": token}).get("relocation")
                    if (not current or current["handoff_id"] != candidate["handoff_id"]
                            or current["expires_at"] <= time.time()):
                        with relocation_lock:
                            relocation[:] = [current] if current else []
                        continue
                    move = candidate
                    break
                elif draining_owner:
                    if args.resume_command:
                        resumed = subprocess.run(args.resume_command, shell=True, env=env,
                                                 timeout=max(1.0, args.stop_timeout))
                        if resumed.returncode:
                            raise RuntimeError("owner could not reopen admission after cancelled handoff")
                    draining_owner = False
            if move is None:
                break
            print(f"ollama-unify: draining {args.owner} for priority {move['requested_priority']} "
                  f"request by {move['requested_by']}; destination {move['to_gpu_uuid']}", file=sys.stderr)
            if not stop_child():
                raise RuntimeError("owner did not stop gracefully; retaining its lease until process exit")
            child = None
            if stop.is_set():
                break
            # The broker independently verifies source CUDA release and the reserved target.
            try:
                moved = send_control({"action": "relocate", "token": token,
                                      "handoff_id": move["handoff_id"]})
                lease = moved["lease"]
            except Exception:
                current = send_control({"action": "heartbeat", "token": token})
                if (current.get("relocation") is None
                        and current["lease"]["state"] == "active"
                        and current["lease"]["gpu_uuids"] == lease["gpu_uuids"]):
                    # Cancellation can arrive after the owner stopped. Reload
                    # its original, still-owned scope once through prepare;
                    # health admission remains mandatory, with no CUDA retry.
                    lease = send_control({"action": "prepare", "token": token})["lease"]
                    print("ollama-unify: handoff cancelled; restarting the original owned scope",
                          file=sys.stderr)
                else:
                    raise
            with relocation_lock:
                relocation.clear()
        if heartbeat_errors:
            raise RuntimeError("lease heartbeat failed: " + heartbeat_errors[-1])
        return 0
    finally:
        # Heartbeat continues while owned workers exit. Never release ownership
        # merely because the supervisor received a signal or a drain timed out.
        if not stop_child():
            print("ollama-unify: waiting for owned processes; no forced kill or lease release", file=sys.stderr)
            while process_group_alive(child):
                child.poll()
                time.sleep(0.5)
        try:
            while True:
                try:
                    send_control({"action": "release", "token": token})
                    break
                except Exception as exc:
                    try:
                        status = send_control({"action": "status"})
                        if not any(item.get("token") == token for item in status.get("leases", [])):
                            print("ollama-unify: reservation was already removed after worker exit", file=sys.stderr)
                            break
                    except Exception:
                        pass
                    print(f"ollama-unify: retaining reservation until CUDA release can be verified: {exc}",
                          file=sys.stderr)
                    time.sleep(5)
        finally:
            heartbeat_stop.set()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def lease_run(args: argparse.Namespace) -> int:
    if not args.ready_command:
        raise RuntimeError("run requires --ready-command so Ollama cannot reload before external CUDA allocation")
    if getattr(args, "exclusive", False) and args.placement != "fixed":
        raise ValueError("exclusive run requires fixed placement")
    if args.placement == "flexible":
        if getattr(args, "exclusive", False):
            raise ValueError("exclusive leases require fixed placement")
        return lease_run_flexible(args)
    owner = args.owner or f"{os.environ.get('USER', 'user')}:{os.getpid()}"
    acquired = send_control({"action": "acquire", "owner": owner,
                             "requested_mib": args.vram_mib, "ttl": args.ttl,
                             "gpu_uuids": args.gpu,
                             "justification": args.justification,
                             "expected_duration_seconds": args.expected_duration,
                             "priority": args.priority, "evacuation_id": args.evacuation_id,
                             "exclusive": getattr(args, "exclusive", False)})
    print_lease_recommendations(acquired)
    token = acquired["lease"]["token"]
    env = os.environ.copy()
    env["OLLAMA_UNIFY_GPU_LEASE"] = token
    assigned_gpus = acquired["lease"].get("gpu_uuids") or []
    if assigned_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned_gpus)
        env["HIP_VISIBLE_DEVICES"] = "-1"
        env["ROCR_VISIBLE_DEVICES"] = "-1"
    child = None
    stop_heartbeat = threading.Event()
    heartbeat_failed = threading.Event()
    heartbeat_errors: list[str] = []
    try:
        child = subprocess.Popen(args.command, env=env)

        def heartbeat() -> None:
            interval = max(2.0, min(30.0, args.ttl / 3 if args.ttl else 30.0))
            reconnect_grace = heartbeat_reconnect_grace(args.ttl)
            while not stop_heartbeat.wait(interval):
                try:
                    if heartbeat_with_reconnect(
                        token, reconnect_grace, stop_heartbeat
                    ) is None:
                        return
                except Exception as exc:
                    heartbeat_errors.append(str(exc))
                    heartbeat_failed.set()
                    if child is not None and child.poll() is None:
                        child.terminate()
                    return

        threading.Thread(target=heartbeat, daemon=True).start()
        deadline = time.monotonic() + args.ready_timeout
        while (child.poll() is None and not heartbeat_failed.is_set()
               and time.monotonic() < deadline):
            ready = subprocess.run(args.ready_command, shell=True, env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ready.returncode == 0:
                break
            time.sleep(1.0)
        if heartbeat_failed.is_set():
            raise RuntimeError(
                "lease heartbeat failed during startup: "
                + (heartbeat_errors[-1] if heartbeat_errors else "unknown error")
            )
        if child.poll() is not None:
            raise RuntimeError("external command exited before its readiness check passed")
        if time.monotonic() >= deadline:
            raise TimeoutError("external workload did not become ready before timeout")
        send_control({"action": "ready", "token": token, "workload_pid": child.pid})
        try:
            return_code = child.wait()
            if heartbeat_failed.is_set():
                raise RuntimeError(
                    "lease heartbeat failed: "
                    + (heartbeat_errors[-1] if heartbeat_errors else "unknown error")
                )
            return return_code
        except KeyboardInterrupt:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            return 130
    finally:
        stop_heartbeat.set()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        try:
            send_control({"action": "release", "token": token})
        except Exception as exc:
            print(f"ollama-unify lease release failed: {exc}", file=sys.stderr)


def self_test() -> int:
    global MAX_CONTEXT
    MAX_CONTEXT = 8192
    # Test request shaping without requiring a synthetic installed model.
    # Exact model metadata admission is covered by the model-context suite.
    original = json.dumps({"options": {
        "num_gpu": 999, "main_gpu": 2, "num_ctx": 262144,
    }}).encode()
    clamped = json.loads(clamp_request("/api/generate", "application/json", original))
    assert clamped["options"]["num_gpu"] == -1
    assert "main_gpu" not in clamped["options"]
    assert clamped["options"]["num_ctx"] == 8192
    assert split_address("0.0.0.0") == ("0.0.0.0", 11434)
    assert split_address("127.0.0.1:11435") == ("127.0.0.1", 11435)
    document = discovery_document()
    assert document["schema"] == "io.ollama-unify.gpu-negotiator.discovery.v1"
    assert document["commands"]["discover"] == "docker gpu discover"
    assert document["heartbeat_reconnect_grace_seconds"] > 0
    assert "num_gpu" in agent_instructions_text()
    print("negotiator self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic Ollama/external CUDA lease negotiator")
    sub = parser.add_subparsers(dest="command_name", required=True)
    sub.add_parser("serve")
    sub.add_parser("status")
    sub.add_parser("self-test")
    sub.add_parser("discover")
    sub.add_parser("agent-instructions")
    acquire = sub.add_parser("acquire")
    acquire.add_argument("--owner", required=True)
    acquire.add_argument("--justification", default="", help="recommended: purpose of this reservation")
    acquire.add_argument("--expected-duration", type=int, default=0,
                         help="recommended: expected lease duration in seconds; zero means unknown")
    acquire.add_argument("--vram-mib", type=int, default=0)
    acquire.add_argument("--ttl", type=int, default=DEFAULT_LEASE_TTL)
    acquire.add_argument("--gpu", action="append", default=[])
    acquire.add_argument("--exclusive", action="store_true", help="reserve explicit fixed GPUs without managed-lane co-tenancy")
    acquire.add_argument("--token-only", action="store_true")
    acquire.add_argument("--yield-on-request", action="store_true",
                         help="opt in to cooperative single-GPU handoff; owner must honor discovery intent")
    acquire.add_argument("--priority", type=int)
    acquire.add_argument("--placement", choices=("fixed", "flexible"), default="fixed",
                         help="fixed reserves the exact GPU group; flexible assigns one ordered candidate and permits cooperative relocation")
    acquire.add_argument("--evacuation-id", default="", help="stable ID for opt-in copy-first lower-priority cache evacuation")
    cache = sub.add_parser("set-cache-policy")
    cache.add_argument("model")
    cache.add_argument("--movable", action="store_true")
    cache.add_argument("--priority", type=int, required=True)
    cache.add_argument("--gpu", action="append", default=[])
    cancel = sub.add_parser("cancel-evacuation")
    cancel.add_argument("evacuation_id")
    recover = sub.add_parser("recover-evacuation")
    recover.add_argument("evacuation_id")
    archive = sub.add_parser("archive-evacuation")
    archive.add_argument("evacuation_id")
    scope = sub.add_parser("scope")
    scope.add_argument("token")
    scope.add_argument("--gpu", action="append", required=True)
    for name in ("ready", "prepare", "release"):
        command = sub.add_parser(name)
        command.add_argument("token")
        if name == "prepare":
            command.add_argument("--priority", type=int)
            command.add_argument("--evacuation-id", default="")
            command.add_argument("--vram-mib", type=int, default=None,
                                 help="new total VRAM budget, reserved before growth")
        if name == "ready":
            command.add_argument("--pid", dest="workload_pid", type=int, action="append", default=None,
                                 help="workload parent PID for concurrent-owner CUDA accounting")
        if name == "release":
            command.add_argument(
                "--force", action="store_true",
                help="reclaim the scope without verifying the owner's "
                     "CUDA release; for an owner that is already gone",
            )
    relocate = sub.add_parser("relocate")
    relocate.add_argument("token")
    relocate.add_argument("--handoff-id", required=True)
    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("token")
    heartbeat.add_argument("--watch", action="store_true")
    heartbeat.add_argument("--interval", type=float, default=0.0)
    run = sub.add_parser("run")
    run.add_argument("--owner", required=True)
    run.add_argument("--justification", default="", help="recommended: purpose of this reservation")
    run.add_argument("--expected-duration", type=int, default=0,
                     help="recommended: expected lease duration in seconds; zero means unknown")
    run.add_argument("--vram-mib", type=int, default=0)
    run.add_argument("--ttl", type=int, default=DEFAULT_LEASE_TTL)
    run.add_argument("--gpu", action="append", default=[])
    run.add_argument("--exclusive", action="store_true", help="reserve explicit fixed GPUs without managed-lane co-tenancy")
    run.add_argument("--priority", type=int)
    run.add_argument("--placement", choices=("fixed", "flexible"), default="fixed")
    run.add_argument("--drain-command", default="",
                     help="flexible owner hook: stop new work and finish active requests before graceful restart")
    run.add_argument("--resume-command", default="",
                     help="reopen owner admission if a handoff is cancelled before restart")
    run.add_argument("--stop-timeout", type=float, default=60.0)
    run.add_argument("--evacuation-id", default="")
    run.add_argument("--ready-command", required=True)
    run.add_argument("--ready-timeout", type=float, default=300.0)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command_name == "serve":
        return serve()
    if args.command_name == "self-test":
        return self_test()
    if args.command_name == "discover":
        document = discovery_document()
        try:
            live_status = send_control({"action": "status"})
            summaries = [
                lease_public_summary(raw)
                for raw in live_status.get("leases", [])
                if isinstance(raw, dict)
            ]
            document["active_leases"] = summaries
            # The daemon can have systemd EnvironmentFile overrides that the
            # CLI does not inherit. Publish its effective scope and health.
            for key in ("selected_gpu_ids", "selected_gpu_count", "gpus", "gpu_health", "inference_hold",
                        "unregistered_gpu_quarantine", "lease_policy", "lease_handoff_requests"):
                if key in live_status:
                    document[key] = live_status[key]
            selected = set(document.get("selected_gpu_ids", []))
            for device in document.get("gpus", []):
                device["selected_for_ollama"] = not selected or device.get("uuid") in selected
            if "enabled" in live_status.get("parallel_pool", {}):
                document["parallel_pool"]["enabled"] = live_status["parallel_pool"]["enabled"]
            document["warnings"] = lease_visibility_warnings(summaries) + gpu_health_warnings(document["gpu_health"])
            if document.get("unregistered_gpu_quarantine"):
                document["warnings"].append(
                    "Unregistered CUDA activity or unavailable process telemetry: "
                    "model load/unload deferred on "
                    + ", ".join(document["unregistered_gpu_quarantine"])
                )
        except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError):
            document["warnings"].insert(0, (
                LEASE_COORDINATION_WARNING
                + " Live lease state is unavailable; do not assume GPUs are unleased."
            ))
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    if args.command_name == "agent-instructions":
        print(agent_instructions_text(), end="")
        return 0
    if args.command_name == "status":
        result = send_control({"action": "status"})
    elif args.command_name == "acquire":
        result = send_control({"action": "acquire", "owner": args.owner,
                               "requested_mib": args.vram_mib, "ttl": args.ttl,
                               "gpu_uuids": args.gpu,
                               "justification": args.justification,
                               "expected_duration_seconds": args.expected_duration,
                               "exclusive": getattr(args, "exclusive", False),
                               "yield_on_request": args.yield_on_request,
                               "placement": args.placement,
                               "priority": args.priority, "evacuation_id": args.evacuation_id})
    elif args.command_name == "relocate":
        result = send_control({"action": "relocate", "token": args.token,
                               "handoff_id": args.handoff_id})
    elif args.command_name == "set-cache-policy":
        result = send_control({"action": "set_cache_policy", "model": args.model,
            "movable": args.movable, "priority": args.priority, "gpu_uuids": args.gpu})
    elif args.command_name == "cancel-evacuation":
        result = send_control({"action": "cancel_evacuation", "evacuation_id": args.evacuation_id})
    elif args.command_name == "recover-evacuation":
        result = send_control({"action": "recover_evacuation", "evacuation_id": args.evacuation_id})
    elif args.command_name == "archive-evacuation":
        result = send_control({"action": "archive_evacuation", "evacuation_id": args.evacuation_id})
    elif args.command_name == "heartbeat" and args.watch:
        if args.interval < 0:
            parser.error("heartbeat --interval must be zero (automatic) or positive")
        return watch_heartbeat(args.token, args.interval)
    elif args.command_name == "scope":
        result = send_control({
            "action": "scope", "token": args.token, "gpu_uuids": args.gpu,
        })
    elif args.command_name in ("ready", "prepare", "release", "heartbeat"):
        control: dict[str, Any] = {
            "action": args.command_name, "token": args.token,
        }
        if args.command_name == "release" and getattr(args, "force", False):
            control["force"] = True
        if args.command_name == "prepare":
            control.update(priority=args.priority, evacuation_id=args.evacuation_id)
            if args.vram_mib is not None:
                control["requested_mib"] = args.vram_mib
        if args.command_name == "ready" and args.workload_pid is not None:
            control["workload_pids"] = args.workload_pid
        result = send_control(control)
    elif args.command_name == "run":
        if not args.command:
            parser.error("run requires a command after --")
        if args.command[0] == "--":
            args.command = args.command[1:]
        return lease_run(args)
    else:
        parser.error("unknown command")
    if args.command_name == "acquire":
        print_lease_recommendations(result)
    if args.command_name == "acquire" and args.token_only:
        print(result["lease"]["token"])
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"ollama-unify negotiator: {exc}", file=sys.stderr)
        raise SystemExit(1)

NEGOTIATOR
}

render_gpu_negotiator_cli() {
  cat <<'LEASE_CLI'
#!/bin/sh
# Generated by ollama-unify.
exec /usr/local/libexec/ollama-unify-gpu-negotiator "$@"
LEASE_CLI
}

render_docker_gpu_lease_plugin() {
  cat <<'DOCKER_PLUGIN'
#!/bin/sh
# Generated by ollama-unify. Docker CLI plugin entrypoint.
if [ "${1:-}" = "docker-cli-plugin-metadata" ]; then
  printf '%s\n' '{"SchemaVersion":"0.1.0","Vendor":"ollama-unify","Version":"1.0.0","ShortDescription":"Negotiate CUDA VRAM with Ollama","URL":"https://github.com/robit-man/ollama-unify"}'
  exit 0
fi
if [ "${1:-}" = "gpu" ]; then shift; fi
exec "${OLLAMA_UNIFY_GPU_LEASE_CLI:-/usr/local/bin/ollama-unify-gpu-lease}" "$@"
DOCKER_PLUGIN
}

render_gpu_tray_indicator() {
  cat <<'TRAY_INDICATOR'
#!/usr/bin/env python3
# Generated by ollama-unify.
"""System tray indicator for the ollama-unify GPU lease broker.

Shows every lease holder and broker-owned Ollama lane, the GPUs they occupy,
and the operator actions the broker accepts for each. Reads and acts through
the broker control socket, so it needs membership in the socket's group.
"""

from __future__ import annotations

import argparse
import grp
import json
import os
import pwd
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from typing import Any

CONFIG_PATH = "/etc/default/ollama-unify-negotiator"
DEFAULT_SOCKET = "/run/ollama-unify/gpu-negotiator.sock"
TRAY_UNIT = "ollama-unify-tray.service"
POLL_SECONDS = 5.0
STATUS_TIMEOUT = 20.0
ACTION_TIMEOUT = 60.0
# Slack for lease transitions queued behind another one on the broker.
TRANSITION_MARGIN_SECONDS = 60.0
INVENTORY_REFRESH_SECONDS = 60.0
ICON_OK = "video-display-symbolic"
ICON_ATTENTION = "dialog-warning-symbolic"
ICON_OFFLINE = "network-offline-symbolic"
# Hidden items kept ready per menu group. Created only alongside an
# unavoidable layout change, they let entries swap (a lane replaced by
# another within one poll) without changing the exported layout again.
SPARE_SLOTS = 2
# Groups whose size is fixed by construction never need spares.
FIXED_GROUPS = ("copy", "commands")
# Most recently active clients shown; the broker keeps a longer history.
CLIENT_MENU_LIMIT = 20
# Leave room for menu padding, check marks, and submenu arrows. The remaining
# label budget is derived from the narrowest monitor, so moving the indicator
# between unequal screens cannot create an off-screen menu.
TRAY_SCREEN_MARGIN_PX = 192
TRAY_FALLBACK_SCREEN_WIDTH_PX = 1024

Gtk: Any = None
Gdk: Any = None
GLib: Any = None
AppIndicator: Any = None


def config_value(name: str, default: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        with open(CONFIG_PATH, encoding="utf-8") as stream:
            for line in stream:
                key, separator, value = line.strip().partition("=")
                if separator and key == name:
                    return value.strip().strip('"')
    except OSError:
        pass
    return default


def control(payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    path = config_value("OLLAMA_UNIFY_SOCKET", DEFAULT_SOCKET)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(path)
        client.sendall(json.dumps(payload).encode() + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
    if not data:
        raise RuntimeError("broker closed the control connection")
    result = json.loads(data)
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "broker rejected the request")
    return result


def gpu_inventory() -> dict[str, dict[str, str]]:
    try:
        output = subprocess.run([
            "nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader",
        ], check=True, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    inventory = {}
    for raw in output.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            continue
        bus = fields[3].split(":", 1)[-1].rsplit(".", 1)[0]
        inventory[fields[1]] = {"index": fields[0], "name": fields[2], "bus": bus}
    return inventory


def describe_process(pid: int) -> str:
    """Return a short name plus the systemd unit or container that owns pid."""
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as stream:
            name = stream.read().strip()
    except OSError:
        return f"pid {pid} (exited)"
    unit = ""
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as stream:
            path = stream.read().strip().rpartition("::")[-1]
        for part in reversed(path.split("/")):
            if part.startswith("docker-") and part.endswith(".scope"):
                unit = "container " + part[len("docker-"):-len(".scope")][:12]
                break
            if part.endswith(".service"):
                unit = part
                break
    except OSError:
        pass
    return f"{name} (pid {pid}, {unit})" if unit else f"{name} (pid {pid})"


def config_seconds(name: str, default: float) -> float:
    try:
        return max(0.0, float(config_value(name, str(default))))
    except ValueError:
        return default


def transition_timeout() -> float:
    """Bound a release or prepare by the broker's own configured phases.

    Both can drain in-flight requests, unload models, and wait for foreign
    CUDA memory to settle, each under its own configurable deadline.
    """
    return (
        config_seconds("OLLAMA_UNIFY_DRAIN_TIMEOUT", 300.0)
        + config_seconds("OLLAMA_UNIFY_UNLOAD_TIMEOUT", 120.0)
        + config_seconds("OLLAMA_UNIFY_ANON_MAX_DRAIN", 15.0)
        + TRANSITION_MARGIN_SECONDS
    )


def memory(mib: Any) -> str:
    """Format VRAM legibly on any device, from small cards to large ones."""
    mib = max(0, int(mib or 0))
    if mib < 1024:
        return f"{mib} MB"
    if mib < 10 * 1024:
        return f"{mib / 1024:.1f} GB"
    return f"{mib / 1024:.0f} GB"


def duration(seconds: float) -> str:
    seconds = abs(int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 90 * 60:
        return f"{seconds // 60} min"
    if seconds < 36 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def age(seconds: float) -> str:
    # Coarse buckets keep the menu stable between polls.
    seconds = int(seconds)
    if seconds < 60:
        return "under a minute ago"
    return f"{duration(seconds)} ago"


def gpu_name(gpu_uuid: str | None, inventory: dict[str, dict[str, str]]) -> str:
    if not gpu_uuid:
        return "any GPU"
    info = inventory.get(gpu_uuid)
    return f"GPU{info['index']}" if info else gpu_uuid[:12]


def gpu_list(gpu_uuids: list[str], inventory: dict[str, dict[str, str]]) -> str:
    return " + ".join(gpu_name(value, inventory) for value in gpu_uuids) or "all GPUs"


def wrapped(prefix: str, text: str, width: int = 64) -> list[str]:
    lines = textwrap.wrap(text, width) or [""]
    return [prefix + lines[0]] + ["    " + line for line in lines[1:]]


def ellipsized_text(text: str, max_pixels: int, measure: Any) -> str:
    """Return the longest measured prefix that fits, with a visible ellipsis."""
    if max_pixels <= 0 or measure(text) <= max_pixels:
        return text
    ellipsis = "…"
    if measure(ellipsis) > max_pixels:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if measure(text[:middle] + ellipsis) <= max_pixels:
            low = middle
        else:
            high = middle - 1
    return text[:low] + ellipsis


def tray_label_width_pixels() -> int:
    """Fit labels on every monitor, including smaller secondary displays."""
    widths = []
    try:
        display = Gdk.Display.get_default()
        if display is not None:
            for index in range(display.get_n_monitors()):
                monitor = display.get_monitor(index)
                if monitor is not None:
                    workarea = monitor.get_workarea()
                    if workarea.width > 0:
                        widths.append(int(workarea.width))
    except (AttributeError, TypeError):
        pass
    screen_width = min(widths) if widths else TRAY_FALLBACK_SCREEN_WIDTH_PX
    return max(32, screen_width - TRAY_SCREEN_MARGIN_PX)


def action(label: str, request: dict[str, Any], confirm: str | None = None,
           timeout: float = ACTION_TIMEOUT) -> dict[str, Any]:
    return {"label": label, "request": request, "confirm": confirm,
            "timeout": timeout}


def lease_actions(lease: dict[str, Any], inventory: dict[str, dict[str, str]]
                  ) -> list[dict[str, Any]]:
    token = lease.get("token") or ""
    owner = lease.get("owner") or "unknown"
    state = lease.get("state")
    actions = []
    if state == "pending":
        actions.append(action(
            "Mark ready", {"action": "ready", "token": token},
            f"Mark {owner}'s lease ready?\n\nOnly do this once its CUDA "
            "allocation is fully loaded. Ready single-GPU scopes admit "
            "Ollama lanes into their remaining free VRAM.",
        ))
    if state == "active":
        actions.append(action(
            "Prepare for resize…", {"action": "prepare", "token": token},
            f"Move {owner}'s lease back to pending so it can grow its VRAM?\n\n"
            "The broker checks this lease's reserved budget, reclaims only "
            "necessary managed capacity, and preserves other owners. Use "
            "prepare --vram-mib NEW_TOTAL to increase the budget. The owner "
            "must call ready again after its allocation settles.",
            transition_timeout(),
        ))
    if state in ("pending", "active"):
        actions.append(action(
            "Renew heartbeat", {"action": "heartbeat", "token": token},
        ))
        actions.append(action(
            "Revoke lease…",
            {"action": "revoke", "token": token, "reason": "operator revoke from tray"},
            f"Revoke {owner}'s lease?\n\nIts next heartbeat fails, telling it "
            "to free its CUDA memory and release. Its GPUs stay blocked and "
            "new leases wait until it does; if it stops heartbeating, the "
            "broker abandons the lease after the revoke deadline.",
        ))
    actions.append(action(
        "Release lease…", {"action": "release", "token": token},
        f"End {owner}'s lease?\n\nThe broker waits for the owner's CUDA "
        "contexts to exit and refuses if cleanup cannot be verified. "
        "Shared peers keep their leases and allocations. This can take a few minutes.",
        transition_timeout(),
    ))
    actions.append(action(
        "Force release…", {"action": "release", "token": token, "force": True},
        f"Force-end {owner}'s lease without verifying its CUDA memory was "
        "freed?\n\nUse this only when the owner is gone. Memory still "
        "resident stays allocated, and the GPUs return to Ollama placement.",
    ))
    return actions


def lane_actions(lane: dict[str, Any], gpu: str) -> list[dict[str, Any]]:
    lane_id = lane.get("id") or ""
    label = f"{lane_id} ({lane.get('model') or 'no model'} on {gpu})"
    actions = []
    if not lane.get("in_flight"):
        actions.append(action(
            "Stop lane…", {"action": "stop_lane", "lane_id": lane_id},
            f"Stop Ollama lane {label}?\n\nThe broker starts a new lane the "
            "next time this model is requested.",
        ))
    actions.append(action(
        "Force stop lane…",
        {"action": "stop_lane", "lane_id": lane_id, "force": True},
        f"Force-stop Ollama lane {label}?\n\nRequests it is serving now "
        f"({lane.get('in_flight') or 0}) fail immediately.",
    ))
    return actions


def toggle(key: str, label: str, active: bool, enabled: bool,
           spec: dict[str, Any]) -> dict[str, Any]:
    return {"key": key, "label": label, "active": active, "enabled": enabled,
            "spec": spec}


def gpu_toggle_label(gpu_uuid: str, inventory: dict[str, dict[str, str]],
                     note: str = "") -> str:
    name = inventory.get(gpu_uuid, {}).get("name")
    label = gpu_name(gpu_uuid, inventory) + (f" · {name}" if name else "")
    return label + (f" ({note})" if note else "")


def flipped(brokered: list[str], current: list[str], gpu_uuid: str) -> list[str]:
    """Return current with gpu_uuid toggled, in brokered order."""
    return [value for value in brokered
            if (value in current) != (value == gpu_uuid)]


def model_gpu_submenu(model: str, policy: dict[str, list[str]],
                      brokered: list[str], reserved: dict[str, str],
                      inventory: dict[str, dict[str, str]]) -> dict[str, Any]:
    allowed = policy.get(model)
    effective = [value for value in brokered if allowed is None or value in allowed]
    toggles = []
    for gpu_uuid in brokered:
        active = gpu_uuid in effective
        verb = "Disallow" if active else "Allow"
        toggles.append(toggle(
            gpu_uuid,
            gpu_toggle_label(gpu_uuid, inventory, (
                f"exclusive to {reserved[gpu_uuid]}" if gpu_uuid in reserved else ""
            )),
            active,
            # The last allowed GPU cannot be switched off; clear instead.
            not (active and len(effective) == 1),
            action(
                f"{verb} {gpu_name(gpu_uuid, inventory)} for {model}",
                {"action": "set_model_gpus", "model": model,
                 "gpu_uuids": flipped(brokered, effective, gpu_uuid)},
            ),
        ))
    return {
        "key": "gpu-policy",
        "title": "Allowed GPUs: " + (
            "all" if allowed is None else gpu_list(allowed, inventory)
        ),
        "details": wrapped("", f"Applies to every {model} lane; lanes on a "
                           "disallowed GPU move once their requests finish."),
        "toggles": toggles,
        "actions": [action(
            "Allow all GPUs",
            {"action": "set_model_gpus", "model": model, "gpu_uuids": None},
        )] if allowed is not None else [],
    }


def lease_gpu_submenu(lease: dict[str, Any], leases: list[dict[str, Any]],
                      brokered: list[str],
                      inventory: dict[str, dict[str, str]]) -> dict[str, Any]:
    token = lease.get("token") or ""
    owner = lease.get("owner") or "unknown"
    scope = [str(value) for value in lease.get("gpu_uuids") or []]
    def conflicts(new_scope: list[str]) -> bool:
        for other in leases:
            if other is lease or other.get("state") not in ("pending", "active", "revoking"):
                continue
            other_scope = other.get("gpu_uuids") or []
            if other_scope and not set(new_scope).intersection(other_scope):
                continue
            if (len(new_scope) != 1 or len(other_scope) != 1
                    or lease.get("exclusive") or other.get("exclusive")
                    or not lease.get("requested_mib") or not other.get("requested_mib")
                    or other.get("state") == "revoking"):
                return True
        return False
    held = {
        gpu_uuid: other.get("owner")
        for other in leases if other is not lease
        and other.get("state") in ("pending", "active", "revoking")
        for gpu_uuid in other.get("gpu_uuids") or []
    }
    toggles = []
    for gpu_uuid in brokered:
        active = gpu_uuid in scope
        name = gpu_name(gpu_uuid, inventory)
        new_scope = flipped(brokered, scope, gpu_uuid)
        if active:
            confirm = (
                f"Remove {name} from {owner}'s lease?\n\nThe broker refuses if "
                "the owner's CUDA memory grew on that GPU. A lease narrowed to "
                "one GPU is no longer exclusive and shares free VRAM with Ollama."
            )
        else:
            confirm = (
                f"Add {name} to {owner}'s lease?\n\nThe broker reserves it for "
                "this lease; the owner only uses it once its process sees that "
                "GPU. A lease of two or more GPUs is exclusive, so Ollama lanes "
                "on its GPUs stop."
            )
        toggles.append(toggle(
            gpu_uuid,
            gpu_toggle_label(gpu_uuid, inventory, (
                f"leased by {held[gpu_uuid]}" if gpu_uuid in held else ""
            )),
            active,
            (active or not conflicts(new_scope)) and not (active and len(scope) == 1),
            action(
                f"{'Remove' if active else 'Add'} {name} "
                f"{'from' if active else 'to'} lease",
                {"action": "scope", "token": token, "gpu_uuids": new_scope},
                confirm,
            ),
        ))
    return {
        "key": "lease-gpus",
        "title": "GPUs in lease: " + (gpu_list(scope, inventory) if scope
                                      else "host-wide (unscoped)"),
        "details": [],
        "toggles": toggles,
        "actions": [],
    }


def client_gpu_submenu(client: dict[str, Any], policy: dict[str, list[str]],
                       brokered: list[str], inventory: dict[str, dict[str, str]]
                       ) -> dict[str, Any]:
    key = str(client["key"])
    allowed = policy.get(key)
    effective = [gpu for gpu in brokered if allowed is None or gpu in allowed]
    label = (client.get("identity") or {}).get("label", key)
    actions = []
    if allowed is not None:
        actions.append(action("Allow all GPUs", {
            "action": "set_client_gpus", "client_key": key, "gpu_uuids": None,
        }))
    if allowed != []:
        actions.append(action("Pause GPU requests", {
            "action": "set_client_gpus", "client_key": key, "gpu_uuids": [],
        }))
    return {
        "key": "client-gpus",
        "title": "Allowed GPUs: " + (
            "all" if allowed is None else gpu_list(allowed, inventory)
            if allowed else "none (paused)"),
        "details": wrapped("", "Running requests finish. Queued and future requests "
                           "use these GPUs, subject to model policy and reservations. "
                           "Selecting none pauses new GPU requests."),
        "toggles": [toggle(
            gpu, gpu_toggle_label(gpu, inventory), gpu in effective, True,
            action(f"{'Disallow' if gpu in effective else 'Allow'} "
                   f"{gpu_name(gpu, inventory)} for {label}", {
                       "action": "set_client_gpu", "client_key": key,
                       "gpu_uuid": gpu, "allowed": gpu not in effective,
                   }),
        ) for gpu in brokered],
        "actions": actions,
    }


def client_entry(client: dict[str, Any], lanes: list[dict[str, Any]],
                 inventory: dict[str, dict[str, str]], now: float,
                 policy: dict[str, list[str]], brokered: list[str],
                 ) -> dict[str, Any]:
    identity = client.get("identity") or {}
    live = {lane.get("id") for lane in lanes}
    details = []
    for label, value in (
        ("Declared as", identity.get("declared")),
        ("User", identity.get("user")),
        ("Container", identity.get("container")),
        ("Unit", identity.get("unit")),
        ("Process", f"{identity.get('process')} (pid {identity.get('pid')})"
         if identity.get("pid") else (
             f"{identity.get('process')} (inferred from its unit)"
             if identity.get("process") else None)),
        ("Processes", ", ".join(identity.get("candidate_processes") or [])),
        ("Address", identity.get("address")),
        ("User agent", identity.get("user_agent")),
    ):
        if value:
            details += wrapped(f"{label}: ", str(value))
    details.append("Saved GPU policy · no recent requests" if client.get("policy_only") else (
        f"Requests: {client.get('requests')} · last "
        f"{age(now - float(client.get('last_seen') or now))}"
    ))
    details += [
        f"Model: {model} · {count} req"
        for model, count in sorted((client.get("models") or {}).items(),
                                   key=lambda item: -item[1])
    ]
    for usage in client.get("lanes") or []:
        where = (gpu_list(usage.get("gpu_uuids") or [usage.get("gpu_uuid")], inventory)
                 if usage.get("kind") == "managed" else "base Ollama")
        state = "live" if usage.get("id") in live else "ended"
        details.append(
            f"Lane {usage.get('id')}: {usage.get('model')} on {where} · "
            f"{usage.get('requests')} req · {state}"
        )
    models = ", ".join(list(client.get("models") or {})[:2])
    return {
        "key": str(client.get("key")),
        "title": (f"{identity.get('label', client.get('key'))} · "
                  f"{client.get('requests')} req"
                  + (f" · {models}" if models else "")),
        "details": details,
        "actions": [],
        "submenus": [client_gpu_submenu(client, policy, brokered, inventory)],
    }


def lease_details(lease: dict[str, Any], summary: dict[str, Any],
                  inventory: dict[str, dict[str, str]], now: float) -> list[str]:
    details = [
        f"Owner: {lease.get('owner')}",
        f"State: {lease.get('state')}",
        f"GPUs: {gpu_list(lease.get('gpu_uuids') or [], inventory)}"
        + (" (exclusive)" if lease.get("exclusive") is True or len(lease.get("gpu_uuids") or []) > 1 else ""),
        f"Requested: {memory(lease.get('requested_mib'))}",
    ]
    details += wrapped("Why: ", summary.get("justification") or "")
    created = float(lease.get("created_at") or 0)
    if created:
        details.append(f"Started: {age(now - created)}")
    remaining = summary.get("seconds_until_expected_release")
    if remaining is None:
        details.append("Expected release: unknown (legacy lease)")
    elif remaining < 0:
        details.append(f"Expected release: overdue by {duration(remaining)}")
    else:
        details.append(f"Expected release: in {duration(remaining)}")
    heartbeat = float(lease.get("heartbeat_at") or 0)
    if heartbeat:
        details.append(
            f"Heartbeat: {age(now - heartbeat)} (TTL {lease.get('ttl')}s)"
        )
    return details


def build_menu_model(
    status: dict[str, Any] | None,
    error: str | None,
    inventory: dict[str, dict[str, str]],
    selected_gpus: list[str],
    process_names: dict[int, str],
    now: float,
) -> dict[str, Any]:
    """Turn a broker status payload into a toolkit-free menu description."""
    if status is None:
        return {
            "icon": ICON_OFFLINE, "label": "offline",
            "summary": [f"Broker unreachable: {error or 'no status'}"],
            "gpus": [], "leases": [], "lanes": [], "clients": [],
        }
    leases = status.get("leases") or []
    summaries = status.get("lease_summaries") or []
    summary_by_key = {
        (item.get("owner"), float(item.get("created_at") or 0)): item
        for item in summaries
    }
    lanes = [lane for lane in (status.get("parallel_pool") or {}).get("lanes", [])
             if lane.get("kind") == "managed"]
    foreign: dict[str, list[tuple[int, int]]] = {}
    for key, used_mib in (status.get("foreign_gpu_processes") or {}).items():
        pid, separator, gpu_uuid = key.partition("@")
        if separator and pid.isdigit():
            foreign.setdefault(gpu_uuid, []).append((int(pid), int(used_mib)))

    brokered = list(selected_gpus) or [
        str(device.get("uuid")) for device in status.get("gpus") or []
    ]
    policy = status.get("model_gpu_policy") or {}
    reserved = {
        gpu_uuid: str(lease.get("owner"))
        for lease in leases if lease.get("exclusive") is True or len(lease.get("gpu_uuids") or []) > 1
        for gpu_uuid in lease.get("gpu_uuids") or []
    }
    attention = bool(status.get("draining"))
    lease_entries = []
    for lease in sorted(leases, key=lambda item: float(item.get("created_at") or 0)):
        summary = summary_by_key.get(
            (lease.get("owner"), float(lease.get("created_at") or 0)), {}
        )
        overdue = summary.get("horizon_status") == "overdue"
        if lease.get("state") in ("pending", "revoking") or overdue:
            attention = True
        flag = " ⚠" if lease.get("state") == "revoking" or overdue else ""
        lease_entries.append({
            "key": f"{lease.get('owner')}@{float(lease.get('created_at') or 0)}",
            "title": (
                f"{lease.get('owner')} · "
                f"{gpu_list(lease.get('gpu_uuids') or [], inventory)} · "
                f"{lease.get('state')}{flag}"
            ),
            "details": lease_details(lease, summary, inventory, now),
            "actions": lease_actions(lease, inventory),
            "submenus": [lease_gpu_submenu(lease, leases, brokered, inventory)]
            if lease.get("state") in ("pending", "active") else [],
        })

    lane_entries = []
    for lane in sorted(lanes, key=lambda item: str(item.get("id"))):
        scope = lane.get("gpu_uuids") or [lane.get("gpu_uuid")]
        gpu = gpu_list(scope, inventory)
        lane_entries.append({
            "key": str(lane.get("id")),
            "title": f"{lane.get('model')} · {gpu} · {lane.get('state')}",
            "details": [
                f"Lane: {lane.get('id')}",
                f"Model: {lane.get('model')}",
                f"GPU: {gpu}",
                f"State: {lane.get('state')}",
                f"Serving: {lane.get('in_flight') or 0} of {lane.get('parallel')}",
                f"Reserved: {memory(sum((lane.get('reserved_mib_by_gpu') or {}).values()) or lane.get('reserved_mib'))}",
            ] + ([f"Context: {lane['resolved_context_length']} tokens"]
                 if lane.get("resolved_context_length") else [])
            + [f"Started for: {(lane.get('triggered_by') or {}).get('label', 'unknown')}"]
            + [
                f"Used by: {usage.get('label')} · {usage.get('requests')} req · "
                f"{age(now - float(usage.get('last_seen') or now))}"
                for usage in lane.get("clients") or []
            ],
            "actions": lane_actions(lane, gpu),
            "submenus": [model_gpu_submenu(
                str(lane.get("model")), policy, brokered, reserved, inventory,
            )] if lane.get("model") else [],
        })

    client_policy = status.get("client_gpu_policy") or {}
    clients = status.get("clients") or []
    # Saved policies remain editable even after their attribution expires.
    visible_clients = [client for client in clients if client.get("key") in client_policy]
    visible_clients += [client for client in clients
                        if client.get("key") not in client_policy][:CLIENT_MENU_LIMIT]
    client_entries = [client_entry(client, lanes, inventory, now, client_policy, brokered)
                      for client in visible_clients]

    gpu_entries = []
    for device in status.get("gpus") or []:
        gpu_uuid = str(device.get("uuid") or "")
        info = inventory.get(gpu_uuid, {})
        holders = [lease.get("owner") for lease in leases
                   if gpu_uuid in (lease.get("gpu_uuids") or [])]
        gpu_lanes = [lane for lane in lanes if gpu_uuid in (
            lane.get("gpu_uuids") or [lane.get("gpu_uuid")])]
        brokered = not selected_gpus or gpu_uuid in selected_gpus
        title = (
            f"{gpu_name(gpu_uuid, inventory)} · {info.get('name', 'GPU')} · "
            f"{memory(device.get('used_mib'))} / {memory(device.get('total_mib'))}"
        )
        if holders:
            title += " · leased"
        elif not brokered:
            title += " · not brokered"
        details = [
            f"UUID: {gpu_uuid}",
            f"PCI bus: {info.get('bus', 'unknown')}",
            f"Free: {memory(device.get('free_mib'))}",
            "Lease: " + (", ".join(holders) if holders else "none"),
        ]
        details += [
            f"Ollama: {lane.get('model')} "
            f"({memory((lane.get('reserved_mib_by_gpu') or {}).get(gpu_uuid, lane.get('reserved_mib')))})"
            for lane in gpu_lanes
        ]
        details += [
            f"CUDA: {process_names.get(pid, f'pid {pid}')} · {memory(used)}"
            for pid, used in sorted(foreign.get(gpu_uuid, []),
                                    key=lambda item: -item[1])
        ]
        gpu_entries.append({
            "key": gpu_uuid, "title": title, "details": details, "actions": [],
        })

    state = "draining" if status.get("draining") else "running"
    summary_lines = [
        f"Broker {state} · {len(lease_entries)} lease(s) · "
        f"{len(lane_entries)} Ollama lane(s)"
    ]
    overdue_owners = [
        str(lease.get("owner") or "unknown")
        for lease in leases
        if summary_by_key.get(
            (lease.get("owner"), float(lease.get("created_at") or 0)), {}
        ).get("horizon_status") == "overdue"
    ]
    if overdue_owners:
        summary_lines.append(
            "Lease expected-release overdue: " + ", ".join(overdue_owners)
        )
    if status.get("draining"):
        summary_lines += wrapped("Reason: ", str(status.get("last_reason") or ""))
    if not status.get("backend_available", True):
        summary_lines.append("Ollama backend unavailable")
        attention = True
    return {
        "icon": ICON_ATTENTION if attention else ICON_OK,
        "label": f"{len(lease_entries)}L · {len(lane_entries)}O",
        "summary": summary_lines,
        "gpus": gpu_entries,
        "leases": lease_entries,
        "lanes": lane_entries,
        "clients": client_entries,
    }


def menu_rows(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe the menu as keyed rows in fixed, homogeneous groups.

    The tray maps each group to a pool of reusable items, so the order and
    kind of groups never change and updates stay property-only.
    """
    def row(group: str, key: str, kind: str, label: str = "",
            **extra: Any) -> dict[str, Any]:
        return {"group": group, "key": key, "kind": kind, "label": label,
                **extra}

    def entry_rows(entry: dict[str, Any], copy: bool = True
                   ) -> list[dict[str, Any]]:
        rows = [row("details", f"detail:{index}", "info", line)
                for index, line in enumerate(entry["details"])]
        rows += [
            row("toggles", item["key"], "toggle", item["label"],
                active=item["active"], enabled=item["enabled"],
                spec=item["spec"])
            for item in entry.get("toggles") or []
        ]
        submenus = entry.get("submenus") or []
        if entry["actions"] or submenus:
            rows.append(row("actions:separator", "separator", "separator"))
        rows += [
            row("submenus", sub["key"], "entry", sub["title"],
                rows=entry_rows(sub, copy=False))
            for sub in submenus
        ]
        rows += [
            row("actions", f"action:{spec['label']}", "action", spec["label"],
                spec=spec)
            for spec in entry["actions"]
        ]
        if copy:
            rows.append(row(
                "copy", "copy", "copy", "Copy details",
                text="\n".join([entry["title"]] + entry["details"]),
            ))
        return rows

    rows = [row("summary", f"summary:{index}", "info", line)
            for index, line in enumerate(model["summary"])]
    for heading, section, empty in (
        ("GPUs", "gpus", "No GPUs reported"),
        ("Leases", "leases", "No active leases"),
        ("Ollama lanes", "lanes", "No Ollama lanes running"),
        ("Clients", "clients", "No clients seen since broker start"),
    ):
        rows += [
            row(f"{section}:separator", "separator", "separator"),
            row(f"{section}:heading", "heading", "info", heading),
        ]
        if not model[section]:
            rows.append(row(f"{section}:heading", "empty", "info", "    " + empty))
        rows += [
            row(section, entry["key"], "entry", entry["title"],
                rows=entry_rows(entry))
            for entry in model[section]
        ]
    rows += [
        row("commands:separator", "separator", "separator"),
        row("commands", "refresh", "command", "Refresh now", command="refresh"),
        row("commands", "quit", "command", "Quit indicator", command="quit"),
    ]
    return rows


def load_toolkit() -> None:
    global Gtk, Gdk, GLib, AppIndicator
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3, Gdk as gdk, GLib as glib, Gtk as gtk
    Gtk, Gdk, GLib, AppIndicator = gtk, gdk, glib, AyatanaAppIndicator3


class TrayApp:
    def __init__(self, poll: bool = True) -> None:
        self.indicator = AppIndicator.Indicator.new(
            "ollama-unify-gpu-broker", ICON_OFFLINE,
            AppIndicator.IndicatorCategory.SYSTEM_SERVICES,
        )
        self.indicator.set_title("ollama-unify GPU broker")
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        self.menu = Gtk.Menu()
        self.indicator.set_menu(self.menu)
        self.signature: str | None = None
        self.icon: str | None = None
        self.label: str | None = None
        # Per menu (identified by the slot path leading to it), each row group
        # owns an ordered pool of item slots. Slots are hidden and reused
        # rather than removed, so steady-state updates never change the
        # exported menu layout; a layout change would make the shell rebuild
        # every submenu item and close whatever the user has open.
        self.slots: dict[tuple, dict[str, list[dict[str, Any]]]] = {}
        self.group_order: dict[tuple, list[str]] = {}
        self.generation = 0
        self.layout_changed = False
        # Set while the tray itself changes a check item, whose GTK
        # set_active emits the same activate signal as a user click.
        self.applying = False
        self.label_width_pixels = tray_label_width_pixels()
        self.inventory: dict[str, dict[str, str]] = {}
        self.inventory_at = 0.0
        self.selected_gpus = [
            value for value in
            config_value("OLLAMA_UNIFY_SELECTED_GPUS", "").split(",") if value
        ]
        self.wake = threading.Event()
        self.render(build_menu_model(None, "connecting", {}, [], {}, time.time()))
        if poll:
            threading.Thread(target=self.poll_loop, daemon=True).start()

    def poll_loop(self) -> None:
        while True:
            status, error = None, None
            try:
                status = control({"action": "status"}, STATUS_TIMEOUT)
            except (OSError, ValueError, RuntimeError) as exc:
                error = str(exc)
            if (not self.inventory
                    or time.monotonic() - self.inventory_at > INVENTORY_REFRESH_SECONDS):
                self.inventory = gpu_inventory()
                self.inventory_at = time.monotonic()
            process_names = {}
            for key in (status or {}).get("foreign_gpu_processes") or {}:
                pid = key.partition("@")[0]
                if pid.isdigit():
                    process_names[int(pid)] = describe_process(int(pid))
            model = build_menu_model(
                status, error, self.inventory, self.selected_gpus,
                process_names, time.time(),
            )
            GLib.idle_add(self.render, model)
            self.wake.wait(POLL_SECONDS)
            self.wake.clear()

    def render(self, model: dict[str, Any]) -> bool:
        signature = json.dumps(model, sort_keys=True, default=str)
        if signature == self.signature:
            return False
        self.signature = signature
        if model["icon"] != self.icon:
            self.icon = model["icon"]
            self.indicator.set_icon_full(self.icon, "ollama-unify GPU broker")
        if model.get("label") != self.label:
            self.label = model.get("label")
            self.indicator.set_label(self.label or "", "00L · 00O")
        self.generation += 1
        self.layout_changed = False
        self.label_width_pixels = tray_label_width_pixels()
        self.sync_menu(self.menu, menu_rows(model), ())
        if self.layout_changed:
            self.add_spares()
        return False

    def sync_menu(self, menu: Any, rows: list[dict[str, Any]],
                  path: tuple) -> None:
        """Apply rows to a menu using only property changes when possible.

        A row keeps its slot while its key lives. A vanished row hides its
        slot; a new row takes a slot hidden in an earlier update, so an open
        submenu never switches to another entry's data. Only a group growing
        past every size it has had creates an item.
        """
        by_group: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_group.setdefault(row["group"], []).append(row)
        order = self.group_order.setdefault(path, [])
        previous = None
        for group in by_group:
            if group not in order:
                order.insert(order.index(previous) + 1 if previous else 0, group)
            previous = group
        pools = self.slots.setdefault(path, {})
        offset = 0
        for group in order:
            pool = pools.setdefault(group, [])
            wanted = by_group.get(group, [])
            keys = {row["key"] for row in wanted}
            for slot in pool:
                if slot["row"] is not None and slot["row"]["key"] not in keys:
                    slot["row"] = None
                    slot["freed"] = self.generation
                    slot["widget"].hide()
            for row in wanted:
                slot = next(
                    (slot for slot in pool if slot["row"] is not None
                     and slot["row"]["key"] == row["key"]), None,
                ) or next(
                    (slot for slot in pool if slot["row"] is None
                     and slot["freed"] < self.generation), None,
                )
                if slot is None:
                    slot = self.append_slot(menu, path, group, row["kind"], offset)
                slot["row"] = row
                self.apply_row(slot)
            offset += len(pool)

    def append_slot(self, menu: Any, path: tuple, group: str, kind: str,
                    offset: int) -> dict[str, Any]:
        pool = self.slots[path][group]
        slot = self.create_slot(kind, path + ((group, len(pool)),))
        menu.insert(slot["widget"], offset + len(pool))
        pool.append(slot)
        self.layout_changed = True
        return slot

    def add_spares(self) -> None:
        """Top up hidden slots while the layout is changing anyway."""
        for path in list(self.slots):
            menu = self.menu_at(path)
            offset = 0
            for group in list(self.group_order[path]):
                pool = self.slots[path][group]
                kind = pool[0]["kind"] if pool else None
                if kind and kind != "separator" and group not in FIXED_GROUPS:
                    template = max(
                        (slot["row"] for slot in pool
                         if slot["row"] is not None and kind == "entry"),
                        key=lambda row: len(row["rows"]), default=None,
                    )
                    free = sum(1 for slot in pool if slot["row"] is None)
                    for _ in range(SPARE_SLOTS - free):
                        spare = self.append_slot(menu, path, group, kind, offset)
                        if template is not None:
                            # Prebuild the submenu so reusing the spare
                            # only relabels items.
                            self.sync_menu(spare["widget"].get_submenu(),
                                           template["rows"], spare["path"])
                            self.release(spare["path"])
                offset += len(pool)

    def release(self, path: tuple) -> None:
        for pool in self.slots.get(path, {}).values():
            for slot in pool:
                slot["row"] = None
                slot["widget"].hide()

    def menu_at(self, path: tuple) -> Any:
        if not path:
            return self.menu
        return self.slot_at(path)["widget"].get_submenu()

    def create_slot(self, kind: str, slot_path: tuple) -> dict[str, Any]:
        if kind == "separator":
            widget = Gtk.SeparatorMenuItem()
        elif kind == "toggle":
            widget = Gtk.CheckMenuItem.new_with_mnemonic("")
            widget.connect("activate", lambda _item: self.activate(slot_path))
        else:
            widget = Gtk.MenuItem.new_with_mnemonic("")
            if kind == "entry":
                widget.set_submenu(Gtk.Menu())
            elif kind != "info":
                widget.connect(
                    "activate", lambda _item: self.activate(slot_path)
                )
        return {"widget": widget, "row": None, "freed": 0, "path": slot_path,
                "kind": kind}

    def apply_row(self, slot: dict[str, Any]) -> None:
        widget, row = slot["widget"], slot["row"]
        if row["kind"] != "separator":
            full_text = row["label"]
            label = widget.get_child()

            def measure(value: str) -> int:
                if label is None or not hasattr(label, "create_pango_layout"):
                    return len(value) * 8
                return label.create_pango_layout(value).get_pixel_size()[0]

            visible_text = ellipsized_text(
                full_text, self.label_width_pixels, measure,
            )
            text = visible_text.replace("_", "__")
            if widget.get_label() != text:
                widget.set_label(text)
            tooltip = full_text if visible_text != full_text else None
            widget.set_tooltip_text(tooltip)
            if label is not None:
                label.set_tooltip_text(tooltip)
            sensitive = row.get("enabled", row["kind"] != "info")
            if widget.get_sensitive() != sensitive:
                widget.set_sensitive(sensitive)
            if row["kind"] == "toggle":
                self.set_toggle(widget, row["active"])
        if not widget.get_visible():
            widget.show()
        if row["kind"] == "entry":
            self.sync_menu(widget.get_submenu(), row["rows"], slot["path"])

    def set_toggle(self, widget: Any, active: bool) -> None:
        if widget.get_active() != active:
            self.applying = True
            try:
                widget.set_active(active)
            finally:
                self.applying = False

    def slot_at(self, slot_path: tuple) -> dict[str, Any] | None:
        *parent, (group, index) = slot_path
        pool = self.slots.get(tuple(parent), {}).get(group, [])
        return pool[index] if index < len(pool) else None

    def activate(self, slot_path: tuple) -> None:
        # Read the slot's row at click time so a reused item acts on the
        # entry it currently shows.
        if self.applying:
            return
        slot = self.slot_at(slot_path)
        row = slot["row"] if slot else None
        if row is None:
            return
        if row["kind"] == "toggle":
            # Show the broker's state until it confirms the change; the next
            # poll renders the result.
            self.set_toggle(slot["widget"], row["active"])
            self.trigger(row["spec"])
        elif row["kind"] == "action":
            self.trigger(row["spec"])
        elif row["kind"] == "copy":
            Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).set_text(row["text"], -1)
        elif row["command"] == "refresh":
            self.wake.set()
        elif row["command"] == "quit":
            Gtk.main_quit()

    def trigger(self, spec: dict[str, Any]) -> None:
        if spec["confirm"] and not self.confirm(spec["label"], spec["confirm"]):
            return
        threading.Thread(target=self.run_action, args=(spec,), daemon=True).start()

    def confirm(self, title: str, text: str) -> bool:
        heading, _, body = text.partition("\n\n")
        dialog = Gtk.MessageDialog(
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK_CANCEL, text=heading,
        )
        dialog.set_title(title.rstrip("…"))
        if body:
            dialog.format_secondary_text(body)
        dialog.set_keep_above(True)
        response = dialog.run()
        dialog.destroy()
        return response == Gtk.ResponseType.OK

    def run_action(self, spec: dict[str, Any]) -> None:
        try:
            control(spec["request"], spec["timeout"])
            GLib.idle_add(self.finish, spec, None)
        except (OSError, ValueError, RuntimeError) as exc:
            GLib.idle_add(self.finish, spec, str(exc))

    def finish(self, spec: dict[str, Any], error: str | None) -> bool:
        label = spec["label"].rstrip("…")
        if error is None:
            try:
                subprocess.Popen([
                    "notify-send", "--app-name=ollama-unify",
                    "GPU broker", f"{label}: done",
                ])
            except OSError:
                pass
        else:
            dialog = Gtk.MessageDialog(
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.CLOSE, text=f"{label} failed",
            )
            dialog.format_secondary_text(error)
            dialog.set_keep_above(True)
            dialog.run()
            dialog.destroy()
        self.wake.set()
        return False


def run_indicator() -> int:
    try:
        load_toolkit()
    except (ImportError, ValueError) as exc:
        print(f"ollama-unify-tray: GTK/AppIndicator unavailable: {exc}",
              file=sys.stderr)
        return 1
    TrayApp()
    for signum in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signum, Gtk.main_quit)
    Gtk.main()
    return 0


def start_sessions(group: str) -> int:
    """Start the tray in graphical sessions of users allowed on the socket.

    Runs from the broker unit with full privileges so the indicator comes up
    with the broker. Failures never affect the broker.
    """
    try:
        gid = grp.getgrnam(group).gr_gid
    except KeyError:
        print(f"ollama-unify-tray: group {group!r} does not exist", file=sys.stderr)
        return 0

    def run(argv: list[str]) -> str:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode:
            print(f"ollama-unify-tray: {' '.join(argv)} failed "
                  f"({result.returncode}): {result.stderr.strip()}",
                  file=sys.stderr)
        return result.stdout

    users = set()
    for line in run(["loginctl", "list-sessions", "--no-legend"]).splitlines():
        fields = line.split()
        if not fields:
            continue
        props = dict(
            entry.split("=", 1) for entry in run([
                "loginctl", "show-session", fields[0],
                "-p", "Name", "-p", "Type", "-p", "Class", "-p", "State",
            ]).splitlines() if "=" in entry
        )
        if (props.get("Class") == "user"
                and props.get("Type") in ("x11", "wayland")
                and props.get("State") in ("active", "online")):
            users.add(props.get("Name", ""))
    for user in sorted(value for value in users if value):
        try:
            account = pwd.getpwnam(user)
        except KeyError:
            continue
        if gid not in os.getgrouplist(user, account.pw_gid):
            continue
        # Talk to the user's own bus directly. `systemctl --machine=user@.host`
        # fails inside the broker unit's namespaced environment.
        runtime_dir = f"/run/user/{account.pw_uid}"
        if not os.path.exists(f"{runtime_dir}/bus"):
            print(f"ollama-unify-tray: no user bus for {user}", file=sys.stderr)
            continue
        user_systemctl = [
            "runuser", "-u", user, "--", "env",
            f"XDG_RUNTIME_DIR={runtime_dir}",
            f"DBUS_SESSION_BUS_ADDRESS=unix:path={runtime_dir}/bus",
            "systemctl", "--user",
        ]
        run(user_systemctl + ["daemon-reload"])
        run(user_systemctl + ["--no-block", "restart", TRAY_UNIT])
        print(f"ollama-unify-tray: requested restart of {TRAY_UNIT} for {user}",
              file=sys.stderr)
    if not users:
        print("ollama-unify-tray: no active graphical sessions", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="ollama-unify-tray")
    commands = parser.add_subparsers(dest="command")
    sessions = commands.add_parser(
        "start-sessions", help="start the tray in eligible graphical sessions",
    )
    sessions.add_argument("--group", default="ollama-gpu")
    args = parser.parse_args()
    if args.command == "start-sessions":
        return start_sessions(args.group)
    return run_indicator()


if __name__ == "__main__":
    sys.exit(main())
TRAY_INDICATOR
}

render_gpu_tray_user_unit() {
  cat <<TRAY_UNIT
# Generated by ollama-unify.
[Unit]
Description=ollama-unify GPU broker tray indicator
Documentation=https://github.com/robit-man/ollama-unify
PartOf=graphical-session.target
After=graphical-session.target

[Service]
Type=simple
ExecStart=$SAFETY_TRAY_PATH
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=graphical-session.target
TRAY_UNIT
}

render_gpu_agent_instructions() {
  cat <<'AGENT_INSTRUCTIONS'
# Host CUDA negotiation

This host runs the ollama-unify GPU lease broker. Before creating, starting, or resizing any Docker/container/service deployment that uses CUDA:

1. Run `docker gpu discover` and inspect the selected GPUs, active lease summaries, coordination warnings, and current policy.
2. Launch long-running CUDA services with `docker gpu run --owner NAME --justification PURPOSE --expected-duration SECONDS --vram-mib MIB --gpu GPU_UUID --ready-command 'CHECK' -- COMMAND`. Repeat `--gpu` for each reserved device. The readiness check must pass only after CUDA models are resident.
3. For an independently supervised service, use scoped `acquire --owner NAME --justification PURPOSE --expected-duration SECONDS --gpu GPU_UUID`, set the child's `CUDA_VISIBLE_DEVICES` to exactly the same UUIDs, load the CUDA models, call `ready`, call `prepare` before any VRAM growth, and call `release` only after freeing CUDA memory.
4. Lease registration requires a specific owner. Provide --justification PURPOSE and --expected-duration SECONDS when setting up workloads to help other agents coordinate; these are recommendations, and missing metadata must not halt a basic deployment. Omitted purpose and release time remain unknown and do not affect heartbeat TTL. These fields and the GPU scope are visible to other local agents; tokens are not exposed in discovery. Review current lessees before acquiring.
5. Positive --vram-mib single-GPU leases share memory when their combined promises and live usage fit; there is no lease-count cap. Zero-budget, explicit --exclusive and multi-GPU owners retain exclusive external scopes. Call prepare TOKEN --vram-mib NEW_TOTAL before exceeding the budget, and ready TOKEN --pid WORKLOAD_PID after allocation. The run wrapper registers its child automatically. Pending and revoking scoped leases defer Ollama placement. After `ready`, the broker can place Ollama lanes in measured free VRAM on single-GPU scopes. Explicit --exclusive single-GPU scopes and multi-GPU scopes stay exclusive until release: the broker retires Ollama lanes there at acquire and never places, loads, or reclaims lanes on them, because lane churn during peer-to-peer (NVLink/NCCL) traffic is unsafe. Explicit --exclusive requires fixed placement and cannot combine yield or cache evacuation; its reservation is not discarded while scoped CUDA contexts remain or telemetry is invalid. The active workload must call `prepare` before any VRAM growth. Unscoped leases retain a host-wide drain because placement cannot be proven. A non-exclusive revoked lease whose owner stops heartbeating can be abandoned once the revoke deadline passes; explicit exclusive leases additionally require verified absence of every scoped CUDA context. Always `release` rather than letting an owner exit.
6. Never bypass the broker by assuming free VRAM from a static scan. Unregistered CUDA activity quarantines affected GPUs: broker model load/unload is deferred until it exits. This cannot prevent its initial allocation or replace cooperative leases.
7. Ollama API `num_gpu` counts offloaded layers, not physical GPUs. Keep it automatic (`-1`).
8. Clients that require exact Ollama placement must send an ordered hard allowlist as `gpu_uuids` on capacity requests and `X-Ollama-Unify-GPU-UUIDs` on inference requests. The broker uses the ordered intersection with its live selected GPUs, rejects an empty intersection, and never falls back outside the allowlist.

Fixed placement preserves exact GPU requirements. For a restartable service, opt into `--placement flexible` with ordered `--gpu` candidates and `--priority` (default0, range-1000..1000). A strictly higher-priority fixed request can ask it to move to a reserved eligible destination. The owner must drain requests, gracefully stop CUDA workers and use `relocate TOKEN --handoff-id ID`, then reload on the returned UUID and call `ready`. `docker gpu run --placement flexible` supervises this sequence with `--drain-command` and `--resume-command`. A lease warning describes coordinated capacity, not a disabled GPU. No context is live-retargeted and no fixed owner is forcibly moved.

Machine-readable discovery: `/usr/local/share/ollama-unify/gpu-negotiator.json` or `http://127.0.0.1:11434/.well-known/ollama-unify-gpu-negotiator`.
AGENT_INSTRUCTIONS
}

render_global_codex_gpu_block() {
  cat <<'CODEX_BLOCK'
<!-- BEGIN ollama-unify GPU negotiator -->
## CUDA deployment policy

Before changing or starting any Docker/container/service workload that uses CUDA on this host, run `docker gpu discover` and inspect its warnings and active lease summaries. New leases require `--owner`; suggest `--justification` and `--expected-duration` when setting up workloads, but missing purpose or duration must not halt basic deployments. Owner, purpose, GPU scope, and expected release are visible to other local agents; omitted metadata remains unknown and does not change heartbeat TTL. Use scoped `docker gpu run --gpu GPU_UUID` for foreground deployments or the scoped acquire → ready → prepare/resize → release protocol for independently supervised services. The external workload must use exactly its reserved GPU UUIDs. Fixed placement preserves an exact GPU group. Restartable single-GPU services can opt in to `--placement flexible` with ordered candidates, a priority, and drain/readiness hooks; a higher-priority fixed request waits for a reserved destination, graceful stop, verified CUDA release and readiness before admission. Fixed owners are never silently moved. Pending and revoking scopes defer Ollama placement; bounded pending single-GPU budgets may coexist with other leases. Positive single-GPU VRAM budgets may share with other leases and Ollama lanes when all promises fit. Use prepare --vram-mib NEW_TOTAL before exceeding the budget, and ready --pid WORKLOAD_PID for independently supervised concurrent loaders. Zero-budget, explicit-exclusive and multi-GPU scopes remain exclusive. Do not rely on static free-VRAM scans or bypass the broker; anonymous CUDA allocation is only best-effort. Full instructions are at `/usr/local/share/ollama-unify/AGENTS.md`.
<!-- END ollama-unify GPU negotiator -->
CODEX_BLOCK
}

detect_ollama_proxy_listen() {
  local listen="${OLLAMA_SAFE_NEGOTIATOR_LISTEN:-}" effective_env
  if [ -z "$listen" ] && [ -r "$SAFETY_NEGOTIATOR_CONFIG_PATH" ]; then
    listen=$(awk -F= '$1 == "OLLAMA_UNIFY_LISTEN" { sub(/^[^=]*=/, ""); gsub(/^"|"$/, ""); print; exit }' \
      "$SAFETY_NEGOTIATOR_CONFIG_PATH" 2>/dev/null || true)
  fi
  if [ -z "$listen" ]; then
    effective_env=$(systemctl show ollama.service -p Environment --value 2>/dev/null || true)
    listen=$(printf '%s\n' "$effective_env" | grep -o 'OLLAMA_HOST=[^ "[:space:]]*' \
      | tail -n 1 | cut -d= -f2- || true)
  fi
  listen="${listen#http://}"
  listen="${listen#https://}"
  listen="${listen%/}"
  [ -n "$listen" ] || listen="127.0.0.1"
  if [ "$listen" = "$SAFETY_OLLAMA_BACKEND" ]; then listen="127.0.0.1:11434"; fi
  [[ "$listen" == *:* ]] || listen="${listen}:11434"
  [[ "$listen" =~ ^[A-Za-z0-9_.:-]+$ ]] \
    || { err "invalid negotiator listen address: $listen"; exit 2; }
  printf '%s' "$listen"
}

install_gpu_negotiator() {
  local sudo_pfx="$1"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  command -v python3 >/dev/null 2>&1 \
    || { err "python3 is required for the streaming GPU negotiator"; exit 2; }

  local proxy_listen service_user service_group access_group unit_dir config_dir helper_dir cli_dir
  local plugin_dir selected_ids model_store ollama_binary configured_environment backend_port
  local drain_timeout pending_timeout unload_timeout lease_ttl heartbeat_reconnect_grace
  local anon_poll anon_settle anon_max_drain
  local client_history_ttl client_history_limit client_lane_history_limit
  local pool_enabled pool_max_servers pool_port_start pool_instance_parallel
  local pool_idle_timeout pool_ready_timeout pool_load_timeout
  local pool_vram_reserve pool_host_reserve pool_model_overhead
  proxy_listen=$(detect_ollama_proxy_listen)
  service_user=$(systemctl show ollama.service -p User --value 2>/dev/null || true)
  service_group=$(systemctl show ollama.service -p Group --value 2>/dev/null || true)
  [ -n "$service_user" ] || service_user="ollama"
  [ -n "$service_group" ] || service_group="$service_user"
  access_group="${OLLAMA_SAFE_NEGOTIATOR_GROUP:-$service_group}"
  if [ "$EUID" -ne 0 ]; then access_group="${OLLAMA_SAFE_NEGOTIATOR_GROUP:-$(id -gn)}"; fi
  [[ "$service_user" =~ ^[A-Za-z0-9_.@-]+$ ]] \
    || { err "cannot install negotiator for unsafe service user value: $service_user"; exit 2; }
  [[ "$service_group" =~ ^[A-Za-z0-9_.@-]+$ ]] \
    || { err "cannot install negotiator for unsafe service group value: $service_group"; exit 2; }
  [[ "$access_group" =~ ^[A-Za-z0-9_.@-]+$ ]] \
    || { err "cannot install negotiator for unsafe access group value: $access_group"; exit 2; }

  drain_timeout="${OLLAMA_SAFE_NEGOTIATOR_DRAIN_TIMEOUT:-300}"
  pending_timeout="${OLLAMA_SAFE_NEGOTIATOR_PENDING_TIMEOUT:-300}"
  revoke_timeout="${OLLAMA_SAFE_NEGOTIATOR_REVOKE_TIMEOUT:-300}"
  unload_timeout="${OLLAMA_SAFE_NEGOTIATOR_UNLOAD_TIMEOUT:-120}"
  lease_ttl="${OLLAMA_SAFE_NEGOTIATOR_LEASE_TTL:-300}"
  heartbeat_reconnect_grace="${OLLAMA_SAFE_HEARTBEAT_RECONNECT_GRACE:-90}"
  anon_poll="${OLLAMA_SAFE_NEGOTIATOR_ANON_POLL:-0.5}"
  anon_settle="${OLLAMA_SAFE_NEGOTIATOR_ANON_SETTLE:-2}"
  anon_max_drain="${OLLAMA_SAFE_NEGOTIATOR_ANON_MAX_DRAIN:-15}"
  client_history_ttl="${OLLAMA_SAFE_CLIENT_HISTORY_TTL:-3600}"
  client_history_limit="${OLLAMA_SAFE_CLIENT_HISTORY_LIMIT:-256}"
  client_lane_history_limit="${OLLAMA_SAFE_CLIENT_LANE_HISTORY_LIMIT:-32}"
  configured_environment=$(systemctl show ollama.service -p Environment --value 2>/dev/null || true)
  model_store="${OLLAMA_SAFE_MODEL_STORE:-}"
  if [ -z "$model_store" ]; then
    model_store=$(printf '%s\n' "$configured_environment" \
      | grep -o 'OLLAMA_MODELS=[^ "[:space:]]*' | tail -n 1 | cut -d= -f2- || true)
  fi
  [ -n "$model_store" ] || model_store="/usr/share/ollama/.ollama/models"
  ollama_binary="${OLLAMA_SAFE_POOL_OLLAMA_BINARY:-$(command -v ollama || true)}"
  [ -n "$ollama_binary" ] || ollama_binary="/usr/local/bin/ollama"
  pool_enabled="${OLLAMA_SAFE_POOL_ENABLED:-$([ "$SAFETY_BACKEND" = cuda ] && printf 1 || printf 0)}"
  pool_max_servers="${OLLAMA_SAFE_POOL_MAX_SERVERS:-$(( ${#SAFETY_DEVICE_IDS[@]} * 2 ))}"
  backend_port="${SAFETY_OLLAMA_BACKEND##*:}"
  pool_port_start="${OLLAMA_SAFE_POOL_PORT_START:-$((backend_port + 1))}"
  pool_instance_parallel="${OLLAMA_SAFE_POOL_INSTANCE_PARALLEL:-1}"
  pool_resume_ttl="${OLLAMA_SAFE_POOL_RESUME_TTL:-30}"
  pool_idle_timeout="${OLLAMA_SAFE_POOL_IDLE_TIMEOUT:-300}"
  pool_ready_timeout="${OLLAMA_SAFE_POOL_READY_TIMEOUT:-30}"
  pool_load_timeout="${OLLAMA_SAFE_POOL_LOAD_TIMEOUT:-$drain_timeout}"
  request_activity_ttl="${OLLAMA_SAFE_REQUEST_ACTIVITY_TTL:-$drain_timeout}"
  request_detached_ttl="${OLLAMA_SAFE_REQUEST_DETACHED_TTL:-30}"
  request_cancel_grace="${OLLAMA_SAFE_REQUEST_CANCEL_GRACE:-5}"
  pool_vram_reserve="${OLLAMA_SAFE_POOL_VRAM_RESERVE_MIB:-8192}"
  pool_host_reserve="${OLLAMA_SAFE_POOL_HOST_RESERVE_MIB:-2048}"
  pool_model_overhead="${OLLAMA_SAFE_POOL_MODEL_OVERHEAD_PERCENT:-110}"
  [[ "$drain_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_DRAIN_TIMEOUT must be numeric"; exit 2; }
  [[ "$pending_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_PENDING_TIMEOUT must be numeric"; exit 2; }
  [[ "$unload_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_UNLOAD_TIMEOUT must be numeric"; exit 2; }
  [[ "$lease_ttl" =~ ^[0-9]+$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_LEASE_TTL must be an unsigned integer"; exit 2; }
  [[ "$heartbeat_reconnect_grace" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_HEARTBEAT_RECONNECT_GRACE must be numeric"; exit 2; }
  [[ "$anon_poll" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_ANON_POLL must be numeric"; exit 2; }
  [[ "$anon_settle" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_ANON_SETTLE must be numeric"; exit 2; }
  [[ "$anon_max_drain" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_NEGOTIATOR_ANON_MAX_DRAIN must be numeric"; exit 2; }
  [[ "$client_history_ttl" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_CLIENT_HISTORY_TTL must be numeric"; exit 2; }
  for value_name in client_history_limit client_lane_history_limit; do
    [[ "${!value_name}" =~ ^[0-9]+$ ]] \
      || { err "${value_name} must be an unsigned integer"; exit 2; }
    [ "${!value_name}" -ge 1 ] \
      || { err "${value_name} must be at least 1"; exit 2; }
  done
  [[ "$pool_enabled" =~ ^[01]$ ]] \
    || { err "OLLAMA_SAFE_POOL_ENABLED must be 0 or 1"; exit 2; }
  for value_name in pool_max_servers pool_port_start pool_instance_parallel pool_vram_reserve pool_host_reserve pool_model_overhead; do
    [[ "${!value_name}" =~ ^[0-9]+$ ]] \
      || { err "${value_name} must be an unsigned integer"; exit 2; }
  done
  [ "$pool_instance_parallel" -ge 1 ] \
    || { err "OLLAMA_SAFE_POOL_INSTANCE_PARALLEL must be at least 1"; exit 2; }
  [ "$pool_model_overhead" -ge 100 ] \
    || { err "OLLAMA_SAFE_POOL_MODEL_OVERHEAD_PERCENT must be at least 100"; exit 2; }
  [[ "$pool_idle_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_POOL_IDLE_TIMEOUT must be numeric"; exit 2; }
  [[ "$pool_ready_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_POOL_READY_TIMEOUT must be numeric"; exit 2; }
  [[ "$pool_load_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_POOL_LOAD_TIMEOUT must be numeric"; exit 2; }
  [[ "$request_activity_ttl" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_REQUEST_ACTIVITY_TTL must be numeric"; exit 2; }
  [[ "$request_detached_ttl" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_REQUEST_DETACHED_TTL must be numeric"; exit 2; }
  [[ "$request_cancel_grace" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_REQUEST_CANCEL_GRACE must be numeric"; exit 2; }
  [[ "$pool_resume_ttl" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || { err "OLLAMA_SAFE_POOL_RESUME_TTL must be numeric"; exit 2; }
  [ -x "$ollama_binary" ] \
    || { err "managed pool Ollama binary is not executable: $ollama_binary"; exit 2; }
  [[ "$model_store" != *'"'* ]] \
    || { err "model store path may not contain a double quote"; exit 2; }

  unit_dir="${SAFETY_NEGOTIATOR_UNIT_PATH%/*}"
  config_dir="${SAFETY_NEGOTIATOR_CONFIG_PATH%/*}"
  helper_dir="${SAFETY_NEGOTIATOR_PATH%/*}"
  cli_dir="${SAFETY_NEGOTIATOR_CLI_PATH%/*}"
  plugin_dir="${SAFETY_DOCKER_PLUGIN_PATH%/*}"
  selected_ids=$(csv_from_array "${SAFETY_DEVICE_IDS[@]}")
  "${elevate[@]}" mkdir -p "$unit_dir" "$config_dir" "$helper_dir" "$cli_dir" \
    "$plugin_dir" "$SAFETY_DISCOVERY_DIR"

  render_gpu_negotiator_script | "${elevate[@]}" tee "$SAFETY_NEGOTIATOR_PATH" >/dev/null
  render_gpu_negotiator_cli | "${elevate[@]}" tee "$SAFETY_NEGOTIATOR_CLI_PATH" >/dev/null
  render_docker_gpu_lease_plugin | "${elevate[@]}" tee "$SAFETY_DOCKER_PLUGIN_PATH" >/dev/null
  render_gpu_tray_indicator | "${elevate[@]}" tee "$SAFETY_TRAY_PATH" >/dev/null
  "${elevate[@]}" mkdir -p "${SAFETY_TRAY_UNIT_PATH%/*}"
  render_gpu_tray_user_unit | "${elevate[@]}" tee "$SAFETY_TRAY_UNIT_PATH" >/dev/null
  "${elevate[@]}" chmod 0644 "$SAFETY_TRAY_UNIT_PATH"
  "${elevate[@]}" systemctl --global enable "${SAFETY_TRAY_UNIT_PATH##*/}" >/dev/null 2>&1 \
    || warn "could not enable the tray indicator for graphical sessions"
  if "${elevate[@]}" test -f "$SAFETY_LEGACY_DOCKER_PLUGIN_PATH" \
    && "${elevate[@]}" grep -q '^# Generated by ollama-unify' "$SAFETY_LEGACY_DOCKER_PLUGIN_PATH"; then
    "${elevate[@]}" rm -f "$SAFETY_LEGACY_DOCKER_PLUGIN_PATH"
  fi
  render_gpu_agent_instructions | "${elevate[@]}" tee "$SAFETY_AGENT_INSTRUCTIONS_PATH" >/dev/null
  "${elevate[@]}" chmod 0755 "$SAFETY_NEGOTIATOR_PATH" "$SAFETY_NEGOTIATOR_CLI_PATH" \
    "$SAFETY_DOCKER_PLUGIN_PATH" "$SAFETY_TRAY_PATH"
  "${elevate[@]}" chmod 0644 "$SAFETY_AGENT_INSTRUCTIONS_PATH"

  {
    printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf 'OLLAMA_UNIFY_BACKEND="%s"\n' "$SAFETY_OLLAMA_BACKEND"
    printf 'OLLAMA_UNIFY_BACKEND_TYPE="%s"\n' "$SAFETY_BACKEND"
    printf 'OLLAMA_UNIFY_SELECTED_GPUS="%s"\n' "$selected_ids"
    printf 'OLLAMA_UNIFY_LISTEN="%s"\n' "$proxy_listen"
    printf 'OLLAMA_UNIFY_SOCKET="%s"\n' "$SAFETY_NEGOTIATOR_SOCKET"
    printf 'OLLAMA_UNIFY_MAX_CONTEXT="%s"\n' "$SAFETY_CONTEXT_LENGTH"
    printf 'OLLAMA_UNIFY_AUTO_MODEL_CONTEXT="%s"\n' "${OLLAMA_SAFE_AUTO_MODEL_CONTEXT:-1}"
    printf 'OLLAMA_UNIFY_CONTEXT_HARD_LIMIT="%s"\n' "${OLLAMA_SAFE_CONTEXT_HARD_LIMIT:-0}"
    printf 'OLLAMA_UNIFY_DRAIN_TIMEOUT="%s"\n' "$drain_timeout"
    printf 'OLLAMA_UNIFY_PENDING_TIMEOUT="%s"\n' "$pending_timeout"
    printf 'OLLAMA_UNIFY_REVOKE_TIMEOUT="%s"\n' "$revoke_timeout"
    printf 'OLLAMA_UNIFY_UNLOAD_TIMEOUT="%s"\n' "$unload_timeout"
    printf 'OLLAMA_UNIFY_LEASE_TTL="%s"\n' "$lease_ttl"
    printf 'OLLAMA_UNIFY_HEARTBEAT_RECONNECT_GRACE="%s"\n' "$heartbeat_reconnect_grace"
    printf 'OLLAMA_UNIFY_ANON_POLL="%s"\n' "$anon_poll"
    printf 'OLLAMA_UNIFY_ANON_SETTLE="%s"\n' "$anon_settle"
    printf 'OLLAMA_UNIFY_ANON_MAX_DRAIN="%s"\n' "$anon_max_drain"
    printf 'OLLAMA_UNIFY_FOREIGN_RELEASE_TOLERANCE_MIB="256"\n'
    printf 'OLLAMA_UNIFY_CLIENT_HISTORY_TTL="%s"\n' "$client_history_ttl"
    printf 'OLLAMA_UNIFY_CLIENT_HISTORY_LIMIT="%s"\n' "$client_history_limit"
    printf 'OLLAMA_UNIFY_CLIENT_LANE_HISTORY_LIMIT="%s"\n' "$client_lane_history_limit"
    printf 'OLLAMA_UNIFY_POOL_ENABLED="%s"\n' "$pool_enabled"
    printf 'OLLAMA_UNIFY_POOL_MAX_SERVERS="%s"\n' "$pool_max_servers"
    printf 'OLLAMA_UNIFY_POOL_MAX_QUEUE="64"\n'
    printf 'OLLAMA_UNIFY_POOL_RESUME_TTL="%s"\n' "$pool_resume_ttl"
    printf 'OLLAMA_UNIFY_RETAINED_REQUEST_MAX_BODY_BYTES="16777216"\n'
    printf 'OLLAMA_UNIFY_RETAINED_REQUEST_MAX_TOTAL_BYTES="134217728"\n'
    printf 'OLLAMA_UNIFY_COMPLETED_RESPONSE_TTL="120"\n'
    printf 'OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_ENTRIES="64"\n'
    printf 'OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_BODY_BYTES="8388608"\n'
    printf 'OLLAMA_UNIFY_COMPLETED_RESPONSE_MAX_TOTAL_BYTES="67108864"\n'
    printf 'OLLAMA_UNIFY_POOL_PORT_START="%s"\n' "$pool_port_start"
    printf 'OLLAMA_UNIFY_POOL_INSTANCE_PARALLEL="%s"\n' "$pool_instance_parallel"
    printf 'OLLAMA_UNIFY_POOL_IDLE_TIMEOUT="%s"\n' "$pool_idle_timeout"
    printf 'OLLAMA_UNIFY_POOL_READY_TIMEOUT="%s"\n' "$pool_ready_timeout"
    printf 'OLLAMA_UNIFY_POOL_LOAD_TIMEOUT="%s"\n' "$pool_load_timeout"
    printf 'OLLAMA_UNIFY_REQUEST_ACTIVITY_TTL="%s"\n' "$request_activity_ttl"
    printf 'OLLAMA_UNIFY_REQUEST_DETACHED_TTL="%s"\n' "$request_detached_ttl"
    printf 'OLLAMA_UNIFY_REQUEST_CANCEL_GRACE="%s"\n' "$request_cancel_grace"
    printf 'OLLAMA_UNIFY_POOL_VRAM_RESERVE_MIB="%s"\n' "$pool_vram_reserve"
    printf 'OLLAMA_UNIFY_POOL_HOST_RESERVE_MIB="%s"\n' "$pool_host_reserve"
    printf 'OLLAMA_UNIFY_POOL_MODEL_OVERHEAD_PERCENT="%s"\n' "$pool_model_overhead"
    printf 'OLLAMA_UNIFY_OLLAMA_BINARY="%s"\n' "$ollama_binary"
    printf 'OLLAMA_UNIFY_MODELS="%s"\n' "$model_store"
  } | "${elevate[@]}" tee "$SAFETY_NEGOTIATOR_CONFIG_PATH" >/dev/null
  "${elevate[@]}" chmod 0644 "$SAFETY_NEGOTIATOR_CONFIG_PATH"
  "${elevate[@]}" "$SAFETY_NEGOTIATOR_PATH" discover \
    | "${elevate[@]}" tee "$SAFETY_DISCOVERY_PATH" >/dev/null
  "${elevate[@]}" chmod 0644 "$SAFETY_DISCOVERY_PATH"

  {
    printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    cat <<UNIT
[Unit]
Description=Ollama dynamic GPU lease negotiator and API proxy
Documentation=https://github.com/robit-man/ollama-unify
Wants=ollama.service network-online.target
After=ollama.service network-online.target
StartLimitIntervalSec=5min
StartLimitBurst=5

[Service]
Type=simple
User=$service_user
Group=$access_group
EnvironmentFile=$SAFETY_NEGOTIATOR_CONFIG_PATH
RuntimeDirectory=ollama-unify
RuntimeDirectoryMode=0750
StateDirectory=ollama-unify
StateDirectoryMode=0750
ExecStartPre=$SAFETY_NEGOTIATOR_PATH self-test
ExecStart=$SAFETY_NEGOTIATOR_PATH serve
# Bring the tray indicator up in eligible desktop sessions with the broker.
ExecStartPost=-+$SAFETY_TRAY_PATH start-sessions --group $access_group
Restart=on-failure
RestartSec=5s
KillMode=mixed
TimeoutStopSec=45s
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
LockPersonality=yes
UMask=0007

[Install]
WantedBy=multi-user.target
UNIT
  } | "${elevate[@]}" tee "$SAFETY_NEGOTIATOR_UNIT_PATH" >/dev/null

  "${elevate[@]}" "$SAFETY_NEGOTIATOR_PATH" self-test >/dev/null
  ok "dynamic GPU negotiator installed: $SAFETY_NEGOTIATOR_PATH"
  ok "GPU lease client installed: $SAFETY_NEGOTIATOR_CLI_PATH"
  ok "Docker discovery plugin installed: $SAFETY_DOCKER_PLUGIN_PATH"
  ok "GPU broker tray indicator installed: $SAFETY_TRAY_PATH"
  ok "agent discovery manifest installed: $SAFETY_DISCOVERY_PATH"
  say "  Ollama backend: $SAFETY_OLLAMA_BACKEND; negotiated API: $proxy_listen"
}

refresh_installed_gpu_discovery() {
  local sudo_pfx="$1"
  local -a elevate=()
  local temp_file proxy_listen proxy_port
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  temp_file=$(mktemp "${TMPDIR:-/tmp}/ollama-unify-discovery.XXXXXX")
  proxy_listen=$(detect_ollama_proxy_listen)
  proxy_port="${proxy_listen##*:}"
  if ! python3 - "$proxy_port" "$temp_file" <<'PY'
import json
import sys
import urllib.request

try:
    port = int(sys.argv[1])
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/.well-known/ollama-unify-gpu-negotiator",
        timeout=5,
    ) as response:
        document = json.load(response)
    if not isinstance(document, dict):
        raise ValueError("discovery response is not an object")
    if document.get("schema") != "io.ollama-unify.gpu-negotiator.discovery.v1":
        raise ValueError("unexpected discovery schema")
    if document.get("available") is not True:
        raise ValueError("negotiated API reports unavailable")
    if document.get("backend_available") is not True:
        raise ValueError("Ollama backend reports unavailable")
    protocol = document.get("parallel_pool", {}).get("admission_protocol", {})
    if protocol.get("logical_request_header") != "X-Ollama-Unify-Logical-Request-Id":
        raise ValueError("installed broker lacks the logical admission protocol")
    if protocol.get("gpu_uuids_header") != "X-Ollama-Unify-GPU-UUIDs":
        raise ValueError("installed broker lacks hard per-request GPU selection")
    with open(sys.argv[2], "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    print(f"ollama-unify live discovery refresh: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    rm -f "$temp_file"
    return 1
  fi
  "${elevate[@]}" install -m 0644 "$temp_file" "$SAFETY_DISCOVERY_PATH"
  rm -f "$temp_file"
  ok "live agent discovery manifest refreshed: $SAFETY_DISCOVERY_PATH"
}

install_global_codex_gpu_instructions() {
  local sudo_pfx="$1" enabled="${OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY:-1}"
  [ "$enabled" = 1 ] || { [ "$enabled" = 0 ] && return; err "OLLAMA_SAFE_INSTALL_AGENT_DISCOVERY must be 0 or 1"; exit 2; }

  local agent_user agent_group agent_home agent_dir agent_file temp_file
  agent_user="${SUDO_USER:-${USER:-}}"
  [ -n "$agent_user" ] || return
  agent_home=$(getent passwd "$agent_user" 2>/dev/null | awk -F: 'NR == 1 { print $6 }')
  [ -n "$agent_home" ] || return
  agent_dir="$agent_home/.codex"
  [ -d "$agent_dir" ] || return
  agent_file="$agent_dir/AGENTS.md"
  if [ -L "$agent_file" ]; then
    warn "skipping global Codex discovery because $agent_file is a symlink"
    return
  fi
  agent_group=$(id -gn "$agent_user" 2>/dev/null || printf '%s' "$agent_user")
  temp_file=$(mktemp)
  if [ -f "$agent_file" ]; then
    awk '
      $0 == "<!-- BEGIN ollama-unify GPU negotiator -->" { skip=1; next }
      $0 == "<!-- END ollama-unify GPU negotiator -->" { skip=0; next }
      !skip { print }
    ' "$agent_file" > "$temp_file"
  fi
  if [ -s "$temp_file" ]; then printf '\n' >> "$temp_file"; fi
  render_global_codex_gpu_block >> "$temp_file"

  if [ "$EUID" -eq 0 ] || [ "$agent_user" != "${USER:-}" ]; then
    local -a elevate=()
    [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
    "${elevate[@]}" install -o "$agent_user" -g "$agent_group" -m 0644 "$temp_file" "$agent_file"
  else
    install -m 0644 "$temp_file" "$agent_file"
  fi
  rm -f "$temp_file"
  ok "global Codex CUDA discovery installed: $agent_file"
}

render_safety_service_directives() {
  if [ "$SYSTEMD_VERSION" -ge 235 ]; then
    printf '%s\n' "UnsetEnvironment=OLLAMA_LLM_LIBRARY"
    if [ "$SAFETY_VRAM_RESERVE_MIB" -eq 0 ]; then
      printf '%s\n' "UnsetEnvironment=OLLAMA_GPU_OVERHEAD"
    fi
    if [ "$SAFETY_GPU_PREFERRED" = 1 ]; then
      # llama.cpp treats mere presence as true for these CUDA flags. They must
      # be absent, not assigned the string "0".
      printf '%s\n' "UnsetEnvironment=GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_CUDA_REGISTER_HOST LLAMA_ARG_FIT_TARGET"
    else
      printf '%s\n' "UnsetEnvironment=GGML_CUDA_NO_PINNED LLAMA_ARG_N_GPU_LAYERS LLAMA_ARG_SPLIT_MODE LLAMA_ARG_FIT LLAMA_ARG_FIT_TARGET"
    fi
  fi
  if [ "$SAFETY_BACKEND" = "rocm" ] && [ "$SYSTEMD_VERSION" -ge 235 ]; then
    printf '%s\n' "UnsetEnvironment=CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL"
  fi
  render_safety_environment_directives
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    # The negotiator owns the public Ollama port and serializes model loads
    # around external GPU leases. Ollama itself is reachable only through the
    # loopback backend, so cooperative draining cannot be bypassed remotely.
    printf '%s\n' "Environment=\"OLLAMA_HOST=${SAFETY_OLLAMA_BACKEND}\""
  fi
  if [ "$SYSTEMD_VERSION" -ge 231 ]; then
    printf '%s\n' "MemoryAccounting=yes" "MemoryHigh=${SAFETY_HOST_MEMORY_HIGH_MIB}M" \
      "MemoryMax=${SAFETY_HOST_MEMORY_MAX_MIB}M"
  fi
  if [ "$SYSTEMD_VERSION" -ge 232 ]; then printf '%s\n' "MemorySwapMax=${SAFETY_SWAP_MAX}"; fi
  if [ "$SYSTEMD_VERSION" -ge 243 ]; then printf '%s\n' "OOMPolicy=stop"; fi
  if [ "$SYSTEMD_VERSION" -ge 227 ]; then
    printf '%s\n' "CPUAccounting=yes" "CPUWeight=${SAFETY_CPU_WEIGHT}" \
      "CPUQuota=${SAFETY_CPU_QUOTA_PERCENT}%" "IOAccounting=yes" "IOWeight=${SAFETY_IO_WEIGHT}"
  fi
  if [ "$SYSTEMD_VERSION" -ge 247 ]; then
    printf '%s\n' "ManagedOOMMemoryPressure=kill" \
      "ManagedOOMMemoryPressureLimit=${SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT}%" \
      "ManagedOOMSwap=kill"
  fi
  printf '%s\n' "OOMScoreAdjust=750" "Nice=10" "KillMode=control-group"
  if [ "$SYSTEMD_VERSION" -ge 243 ]; then
    printf '%s\n' "ExecCondition=${SAFETY_PREFLIGHT_PATH} ${SAFETY_STARTUP_HEADROOM_MIB} ${SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT}"
  else
    printf '%s\n' "ExecStartPre=${SAFETY_PREFLIGHT_PATH} ${SAFETY_STARTUP_HEADROOM_MIB} ${SAFETY_MEMORY_PRESSURE_LIMIT_PERCENT}"
  fi
  printf '%s\n' "Restart=${SAFETY_RESTART_POLICY}" "RestartSec=60s"
  if [ ${#SAFETY_PREFLIGHT_DIRECTIVES[@]} -gt 0 ]; then printf '%s\n' "${SAFETY_PREFLIGHT_DIRECTIVES[@]}"; fi
}

install_systemd_safety_policy() {
  local sudo_pfx="$1"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  local dropin_dir="/etc/systemd/system/ollama.service.d"
  local safety_file="$dropin_dir/zzz-ollama-unify-safety.conf"
  local preflight_dir="${SAFETY_PREFLIGHT_PATH%/*}"

  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    command -v python3 >/dev/null 2>&1 \
      || { err "python3 is required before Ollama can be routed through the GPU negotiator"; exit 2; }
  fi

  "${elevate[@]}" mkdir -p "$dropin_dir" "$preflight_dir"
  render_safety_preflight_script | "${elevate[@]}" tee "$SAFETY_PREFLIGHT_PATH" >/dev/null
  "${elevate[@]}" chmod 0755 "$SAFETY_PREFLIGHT_PATH"
  render_gpu_preflight_script | "${elevate[@]}" tee "$SAFETY_GPU_PREFLIGHT_PATH" >/dev/null
  "${elevate[@]}" chmod 0755 "$SAFETY_GPU_PREFLIGHT_PATH"
  {
    printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf '[Unit]\nStartLimitIntervalSec=5min\nStartLimitBurst=2\n\n[Service]\n'
    render_safety_service_directives
  } | "${elevate[@]}" tee "$safety_file" >/dev/null

  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    install_gpu_negotiator "$sudo_pfx"
    install_global_codex_gpu_instructions "$sudo_pfx"
  fi

  "${elevate[@]}" systemctl daemon-reload
  "${elevate[@]}" systemctl enable ollama.service >/dev/null
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    "${elevate[@]}" systemctl enable ollama-unify-negotiator.service >/dev/null
  fi
  if [ "$SYSTEMD_VERSION" -ge 247 ]; then
    if "${elevate[@]}" systemctl enable --now systemd-oomd.service >/dev/null 2>&1; then
      ok "systemd-oomd is enabled for proactive memory-pressure kills"
    else
      warn "systemd-oomd could not be enabled; MemoryMax remains active, but PSI-based killing is unavailable"
    fi
  fi
  if command -v systemd-analyze >/dev/null 2>&1; then
    if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
      "${elevate[@]}" systemd-analyze verify ollama.service ollama-unify-negotiator.service
    else
      "${elevate[@]}" systemd-analyze verify ollama.service
    fi
  fi
  ok "memory-pressure preflight installed: $SAFETY_PREFLIGHT_PATH"
  ok "late-priority safety drop-in installed: $safety_file"
  install_reconcile_watchdog "$sudo_pfx"
}

install_safety_only() {
  banner
  [ "$HOST_SERVICE_MANAGER" = "systemd" ] \
    || { err "--install-safety requires a systemd host"; exit 2; }
  systemctl cat ollama.service >/dev/null 2>&1 \
    || { err "ollama.service was not found"; exit 2; }
  require awk
  build_safety_profile
  print_safety_profile

  local SUDO=""
  if [ "$EUID" -ne 0 ]; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo is required to install the systemd policy"; exit 2; }
    SUDO="sudo"
    $SUDO -n true 2>/dev/null || $SUDO -v
  fi

  local was_active=0 negotiator_was_active=0
  if systemctl is-active ollama-unify-negotiator.service >/dev/null 2>&1; then
    negotiator_was_active=1
    $SUDO systemctl stop ollama-unify-negotiator.service
    ok "ollama-unify-negotiator.service stopped while its policy is replaced"
  fi
  if systemctl is-active ollama.service >/dev/null 2>&1; then
    was_active=1
    $SUDO systemctl stop ollama.service
    ok "ollama.service stopped while its policy is replaced"
  fi

  hdr "Installing Ollama safety policy"
  install_systemd_safety_policy "$SUDO"

  if [ "$was_active" = 1 ]; then
    $SUDO systemctl start ollama.service || true
    if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
      $SUDO systemctl start ollama-unify-negotiator.service || true
    fi
    if systemctl is-active ollama.service >/dev/null 2>&1; then
      ok "ollama.service restarted under the new policy"
      if systemctl is-active ollama-unify-negotiator.service >/dev/null 2>&1; then
        refresh_installed_gpu_discovery "$SUDO" \
          || warn "live discovery refresh failed; inspect the negotiated API before trusting the static manifest"
      else
        warn "ollama-unify-negotiator.service did not become active; static discovery was not refreshed"
      fi
    else
      warn "ollama.service remains stopped because the safety condition refused the restart"
    fi
  else
    say "  ollama.service was already stopped and has been left stopped."
    if [ "$negotiator_was_active" = 1 ]; then
      warn "the negotiator was active without Ollama and has been left stopped with it"
    fi
  fi
}

preview_safety_profile() {
  banner
  build_safety_profile
  print_safety_profile
  if [ "$HOST_SERVICE_MANAGER" = "systemd" ]; then
    hdr "Generated late-priority systemd drop-in"
    printf '%s\n' "# Managed by ollama-unify" "[Unit]" "StartLimitIntervalSec=5min" \
      "StartLimitBurst=2" "" "[Service]"
    render_safety_service_directives
  else
    hdr "Generated portable Ollama environment policy"
    render_safety_shell_exports
  fi
}

classify_host() {
  banner
  build_safety_profile
  print_safety_profile
}

print_environment_policy() {
  build_safety_profile
  render_safety_shell_exports
}

# ─────────────────────────────────────────── store discovery (read-only)
declare -a STORE_PATHS=()
canonical_path() {
  local path="$1"
  if command -v realpath >/dev/null 2>&1 && realpath -m -- "$path" >/dev/null 2>&1; then
    realpath -m -- "$path"
    return
  fi
  if readlink -m -- "$path" >/dev/null 2>&1; then
    readlink -m -- "$path"
    return
  fi
  if [ -d "$path" ]; then
    (cd "$path" 2>/dev/null && pwd -P)
    return
  fi
  [[ "$path" == /* ]] || path="$PWD/$path"
  local parent="${path%/*}" base="${path##*/}"
  if [ -d "$parent" ]; then
    printf '%s/%s\n' "$(cd "$parent" 2>/dev/null && pwd -P)" "$base"
  else
    printf '%s\n' "$path"
  fi
}

add_store() {
  local p="$1"
  [ -n "$p" ] || return 0
  # canonicalize without requiring existence
  p="$(canonical_path "$p")"
  [ -d "$p" ] || return 0
  for existing in "${STORE_PATHS[@]:-}"; do [ "$existing" = "$p" ] && return 0; done
  STORE_PATHS+=("$p")
}

discover_stores() {
  hdr "Scanning for ollama model stores…"
  # default locations
  add_store "${HOME}/.ollama/models"
  add_store "/usr/share/ollama/.ollama/models"
  add_store "/var/lib/ollama/.ollama/models"
  add_store "/root/.ollama/models"
  # shell env
  [ -n "${OLLAMA_MODELS:-}" ] && add_store "$OLLAMA_MODELS"
  # /etc/default/ollama
  if [ -r /etc/default/ollama ]; then
    local v
    v=$(grep -E '^OLLAMA_MODELS=' /etc/default/ollama 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"'' || true)
    add_store "$v"
  fi
  # /etc/environment
  if [ -r /etc/environment ]; then
    local v
    v=$(grep -E '^OLLAMA_MODELS=' /etc/environment 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"'' || true)
    add_store "$v"
  fi
  # systemd unit + drop-ins
  if [ "$HAS_SYSTEMD" = 1 ]; then
    if systemctl cat ollama >/dev/null 2>&1; then
      local v
      v=$(systemctl cat ollama 2>/dev/null \
        | sed -n 's/.*OLLAMA_MODELS=\([^"[:space:]]*\).*/\1/p' | tail -1)
      add_store "$v"
    fi
  fi
  # running runner cmdlines
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r path; do add_store "$path"; done < <(
      pgrep -af 'ollama runner' 2>/dev/null \
        | sed -n 's#.*\(/[^ ]*/blobs/sha256-[a-f0-9]*\).*#\1#p' \
        | sed 's|/blobs/.*||' | sort -u
    )
  fi
}

count_manifests() { find "$1/manifests" -type f 2>/dev/null | wc -l; }
count_blobs()     { find "$1/blobs"     -type f 2>/dev/null | wc -l; }
fs_of()           { df -Pk "$1" 2>/dev/null | awk 'END {print $1" ("$NF")"}'; }
dev_of()          { df -Pk "$1" 2>/dev/null | awk 'END {print $1}'; }
mount_of()        { df -Pk "$1" 2>/dev/null | awk 'END {print $NF}'; }
own_of()          { stat -c '%U:%G' "$1" 2>/dev/null || stat -f '%Su:%Sg' "$1" 2>/dev/null || echo "?"; }
size_of() {
  if du -sb /dev/null >/dev/null 2>&1; then du -sb "$1" 2>/dev/null | awk '{print $1}'
  else du -sk "$1" 2>/dev/null | awk '{print $1 * 1024}'; fi
}
human() {
  local b="${1:-0}"; local u=(B K M G T P); local i=0
  while [ "$b" -ge 1024 ] && [ $i -lt 5 ]; do b=$(( b / 1024 )); i=$(( i + 1 )); done
  printf '%s%s' "$b" "${u[$i]}"
}

# ────────────────────────── store table printing + reference detection
declare -a SERVICE_PIDS=() MANUAL_SERVE_PIDS=()
detect_daemons() {
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl is-active ollama >/dev/null 2>&1; then
    SERVICE_PIDS+=("$(systemctl show -p MainPID --value ollama 2>/dev/null)")
  fi
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r pid; do
      # skip the systemd MainPID we already recorded
      local skip=0
      for sp in "${SERVICE_PIDS[@]:-}"; do [ "$pid" = "$sp" ] && skip=1; done
      [ $skip = 1 ] && continue
      MANUAL_SERVE_PIDS+=("$pid")
    done < <(pgrep -f 'ollama serve' 2>/dev/null || true)
  fi
}

print_store_table() {
  printf '%-4s %-45s %-8s %-22s %-9s %-9s %s\n' "  #" "Path" "Size" "Filesystem" "Manifests" "Blobs" "Owner"
  printf '%-4s %-45s %-8s %-22s %-9s %-9s %s\n' " ──" "────" "────" "──────────" "─────────" "─────" "─────"
  local i=0
  for s in "${STORE_PATHS[@]}"; do
    i=$((i + 1))
    local m b sz fs own
    m=$(count_manifests "$s"); b=$(count_blobs "$s")
    sz=$(human "$(size_of "$s")"); fs=$(fs_of "$s"); own=$(own_of "$s")
    printf '%-4s %-45s %-8s %-22s %-9s %-9s %s\n' "[$i]" "$s" "$sz" "$fs" "$m" "$b" "$own"
  done
}

# ─────────────────────────────────────── mount-point candidates for dest
print_dest_candidates() {
  hdr "Available mount points for unified storage:"
  df -hP 2>/dev/null \
    | awk 'NR==1 {print "  "$0; next} ($NF ~ /^\/(home|Users|srv|var|opt|mnt|media|data|Volumes)(\/|$)/) || $NF=="/" {if (!seen[$NF]++) print "  "$0}'
}

# ────────────────────────────────────────── transfer-strategy selection
# Echoes one of: "mv" (same fs), "reflink" (CoW), "rsync" (cross-fs)
transfer_strategy() {
  local src="$1" dst="$2"
  local sd dd
  sd=$(dev_of "$src" 2>/dev/null || echo a)
  dd=$(dev_of "$dst" 2>/dev/null || echo b)
  if [ "$sd" = "$dd" ]; then
    echo "mv"; return
  fi
  # Reflink test when GNU cp exposes it; otherwise use rsync cross-filesystem.
  local probe="$dst/.reflink_probe_$$" cp_help=""
  cp_help=$(cp --help 2>&1 || true)
  if [[ "$cp_help" == *--reflink* ]] && cp --reflink=always /dev/null "$probe" 2>/dev/null; then
    rm -f "$probe"
    echo "reflink"; return
  fi
  rm -f "$probe" 2>/dev/null || true
  echo "rsync"
}

# ──────────────────────────────────────── execute transfer per strategy
transfer_store() {
  local src="$1" dst="$2" strategy="$3" sudo_pfx="$4"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  case "$strategy" in
    mv)
      ok "same filesystem detected → using mv (instant)"
      # mv each subdir's contents so we merge into existing dst structure
      "${elevate[@]}" mkdir -p "$dst/manifests" "$dst/blobs"
      if [ -d "$src/manifests" ]; then
        while IFS= read -r -d '' item; do
          "${elevate[@]}" mv -n "$item" "$dst/manifests/"
        done < <(find "$src/manifests" -mindepth 1 -maxdepth 1 -print0)
      fi
      if [ -d "$src/blobs" ]; then
        while IFS= read -r -d '' item; do
          "${elevate[@]}" mv -n "$item" "$dst/blobs/"
        done < <(find "$src/blobs" -mindepth 1 -maxdepth 1 -print0)
      fi
      ;;
    reflink)
      ok "reflink-capable filesystem detected → using cp --reflink=auto"
      "${elevate[@]}" cp -a --reflink=auto -n "$src/." "$dst/"
      ;;
    rsync)
      ok "cross-filesystem copy → using rsync (local-copy tuned)"
      local rsync_help
      rsync_help=$(rsync --help 2>&1)
      local -a rsync_args=(-a --ignore-existing)
      [[ "$rsync_help" == *--hard-links* ]] && rsync_args+=(-H)
      [[ "$rsync_help" == *--whole-file* ]] && rsync_args+=(--whole-file)
      [[ "$rsync_help" == *--inplace* ]] && rsync_args+=(--inplace)
      [[ "$rsync_help" == *--no-compress* ]] && rsync_args+=(--no-compress)
      if [[ "$rsync_help" == *--info* ]]; then rsync_args+=(--info=progress2); else rsync_args+=(--progress); fi
      "${elevate[@]}" rsync "${rsync_args[@]}" "$src/" "$dst/"
      ;;
  esac
}

plan_requires_sudo() {
  local do_systemd="$1" do_service_user="$2" destination="$3"
  if [ "$do_systemd" = 1 ] || [ "$do_service_user" = 1 ] \
    || [[ "$destination" =~ ^/(srv|var|opt|usr|etc) ]]; then
    return 0
  fi
  [ ${#SERVICE_PIDS[@]} -gt 0 ] && [ -n "${SERVICE_PIDS[0]:-}" ]
}

# ───────────────────────────────────────────────────────────── main flow
# ──────────────────────────────── Ollama update reconciliation
# An Ollama upgrade rewrites /etc/systemd/system/ollama.service and restarts the
# daemon. The unit it installs carries no OLLAMA_HOST, so the pinned loopback
# backend exists only in our late-priority drop-in. If that drop-in is lost or
# drifts, Ollama falls back to its built-in 0.0.0.0:11434 and collides head-on
# with the negotiator that already owns that address. These helpers detect the
# drift, repair it, and drive upgrades inside a safe stop → repin → start
# envelope so the proxy never races the daemon it fronts.

DRIFT_FINDINGS=()

ollama_binary_path() {
  local exec_start path
  exec_start=$(systemctl show ollama.service -p ExecStart --value 2>/dev/null || true)
  path=$(printf '%s' "$exec_start" | grep -oE 'path=[^ ;]+' | head -n1 | cut -d= -f2- || true)
  if [ -n "$path" ] && [ -x "$path" ]; then printf '%s' "$path"; return 0; fi
  command -v ollama 2>/dev/null || return 1
}

ollama_installed_version() {
  local bin ver
  bin=$(ollama_binary_path) || return 1
  ver=$("$bin" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)
  [ -n "$ver" ] || return 1
  printf '%s' "$ver"
}

ollama_latest_version() {
  [ "$HAS_CURL" = 1 ] || return 1
  curl -fsSL --max-time 10 "$SAFETY_OLLAMA_RELEASE_API" 2>/dev/null \
    | grep -oE '"tag_name"[[:space:]]*:[[:space:]]*"[^"]+"' | head -n1 \
    | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1
}

# true when $1 sorts strictly before $2 under version ordering
version_lt() {
  [ "$1" = "$2" ] && return 1
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]
}

port_holder_pid() {
  local port="$1"
  command -v ss >/dev/null 2>&1 || return 1
  ss -tlnpH 2>/dev/null | awk -v p=":${port}\$" '$4 ~ p {print; exit}' \
    | grep -oE 'pid=[0-9]+' | head -n1 | cut -d= -f2
}

effective_ollama_host() {
  systemctl show ollama.service -p Environment --value 2>/dev/null \
    | tr ' ' '\n' | grep -E '^OLLAMA_HOST=' | tail -n1 | cut -d= -f2- | tr -d '"'
}

negotiator_config_value() {
  local key="$1"
  [ -r "$SAFETY_NEGOTIATOR_CONFIG_PATH" ] || return 1
  awk -F= -v k="$key" '$1 == k { sub(/^[^=]*=/, ""); gsub(/^"|"$/, ""); print; exit }' \
    "$SAFETY_NEGOTIATOR_CONFIG_PATH"
}

state_value() {
  local key="$1"
  [ -r "$SAFETY_STATE_PATH" ] || return 1
  awk -F= -v k="$key" '$1 == k { sub(/^[^=]*=/, ""); gsub(/^"|"$/, ""); print; exit }' \
    "$SAFETY_STATE_PATH"
}

render_reconcile_state() {
  local version
  version=$(ollama_installed_version || printf 'unknown')
  printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'OLLAMA_UNIFY_PINNED_BACKEND="%s"\n' "$SAFETY_OLLAMA_BACKEND"
  printf 'OLLAMA_UNIFY_PINNED_LISTEN="%s"\n' "$(detect_ollama_proxy_listen)"
  printf 'OLLAMA_UNIFY_OLLAMA_VERSION="%s"\n' "$version"
  printf 'OLLAMA_UNIFY_OLLAMA_BINARY="%s"\n' "$(ollama_binary_path || printf '')"
}

# Populates DRIFT_FINDINGS with everything that would break the proxy↔backend pair.
collect_policy_drift() {
  DRIFT_FINDINGS=()
  local dropin="/etc/systemd/system/ollama.service.d/zzz-ollama-unify-safety.conf"
  local host backend listen backend_port holder_pid main_pid holder_comm

  [ -r "$dropin" ] || DRIFT_FINDINGS+=(
    "safety drop-in is missing ($dropin); Ollama would fall back to its built-in 0.0.0.0:11434")

  host=$(effective_ollama_host || true)
  if [ -z "$host" ]; then
    DRIFT_FINDINGS+=("ollama.service exposes no OLLAMA_HOST; the daemon would bind its default address")
  elif [ "$host" != "$SAFETY_OLLAMA_BACKEND" ]; then
    DRIFT_FINDINGS+=("ollama.service OLLAMA_HOST is $host; the pinned backend is $SAFETY_OLLAMA_BACKEND")
  fi

  if [ -r "$SAFETY_NEGOTIATOR_CONFIG_PATH" ]; then
    backend=$(negotiator_config_value OLLAMA_UNIFY_BACKEND || true)
    listen=$(negotiator_config_value OLLAMA_UNIFY_LISTEN || true)
    if [ -n "$backend" ] && [ "$backend" != "$SAFETY_OLLAMA_BACKEND" ]; then
      DRIFT_FINDINGS+=("negotiator OLLAMA_UNIFY_BACKEND is $backend; the pinned backend is $SAFETY_OLLAMA_BACKEND")
    fi
    if [ -n "$listen" ] && [ "$listen" = "$SAFETY_OLLAMA_BACKEND" ]; then
      DRIFT_FINDINGS+=("negotiator listen address equals the backend address ($listen); the proxy would loop onto itself")
    fi
  fi

  backend_port="${SAFETY_OLLAMA_BACKEND##*:}"
  holder_pid=$(port_holder_pid "$backend_port" || true)
  main_pid=$(systemctl show ollama.service -p MainPID --value 2>/dev/null || printf '0')
  if [ -n "$holder_pid" ] && [ "$holder_pid" != "${main_pid:-0}" ]; then
    holder_comm=$(ps -o comm= -p "$holder_pid" 2>/dev/null || printf 'unknown')
    DRIFT_FINDINGS+=("backend port $backend_port is held by PID $holder_pid ($holder_comm), not by ollama.service")
  fi
}

# Returns 0 when the negotiated stack answers end to end.
verify_negotiated_stack() {
  local listen probe host port
  systemctl is-active ollama.service >/dev/null 2>&1 \
    || { warn "ollama.service is not active"; return 1; }
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    systemctl is-active ollama-unify-negotiator.service >/dev/null 2>&1 \
      || { warn "ollama-unify-negotiator.service is not active"; return 1; }
  fi
  listen=$(negotiator_config_value OLLAMA_UNIFY_LISTEN 2>/dev/null || printf '')
  [ -n "$listen" ] || listen="$SAFETY_OLLAMA_BACKEND"
  host="${listen%:*}"; port="${listen##*:}"
  [ "$host" = "0.0.0.0" ] && host="127.0.0.1"
  [ "$HAS_CURL" = 1 ] || { warn "curl is unavailable; skipping the end-to-end probe"; return 0; }
  probe=$(curl -fsS --max-time 15 "http://${host}:${port}/api/tags" 2>/dev/null || true)
  [ -n "$probe" ] || { warn "the negotiated API at ${host}:${port} did not answer /api/tags"; return 1; }
  ok "negotiated API answers on ${host}:${port}"
}

# Ordered cycle: the proxy must release its socket before the backend moves.
cycle_negotiated_stack() {
  local sudo_pfx="$1"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  "${elevate[@]}" systemctl stop ollama-unify-negotiator.service 2>/dev/null || true
  "${elevate[@]}" systemctl restart ollama.service || {
    err "ollama.service refused to start; the safety condition or the backend port is still blocked"
    return 1
  }
  if [ "$SAFETY_NEGOTIATOR_ENABLED" = 1 ]; then
    "${elevate[@]}" systemctl start ollama-unify-negotiator.service || {
      err "ollama-unify-negotiator.service refused to start"
      return 1
    }
  fi
}

report_update_status() {
  local installed latest recorded
  installed=$(ollama_installed_version || printf '')
  recorded=$(state_value OLLAMA_UNIFY_OLLAMA_VERSION 2>/dev/null || printf '')
  hdr "Ollama release status"
  if [ -n "$installed" ]; then
    say "  Installed: $installed ($(ollama_binary_path || printf 'binary not found'))"
  else
    warn "  Installed: could not read a version from the Ollama binary"
  fi
  if [ -n "$recorded" ] && [ -n "$installed" ] && [ "$recorded" != "$installed" ]; then
    warn "  Ollama moved $recorded → $installed since the policy was last applied"
  fi
  latest=$(ollama_latest_version || printf '')
  if [ -z "$latest" ]; then
    say "  Latest:    unavailable (no network or GitHub API unreachable)"
  elif [ -z "$installed" ]; then
    say "  Latest:    $latest"
  elif version_lt "$installed" "$latest"; then
    warn "  Latest:    $latest — an update is available; run --update-ollama to take it safely"
  else
    ok "  Latest:    $latest — Ollama is current"
  fi
}

report_policy_drift() {
  collect_policy_drift
  hdr "Pinned topology"
  say "  Ollama backend:  $SAFETY_OLLAMA_BACKEND"
  say "  Negotiated API:  $(negotiator_config_value OLLAMA_UNIFY_LISTEN 2>/dev/null || printf 'not installed')"
  if [ ${#DRIFT_FINDINGS[@]} -eq 0 ]; then
    ok "no policy drift; the proxy and the backend agree on their addresses"
    return 0
  fi
  hdr "Policy drift"
  local finding
  for finding in "${DRIFT_FINDINGS[@]}"; do warn "  $finding"; done
  return 1
}

check_ollama_update() {
  banner
  [ "$HOST_SERVICE_MANAGER" = "systemd" ] \
    || { err "--check-update requires a systemd host"; exit 2; }
  build_safety_profile
  report_update_status
  if report_policy_drift; then
    say ""
    ok "nothing to reconcile"
  else
    say ""
    warn "run --reconcile to repin Ollama and restart the pair in the correct order"
  fi
}

reconcile_after_update() {
  banner
  if [ -x /usr/local/sbin/ollama-version-guard ]; then
    err "Ollama release reconciliation is managed by the host version guard. Run sudo ollama-version-guard ensure."
    err "The broker and running Ollama have been left untouched."
    return 2
  fi
  [ "$HOST_SERVICE_MANAGER" = "systemd" ] \
    || { err "--reconcile requires a systemd host"; exit 2; }
  systemctl cat ollama.service >/dev/null 2>&1 \
    || { err "ollama.service was not found"; exit 2; }
  build_safety_profile
  report_update_status

  local drifted=0
  report_policy_drift || drifted=1

  local installed recorded
  installed=$(ollama_installed_version || printf '')
  recorded=$(state_value OLLAMA_UNIFY_OLLAMA_VERSION 2>/dev/null || printf '')
  [ -n "$installed" ] && [ -n "$recorded" ] && [ "$installed" != "$recorded" ] && drifted=1

  if [ "$drifted" = 0 ]; then
    say ""
    ok "policy already matches the running stack; nothing was changed"
    verify_negotiated_stack || true
    return 0
  fi

  local SUDO=""
  if [ "$EUID" -ne 0 ]; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo is required to reconcile the systemd policy"; exit 2; }
    SUDO="sudo"
    $SUDO -n true 2>/dev/null || $SUDO -v
  fi

  hdr "Reapplying the pinned safety policy"
  $SUDO systemctl stop ollama-unify-negotiator.service 2>/dev/null || true
  $SUDO systemctl stop ollama.service 2>/dev/null || true
  install_systemd_safety_policy "$SUDO"

  hdr "Restarting the negotiated pair"
  cycle_negotiated_stack "$SUDO" || exit 1
  verify_negotiated_stack || exit 1
  ok "Ollama is repinned to $SAFETY_OLLAMA_BACKEND behind the negotiator"
}

update_ollama() {
  banner
  if [ -x /usr/local/sbin/ollama-version-guard ]; then
    err "Ollama upgrades are permission-gated on this host. Use ollama-version-guard approve/stage/activate with explicit operator permission."
    err "The broker and running Ollama have been left untouched."
    return 2
  fi
  [ "$HOST_SERVICE_MANAGER" = "systemd" ] \
    || { err "--update-ollama requires a systemd host"; exit 2; }
  [ "$HAS_CURL" = 1 ] || { err "curl is required to download an Ollama update"; exit 2; }
  build_safety_profile
  report_update_status

  local installed latest
  installed=$(ollama_installed_version || printf '')
  latest=$(ollama_latest_version || printf '')
  if [ -z "$latest" ]; then
    err "the latest Ollama release could not be determined; refusing to run the installer blind"
    exit 2
  fi
  if [ -n "$installed" ] && ! version_lt "$installed" "$latest"; then
    say ""
    ok "Ollama $installed is already current; nothing to update"
    exit 0
  fi

  say ""
  say "  The official installer rewrites /etc/systemd/system/ollama.service and restarts"
  say "  the daemon. ollama-unify will stop the negotiator first, let the installer run,"
  say "  then repin the backend to $SAFETY_OLLAMA_BACKEND before the proxy comes back."
  if [ -t 0 ] && [ "${OLLAMA_UNIFY_ASSUME_YES:-0}" != "1" ]; then
    confirm "Update Ollama ${installed:-unknown} → $latest now?" "Y" \
      || { say "  Left unchanged."; exit 0; }
  fi

  local SUDO=""
  if [ "$EUID" -ne 0 ]; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo is required to update Ollama"; exit 2; }
    SUDO="sudo"
    $SUDO -n true 2>/dev/null || $SUDO -v
  fi

  hdr "Quiescing the negotiated pair"
  $SUDO systemctl stop ollama-unify-negotiator.service 2>/dev/null || true
  ok "negotiator stopped; the public address is free while the installer runs"
  $SUDO systemctl stop ollama.service 2>/dev/null || true
  ok "ollama.service stopped"

  hdr "Running the official Ollama installer"
  if ! curl -fsSL "$SAFETY_OLLAMA_INSTALL_URL" | $SUDO sh; then
    err "the Ollama installer failed; reconciling the previous policy before exiting"
    install_systemd_safety_policy "$SUDO"
    cycle_negotiated_stack "$SUDO" || true
    exit 1
  fi

  hdr "Repinning Ollama behind the negotiator"
  install_systemd_safety_policy "$SUDO"
  cycle_negotiated_stack "$SUDO" || exit 1
  verify_negotiated_stack || exit 1
  ok "Ollama updated to $(ollama_installed_version || printf '%s' "$latest") and repinned to $SAFETY_OLLAMA_BACKEND"
}

# Standalone repair helper: no repo checkout, no python, reads the installed state.
render_reconcile_helper() {
  cat <<'RECONCILE'
#!/usr/bin/env bash
# Managed by ollama-unify — repins Ollama after an out-of-band upgrade.
#
# The Ollama installer rewrites ollama.service and restarts the daemon. The
# loopback backend address lives only in the ollama-unify drop-in, so an upgrade
# that clears or bypasses it drops Ollama back onto 0.0.0.0:11434 — the address
# the negotiator already owns. This helper re-asserts the pin and restarts the
# pair in the only safe order: proxy down, backend up, proxy up.
set -euo pipefail

# A host version guard owns release admission and rollback. Never accept an
# out-of-band executable as a new baseline or restart GPU services around it.
if [ -x /usr/local/sbin/ollama-version-guard ]; then
  exec /usr/local/sbin/ollama-version-guard ensure
fi

STATE="/usr/local/share/ollama-unify/state.env"
DROPIN="/etc/systemd/system/ollama.service.d/zzz-ollama-unify-safety.conf"
NEG_CONF="/etc/default/ollama-unify-negotiator"

log() { printf 'ollama-unify-reconcile: %s\n' "$*"; }

[ -r "$STATE" ] || { log "no installed state at $STATE; nothing to reconcile"; exit 0; }
# shellcheck disable=SC1090
. "$STATE"

BACKEND="${OLLAMA_UNIFY_PINNED_BACKEND:-}"
[ -n "$BACKEND" ] || { log "state carries no pinned backend; nothing to reconcile"; exit 0; }

changed=0

current_version="$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1 || true)"
if [ -n "$current_version" ] && [ "$current_version" != "${OLLAMA_UNIFY_OLLAMA_VERSION:-}" ]; then
  log "Ollama moved ${OLLAMA_UNIFY_OLLAMA_VERSION:-unknown} -> $current_version"
  changed=1
fi

if [ ! -r "$DROPIN" ]; then
  log "safety drop-in missing; restoring the backend pin only"
  mkdir -p "$(dirname "$DROPIN")"
  printf '# Restored by ollama-unify-reconcile\n[Service]\nEnvironment="OLLAMA_HOST=%s"\n' \
    "$BACKEND" > "$DROPIN"
  log "run 'ollama-unify.sh --install-safety' to restore the full containment policy"
  changed=1
elif ! grep -qF "OLLAMA_HOST=${BACKEND}" "$DROPIN"; then
  log "drop-in OLLAMA_HOST drifted; repinning to $BACKEND"
  if grep -qE '^Environment="OLLAMA_HOST=' "$DROPIN"; then
    sed -i -E "s|^Environment=\"OLLAMA_HOST=.*\"|Environment=\"OLLAMA_HOST=${BACKEND}\"|" "$DROPIN"
  else
    printf 'Environment="OLLAMA_HOST=%s"\n' "$BACKEND" >> "$DROPIN"
  fi
  changed=1
fi

if [ -w "$NEG_CONF" ] && ! grep -qF "OLLAMA_UNIFY_BACKEND=\"${BACKEND}\"" "$NEG_CONF"; then
  log "negotiator backend drifted; repinning to $BACKEND"
  sed -i -E "s|^OLLAMA_UNIFY_BACKEND=.*|OLLAMA_UNIFY_BACKEND=\"${BACKEND}\"|" "$NEG_CONF"
  changed=1
fi

if [ "$changed" = 0 ]; then
  log "no drift detected"
  exit 0
fi

systemctl daemon-reload
systemctl stop ollama-unify-negotiator.service 2>/dev/null || true
systemctl restart ollama.service
if systemctl list-unit-files ollama-unify-negotiator.service >/dev/null 2>&1; then
  systemctl start ollama-unify-negotiator.service || log "negotiator failed to start"
fi

if [ -n "$current_version" ]; then
  sed -i -E "s|^OLLAMA_UNIFY_OLLAMA_VERSION=.*|OLLAMA_UNIFY_OLLAMA_VERSION=\"${current_version}\"|" "$STATE" || true
fi
log "reconciled; Ollama pinned to $BACKEND"
RECONCILE
}

render_reconcile_units() {
  local binary="$1"
  cat <<UNIT
# Managed by ollama-unify — generated $(date '+%Y-%m-%dT%H:%M:%S%z')
[Unit]
Description=Repin Ollama behind the ollama-unify negotiator after an upgrade
Documentation=https://github.com/robit-man/ollama-unify
After=ollama.service

[Service]
Type=oneshot
ExecStart=$SAFETY_RECONCILE_HELPER_PATH
UNIT
}

render_reconcile_path_unit() {
  local binary="$1"
  cat <<UNIT
# Managed by ollama-unify — generated $(date '+%Y-%m-%dT%H:%M:%S%z')
[Unit]
Description=Watch the Ollama binary for upgrades that clear the pinned backend
Documentation=https://github.com/robit-man/ollama-unify

[Path]
PathChanged=$binary
Unit=ollama-unify-reconcile.service

[Install]
WantedBy=multi-user.target
UNIT
}

install_reconcile_watchdog() {
  local sudo_pfx="$1"
  local -a elevate=()
  [ -n "$sudo_pfx" ] && elevate=("$sudo_pfx")
  local binary helper_dir
  binary=$(ollama_binary_path || printf '')
  helper_dir="${SAFETY_RECONCILE_HELPER_PATH%/*}"

  "${elevate[@]}" mkdir -p "$helper_dir" "$SAFETY_DISCOVERY_DIR"
  render_reconcile_helper | "${elevate[@]}" tee "$SAFETY_RECONCILE_HELPER_PATH" >/dev/null
  "${elevate[@]}" chmod 0755 "$SAFETY_RECONCILE_HELPER_PATH"
  render_reconcile_state | "${elevate[@]}" tee "$SAFETY_STATE_PATH" >/dev/null
  "${elevate[@]}" chmod 0644 "$SAFETY_STATE_PATH"
  ok "reconcile helper installed: $SAFETY_RECONCILE_HELPER_PATH"

  if [ -z "$binary" ]; then
    warn "the Ollama binary could not be located; the upgrade watchdog was not installed"
    return 0
  fi
  render_reconcile_units "$binary" | "${elevate[@]}" tee "$SAFETY_RECONCILE_SERVICE_PATH" >/dev/null
  render_reconcile_path_unit "$binary" | "${elevate[@]}" tee "$SAFETY_RECONCILE_PATH_UNIT_PATH" >/dev/null
  "${elevate[@]}" systemctl daemon-reload
  "${elevate[@]}" systemctl enable --now ollama-unify-reconcile.path >/dev/null 2>&1 \
    || warn "ollama-unify-reconcile.path could not be enabled"
  ok "upgrade watchdog armed on $binary"
}

main() {
  case "${1:-}" in
    --classify)
      classify_host
      exit 0
      ;;
    --safety-preview)
      preview_safety_profile
      exit 0
      ;;
    --print-env)
      print_environment_policy
      exit 0
      ;;
    --install-safety)
      detect_host_profile
      install_safety_only
      exit 0
      ;;
    --check-update)
      detect_host_profile
      check_ollama_update
      exit 0
      ;;
    --reconcile)
      detect_host_profile
      reconcile_after_update
      exit 0
      ;;
    --update-ollama)
      detect_host_profile
      update_ollama
      exit 0
      ;;
    -h|--help)
      say "Usage: ./ollama-unify.sh [--classify|--safety-preview|--print-env|--install-safety]"
      say "                         [--check-update|--reconcile|--update-ollama]"
      say "  --classify        Classify the host, accelerators, backend, and risk tier without changes."
      say "  --safety-preview  Classify the host and print the generated safety policy without changes."
      say "  --print-env       Print shell exports for the selected scheduler/backend policy."
      say "  --install-safety  Install systemd containment and the dynamic GPU negotiator; do not migrate models."
      say "  --check-update    Report the installed vs latest Ollama release and any pinning drift."
      say "  --reconcile       Repin Ollama behind the negotiator after an out-of-band Ollama upgrade."
      say "  --update-ollama   Update Ollama inside a safe stop → repin → start envelope, then verify."
      exit 0
      ;;
    "") ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac

  require_migration_tools
  banner
  discover_stores
  detect_daemons

  if [ ${#STORE_PATHS[@]} -eq 0 ]; then
    warn "No ollama model directories found. Nothing to unify."
    exit 0
  fi

  print_store_table

  hdr "Detected daemons:"
  if [ ${#SERVICE_PIDS[@]} -gt 0 ] && [ -n "${SERVICE_PIDS[0]:-}" ]; then
    say "  • ollama.service active (PID ${SERVICE_PIDS[0]})"
  else
    say "  • ollama.service: not active"
  fi
  if [ ${#MANUAL_SERVE_PIDS[@]} -gt 0 ]; then
    say "  • ${#MANUAL_SERVE_PIDS[@]} manual 'ollama serve' processes (PIDs: ${MANUAL_SERVE_PIDS[*]})"
  fi

  build_safety_profile
  print_safety_profile

  local ALREADY_UNIFIED=0
  if [ ${#STORE_PATHS[@]} -eq 1 ]; then
    ok "Only one store found — your setup is already unified at: ${STORE_PATHS[0]}"
    ALREADY_UNIFIED=1
  fi

  local DEST default_dst max_free
  if [ "$ALREADY_UNIFIED" = 1 ]; then
    DEST="${STORE_PATHS[0]}"
  else
    print_dest_candidates

    # destination suggestion: largest fast mount that's already a store, else /srv/ollama/models
    default_dst="${STORE_PATHS[0]}"
    max_free=0
    for s in "${STORE_PATHS[@]}"; do
      local free
      free=$(df -Pk "$s" 2>/dev/null | awk 'END {print $4}')
      [ -z "$free" ] && continue
      if [ "$free" -gt "$max_free" ]; then max_free=$free; default_dst="$s"; fi
    done

    hdr "Destination"
    DEST=$(ask "Where to unify all models?" "$default_dst")
    DEST="$(canonical_path "$DEST")"
  fi

  # opt-ins
  hdr "Options"
  local DO_SYSTEMD=0 DO_SYSTEMD_MODELS=0 DO_SAFETY=0 DO_SYMLINKS=0 DO_BASHRC=0 DO_SERVICE_USER=0
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl cat ollama >/dev/null 2>&1; then
    if confirm "Update ollama.service via drop-in to use OLLAMA_MODELS=$DEST?" "Y"; then
      DO_SYSTEMD=1
      DO_SYSTEMD_MODELS=1
    fi

    if confirm "Apply this classified backend and OOM safety policy to ollama.service?" "Y"; then
      DO_SYSTEMD=1
      DO_SAFETY=1
    fi

    if [ "$DO_SYSTEMD" = 1 ]; then
      local cur_user
      cur_user=$(systemctl show -p User --value ollama 2>/dev/null)
      if [ -n "$cur_user" ] && [ "$cur_user" != "$USER" ]; then
        if confirm "Service runs as '$cur_user'. Change to '$USER' for single-user simplification?" "Y"; then
          DO_SERVICE_USER=1
        fi
      fi
    fi
  fi
  if [ "$ALREADY_UNIFIED" = 0 ]; then
    if confirm "Replace each original store path with a symlink to $DEST (backward compat)?" "Y"; then
      DO_SYMLINKS=1
    fi
  fi
  local SHELL_RC=""
  case "${SHELL##*/}" in
    bash) SHELL_RC="$HOME/.bashrc" ;;
    zsh)  SHELL_RC="$HOME/.zshrc" ;;
    fish) SHELL_RC="$HOME/.config/fish/config.fish" ;;
    *)    SHELL_RC="$HOME/.bashrc" ;;
  esac
  if confirm "Add 'export OLLAMA_MODELS=$DEST' to $SHELL_RC?" "Y"; then
    DO_BASHRC=1
  fi

  if [ "$ALREADY_UNIFIED" = 1 ] && [ "$DO_SYSTEMD" = 0 ] && [ "$DO_BASHRC" = 0 ]; then
    ok "No changes selected."
    exit 0
  fi

  # plan summary
  hdr "Plan"
  say "  Destination: $DEST"
  local total_bytes=0
  for s in "${STORE_PATHS[@]}"; do
    [ "$s" = "$DEST" ] && continue
    local strategy bytes
    strategy=$(transfer_strategy "$s" "$DEST" 2>/dev/null || echo rsync)
    bytes=$(size_of "$s")
    total_bytes=$((total_bytes + bytes))
    say "    $s  →  $DEST   ($(human "$bytes"), strategy: ${C_GRN}$strategy${C_RST})"
  done
  say "  Total to move: $(human "$total_bytes")"
  [ "$DO_SYSTEMD_MODELS" = 1 ] && say "  • Point ollama.service at the unified model store"
  [ "$DO_SAFETY"       = 1 ] && say "  • Install fail-closed memory-pressure, CPU, I/O, and OOM guardrails"
  [ "$DO_SERVICE_USER" = 1 ] && say "  • Change service User/Group to $USER"
  [ "$DO_SYMLINKS"     = 1 ] && say "  • Symlink originals → destination"
  [ "$DO_BASHRC"       = 1 ] && say "  • Add OLLAMA_MODELS export to $SHELL_RC"
  say "  • Stop running daemons, perform transfer, restart ollama.service"

  echo
  confirm "Proceed?" "N" || { warn "Aborted."; exit 1; }

  # ── sudo gate
  local SUDO=""
  if plan_requires_sudo "$DO_SYSTEMD" "$DO_SERVICE_USER" "$DEST"; then
    [ "$HAS_SUDO" = 1 ] || { err "sudo required but not installed."; exit 2; }
    SUDO="sudo"
    say "${C_DIM}(sudo will prompt for your password)${C_RST}"
    $SUDO -n true 2>/dev/null || $SUDO -v
  fi

  # ── stop daemons
  hdr "Stopping daemons"
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl is-active ollama-unify-negotiator.service >/dev/null 2>&1; then
    $SUDO systemctl stop ollama-unify-negotiator.service
    ok "ollama-unify-negotiator.service stopped"
  fi
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl is-active ollama >/dev/null 2>&1; then
    $SUDO systemctl stop ollama
    ok "ollama.service stopped"
  fi
  if [ ${#MANUAL_SERVE_PIDS[@]} -gt 0 ]; then
    for pid in "${MANUAL_SERVE_PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${MANUAL_SERVE_PIDS[@]}"; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    ok "killed ${#MANUAL_SERVE_PIDS[@]} manual ollama serve(s)"
  fi
  # also nuke any remaining runners
  pkill -KILL -f 'ollama runner' 2>/dev/null || true
  sleep 1

  if [ "$ALREADY_UNIFIED" = 0 ]; then
    # ── ensure destination exists
    $SUDO mkdir -p "$DEST/manifests" "$DEST/blobs"

    # ── transfer each source
    hdr "Transferring"
    local started_at; started_at=$(date +%s)
    for s in "${STORE_PATHS[@]}"; do
      [ "$s" = "$DEST" ] && continue
      say "${C_BOLD}→ $s${C_RST}"
      local strategy
      strategy=$(transfer_strategy "$s" "$DEST")

      # special case: store has 0 manifests = orphan blobs, archive instead of merging
      local mcount; mcount=$(count_manifests "$s")
      if [ "$mcount" -eq 0 ] && [ "$s" != "$DEST" ]; then
        warn "  $s has 0 manifests (orphan blobs only) — archiving as .orphan-blobs"
        $SUDO mv "$s" "${s}.orphan-blobs"
        continue
      fi

      transfer_store "$s" "$DEST" "$strategy" "$SUDO"

      # post-transfer: rename original to .bak (or remove if empty after mv-merge)
      if [ -d "$s" ]; then
        if [ "$strategy" = "mv" ]; then
          # source dir likely empty now; just rmdir
          find "$s" -type d -empty -delete 2>/dev/null || true
          if [ -d "$s" ]; then
            $SUDO mv "$s" "${s}.bak"
          fi
        else
          $SUDO mv "$s" "${s}.bak"
        fi
        ok "  archived original: ${s}.bak"
      fi
    done
    local elapsed=$(( $(date +%s) - started_at ))
    local throughput=$(( total_bytes / (elapsed > 0 ? elapsed : 1) ))
    ok "transfer complete in ${elapsed}s  (~$(human "$throughput")/s)"
  fi

  # ── ownership + permissions on destination
  if [ "$ALREADY_UNIFIED" = 0 ] || [ "$DO_SERVICE_USER" = 1 ]; then
    hdr "Finalizing destination"
    if [ "$DO_SERVICE_USER" = 1 ]; then
      $SUDO chown -R "$USER:$(id -gn)" "$DEST"
    fi
    $SUDO chmod -R u+rwX,g+rX,o+rX "$DEST"
    ok "permissions normalized"
  fi

  # ── systemd drop-in
  if [ "$DO_SYSTEMD" = 1 ]; then
    hdr "Updating systemd"
    local dropin_dir="/etc/systemd/system/ollama.service.d"
    local dropin_file="$dropin_dir/zz-ollama-unify.conf"
    $SUDO mkdir -p "$dropin_dir"
    {
      printf '# Managed by ollama-unify — generated %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
      printf '[Service]\n'
      if [ "$DO_SYSTEMD_MODELS" = 1 ]; then
        printf 'Environment="OLLAMA_MODELS=%s"\n' "$DEST"
      fi
      if [ "$DO_SERVICE_USER" = 1 ]; then
        printf 'User=%s\n' "$USER"
        printf 'Group=%s\n' "$(id -gn)"
      fi
    } | $SUDO tee "$dropin_file" >/dev/null
    ok "late-priority drop-in installed: $dropin_file"
    if [ "$DO_SAFETY" = 1 ]; then
      install_systemd_safety_policy "$SUDO"
    else
      $SUDO systemctl daemon-reload
      if command -v systemd-analyze >/dev/null 2>&1; then
        $SUDO systemd-analyze verify ollama.service
      fi
    fi
  fi

  # ── symlinks for backward compat
  if [ "$DO_SYMLINKS" = 1 ]; then
    hdr "Creating backward-compat symlinks"
    for s in "${STORE_PATHS[@]}"; do
      [ "$s" = "$DEST" ] && continue
      [ -e "$s" ] && continue   # something still here (.bak rename failed?)
      $SUDO ln -s "$DEST" "$s"
      $SUDO chown -h "$USER:$(id -gn)" "$s" 2>/dev/null || true
      ok "  $s → $DEST"
    done
  fi

  # ── shell rc
  if [ "$DO_BASHRC" = 1 ]; then
    # grep completes before this distinct append; it does not read while writing.
    # shellcheck disable=SC2094
    if ! grep -q "OLLAMA_MODELS=$DEST" "$SHELL_RC" 2>/dev/null; then
      {
        printf '\n# Unified Ollama model store (added by ollama-unify on %s)\n' "$(date +%F)"
        if [[ "$SHELL_RC" == *config.fish ]]; then
          printf 'set -gx OLLAMA_MODELS %s\n' "$DEST"
        else
          printf 'export OLLAMA_MODELS=%s\n' "$DEST"
        fi
      } >> "$SHELL_RC"
      ok "added export to $SHELL_RC (new shells only)"
    fi
  fi

  # ── restart service + verify
  if [ "$HAS_SYSTEMD" = 1 ] && systemctl cat ollama >/dev/null 2>&1; then
    hdr "Restarting ollama.service"
    if $SUDO systemctl start ollama; then
      if systemctl is-enabled ollama-unify-negotiator.service >/dev/null 2>&1; then
        $SUDO systemctl start ollama-unify-negotiator.service || true
      fi
      sleep 3
      if systemctl is-active ollama >/dev/null 2>&1; then
        ok "ollama.service is active"
      else
        local service_result
        service_result=$(systemctl show ollama.service -p Result --value 2>/dev/null || true)
        if [ "$service_result" = "exec-condition" ]; then
          warn "ollama.service was safely held inactive by the memory-pressure condition"
        else
          err "ollama.service stopped after launch — check 'journalctl -u ollama -n 50'"
        fi
      fi
    else
      warn "ollama.service was left stopped; the safety preflight may have refused an unsafe restart"
      warn "check: journalctl -u ollama -n 50"
    fi
  fi

  hdr "Verification"
  local mcount; mcount=$(count_manifests "$DEST")
  ok "$mcount manifests at $DEST"
  if [ "$HAS_CURL" = 1 ]; then
    local api_count
    api_count=$(curl -s --max-time 5 http://127.0.0.1:11434/api/tags 2>/dev/null \
      | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | wc -l || true)
    [ -n "$api_count" ] && [ "$api_count" -gt 0 ] && ok "/api/tags reports $api_count models"
  fi

  hdr "Backups preserved (verify everything works, then reclaim):"
  for s in "${STORE_PATHS[@]}"; do
    [ "$s" = "$DEST" ] && continue
    [ -d "${s}.bak"           ] && say "  ${s}.bak  ($(du -sh "${s}.bak"           2>/dev/null | awk '{print $1}'))"
    [ -d "${s}.orphan-blobs"  ] && say "  ${s}.orphan-blobs  ($(du -sh "${s}.orphan-blobs"  2>/dev/null | awk '{print $1}'))"
  done

  hdr "Done."
  say "  Unified store:  $DEST"
  say "  Next steps:"
  say "    • open a new shell (or source $SHELL_RC) to pick up OLLAMA_MODELS"
  say "    • run: ${C_BOLD}ollama list${C_RST}    to confirm all models appear"
  say "    • after verifying, ${C_BOLD}rm -rf${C_RST} the .bak directories to reclaim disk"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
