#!/usr/bin/env bash
# =============================================================================
# tunnel_mcp.sh — Run the MCP server locally (WSL) and forward its port to
#                 the GCE VM via a persistent SSH reverse tunnel.
#
# Architecture:
#
#   WSL (local)                         GCE VM
#   ───────────────────                 ──────────────────────────
#   MCP server :8765  ◄── SSH -R ────── :8765 (localhost only)
#                                            ▲
#                                       OpenClaw calls
#                                       http://127.0.0.1:8765/mcp
#
# OpenClaw's config does not change — it still calls localhost:8765.
# The actual server runs in WSL with your full disk space and venv.
#
# Usage:
#   chmod +x scripts/tunnel_mcp.sh
#   ./scripts/tunnel_mcp.sh <gcp-project> <vm-instance> [zone]
#
# Run persistently in the background (survives terminal close):
#   mkdir -p logs
#   nohup ./scripts/tunnel_mcp.sh my-project openclaw-vm \
#       > logs/tunnel.log 2>&1 &
#   echo $! > logs/tunnel.pid
#   disown
#
# Stop it:
#   kill $(cat logs/tunnel.pid)
#
# Or use the companion systemd unit (recommended for always-on):
#   see scripts/kalshi-mcp-tunnel.service
# =============================================================================

set -euo pipefail

GCP_PROJECT="${1:?Usage: $0 <gcp-project> <vm-instance> [zone]}"
VM_INSTANCE="${2:?Usage: $0 <gcp-project> <vm-instance> [zone]}"
ZONE="${3:-us-central1-a}"
MCP_PORT=8765

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_DIR/venv"
PYTHON="$VENV/bin/python"

# ── Resolve gcloud to an absolute path ───────────────────────────────────────
# systemd runs with a minimal PATH that excludes the Cloud SDK bin directory.
# We probe common install locations so the script works under systemd without
# requiring Environment=PATH= to be exactly right in the unit file.
# You can also override by setting GCLOUD_PATH before running the script.
_find_gcloud() {
    # Already on PATH (interactive shell, or Environment=PATH= in unit worked)
    if command -v gcloud &>/dev/null; then
        command -v gcloud
        return
    fi
    # Common install locations: apt, manual user install, snap, homebrew-on-linux
    local candidates=(
        "/usr/lib/google-cloud-sdk/bin/gcloud"
        "/usr/share/google-cloud-sdk/bin/gcloud"
        "/snap/bin/gcloud"
        "/home/giria/google-cloud-sdk/bin/gcloud"
        "$HOME/google-cloud-sdk/bin/gcloud"
        "/opt/google-cloud-sdk/bin/gcloud"
        "/usr/local/lib/google-cloud-sdk/bin/gcloud"
        "/usr/local/google-cloud-sdk/bin/gcloud"
    )
    local c
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && { echo "$c"; return; }
    done
    echo ""
}

GCLOUD="${GCLOUD_PATH:-$(_find_gcloud)}"
if [[ -z "$GCLOUD" ]]; then
    echo "ERROR: gcloud binary not found." >&2
    echo "  Fix option 1 — set GCLOUD_PATH in the systemd unit:" >&2
    echo "    Environment=GCLOUD_PATH=/full/path/to/gcloud" >&2
    echo "  Fix option 2 — add gcloud's bin dir to PATH in the unit:" >&2
    echo "    Environment=PATH=/your/sdk/bin:/usr/local/bin:/usr/bin:/bin" >&2
    echo "  Find the path by running in your terminal:  which gcloud" >&2
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
ts()   { date '+%H:%M:%S'; }
info() { printf '[%s] \033[1;34m[tunnel]\033[0m %s\n' "$(ts)" "$*"; }
ok()   { printf '[%s] \033[1;32m[  ok  ]\033[0m %s\n' "$(ts)" "$*"; }
warn() { printf '[%s] \033[1;33m[ warn ]\033[0m %s\n' "$(ts)" "$*" >&2; }

MCP_PID=""
TUNNEL_PID=""

# ── Clean up both child processes on exit ─────────────────────────────────────
cleanup() {
    info "Shutting down..."
    [[ -n "$MCP_PID"    ]] && kill "$MCP_PID"    2>/dev/null || true
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start the MCP server locally ──────────────────────────────────────────────
start_mcp() {
    # Clear any stale process already holding the port before starting a new one
    local stale
    stale=$(ss -tlnp 2>/dev/null | grep ":$MCP_PORT " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    if [[ -n "$stale" ]]; then
        warn "Port $MCP_PORT already in use by pid=$stale — killing stale process..."
        kill -9 "$stale" 2>/dev/null || true
        sleep 2
    fi

    info "Starting MCP server on localhost:$MCP_PORT..."
    cd "$PROJECT_DIR"
    source "$VENV/bin/activate"
    "$PYTHON" cli.py mcp \
        --transport streamable-http \
        --host 127.0.0.1 \
        --port "$MCP_PORT" &
    MCP_PID=$!
    ok "MCP server started (pid=$MCP_PID)"
}

# ── Wait until the server responds ────────────────────────────────────────────
wait_for_mcp() {
    local attempts=0
    info "Waiting for MCP server to accept connections..."
    while true; do
        # Try the /mcp endpoint; a 4xx still means the server is up
        local http_code
        http_code=$(curl -o /dev/null -sf --max-time 2 -w "%{http_code}" \
            "http://127.0.0.1:$MCP_PORT/mcp" 2>/dev/null || true)
        if [[ "$http_code" =~ ^[1-5] ]]; then
            ok "MCP server is ready"
            return
        fi
        sleep 2
        attempts=$((attempts + 1))
        if [[ $attempts -gt 30 ]]; then
            warn "Server not ready after 60 s — opening tunnel anyway"
            return
        fi
    done
}

# ── Open the reverse SSH tunnel to the GCE VM ────────────────────────────────
#
# -R $MCP_PORT:localhost:$MCP_PORT
#     On the remote VM, bind port $MCP_PORT and forward connections back to
#     localhost:$MCP_PORT on THIS (local/WSL) machine.
#
# -N   No remote command; tunnel only.
# -o ServerAliveInterval=30   Send a keepalive every 30 s so NAT/firewalls
#                             don't silently drop the connection.
# -o ServerAliveCountMax=3    After 3 missed keepalives (~90 s), exit and let
#                             our loop restart the tunnel.
# -o ExitOnForwardFailure=yes Exit immediately if the port binding fails on the
#                             remote side (e.g. port already in use).
#
# gcloud compute ssh with --tunnel-through-iap handles Cloud IAP proxying
# transparently, so no external IP or open firewall rule is needed.
start_tunnel() {
    info "Opening reverse tunnel: VM:$MCP_PORT ← local:$MCP_PORT (via IAP)..."
    "$GCLOUD" compute ssh "$VM_INSTANCE" \
        --project="$GCP_PROJECT" \
        --zone="$ZONE" \
        --tunnel-through-iap \
        -- \
        -N \
        -R "${MCP_PORT}:localhost:${MCP_PORT}" \
        -o "ServerAliveInterval=30" \
        -o "ServerAliveCountMax=3" \
        -o "ExitOnForwardFailure=yes" \
        -o "StrictHostKeyChecking=no" &
    TUNNEL_PID=$!
    ok "Tunnel established (pid=$TUNNEL_PID)"
}

# ── Main ──────────────────────────────────────────────────────────────────────
info "Project dir: $PROJECT_DIR"
info "gcloud:      $GCLOUD"
info "GCP project: $GCP_PROJECT  VM: $VM_INSTANCE  Zone: $ZONE"

start_mcp
wait_for_mcp
start_tunnel

info "Both processes running."
info "  Local MCP:  http://127.0.0.1:$MCP_PORT/mcp"
info "  VM sees:    http://127.0.0.1:$MCP_PORT/mcp  (via reverse tunnel)"

# ── Monitor loop — restart whichever process dies ────────────────────────────
while true; do
    sleep 10

    if ! kill -0 "$MCP_PID" 2>/dev/null; then
        warn "MCP server died — restarting..."
        start_mcp
        wait_for_mcp
        # Bounce the tunnel too so it reconnects to the fresh server
        kill "$TUNNEL_PID" 2>/dev/null || true
        sleep 2
        start_tunnel
    fi

    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        warn "SSH tunnel died — restarting..."
        start_tunnel
    fi
done
