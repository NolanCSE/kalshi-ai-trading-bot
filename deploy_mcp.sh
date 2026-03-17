#!/usr/bin/env bash
# =============================================================================
# deploy_mcp.sh — Deploy the Kalshi MCP server to a GCE VM and install it
#                 as a persistent systemd service.
#
# Usage (run from your LOCAL machine, inside the project root):
#
#   chmod +x deploy_mcp.sh
#   ./deploy_mcp.sh <gcp-project-id> <vm-instance-name> [zone]
#
# Examples:
#   ./deploy_mcp.sh my-gcp-project openclaw-vm us-central1-a
#   ./deploy_mcp.sh my-gcp-project openclaw-vm          # defaults to us-central1-a
#
# What this does:
#   1. Rsyncs the project to ~/kalshi-mcp on the VM via Cloud IAP (no
#      external IP or open firewall ports required for the transfer itself).
#   2. On the VM: creates a venv, installs all Python dependencies.
#   3. Writes and enables a systemd unit that keeps the MCP server running
#      on port 8765 — bound to 127.0.0.1 only, never exposed to internet.
#   4. Prints the exact OpenClaw config snippet to paste into your agent.
#
# Prerequisites (LOCAL machine):
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - rsync installed  (brew install rsync  /  apt install rsync)
#
# Prerequisites (VM):
#   - Python 3.11+ available as python3
#   - systemd (standard on all GCE Debian / Ubuntu images)
#
# Secrets — copy manually once after first deploy, then never touched again:
#   gcloud compute ssh <vm> --tunnel-through-iap -- nano ~/kalshi-mcp/.env
#   sudo systemctl start kalshi-mcp
# =============================================================================

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
GCP_PROJECT="${1:?Usage: $0 <gcp-project-id> <vm-instance> [zone]}"
VM_INSTANCE="${2:?Usage: $0 <gcp-project-id> <vm-instance> [zone]}"
ZONE="${3:-us-central1-a}"
MCP_PORT=8765
REMOTE_DIR="kalshi-mcp"
SERVICE="kalshi-mcp"

# ── Helpers ───────────────────────────────────────────────────────────────────
info() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[  ok  ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ warn ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ FAIL ]\033[0m %s\n' "$*" >&2; exit 1; }

SSH="gcloud compute ssh $VM_INSTANCE --project=$GCP_PROJECT --zone=$ZONE --tunnel-through-iap"

# ── 0. Preflight ──────────────────────────────────────────────────────────────
info "Checking gcloud auth..."
gcloud auth print-access-token --project="$GCP_PROJECT" >/dev/null 2>&1 \
    || die "Not authenticated. Run: gcloud auth login && gcloud config set project $GCP_PROJECT"
ok "Authenticated"

info "Verifying VM is reachable..."
$SSH --command="echo pong" >/dev/null \
    || die "Cannot reach $VM_INSTANCE in zone $ZONE."
ok "VM reachable"

# ── 1. Sync files ─────────────────────────────────────────────────────────────
info "Syncing project to VM:~/$REMOTE_DIR (secrets excluded)..."
$SSH --command="mkdir -p ~/$REMOTE_DIR"

# rsync tunnelled through gcloud's IAP proxy.
# The --rsh trick lets rsync use gcloud ssh as the transport so we never
# need the VM's external IP.
RSYNC_RSH="gcloud compute ssh $VM_INSTANCE \
    --project=$GCP_PROJECT --zone=$ZONE \
    --tunnel-through-iap \
    --ssh-flag=-o --ssh-flag=StrictHostKeyChecking=no --"

rsync -az --progress \
    --exclude=".env" \
    --exclude=".env.*" \
    --exclude="kalshi_private_key" \
    --exclude="venv/" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    --exclude=".git/" \
    --exclude="*.db" \
    --exclude="*.sqlite*" \
    --exclude="logs/" \
    --exclude="*.log" \
    --exclude=".ruff_cache/" \
    --exclude=".cache/" \
    --rsh="$RSYNC_RSH" \
    ./ ":~/$REMOTE_DIR/" \
|| {
    warn "rsync failed. Falling back to tar-pipe transfer..."
    tar cz \
        --exclude=".env" --exclude=".env.*" --exclude="kalshi_private_key" \
        --exclude="venv" --exclude="__pycache__" --exclude="*.pyc" \
        --exclude=".git" --exclude="*.db" --exclude="*.sqlite*" \
        --exclude="logs" --exclude="*.log" \
        -f - . \
    | $SSH --command="tar xz -C ~/$REMOTE_DIR" \
    || die "Both rsync and tar-pipe failed."
}
ok "Files synced"

# ── 2. Bootstrap on VM ────────────────────────────────────────────────────────
info "Bootstrapping VM (venv + systemd)..."

$SSH --command="bash -s" <<REMOTE
set -euo pipefail
PROJECT="\$HOME/$REMOTE_DIR"
VENV="\$PROJECT/venv"
WHOAMI=\$(whoami)

# Python check
python3 --version
python3 -c "import sys; assert sys.version_info >= (3,11), f'Need Python 3.11+, got {sys.version}'" \
    || { echo "FAIL: Python 3.11+ required on the VM."; exit 1; }

# venv + deps
echo "[vm] Setting up venv..."
python3 -m venv "\$VENV"
"\$VENV/bin/pip" install --upgrade pip --quiet
echo "[vm] Installing Python dependencies (first run takes a few minutes)..."
"\$VENV/bin/pip" install -r "\$PROJECT/requirements.txt" --quiet
echo "[vm] Dependencies installed."

# .env placeholder — never overwrites an existing file
if [ ! -f "\$PROJECT/.env" ]; then
    cat > "\$PROJECT/.env" <<'ENVEOF'
# Kalshi AI Trading Bot — API Keys
# Fill in all values, then: sudo systemctl start kalshi-mcp
KALSHI_API_KEY=
XAI_API_KEY=
OPENROUTER_API_KEY=
OPENAI_API_KEY=
SUPABASE_URL=
SUPABASE_ANON_KEY=
ENCRYPTION_KEY=
ENVEOF
    echo "[vm] Created blank .env — you must fill in keys before starting the service."
else
    echo "[vm] .env already exists, not overwriting."
fi

# systemd unit
echo "[vm] Writing systemd unit..."
sudo tee /etc/systemd/system/$SERVICE.service > /dev/null <<UNIT
[Unit]
Description=Kalshi AI Trading Bot MCP Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=\$WHOAMI
WorkingDirectory=\$PROJECT
EnvironmentFile=\$PROJECT/.env
ExecStart=\$VENV/bin/python cli.py mcp --transport streamable-http --host 127.0.0.1 --port $MCP_PORT
Restart=on-failure
RestartSec=15s
TimeoutStopSec=60s
TimeoutStartSec=120s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"
echo "[vm] Service enabled."
REMOTE

ok "VM bootstrap complete"

# ── 3. Restart service if already running ─────────────────────────────────────
if $SSH --command="sudo systemctl is-active $SERVICE" >/dev/null 2>&1; then
    info "Service is running — restarting to pick up new code..."
    $SSH --command="sudo systemctl restart $SERVICE"
    ok "Service restarted"
else
    info "Service not yet started (waiting for .env to be filled in)."
fi

# ── 4. Print instructions ─────────────────────────────────────────────────────
cat <<INSTRUCTIONS

╔══════════════════════════════════════════════════════════════════╗
║             DEPLOYMENT COMPLETE — NEXT STEPS                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Fill in your API keys on the VM:                            ║
║                                                                  ║
║     gcloud compute ssh $VM_INSTANCE \\
║       --project=$GCP_PROJECT --zone=$ZONE \\
║       --tunnel-through-iap -- \\
║       nano ~/kalshi-mcp/.env
║                                                                  ║
║  2. Start the service:                                           ║
║       ... -- sudo systemctl start $SERVICE                      ║
║                                                                  ║
║  3. Confirm it's running:                                        ║
║       ... -- sudo systemctl status $SERVICE                     ║
║       ... -- journalctl -u $SERVICE -f                          ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                   OPENCLAW CONFIGURATION                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Paste this into your OpenClaw agent's MCP server config:       ║
║                                                                  ║
║  {                                                               ║
║    "mcpServers": {                                               ║
║      "kalshi-analyst": {                                         ║
║        "url": "http://127.0.0.1:$MCP_PORT/mcp"                    ║
║      }                                                           ║
║    }                                                             ║
║  }                                                               ║
║                                                                  ║
║  Port $MCP_PORT is bound to 127.0.0.1 only — not in the firewall. ║
║  OpenClaw and the MCP server share the VM's loopback interface. ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                      FUTURE UPDATES                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  To redeploy after any code change, just re-run:                ║
║    ./deploy_mcp.sh $GCP_PROJECT $VM_INSTANCE $ZONE             ║
║                                                                  ║
║  The service restarts automatically. .env is never touched.     ║
╚══════════════════════════════════════════════════════════════════╝
INSTRUCTIONS
