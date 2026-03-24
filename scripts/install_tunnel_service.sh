#!/usr/bin/env bash
# =============================================================================
# install_tunnel_service.sh — Install the kalshi-mcp-tunnel systemd service
#
# This script stamps the service template with the correct paths for THIS
# machine and installs it to /etc/systemd/system/.
#
# Run once after cloning the repo (or after pulling that changes the template):
#   chmod +x scripts/install_tunnel_service.sh
#   ./scripts/install_tunnel_service.sh
#
# Requirements:
#   - Running in the project root directory
#   - .env file exists with GCP_PROJECT, VM_INSTANCE, ZONE
#   - sudo access to install the systemd unit
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_FILE="$SCRIPT_DIR/kalshi-mcp-tunnel.service"
TARGET_FILE="/etc/systemd/system/kalshi-mcp-tunnel.service"

# Helpers
info() { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[  ok  ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ warn ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ FAIL ]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Preflight checks ─────────────────────────────────────────────────────────
info "Running preflight checks..."

# Verify we're in the project root
if [[ ! -f "$PROJECT_DIR/scripts/tunnel_mcp.sh" ]]; then
    die "Must run from project root (where scripts/tunnel_mcp.sh exists)"
fi

# Verify .env exists
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    die ".env file not found. Copy env.template to .env and fill in GCP_PROJECT, VM_INSTANCE, ZONE"
fi

# Verify .env has required variables
source "$PROJECT_DIR/.env"
MISSING=""
[[ -z "${GCP_PROJECT:-}" ]] && MISSING="$MISSING GCP_PROJECT"
[[ -z "${VM_INSTANCE:-}" ]] && MISSING="$MISSING VM_INSTANCE"
if [[ -n "$MISSING" ]]; then
    die "Missing required variables in .env:$MISSING"
fi

# Verify we're running as the user (not root)
if [[ "$(id -u)" -eq 0 ]]; then
    die "Do not run as root. Run as your normal user (sudo is used internally)."
fi

CURRENT_USER="$(whoami)"
CURRENT_DIR="$PROJECT_DIR"
ok "Preflight passed"

# ── Stamp the template ───────────────────────────────────────────────────────
info "Stamping service template for user=$CURRENT_USER project=$CURRENT_DIR"

STAMPED=$(sed -e "s|__USER__|$CURRENT_USER|g" \
               -e "s|__PROJECT_DIR__|$CURRENT_DIR|g" \
               "$TEMPLATE_FILE")

# ── Install to /etc/systemd/system/ ──────────────────────────────────────────
info "Installing service to $TARGET_FILE (requires sudo)..."
echo "$STAMPED" | sudo tee "$TARGET_FILE" > /dev/null

sudo systemctl daemon-reload
ok "Service installed"

# ── Enable (optional) ─────────────────────────────────────────────────────────
if [[ "${1:-}" == "--enable" ]]; then
    info "Enabling service to start on boot..."
    sudo systemctl enable kalshi-mcp-tunnel
    ok "Service enabled"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║              TUNNEL SERVICE INSTALLED SUCCESSFULLY               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  To start the tunnel now:                                        ║
║    sudo systemctl start kalshi-mcp-tunnel                        ║
║    journalctl -u kalshi-mcp-tunnel -f                            ║
║                                                                  ║
║  To start on boot (optional):                                    ║
║    sudo systemctl enable kalshi-mcp-tunnel                       ║
║                                                                  ║
║  To stop:                                                        ║
║    sudo systemctl stop kalshi-mcp-tunnel                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF

info "Done."