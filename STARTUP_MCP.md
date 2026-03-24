# MCP Tunnel Setup

This guide covers setting up the MCP server tunnel to run as a systemd service on WSL/Linux.

## Prerequisites

1. **WSL with systemd enabled** (check with `ps -p 1 -o comm=` — should print "systemd")
2. **gcloud CLI installed** and authenticated (`gcloud auth login`)
3. **.env file** with GCP deployment variables

## Quick Start

### 1. Configure `.env`

Add these variables to your `.env` file (copy from `env.template` if needed):

```bash
# GCP Tunnel Configuration
GCP_PROJECT=your_gcp_project_id_here
VM_INSTANCE=your_vm_instance_name_here
ZONE=us-central1-a
MCP_PORT=8765
```

### 2. Install the systemd service

Run the install script (one-time per machine):

```bash
chmod +x scripts/install_tunnel_service.sh
./scripts/install_tunnel_service.sh
```

This stamps the service template with your machine's username and project path, then installs it to `/etc/systemd/system/`.

### 3. Start the tunnel

```bash
sudo systemctl start kalshi-mcp-tunnel
```

### 4. View logs

```bash
journalctl -u kalshi-mcp-tunnel -f
```

## Common Commands

| Action | Command |
|--------|---------|
| Start | `sudo systemctl start kalshi-mcp-tunnel` |
| Stop | `sudo systemctl stop kalshi-mcp-tunnel` |
| Restart | `sudo systemctl restart kalshi-mcp-tunnel` |
| Enable on boot | `sudo systemctl enable kalshi-mcp-tunnel` |
| View logs | `journalctl -u kalshi-mcp-tunnel -f` |
| Check status | `systemctl status kalshi-mcp-tunnel` |

## Troubleshooting

### Service won't start

Check the logs:
```bash
journalctl -u kalshi-mcp-tunnel -e
```

Common issues:
- **Missing .env variables**: Ensure GCP_PROJECT, VM_INSTANCE are set in `.env`
- **gcloud not found**: The service should find gcloud automatically; if not, ensure it's in your PATH or set `GCLOUD_PATH` in `.env`

### Port already in use

If port 8765 is busy:
```bash
# Find what's using it
sudo ss -tlnp | grep 8765

# Kill it if safe
sudo kill -9 <pid>
```

### Reinstall after updating the template

If you pull changes that modify the service template, reinstall:
```bash
./scripts/install_tunnel_service.sh
```

## Alternative: Run manually (without systemd)

If you prefer to run the tunnel manually in a terminal:

```bash
./scripts/tunnel_mcp.sh
```

Or with explicit arguments (overrides .env):

```bash
./scripts/tunnel_mcp.sh my-gcp-project my-vm us-central1-a
```