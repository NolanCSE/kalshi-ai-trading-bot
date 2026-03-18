sudo systemctl daemon-reload
sudo systemctl restart kalshi-mcp-tunnel
journalctl -u kalshi-mcp-tunnel -f
