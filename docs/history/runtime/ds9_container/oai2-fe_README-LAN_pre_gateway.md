# LAN Access for oai2-fe

This document explains how to access the oai2-fe frontend from other machines on your local network.

## Quick Start

1. **Configure endpoints** (optional):
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and set your host machine's IP address when auto-detection is not enough.
   # Replace 192.168.1.100 with your actual IP
   VITE_WS_URL=ws://192.168.1.100:6008
   VITE_REST_URL=http://192.168.1.100:8082
   ```

2. **Open firewall ports** (run on host machine):
   ```bash
   sudo ufw allow 5173/tcp comment "Vite dev/preview server for oai2-fe"
   sudo ufw allow 6008/tcp comment "WebSocket server for oai2-fe"
   sudo ufw allow 8082/tcp comment "Noesis DS8 REST API for oai2-fe"
   ```

3. **Start the frontend**:
   ```bash
   # Development mode
   npm run dev

   # Or production preview
   npm run build && npm run preview
   ```

4. **Access from another machine**:
   - Open `http://<HOST_IP>:5173` in a web browser
   - Replace `<HOST_IP>` with your host machine's IP address

## How It Works

- **Dynamic WebSocket URL**: The frontend automatically detects the hostname from the browser's location and connects to the WebSocket server on the same machine.
- **Dynamic REST URL**: The frontend automatically calls the DS8 REST API at `http://<browser-host>:8082`. This keeps ROI edits working from localhost and LAN browsers without the Vite proxy.
- **Environment Variables**: You can override the WebSocket and REST connections using `.env` variables:
  - `VITE_WS_URL`: Complete WebSocket URL (e.g., `ws://192.168.1.100:6008`)
  - `VITE_WS_HOST`: WebSocket host only (defaults to browser's hostname)
  - `VITE_WS_PORT`: WebSocket port (defaults to 6008)
  - `VITE_REST_URL`: Complete REST API URL (e.g., `http://192.168.1.100:8082`)
  - `VITE_REST_PORT`: REST API port when using auto-detected browser host (defaults to 8082)
- **HTTPS Support**: If you serve the frontend over HTTPS, it automatically switches to `wss://` for secure WebSocket connections.

## Troubleshooting

- **Connection refused**: Ensure the WebSocket server is running on the host machine and port 6008 is open
- **ROI editor cannot load or save**: Ensure the DS8 REST API is running and port 8082 is open
- **Page won't load**: Check that port 5173 is open and the Vite server is running
- **WebSocket connection fails**: Verify the host machine's IP address in your `.env` file

## Finding Your Host IP

```bash
# On Linux/macOS
ip route get 1.1.1.1 | awk '{print $7; exit}'

# Or
hostname -I | awk '{print $1}'

# On Windows
ipconfig | findstr "IPv4"
```
