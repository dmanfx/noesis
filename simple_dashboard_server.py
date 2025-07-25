#!/usr/bin/env python3
"""
Simple GPU Pipeline Monitoring Dashboard Server

A lightweight dashboard server using Python's built-in HTTP server.
No external dependencies required beyond what's already in the project.

Usage:
    python3 simple_dashboard_server.py [--port 8080] [--interval 1.0]

Features:
- Real-time CPU and GPU metrics
- Simple web interface
- Auto-refresh every 5 seconds
- No WebSocket dependencies

Access the dashboard at: http://localhost:8080
"""

import argparse
import json
import logging
import math
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Import safe join utility
try:
    from utils.interrupt import safe_join
except ImportError:
    # Fallback if utils not available
    def safe_join(thread, timeout=2.0, name=""):
        if thread:
            thread.join(timeout)
            # Cannot set daemon on running thread, just log timeout
            if thread.is_alive():
                print(f"Thread {name} still alive after {timeout}s timeout")
            return not thread.is_alive()

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataclasses import dataclass, asdict
from collections import deque
import psutil
import pynvml

# Basic metrics collector used when realtime_monitor.py is not available
@dataclass
class MetricSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_percent: float
    gpu_temperature: float
    gpu_power_watts: float
    frame_rate: float
    latency_ms: float
    active_streams: int

    def to_dict(self):
        return asdict(self)


class MetricsCollector:
    """Collect simple system metrics using psutil and NVML"""

    def __init__(self, gpu_device: int = 0):
        self.logger = logging.getLogger("MetricsCollector")
        self.gpu_device = gpu_device
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = None

        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
            self.nvml = pynvml
        except Exception as e:
            self.logger.error(f"NVML initialization failed: {e}")
            self.nvml = None

    def collect(self) -> MetricSnapshot:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()

        gpu_util = 0.0
        gpu_mem_percent = 0.0
        gpu_temp = 0.0
        gpu_power = 0.0

        if self.nvml:
            try:
                util = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = util.gpu
                mem_info = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_percent = (mem_info.used / mem_info.total) * 100
                gpu_temp = self.nvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, self.nvml.NVML_TEMPERATURE_GPU
                )
                gpu_power = self.nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            except Exception as e:
                self.logger.error(f"NVML error: {e}")

        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            frame_rate = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        else:
            frame_rate = 0.0

        latency_ms = 0.0

        return MetricSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=mem.percent,
            gpu_utilization=gpu_util,
            gpu_memory_percent=gpu_mem_percent,
            gpu_temperature=gpu_temp,
            gpu_power_watts=gpu_power,
            frame_rate=frame_rate,
            latency_ms=latency_ms,
            active_streams=0,
        )

    def update_frame_time(self):
        current_time = time.time()
        if self.last_frame_time is not None:
            self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time

# Global instances for external cleanup
_dashboard_instance = None
_httpd_instance = None


class MockMetricsCollector:
    """Mock metrics collector for when real monitor is not available"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def collect(self):
        """Generate mock metrics"""
        elapsed = time.time() - self.start_time
        
        class MockSnapshot:
            def __init__(self):
                self.timestamp = time.time()
                self.cpu_percent = 15.0 + 10.0 * (0.5 + 0.5 * math.sin(elapsed / 10))
                self.memory_percent = 45.0 + 5.0 * (0.5 + 0.5 * math.cos(elapsed / 15))
                self.gpu_utilization = 65.0 + 15.0 * (0.5 + 0.5 * math.sin(elapsed / 8))
                self.gpu_memory_percent = 75.0 + 10.0 * (0.5 + 0.5 * math.cos(elapsed / 12))
                self.gpu_temperature = 65 + int(5 * (0.5 + 0.5 * math.sin(elapsed / 20)))
                self.gpu_power_watts = 150.0 + 50.0 * (0.5 + 0.5 * math.cos(elapsed / 18))
                self.frame_rate = 28.0 + 4.0 * (0.5 + 0.5 * math.sin(elapsed / 6))
                self.latency_ms = 15.0 + 5.0 * (0.5 + 0.5 * math.cos(elapsed / 7))
                self.active_streams = 2
            
            def to_dict(self):
                return {
                    'timestamp': self.timestamp,
                    'cpu_percent': self.cpu_percent,
                    'memory_percent': self.memory_percent,
                    'gpu_utilization': self.gpu_utilization,
                    'gpu_memory_percent': self.gpu_memory_percent,
                    'gpu_temperature': self.gpu_temperature,
                    'gpu_power_watts': self.gpu_power_watts,
                    'frame_rate': self.frame_rate,
                    'latency_ms': self.latency_ms,
                    'active_streams': self.active_streams
                }
        
        return MockSnapshot()


class SimpleDashboard:
    """Simple dashboard that collects and serves metrics"""
    
    def __init__(self, update_interval: float = 1.0, history_size: int = 300):
        self.update_interval = update_interval
        self.history_size = history_size
        self.metrics_history = []
        self.current_metrics = None
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        
        # Initialize metrics collector
        try:
            self.collector = MetricsCollector()
        except Exception:
            self.collector = MockMetricsCollector()
        self.alert_manager = None
        
        self.logger = logging.getLogger("SimpleDashboard")
    
    def start(self):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        self.logger.info("Started metrics collection")
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.thread:
            safe_join(self.thread, timeout=2.0, name="dashboard_collection")
        self.logger.info("Stopped metrics collection")
    
    def _collect_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self.collector.collect()
                
                with self.lock:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics.to_dict())
                    
                    # Keep history size limited
                    if len(self.metrics_history) > self.history_size:
                        self.metrics_history.pop(0)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.update_interval)
    
    def get_current_metrics(self):
        """Get current metrics"""
        with self.lock:
            return self.current_metrics.to_dict() if self.current_metrics else {}
    
    def get_metrics_history(self, minutes: int = 5):
        """Get metrics history"""
        cutoff_time = time.time() - (minutes * 60)
        with self.lock:
            return [m for m in self.metrics_history if m.get('timestamp', 0) > cutoff_time]


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard"""
    
    def __init__(self, dashboard, *args, **kwargs):
        self.dashboard = dashboard
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == '/' or path == '/index.html':
                self._serve_dashboard()
            elif path == '/api/current':
                self._serve_current_metrics()
            elif path == '/api/history':
                query_params = parse_qs(parsed_path.query)
                minutes = int(query_params.get('minutes', [5])[0])
                self._serve_metrics_history(minutes)
            else:
                self._serve_404()
        except Exception as e:
            self._serve_error(str(e))
    
    def _serve_dashboard(self):
        """Serve the main dashboard HTML"""
        html = self._generate_dashboard_html()
        self._send_response(200, html, 'text/html')
    
    def _serve_current_metrics(self):
        """Serve current metrics as JSON"""
        metrics = self.dashboard.get_current_metrics()
        self._send_response(200, json.dumps(metrics), 'application/json')
    
    def _serve_metrics_history(self, minutes):
        """Serve metrics history as JSON"""
        history = self.dashboard.get_metrics_history(minutes)
        self._send_response(200, json.dumps(history), 'application/json')
    
    def _serve_404(self):
        """Serve 404 error"""
        self._send_response(404, "Not Found", 'text/plain')
    
    def _serve_error(self, error_msg):
        """Serve error response"""
        self._send_response(500, f"Internal Server Error: {error_msg}", 'text/plain')
    
    def _send_response(self, status_code, content, content_type):
        """Send HTTP response"""
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content.encode('utf-8'))))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _generate_dashboard_html(self):
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Pipeline Monitor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-label {
            font-size: 14px;
            opacity: 0.8;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #00ff88;
        }
        
        .status-warning {
            color: #ffaa00 !important;
        }
        
        .status-critical {
            color: #ff4444 !important;
        }
        
        .info-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .refresh-info {
            text-align: center;
            opacity: 0.7;
            font-size: 14px;
            margin-top: 20px;
        }
        
        .last-updated {
            text-align: center;
            opacity: 0.6;
            font-size: 12px;
            margin-top: 10px;
        }
        
        .loading {
            text-align: center;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GPU Pipeline Monitor</h1>
        
        <div class="metrics-grid" id="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">CPU Usage</div>
                <div class="metric-value" id="cpu-value">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Memory</div>
                <div class="metric-value" id="memory-value">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Usage</div>
                <div class="metric-value" id="gpu-value">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Memory</div>
                <div class="metric-value" id="gpu-mem-value">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Frame Rate</div>
                <div class="metric-value" id="fps-value">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Temp</div>
                <div class="metric-value" id="temp-value">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Power</div>
                <div class="metric-value" id="power-value">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Latency</div>
                <div class="metric-value" id="latency-value">--</div>
            </div>
        </div>
        
        <div class="info-section">
            <h3>📊 System Status</h3>
            <p id="status-text" class="loading">Loading metrics...</p>
            <div class="last-updated" id="last-updated"></div>
        </div>
        
        <div class="refresh-info">
            🔄 Auto-refresh every 5 seconds
        </div>
    </div>
    
    <script>
        function updateMetrics() {
            fetch('/api/current')
                .then(response => response.json())
                .then(data => {
                    if (Object.keys(data).length === 0) {
                        document.getElementById('status-text').textContent = 'No metrics available yet...';
                        return;
                    }
                    
                    // Update metric values
                    document.getElementById('cpu-value').textContent = data.cpu_percent.toFixed(1) + '%';
                    document.getElementById('memory-value').textContent = data.memory_percent.toFixed(1) + '%';
                    document.getElementById('gpu-value').textContent = data.gpu_utilization.toFixed(1) + '%';
                    document.getElementById('gpu-mem-value').textContent = data.gpu_memory_percent.toFixed(1) + '%';
                    document.getElementById('fps-value').textContent = data.frame_rate.toFixed(1);
                    document.getElementById('temp-value').textContent = data.gpu_temperature + '°C';
                    document.getElementById('power-value').textContent = data.gpu_power_watts.toFixed(1) + 'W';
                    document.getElementById('latency-value').textContent = data.latency_ms.toFixed(1) + 'ms';
                    
                    // Update status colors
                    const cpuEl = document.getElementById('cpu-value');
                    cpuEl.className = data.cpu_percent > 15 ? 'metric-value status-critical' : 
                                     data.cpu_percent > 10 ? 'metric-value status-warning' : 
                                     'metric-value';
                    
                    const gpuEl = document.getElementById('gpu-value');
                    gpuEl.className = data.gpu_utilization > 90 ? 'metric-value status-critical' :
                                     data.gpu_utilization > 80 ? 'metric-value status-warning' :
                                     'metric-value';
                    
                    // Update status text
                    let status = '✅ Pipeline running normally';
                    if (data.cpu_percent > 15) status = '⚠️ High CPU usage detected';
                    if (data.gpu_utilization > 90) status = '🔥 GPU at high utilization';
                    if (data.frame_rate < 25) status = '📉 Low frame rate detected';
                    
                    document.getElementById('status-text').textContent = status;
                    document.getElementById('last-updated').textContent = 
                        'Last updated: ' + new Date().toLocaleTimeString();
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                    document.getElementById('status-text').textContent = '❌ Error fetching metrics';
                });
        }
        
        // Initial load
        updateMetrics();
        
        // Auto-refresh every 5 seconds
        setInterval(updateMetrics, 5000);
    </script>
</body>
</html>
        """
    
    def log_message(self, format, *args):
        """Override to reduce log spam"""
        pass


def create_handler_class(dashboard):
    """Create handler class with dashboard instance"""
    def handler(*args, **kwargs):
        DashboardHTTPHandler(dashboard, *args, **kwargs)
    return handler


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Simple GPU Pipeline Monitoring Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 simple_dashboard_server.py                 # Start on port 8080
  python3 simple_dashboard_server.py --port 8888     # Start on custom port
  python3 simple_dashboard_server.py --interval 0.5  # Update every 500ms
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8080, 
        help='Port to serve the dashboard on (default: 8080)'
    )
    
    parser.add_argument(
        '--interval', 
        type=float, 
        default=1.0, 
        help='Metrics update interval in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("simple_dashboard_server")
    
    # Create dashboard
    global _dashboard_instance, _httpd_instance
    logger.info("🚀 Starting Simple GPU Pipeline Dashboard")
    dashboard = SimpleDashboard(update_interval=args.interval)
    _dashboard_instance = dashboard
    dashboard.start()
    
    # Create HTTP server
    handler_class = create_handler_class(dashboard)
    httpd = HTTPServer(('0.0.0.0', args.port), handler_class)
    _httpd_instance = httpd
    
    # Note: Signal handlers are managed by main.py application
    # Dashboard cleanup will be called via stop_dashboard() function
    
    logger.info(f"🌐 Dashboard server started on http://localhost:{args.port}")
    logger.info("📱 Open your browser and navigate to the URL above")
    logger.info("💡 Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard server interrupted by user")
    finally:
        dashboard.stop()
        logger.info("🏁 Dashboard server stopped")


def stop_dashboard():
    """Stop the dashboard server (called by main application)"""
    global _dashboard_instance, _httpd_instance
    if _dashboard_instance:
        _dashboard_instance.stop()
        _dashboard_instance = None
    if _httpd_instance:
        _httpd_instance.shutdown()
        _httpd_instance = None


if __name__ == "__main__":
    main() 
