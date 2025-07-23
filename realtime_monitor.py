"""
Real-Time Performance Monitoring Infrastructure
==============================================

Real-time monitoring dashboard for GPU pipeline performance,
providing live metrics, alerts, and visualization.

Features:
- Live performance metrics
- GPU/CPU utilization tracking
- Memory usage monitoring
- Alert system for anomalies
- Web-based dashboard
"""

import asyncio
import json
import time
import psutil
import pynvml
import torch
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import logging
from datetime import datetime
import aiohttp
from aiohttp import web
import aiohttp_cors
import os


@dataclass
class MetricSnapshot:
    """Single metric snapshot"""
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    memory_percent: float
    memory_mb: float
    gpu_utilization: float
    gpu_memory_percent: float
    gpu_memory_mb: float
    gpu_temperature: float
    gpu_power_watts: float
    frame_rate: float
    latency_ms: float
    active_streams: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class MetricsCollector:
    """Collects system and pipeline metrics"""
    
    def __init__(self, gpu_device: int = 0):
        """Initialize metrics collector"""
        self.gpu_device = gpu_device
        self.logger = logging.getLogger("MetricsCollector")
        
        # Initialize NVML
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
        
        # Pipeline metrics (would be connected to actual pipeline)
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = None
        self.active_streams = 0
        
    def collect(self) -> MetricSnapshot:
        """Collect current metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Memory metrics
        mem = psutil.virtual_memory()
        
        # GPU metrics
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        gpu_temp = pynvml.nvmlDeviceGetTemperature(
            self.gpu_handle, 
            pynvml.NVML_TEMPERATURE_GPU
        )
        
        # Power consumption
        try:
            gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # mW to W
        except:
            gpu_power = 0.0
            
        # Frame rate calculation
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            frame_rate = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        else:
            frame_rate = 0.0
            
        # Latency (placeholder - would be from actual pipeline)
        latency_ms = 0.0
        
        return MetricSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=mem.percent,
            memory_mb=mem.used / 1e6,
            gpu_utilization=gpu_util.gpu,
            gpu_memory_percent=(gpu_mem.used / gpu_mem.total) * 100,
            gpu_memory_mb=gpu_mem.used / 1e6,
            gpu_temperature=gpu_temp,
            gpu_power_watts=gpu_power,
            frame_rate=frame_rate,
            latency_ms=latency_ms,
            active_streams=self.active_streams
        )
        
    def update_frame_time(self):
        """Update frame timing information"""
        current_time = time.time()
        if self.last_frame_time is not None:
            self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
    def cleanup(self):
        """Cleanup resources"""
        try:
            pynvml.nvmlShutdown()
        except:
            pass


class AlertManager:
    """Manages performance alerts"""
    
    def __init__(self):
        """Initialize alert manager"""
        self.logger = logging.getLogger("AlertManager")
        self.thresholds = {
            'cpu_percent': {'max': 15.0, 'critical': 25.0},
            'gpu_utilization': {'min': 40.0, 'critical_min': 20.0},
            'gpu_temperature': {'max': 80.0, 'critical': 85.0},
            'memory_percent': {'max': 80.0, 'critical': 90.0},
            'frame_rate': {'min': 20.0, 'critical_min': 10.0}
        }
        self.alerts = deque(maxlen=100)
        self.alert_callbacks = []
        
    def check_metrics(self, metrics: MetricSnapshot) -> List[Dict[str, Any]]:
        """Check metrics against thresholds"""
        new_alerts = []
        
        # CPU usage
        if metrics.cpu_percent > self.thresholds['cpu_percent']['critical']:
            new_alerts.append({
                'level': 'critical',
                'metric': 'cpu_percent',
                'value': metrics.cpu_percent,
                'threshold': self.thresholds['cpu_percent']['critical'],
                'message': f'Critical CPU usage: {metrics.cpu_percent:.1f}%'
            })
        elif metrics.cpu_percent > self.thresholds['cpu_percent']['max']:
            new_alerts.append({
                'level': 'warning',
                'metric': 'cpu_percent',
                'value': metrics.cpu_percent,
                'threshold': self.thresholds['cpu_percent']['max'],
                'message': f'High CPU usage: {metrics.cpu_percent:.1f}%'
            })
            
        # GPU utilization
        if metrics.gpu_utilization < self.thresholds['gpu_utilization']['critical_min']:
            new_alerts.append({
                'level': 'critical',
                'metric': 'gpu_utilization',
                'value': metrics.gpu_utilization,
                'threshold': self.thresholds['gpu_utilization']['critical_min'],
                'message': f'Critical low GPU utilization: {metrics.gpu_utilization:.1f}%'
            })
        elif metrics.gpu_utilization < self.thresholds['gpu_utilization']['min']:
            new_alerts.append({
                'level': 'warning',
                'metric': 'gpu_utilization',
                'value': metrics.gpu_utilization,
                'threshold': self.thresholds['gpu_utilization']['min'],
                'message': f'Low GPU utilization: {metrics.gpu_utilization:.1f}%'
            })
            
        # GPU temperature
        if metrics.gpu_temperature > self.thresholds['gpu_temperature']['critical']:
            new_alerts.append({
                'level': 'critical',
                'metric': 'gpu_temperature',
                'value': metrics.gpu_temperature,
                'threshold': self.thresholds['gpu_temperature']['critical'],
                'message': f'Critical GPU temperature: {metrics.gpu_temperature}°C'
            })
            
        # Frame rate
        if metrics.frame_rate > 0:  # Only check if we have frame data
            if metrics.frame_rate < self.thresholds['frame_rate']['critical_min']:
                new_alerts.append({
                    'level': 'critical',
                    'metric': 'frame_rate',
                    'value': metrics.frame_rate,
                    'threshold': self.thresholds['frame_rate']['critical_min'],
                    'message': f'Critical low frame rate: {metrics.frame_rate:.1f} FPS'
                })
                
        # Store alerts
        for alert in new_alerts:
            alert['timestamp'] = metrics.timestamp
            self.alerts.append(alert)
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
                    
        return new_alerts
        
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
        
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return list(self.alerts)[-count:]
        
    def update_threshold(self, metric: str, threshold_type: str, value: float):
        """Update alert threshold"""
        if metric in self.thresholds and threshold_type in self.thresholds[metric]:
            self.thresholds[metric][threshold_type] = value
            self.logger.info(f"Updated threshold: {metric}.{threshold_type} = {value}")


class RealtimeMonitor:
    """Main real-time monitoring system"""
    
    def __init__(self, 
                 update_interval: float = 1.0,
                 history_size: int = 3600):
        """
        Initialize monitor.
        
        Args:
            update_interval: Metric update interval in seconds
            history_size: Number of historical snapshots to keep
        """
        self.update_interval = update_interval
        self.history_size = history_size
        self.logger = logging.getLogger("RealtimeMonitor")
        
        # Components
        self.collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Data storage
        self.metrics_history = deque(maxlen=history_size)
        self.current_metrics = None
        
        # Control
        self.running = False
        self.monitor_thread = None
        
        # WebSocket connections
        self.websocket_clients = set()
        
    def start(self):
        """Start monitoring"""
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Real-time monitor started")
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        self.collector.cleanup()
        self.logger.info("Real-time monitor stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self.collector.collect()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Check for alerts
                alerts = self.alert_manager.check_metrics(metrics)
                
                # Broadcast to WebSocket clients
                asyncio.run(self._broadcast_update(metrics, alerts))
                
                # Wait for next interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                
    async def _broadcast_update(self, metrics: MetricSnapshot, alerts: List[Dict]):
        """Broadcast update to WebSocket clients"""
        if not self.websocket_clients:
            return
            
        message = json.dumps({
            'type': 'update',
            'metrics': metrics.to_dict(),
            'alerts': alerts
        })
        
        # Send to all connected clients
        disconnected = set()
        
        for ws in self.websocket_clients:
            try:
                await ws.send_str(message)
            except ConnectionResetError:
                disconnected.add(ws)
                
        # Remove disconnected clients
        self.websocket_clients -= disconnected
        
    def get_current_metrics(self) -> Optional[MetricSnapshot]:
        """Get current metrics"""
        return self.current_metrics
        
    def get_metrics_history(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get metrics history"""
        if not self.metrics_history:
            return []
            
        # Calculate how many samples to return
        samples = int(minutes * 60 / self.update_interval)
        samples = min(samples, len(self.metrics_history))
        
        return [m.to_dict() for m in list(self.metrics_history)[-samples:]]
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.metrics_history:
            return {}
            
        # Convert to lists for calculation
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        gpu_values = [m.gpu_utilization for m in self.metrics_history]
        fps_values = [m.frame_rate for m in self.metrics_history if m.frame_rate > 0]
        
        import numpy as np
        
        return {
            'cpu': {
                'current': cpu_values[-1] if cpu_values else 0,
                'mean': np.mean(cpu_values) if cpu_values else 0,
                'max': np.max(cpu_values) if cpu_values else 0,
                'p95': np.percentile(cpu_values, 95) if cpu_values else 0
            },
            'gpu': {
                'current': gpu_values[-1] if gpu_values else 0,
                'mean': np.mean(gpu_values) if gpu_values else 0,
                'min': np.min(gpu_values) if gpu_values else 0,
                'max': np.max(gpu_values) if gpu_values else 0
            },
            'fps': {
                'current': fps_values[-1] if fps_values else 0,
                'mean': np.mean(fps_values) if fps_values else 0,
                'min': np.min(fps_values) if fps_values else 0
            }
        }


class MonitorWebServer:
    """Web server for monitoring dashboard"""
    
    def __init__(self, monitor: RealtimeMonitor, port: int = 8080):
        """Initialize web server"""
        self.monitor = monitor
        self.port = port
        self.app = web.Application()
        self.logger = logging.getLogger("MonitorWebServer")
        
        # Setup routes
        self._setup_routes()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
            
    def _setup_routes(self):
        """Setup web routes"""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/api/metrics/current', self.handle_current_metrics)
        self.app.router.add_get('/api/metrics/history', self.handle_metrics_history)
        self.app.router.add_get('/api/metrics/summary', self.handle_summary)
        self.app.router.add_get('/api/alerts', self.handle_alerts)
        self.app.router.add_get('/ws', self.handle_websocket)
        
    async def handle_index(self, request):
        """Serve dashboard HTML"""
        html = self._generate_dashboard_html()
        return web.Response(text=html, content_type='text/html')
        
    async def handle_current_metrics(self, request):
        """Get current metrics"""
        metrics = self.monitor.get_current_metrics()
        if metrics:
            return web.json_response(metrics.to_dict())
        else:
            return web.json_response({'error': 'No metrics available'}, status=503)
            
    async def handle_metrics_history(self, request):
        """Get metrics history"""
        minutes = int(request.query.get('minutes', 5))
        history = self.monitor.get_metrics_history(minutes)
        return web.json_response({'history': history})
        
    async def handle_summary(self, request):
        """Get summary statistics"""
        summary = self.monitor.get_summary_stats()
        return web.json_response(summary)
        
    async def handle_alerts(self, request):
        """Get recent alerts"""
        count = int(request.query.get('count', 10))
        alerts = self.monitor.alert_manager.get_recent_alerts(count)
        return web.json_response({'alerts': alerts})
        
    async def handle_websocket(self, request):
        """Handle WebSocket connection"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Add to clients
        self.monitor.websocket_clients.add(ws)
        
        try:
            # Send initial data
            metrics = self.monitor.get_current_metrics()
            if metrics:
                await ws.send_str(json.dumps({
                    'type': 'initial',
                    'metrics': metrics.to_dict(),
                    'history': self.monitor.get_metrics_history(5)
                }))
                
            # Keep connection alive
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle client messages if needed
                    pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')
                    
        finally:
            # Remove from clients
            self.monitor.websocket_clients.discard(ws)
            
        return ws
        
    def _generate_dashboard_html(self):
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>GPU Pipeline Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #888;
            font-size: 0.9em;
        }
        .chart-container {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            height: 300px;
        }
        .alerts {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            max-height: 200px;
            overflow-y: auto;
        }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }
        .alert.warning {
            background: #554400;
        }
        .alert.critical {
            background: #550000;
        }
        .status-good { color: #4CAF50; }
        .status-warning { color: #FFC107; }
        .status-critical { color: #F44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>GPU Pipeline Real-Time Monitor</h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">CPU Usage</div>
                <div class="metric-value" id="cpu-value">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GPU Utilization</div>
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
                <div class="metric-label">Power</div>
                <div class="metric-value" id="power-value">--</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="metrics-chart"></canvas>
        </div>
        
        <div class="alerts">
            <h3>Recent Alerts</h3>
            <div id="alerts-container"></div>
        </div>
    </div>
    
    <script>
        // Initialize chart
        const ctx = document.getElementById('metrics-chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU %',
                    data: [],
                    borderColor: '#FF6384',
                    fill: false
                }, {
                    label: 'GPU %',
                    data: [],
                    borderColor: '#36A2EB',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        
        // WebSocket connection
        let ws = null;
        
        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'update') {
                    updateMetrics(data.metrics);
                    if (data.alerts.length > 0) {
                        updateAlerts(data.alerts);
                    }
                } else if (data.type === 'initial') {
                    updateMetrics(data.metrics);
                    initializeChart(data.history);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                setTimeout(connect, 3000); // Reconnect
            };
        }
        
        function updateMetrics(metrics) {
            // Update metric values
            document.getElementById('cpu-value').textContent = metrics.cpu_percent.toFixed(1) + '%';
            document.getElementById('gpu-value').textContent = metrics.gpu_utilization.toFixed(1) + '%';
            document.getElementById('gpu-mem-value').textContent = metrics.gpu_memory_percent.toFixed(1) + '%';
            document.getElementById('fps-value').textContent = metrics.frame_rate.toFixed(1);
            document.getElementById('temp-value').textContent = metrics.gpu_temperature + '°C';
            document.getElementById('power-value').textContent = metrics.gpu_power_watts.toFixed(1) + 'W';
            
            // Update chart
            const time = new Date(metrics.timestamp * 1000).toLocaleTimeString();
            chart.data.labels.push(time);
            chart.data.datasets[0].data.push(metrics.cpu_percent);
            chart.data.datasets[1].data.push(metrics.gpu_utilization);
            
            // Keep last 60 points
            if (chart.data.labels.length > 60) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            chart.update('none'); // No animation for real-time
            
            // Update colors based on thresholds
            const cpuEl = document.getElementById('cpu-value');
            cpuEl.className = metrics.cpu_percent > 15 ? 'metric-value status-critical' : 
                             metrics.cpu_percent > 10 ? 'metric-value status-warning' : 
                             'metric-value status-good';
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            
            alerts.forEach(alert => {
                const alertEl = document.createElement('div');
                alertEl.className = `alert ${alert.level}`;
                alertEl.textContent = `${new Date(alert.timestamp * 1000).toLocaleTimeString()} - ${alert.message}`;
                container.insertBefore(alertEl, container.firstChild);
            });
            
            // Keep last 10 alerts
            while (container.children.length > 10) {
                container.removeChild(container.lastChild);
            }
        }
        
        function initializeChart(history) {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.data.datasets[1].data = [];
            
            history.forEach(metrics => {
                const time = new Date(metrics.timestamp * 1000).toLocaleTimeString();
                chart.data.labels.push(time);
                chart.data.datasets[0].data.push(metrics.cpu_percent);
                chart.data.datasets[1].data.push(metrics.gpu_utilization);
            });
            
            chart.update();
        }
        
        // Start connection
        connect();
    </script>
</body>
</html>
        """
        
    async def start(self):
        """Start web server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        self.logger.info(f"Monitor web server started on port {self.port}")


async def run_monitor_server(monitor: RealtimeMonitor, port: int = 8080):
    """Run monitoring web server"""
    server = MonitorWebServer(monitor, port)
    await server.start()
    
    # Keep server running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time GPU Pipeline Monitor")
    parser.add_argument('--port', type=int, default=8080,
                        help='Web server port (default: 8080)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Update interval in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start monitor
    monitor = RealtimeMonitor(update_interval=args.interval)
    monitor.start()
    
    print(f"Monitor started. Dashboard available at http://localhost:{args.port}")
    
    try:
        # Run web server
        asyncio.run(run_monitor_server(monitor, args.port))
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        monitor.stop() 