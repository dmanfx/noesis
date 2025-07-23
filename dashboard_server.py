#!/usr/bin/env python3
"""
Standalone GPU Pipeline Monitoring Dashboard Server

This script provides easy access to the real-time monitoring dashboard
for the GPU pipeline optimization system. It starts a web server that
displays live performance metrics, alerts, and system status.

Usage:
    python dashboard_server.py [--port 8080] [--update-interval 1.0]

Features:
- Real-time CPU and GPU metrics
- Live performance charts
- Alert system for threshold violations
- WebSocket-based updates
- Responsive web interface

Access the dashboard at: http://localhost:8080
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from realtime_monitor import RealtimeMonitor, run_monitor_server

# Global monitor instance for external cleanup
_monitor_instance = None


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


async def main():
    """Main entry point for the dashboard server"""
    parser = argparse.ArgumentParser(
        description='GPU Pipeline Monitoring Dashboard Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dashboard_server.py                    # Start on default port 8080
  python dashboard_server.py --port 8888        # Start on custom port
  python dashboard_server.py --debug            # Enable debug logging
  python dashboard_server.py --update-interval 0.5  # Update every 500ms
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8080, 
        help='Port to serve the dashboard on (default: 8080)'
    )
    
    parser.add_argument(
        '--update-interval', 
        type=float, 
        default=1.0, 
        help='Metrics update interval in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--history-size',
        type=int,
        default=3600,
        help='Number of metric snapshots to keep in history (default: 3600)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger("dashboard_server")
    
    # Validate arguments
    if args.port < 1 or args.port > 65535:
        logger.error(f"Invalid port number: {args.port}. Must be between 1 and 65535.")
        sys.exit(1)
    
    if args.update_interval <= 0:
        logger.error(f"Invalid update interval: {args.update_interval}. Must be positive.")
        sys.exit(1)
    
    # Create and start the monitor
    logger.info("🚀 Starting GPU Pipeline Monitoring Dashboard")
    logger.info(f"📊 Update interval: {args.update_interval}s")
    logger.info(f"📈 History size: {args.history_size} snapshots")
    
    try:
        global _monitor_instance
        monitor = RealtimeMonitor(
            update_interval=args.update_interval,
            history_size=args.history_size
        )
        _monitor_instance = monitor
        
        # Start monitoring
        monitor.start()
        logger.info("✅ Performance monitoring started")
        
        # Note: Signal handlers are managed by main.py application
        # Dashboard cleanup will be called via stop() method
        
        # Start web server
        logger.info(f"🌐 Starting web server on http://localhost:{args.port}")
        logger.info(f"📱 Dashboard will be available at: http://localhost:{args.port}")
        logger.info("💡 Press Ctrl+C to stop the server")
        
        await run_monitor_server(monitor, args.port)
        
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard server: {e}")
        if args.debug:
            logger.exception("Full traceback:")
        sys.exit(1)
    
    finally:
        logger.info("🏁 Dashboard server stopped")


def stop_dashboard():
    """Stop the dashboard server (called by main application)"""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop()
        _monitor_instance = None


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 