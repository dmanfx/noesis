import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const restProxyTarget = env.VITE_REST_PROXY_TARGET || process.env.VITE_REST_PROXY_TARGET || 'http://127.0.0.1:8080';
  const uiPort = Number(env.VITE_UI_PORT || process.env.VITE_UI_PORT || 5173);

  return {
    plugins: [react()],
    base: './',
    server: {
      host: true, // or '0.0.0.0' to bind on all interfaces
      port: uiPort,
      proxy: {
        // Dev-only proxy so the browser can call the DS8 REST API via same-origin `/api/...`
        // without requiring CORS. Override with `VITE_REST_PROXY_TARGET` if needed.
        '/api': {
          target: restProxyTarget,
          changeOrigin: true,
        },
      },
      watch: {
        // Avoid watching very large/irrelevant dirs and support optional polling fallback
        ignored: ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**'],
        usePolling: process.env.CHOKIDAR_USEPOLLING === 'true',
        interval: Number(process.env.CHOKIDAR_POLL_INTERVAL || 300),
      },
    },
    preview: {
      host: true,
      port: uiPort,
    },
    build: {
      outDir: 'dist',
      emptyOutDir: true,
    },
  };
});
