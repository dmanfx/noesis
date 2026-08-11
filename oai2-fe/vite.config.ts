import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

function validAllowedHost(host: string): boolean {
  if (/^\d{1,3}(?:\.\d{1,3}){3}$/.test(host)) {
    return host.split('.').every((part) => Number(part) <= 255);
  }
  if (host.length > 253) return false;
  return host.split('.').every((label) => (
    label.length > 0
    && label.length <= 63
    && /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(label)
  ));
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const proxyTarget = env.OAI2_MENON_PROXY_TARGET
    || process.env.OAI2_MENON_PROXY_TARGET
    || 'https://127.0.0.1:3001';
  const uiPort = Number(env.VITE_UI_PORT || process.env.VITE_UI_PORT || 5173);
  const dashboardHost = String(
    env.OAI2_DASHBOARD_HOST || process.env.OAI2_DASHBOARD_HOST || '127.0.0.1',
  ).trim();
  if (!['127.0.0.1', '0.0.0.0'].includes(dashboardHost)) {
    throw new Error('OAI2_DASHBOARD_HOST must be 127.0.0.1 or 0.0.0.0.');
  }
  const allowedHosts = [...new Set(String(
    env.OAI2_DASHBOARD_ALLOWED_HOSTS || process.env.OAI2_DASHBOARD_ALLOWED_HOSTS || '',
  ).split(',').map((candidate) => candidate.trim().toLowerCase()).filter(Boolean))];
  if (!allowedHosts.length && dashboardHost === '127.0.0.1') allowedHosts.push('127.0.0.1');
  if (!allowedHosts.length) {
    throw new Error('A LAN dashboard listener requires OAI2_DASHBOARD_ALLOWED_HOSTS.');
  }
  if (allowedHosts.some((host) => host === '*' || !validAllowedHost(host))) {
    throw new Error('OAI2_DASHBOARD_ALLOWED_HOSTS contains an invalid host name.');
  }
  const parsedProxyTarget = new URL(proxyTarget);
  if (
    !['http:', 'https:'].includes(parsedProxyTarget.protocol)
    || !['127.0.0.1', '[::1]', '::1'].includes(parsedProxyTarget.hostname)
    || parsedProxyTarget.username
    || parsedProxyTarget.password
    || parsedProxyTarget.pathname !== '/'
    || parsedProxyTarget.search
    || parsedProxyTarget.hash
  ) {
    throw new Error('OAI2_MENON_PROXY_TARGET must be an HTTP(S) loopback origin.');
  }
  const commonProxy = {
    target: parsedProxyTarget.origin,
    changeOrigin: true,
    secure: false,
    xfwd: false,
  };
  const proxy = {
    '/api/noesis/ws': {
      ...commonProxy,
      ws: true,
      timeout: 0,
      proxyTimeout: 0,
    },
    '/api': {
      ...commonProxy,
      timeout: 180_000,
      proxyTimeout: 180_000,
    },
  };

  return {
    plugins: [react()],
    base: './',
    server: {
      host: dashboardHost,
      port: uiPort,
      strictPort: true,
      allowedHosts,
      proxy,
      watch: {
        // Avoid watching very large/irrelevant dirs and support optional polling fallback
        ignored: ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**'],
        usePolling: process.env.CHOKIDAR_USEPOLLING === 'true',
        interval: Number(process.env.CHOKIDAR_POLL_INTERVAL || 300),
      },
    },
    preview: {
      host: dashboardHost,
      port: uiPort,
      strictPort: true,
      allowedHosts,
      proxy,
    },
    build: {
      outDir: 'dist',
      emptyOutDir: true,
    },
  };
});
