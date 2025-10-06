import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    host: true,              // or '0.0.0.0' to bind on all interfaces
    port: Number(process.env.PORT || 5173),
    watch: {
      // Avoid watching very large/irrelevant dirs and support optional polling fallback
      ignored: ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**'],
      usePolling: process.env.CHOKIDAR_USEPOLLING === 'true',
      interval: Number(process.env.CHOKIDAR_POLL_INTERVAL || 300),
    },
  },
  preview: {
    host: true,
    port: Number(process.env.PORT || 5173),
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});
