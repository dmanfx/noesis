import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    watch: {
      // Avoid watching very large/irrelevant dirs and support optional polling fallback
      ignored: ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**'],
      usePolling: process.env.CHOKIDAR_USEPOLLING === 'true',
      interval: Number(process.env.CHOKIDAR_POLL_INTERVAL || 300),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});
