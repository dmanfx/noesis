// preload.js - Exposes limited Node.js/Electron APIs to the Renderer securely

const { contextBridge, ipcRenderer } = require('electron');

// Expose a safe subset of APIs needed by the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // We aren't using IPC from Main to Renderer in this version,
  // but preload is still required for contextIsolation.
  // If we needed to send messages *from* main *to* renderer later,
  // we would add listeners here, e.g.:
  // on: (channel, callback) => {
  //     ipcRenderer.on(channel, (event, ...args) => callback(...args));
  // }
});

console.log("Preload script loaded.");