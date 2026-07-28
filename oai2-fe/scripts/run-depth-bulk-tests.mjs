import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { build } from 'esbuild';

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const outputDirectory = await mkdtemp(join(tmpdir(), 'noesis-depth-bulk-tests-'));
const entryPoints = [
  join(projectRoot, 'tests/depthBulkWorker.test.ts'),
  join(projectRoot, 'tests/depthBulkClient.test.ts'),
  join(projectRoot, 'tests/depthBulkLimits.test.ts'),
  join(projectRoot, 'tests/depthDiagnostics.test.ts'),
  join(projectRoot, 'tests/depthNormals.test.ts'),
  join(projectRoot, 'tests/normalDisplay.test.ts'),
  join(projectRoot, 'tests/visibleFloorPlane.test.ts'),
];

const run = (command, args) => new Promise((resolve, reject) => {
  const child = spawn(command, args, {
    cwd: projectRoot,
    stdio: 'inherit',
  });
  child.once('error', reject);
  child.once('exit', (code, signal) => {
    if (signal) {
      reject(new Error(`depth bulk tests terminated by ${signal}`));
      return;
    }
    resolve(code ?? 1);
  });
});

try {
  await build({
    entryPoints,
    bundle: true,
    platform: 'node',
    format: 'esm',
    outExtension: { '.js': '.mjs' },
    outdir: outputDirectory,
    logLevel: 'warning',
  });
  const outputs = entryPoints.map((entry) => (
    join(outputDirectory, basename(entry).replace(/\.ts$/, '.mjs'))
  ));
  const exitCode = await run(process.execPath, ['--test', ...outputs]);
  if (exitCode !== 0) process.exitCode = exitCode;
} finally {
  await rm(outputDirectory, { recursive: true, force: true });
}
