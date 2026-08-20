const positive = (value) => Number.isFinite(Number(value)) && Number(value) > 0.5;

/**
 * Build a presentation-only floor plane from measured PCF floor support.
 *
 * The plane is intentionally independent of authored room rectangles. Direct
 * floor evidence is expanded by only a few grid cells, enclosed sampling gaps
 * are filled, and tiny disconnected flecks are discarded. This lets a room's
 * measured appendages participate without turning unknown space into floor.
 */
export function derivePlanarFloorMask({
  floorSupport,
  rows,
  cols,
  expansionCells = 2,
  minComponentFraction = 0.02,
}) {
  const height = Math.floor(Number(rows));
  const width = Math.floor(Number(cols));
  if (height <= 0 || width <= 0) {
    throw new TypeError('planar floor dimensions must be positive integers');
  }
  const cellCount = height * width;
  if (!floorSupport || floorSupport.length < cellCount) {
    throw new TypeError('planar floor support does not match the requested grid');
  }

  const seeds = new Uint8Array(cellCount);
  let current = new Uint8Array(cellCount);
  let seedCount = 0;
  for (let idx = 0; idx < cellCount; idx += 1) {
    if (!positive(floorSupport[idx])) continue;
    seeds[idx] = 1;
    current[idx] = 1;
    seedCount += 1;
  }
  if (seedCount === 0) return current;

  // Five centimetres at the promoted 2.5 cm PCF grid is enough to join dense
  // floor samples without inventing a broad room envelope.
  const iterations = Math.min(8, Math.max(0, Math.floor(Number(expansionCells) || 0)));
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const next = new Uint8Array(current);
    for (let row = 0; row < height; row += 1) {
      for (let col = 0; col < width; col += 1) {
        const idx = (row * width) + col;
        if (current[idx]) continue;
        let supported = false;
        for (let rowOffset = -1; rowOffset <= 1 && !supported; rowOffset += 1) {
          const neighbourRow = row + rowOffset;
          if (neighbourRow < 0 || neighbourRow >= height) continue;
          for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
            const neighbourCol = col + colOffset;
            if (neighbourCol < 0 || neighbourCol >= width) continue;
            if (current[(neighbourRow * width) + neighbourCol]) {
              supported = true;
              break;
            }
          }
        }
        if (supported) next[idx] = 1;
      }
    }
    current = next;
  }

  // Flood the complement from the raster edge. Any remaining complement cell
  // is enclosed by expanded PCF evidence and can be represented by the plane.
  const exterior = new Uint8Array(cellCount);
  const queue = new Int32Array(cellCount);
  let head = 0;
  let tail = 0;
  const enqueueExterior = (idx) => {
    if (current[idx] || exterior[idx]) return;
    exterior[idx] = 1;
    queue[tail] = idx;
    tail += 1;
  };
  for (let col = 0; col < width; col += 1) {
    enqueueExterior(col);
    enqueueExterior(((height - 1) * width) + col);
  }
  for (let row = 1; row < height - 1; row += 1) {
    enqueueExterior(row * width);
    enqueueExterior((row * width) + width - 1);
  }
  while (head < tail) {
    const idx = queue[head];
    head += 1;
    const row = Math.floor(idx / width);
    const col = idx % width;
    if (row > 0) enqueueExterior(idx - width);
    if (row + 1 < height) enqueueExterior(idx + width);
    if (col > 0) enqueueExterior(idx - 1);
    if (col + 1 < width) enqueueExterior(idx + 1);
  }

  const candidate = new Uint8Array(cellCount);
  for (let idx = 0; idx < cellCount; idx += 1) {
    candidate[idx] = current[idx] || !exterior[idx] ? 1 : 0;
  }

  // Keep the dominant connected floor surface and any separately measured
  // surface that is substantial relative to it. This removes isolated PCF
  // flecks without requiring an authored rectangular crop.
  const labels = new Int32Array(cellCount);
  const componentAreas = [0];
  const componentSupports = [0];
  let componentCount = 0;
  for (let start = 0; start < cellCount; start += 1) {
    if (!candidate[start] || labels[start]) continue;
    componentCount += 1;
    head = 0;
    tail = 0;
    queue[tail] = start;
    tail += 1;
    labels[start] = componentCount;
    let area = 0;
    let support = 0;
    while (head < tail) {
      const idx = queue[head];
      head += 1;
      area += 1;
      support += seeds[idx];
      const row = Math.floor(idx / width);
      const col = idx % width;
      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        const neighbourRow = row + rowOffset;
        if (neighbourRow < 0 || neighbourRow >= height) continue;
        for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
          if (rowOffset === 0 && colOffset === 0) continue;
          const neighbourCol = col + colOffset;
          if (neighbourCol < 0 || neighbourCol >= width) continue;
          const neighbourIndex = (neighbourRow * width) + neighbourCol;
          if (!candidate[neighbourIndex] || labels[neighbourIndex]) continue;
          labels[neighbourIndex] = componentCount;
          queue[tail] = neighbourIndex;
          tail += 1;
        }
      }
    }
    componentAreas.push(area);
    componentSupports.push(support);
  }

  let primary = 1;
  for (let component = 2; component <= componentCount; component += 1) {
    if (
      componentSupports[component] > componentSupports[primary]
      || (
        componentSupports[component] === componentSupports[primary]
        && componentAreas[component] > componentAreas[primary]
      )
    ) primary = component;
  }
  const fraction = Math.min(0.25, Math.max(0, Number(minComponentFraction) || 0));
  const minimumArea = Math.max(16, Math.ceil(componentAreas[primary] * fraction));
  const minimumSupport = Math.max(2, Math.ceil(componentSupports[primary] * fraction));
  const retained = new Uint8Array(componentCount + 1);
  retained[primary] = 1;
  for (let component = 1; component <= componentCount; component += 1) {
    if (
      componentAreas[component] >= minimumArea
      && componentSupports[component] >= minimumSupport
    ) retained[component] = 1;
  }

  const plane = new Uint8Array(cellCount);
  for (let idx = 0; idx < cellCount; idx += 1) {
    plane[idx] = retained[labels[idx]] || 0;
  }
  return plane;
}
