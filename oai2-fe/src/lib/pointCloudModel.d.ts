export type CalibratedPointCloudModel = {
  positions: Float32Array;
  depths: Float32Array;
  confidences: Float32Array;
  sourcePixels: Uint32Array;
  rgbColors: Uint8Array | null;
  count: number;
  stride: number;
  width: number;
  height: number;
  intrinsics: [number, number, number, number];
};

export function buildCalibratedPointCloud(options: {
  depthValues: ArrayLike<number>;
  confidenceValues?: ArrayLike<number> | null;
  maskValues?: ArrayLike<number> | null;
  rgbValues?: ArrayLike<number> | null;
  rgbShape?: [number, number, number] | null;
  width: number;
  height: number;
  intrinsics?: number[] | null;
  maxPoints?: number;
  minDepthM?: number;
  maxDepthM?: number;
  minConfidence?: number;
}): CalibratedPointCloudModel | null;
