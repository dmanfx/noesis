export type HeightMapRequestOptions = {
  camera?: string;
  requestId?: string;
  tsMode?: 'latest' | 'fresh';
  gridResM?: number;
  maxExtentM?: number;
  cacheOnly?: boolean;
};

export type HeightMapResponse = {
  type?: string;
  camera?: string;
  camera_id?: string;
  width?: number;
  height?: number;
  units?: string;
  z_offset?: number;
  z_min?: number;
  z_max?: number;
  data?: number[];
  density?: {
    width: number;
    height: number;
    min: number;
    max: number;
    data: number[];
  };
  distance?: {
    width: number;
    height: number;
    min: number;
    max: number;
    data: number[];
  };
  meta?: {
    generated_at?: string;
    source?: string;
    room_bbox_m?: [number, number, number, number] | null;
  };
  served_from_cache?: boolean;
  grid_res_m?: number;
  max_extent_m?: number;
  ts?: number;
  request_id?: string;
  error?: string;
  ok?: boolean;
};
