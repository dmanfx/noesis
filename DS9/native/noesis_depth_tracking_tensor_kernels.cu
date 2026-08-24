#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

__global__ void sample_roi_values_kernel(
    const float* __restrict__ depth,
    int frame_w,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float* __restrict__ out_values) {
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= sampled_area) {
    return;
  }

  const int sample_y = idx / sampled_cols;
  const int sample_x = idx - (sample_y * sampled_cols);
  const int local_x = min(sample_x * stride, roi_w - 1);
  const int local_y = min(sample_y * stride, roi_h - 1);
  const int frame_x = x0 + local_x;
  const int frame_y = y0 + local_y;
  out_values[idx] = depth[(static_cast<size_t>(frame_y) * static_cast<size_t>(frame_w)) + static_cast<size_t>(frame_x)];
}

__global__ void sample_roi_center_kernel(
    const float* __restrict__ depth,
    int frame_w,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    float* __restrict__ out_center) {
  const int frame_x = x0 + min(roi_w - 1, roi_w / 2);
  const int frame_y = y0 + min(roi_h - 1, roi_h / 2);
  *out_center = depth[(static_cast<size_t>(frame_y) * static_cast<size_t>(frame_w)) + static_cast<size_t>(frame_x)];
}

__device__ bool point_in_capsule(
    float point_x,
    float point_y,
    float ax,
    float ay,
    float bx,
    float by,
    float radius) {
  const float dx = bx - ax;
  const float dy = by - ay;
  const float length_sq = (dx * dx) + (dy * dy);
  float t = 0.0F;
  if (length_sq > 1.0e-6F) {
    t = (((point_x - ax) * dx) + ((point_y - ay) * dy)) / length_sq;
    t = fminf(1.0F, fmaxf(0.0F, t));
  }
  const float nearest_x = ax + (t * dx);
  const float nearest_y = ay + (t * dy);
  const float delta_x = point_x - nearest_x;
  const float delta_y = point_y - nearest_y;
  return (delta_x * delta_x) + (delta_y * delta_y) <= (radius * radius);
}

__device__ bool point_in_pose_contact(
    float point_x,
    float point_y,
    float ax,
    float ay,
    float bx,
    float by,
    float line_radius,
    float ankle_radius) {
  if (point_in_capsule(
          point_x, point_y, ax, ay, bx, by, line_radius)) {
    return true;
  }
  const float ankle_dx = point_x - bx;
  const float ankle_dy = point_y - by;
  return (ankle_dx * ankle_dx) + (ankle_dy * ankle_dy) <=
      (ankle_radius * ankle_radius);
}

// The pose-contact path passes at most two lower-leg capsules as kernel
// parameters.  This deliberately keeps the geometry scalar and avoids a
// host mask upload or a per-call device geometry allocation.
__global__ void sample_pose_capsule_values_kernel(
    const float* __restrict__ depth,
    int frame_w,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    int capsule_count,
    float c0_ax,
    float c0_ay,
    float c0_bx,
    float c0_by,
    float c0_line_radius,
    float c0_ankle_radius,
    float c1_ax,
    float c1_ay,
    float c1_bx,
    float c1_by,
    float c1_line_radius,
    float c1_ankle_radius,
    float* __restrict__ out_values) {
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= sampled_area) {
    return;
  }

  const int sample_y = idx / sampled_cols;
  const int sample_x = idx - (sample_y * sampled_cols);
  const int local_x = min(sample_x * stride, roi_w - 1);
  const int local_y = min(sample_y * stride, roi_h - 1);
  const float point_x = static_cast<float>(x0 + local_x) + 0.5F;
  const float point_y = static_cast<float>(y0 + local_y) + 0.5F;
  bool active = false;
  if (capsule_count > 0) {
    active = point_in_pose_contact(
        point_x,
        point_y,
        c0_ax,
        c0_ay,
        c0_bx,
        c0_by,
        c0_line_radius,
        c0_ankle_radius);
  }
  if (!active && capsule_count > 1) {
    active = point_in_pose_contact(
        point_x,
        point_y,
        c1_ax,
        c1_ay,
        c1_bx,
        c1_by,
        c1_line_radius,
        c1_ankle_radius);
  }
  if (!active) {
    out_values[idx] = CUDART_NAN_F;
    return;
  }

  const int frame_x = x0 + local_x;
  const int frame_y = y0 + local_y;
  out_values[idx] = depth[(static_cast<size_t>(frame_y) * static_cast<size_t>(frame_w)) + static_cast<size_t>(frame_x)];
}

__global__ void sample_masked_roi_values_kernel(
    const float* __restrict__ depth,
    const float* __restrict__ mask,
    int frame_w,
    int mask_w,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float threshold,
    float* __restrict__ out_values) {
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= sampled_area) {
    return;
  }

  const int sample_y = idx / sampled_cols;
  const int sample_x = idx - (sample_y * sampled_cols);
  const int local_x = min(sample_x * stride, roi_w - 1);
  const int local_y = min(sample_y * stride, roi_h - 1);
  const int mask_idx = (local_y * mask_w) + local_x;
  if (mask[mask_idx] <= threshold) {
    out_values[idx] = CUDART_NAN_F;
    return;
  }

  const int frame_x = x0 + local_x;
  const int frame_y = y0 + local_y;
  out_values[idx] = depth[(static_cast<size_t>(frame_y) * static_cast<size_t>(frame_w)) + static_cast<size_t>(frame_x)];
}

__device__ bool mask_eroded3x3(
    const float* __restrict__ mask,
    int mask_w,
    int mask_h,
    int x,
    int y,
    float threshold) {
  for (int dy = -1; dy <= 1; ++dy) {
    const int yy = y + dy;
    if (yy < 0 || yy >= mask_h) {
      return false;
    }
    for (int dx = -1; dx <= 1; ++dx) {
      const int xx = x + dx;
      if (xx < 0 || xx >= mask_w) {
        return false;
      }
      if (mask[(yy * mask_w) + xx] <= threshold) {
        return false;
      }
    }
  }
  return true;
}

__device__ bool in_center_band(
    int x,
    int y,
    int roi_w,
    int roi_h,
    float y0_ratio,
    float y1_ratio,
    float center_width_ratio) {
  int band_y0 = static_cast<int>(floorf(static_cast<float>(roi_h) * y0_ratio));
  int band_y1 = static_cast<int>(ceilf(static_cast<float>(roi_h) * y1_ratio));
  band_y0 = max(0, min(roi_h, band_y0));
  band_y1 = max(band_y0 + 1, min(roi_h, band_y1));
  int band_w = static_cast<int>(roundf(static_cast<float>(roi_w) * center_width_ratio));
  band_w = max(1, min(roi_w, band_w));
  const float center_x = static_cast<float>(roi_w) * 0.5F;
  int band_x0 = static_cast<int>(roundf(center_x - (static_cast<float>(band_w) * 0.5F)));
  int band_x1 = static_cast<int>(roundf(center_x + (static_cast<float>(band_w) * 0.5F)));
  band_x0 = max(0, min(roi_w, band_x0));
  band_x1 = max(band_x0 + 1, min(roi_w, band_x1));
  return y >= band_y0 && y < band_y1 && x >= band_x0 && x < band_x1;
}

__global__ void sample_masked_person_roi_values_kernel(
    const float* __restrict__ depth,
    const float* __restrict__ mask,
    int frame_w,
    int mask_w,
    int mask_h,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float threshold,
    float* __restrict__ out_all,
    float* __restrict__ out_lower,
    float* __restrict__ out_torso) {
  const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= sampled_area) {
    return;
  }

  const int sample_y = idx / sampled_cols;
  const int sample_x = idx - (sample_y * sampled_cols);
  const int local_x = min(sample_x * stride, roi_w - 1);
  const int local_y = min(sample_y * stride, roi_h - 1);
  const int mask_idx = (local_y * mask_w) + local_x;
  const bool active = mask[mask_idx] > threshold;
  const int frame_x = x0 + local_x;
  const int frame_y = y0 + local_y;
  const float value = active
      ? depth[(static_cast<size_t>(frame_y) * static_cast<size_t>(frame_w)) + static_cast<size_t>(frame_x)]
      : CUDART_NAN_F;
  out_all[idx] = value;

  const bool eroded = active && mask_eroded3x3(mask, mask_w, mask_h, local_x, local_y, threshold);
  out_lower[idx] = eroded && in_center_band(local_x, local_y, roi_w, roi_h, 0.88F, 1.0F, 0.35F)
      ? value
      : CUDART_NAN_F;
  out_torso[idx] = eroded && in_center_band(local_x, local_y, roi_w, roi_h, 0.35F, 0.70F, 0.50F)
      ? value
      : CUDART_NAN_F;
}

}  // namespace

extern "C" cudaError_t noesis_sample_roi_values_cuda(
    const float* depth,
    int frame_w,
    int frame_h,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float* out_values,
    float* out_center,
    cudaStream_t stream) {
  if (!depth || !out_values || !out_center) {
    return cudaErrorInvalidValue;
  }
  if (frame_w <= 0 || frame_h <= 0 || roi_w <= 0 || roi_h <= 0 || stride <= 0 ||
      sampled_cols <= 0 || sampled_area <= 0) {
    return cudaErrorInvalidValue;
  }
  if (x0 < 0 || y0 < 0 || x0 + roi_w > frame_w || y0 + roi_h > frame_h) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockSize = 256;
  const int blocks = (sampled_area + kBlockSize - 1) / kBlockSize;
  sample_roi_values_kernel<<<blocks, kBlockSize, 0, stream>>>(
      depth,
      frame_w,
      x0,
      y0,
      roi_w,
      roi_h,
      stride,
      sampled_cols,
      sampled_area,
      out_values);
  sample_roi_center_kernel<<<1, 1, 0, stream>>>(depth, frame_w, x0, y0, roi_w, roi_h, out_center);
  return cudaGetLastError();
}

extern "C" cudaError_t noesis_sample_pose_capsule_values_cuda(
    const float* depth,
    int frame_w,
    int frame_h,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    int capsule_count,
    float c0_ax,
    float c0_ay,
    float c0_bx,
    float c0_by,
    float c0_line_radius,
    float c0_ankle_radius,
    float c1_ax,
    float c1_ay,
    float c1_bx,
    float c1_by,
    float c1_line_radius,
    float c1_ankle_radius,
    float* out_values,
    cudaStream_t stream) {
  if (!depth || !out_values) {
    return cudaErrorInvalidValue;
  }
  if (frame_w <= 0 || frame_h <= 0 || roi_w <= 0 || roi_h <= 0 || stride <= 0 ||
      sampled_cols <= 0 || sampled_area <= 0 || capsule_count <= 0 || capsule_count > 2) {
    return cudaErrorInvalidValue;
  }
  if (x0 < 0 || y0 < 0 || x0 + roi_w > frame_w || y0 + roi_h > frame_h) {
    return cudaErrorInvalidValue;
  }
  constexpr int kBlockSize = 256;
  const int blocks = (sampled_area + kBlockSize - 1) / kBlockSize;
  sample_pose_capsule_values_kernel<<<blocks, kBlockSize, 0, stream>>>(
      depth,
      frame_w,
      x0,
      y0,
      roi_w,
      roi_h,
      stride,
      sampled_cols,
      sampled_area,
      capsule_count,
      c0_ax,
      c0_ay,
      c0_bx,
      c0_by,
      c0_line_radius,
      c0_ankle_radius,
      c1_ax,
      c1_ay,
      c1_bx,
      c1_by,
      c1_line_radius,
      c1_ankle_radius,
      out_values);
  return cudaGetLastError();
}

extern "C" cudaError_t noesis_sample_masked_roi_values_cuda(
    const float* depth,
    const float* mask,
    int frame_w,
    int frame_h,
    int mask_w,
    int mask_h,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float threshold,
    float* out_values,
    float* out_center,
    cudaStream_t stream) {
  if (!depth || !mask || !out_values || !out_center) {
    return cudaErrorInvalidValue;
  }
  if (frame_w <= 0 || frame_h <= 0 || mask_w <= 0 || mask_h <= 0 ||
      roi_w <= 0 || roi_h <= 0 || stride <= 0 || sampled_cols <= 0 || sampled_area <= 0) {
    return cudaErrorInvalidValue;
  }
  if (mask_w != roi_w || mask_h != roi_h) {
    return cudaErrorInvalidValue;
  }
  if (x0 < 0 || y0 < 0 || x0 + roi_w > frame_w || y0 + roi_h > frame_h) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockSize = 256;
  const int blocks = (sampled_area + kBlockSize - 1) / kBlockSize;
  sample_masked_roi_values_kernel<<<blocks, kBlockSize, 0, stream>>>(
      depth,
      mask,
      frame_w,
      mask_w,
      x0,
      y0,
      roi_w,
      roi_h,
      stride,
      sampled_cols,
      sampled_area,
      threshold,
      out_values);
  sample_roi_center_kernel<<<1, 1, 0, stream>>>(depth, frame_w, x0, y0, roi_w, roi_h, out_center);
  return cudaGetLastError();
}

extern "C" cudaError_t noesis_sample_masked_person_roi_values_cuda(
    const float* depth,
    const float* mask,
    int frame_w,
    int frame_h,
    int mask_w,
    int mask_h,
    int x0,
    int y0,
    int roi_w,
    int roi_h,
    int stride,
    int sampled_cols,
    int sampled_area,
    float threshold,
    float* out_all,
    float* out_lower,
    float* out_torso,
    float* out_center,
    cudaStream_t stream) {
  if (!depth || !mask || !out_all || !out_lower || !out_torso || !out_center) {
    return cudaErrorInvalidValue;
  }
  if (frame_w <= 0 || frame_h <= 0 || mask_w <= 0 || mask_h <= 0 ||
      roi_w <= 0 || roi_h <= 0 || stride <= 0 || sampled_cols <= 0 || sampled_area <= 0) {
    return cudaErrorInvalidValue;
  }
  if (mask_w != roi_w || mask_h != roi_h) {
    return cudaErrorInvalidValue;
  }
  if (x0 < 0 || y0 < 0 || x0 + roi_w > frame_w || y0 + roi_h > frame_h) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockSize = 256;
  const int blocks = (sampled_area + kBlockSize - 1) / kBlockSize;
  sample_masked_person_roi_values_kernel<<<blocks, kBlockSize, 0, stream>>>(
      depth,
      mask,
      frame_w,
      mask_w,
      mask_h,
      x0,
      y0,
      roi_w,
      roi_h,
      stride,
      sampled_cols,
      sampled_area,
      threshold,
      out_all,
      out_lower,
      out_torso);
  sample_roi_center_kernel<<<1, 1, 0, stream>>>(depth, frame_w, x0, y0, roi_w, roi_h, out_center);
  return cudaGetLastError();
}
