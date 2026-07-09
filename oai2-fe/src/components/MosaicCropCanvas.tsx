import React, { useEffect, useRef } from 'react';
import { CameraKey, cameraLabel } from '../lib/camera';
import { containRect, resolveMosaicTileCrop, type MosaicLayoutLike } from '../lib/mosaic';

type MosaicCropCanvasProps = {
  camera: CameraKey;
  mosaicLayout?: MosaicLayoutLike | null;
  videoRef?: React.RefObject<HTMLVideoElement>;
  className?: string;
};

export const MosaicCropCanvas: React.FC<MosaicCropCanvasProps> = ({
  camera,
  mosaicLayout,
  videoRef,
  className,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef?.current;
    if (!canvas || !video) return;

    let frameId = 0;
    let stopped = false;

    const draw = () => {
      if (stopped) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const targetW = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const targetH = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      if (ctx) {
        ctx.fillStyle = '#020509';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const videoWidth = video.videoWidth || 0;
        const videoHeight = video.videoHeight || 0;
        const crop = resolveMosaicTileCrop(mosaicLayout, camera, videoWidth, videoHeight);
        if (crop && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
          const dest = containRect(canvas.width, canvas.height, crop.sw, crop.sh);
          ctx.imageSmoothingEnabled = true;
          ctx.drawImage(
            video,
            crop.sx,
            crop.sy,
            crop.sw,
            crop.sh,
            dest.x,
            dest.y,
            dest.w,
            dest.h
          );
        }
      }

      frameId = window.requestAnimationFrame(draw);
    };

    draw();
    return () => {
      stopped = true;
      if (frameId) window.cancelAnimationFrame(frameId);
    };
  }, [camera, mosaicLayout, videoRef]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      aria-label={`${cameraLabel(camera)} video crop`}
    />
  );
};

export default MosaicCropCanvas;
