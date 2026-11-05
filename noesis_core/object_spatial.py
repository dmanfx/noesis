"""Off-callback, original-frame depth samples for selected native objects.

The quantity is the visible surface along a box-center image ray, not an object
centroid, footprint, identity, or physical clearance estimate.
"""
from __future__ import annotations
import math
import numpy as np
from .contracts.object_tracks import ObjectSpatial
from .view_projection import project_camera_point

class ObjectSpatialSampler:
    def __init__(self, depth_store, calibration):
        self.depth_store, self.calibration = depth_store, calibration

    def sample(self, capture, bbox):
        calibration = self.calibration(capture.source_id)
        frame = getattr(calibration, 'frame_contract', None)
        if (calibration is None or calibration.camera_id != capture.camera_id or frame is None or frame.target_frame.frame_id != 'backend_world_m'
            or calibration.unit_scale != 1.0 or not calibration.camera_calibration_sha256
            or not calibration.world_alignment_sha256):
            return None, 'calibration_unavailable'
        # No temporal or camera substitution. The retained native depth storage
        # owns its GPU allocation; no SDK metadata leaves the callback.
        depth, age, _ = self.depth_store.resolve(source_id=capture.source_id,
            frame_id=capture.frame_id, pts_us=capture.temporal[3]//1000,
            max_age_frames=0, wait_ms=0)
        if (depth is None or age or depth.frame_id != capture.frame_id
            or depth.pts_us != capture.temporal[3]//1000
            or depth.source_id != capture.source_id or depth.unit != 'm' or not depth.is_metric
            or (depth.frame_w,depth.frame_h)!=(capture.width,capture.height)):
            return None, 'original_depth_unavailable'
        left,top,width,height=bbox
        u,v=math.floor(left+width/2),math.floor(top+height/2)
        if not 0<=u<capture.width or not 0<=v<capture.height:
            return None, 'original_depth_unavailable'
        K=np.asarray(calibration.intrinsics,dtype=float)
        # Convert the original pipeline pixel to the declared calibration image.
        cw,ch=calibration.image_size
        uv=(u*cw/capture.width,v*ch/capture.height)
        if K.shape!=(3,3) or not np.isfinite(K).all():return None,'calibration_unavailable'
        projection=(calibration.projection_model,calibration.distortion_model,calibration.projection_metadata_status)
        if projection!=('opencv_fisheye','fisheye','declared'):
            return None,'projection_undeclared_or_unsupported'
        D=calibration.distortion_coefficients
        if D is None or len(D)!=4 or not all(math.isfinite(x) for x in D):return None,'calibration_unavailable'
        # Bounded inversion of the declared fisheye polynomial. A fold or a
        # failed forward round-trip is rejected, never a pinhole fallback.
        ray=np.linalg.solve(K,np.array([*uv,1.]))
        xd,yd=ray[0]/ray[2],ray[1]/ray[2];rd=math.hypot(xd,yd)
        def distorted(t):return t*(1+t*t*(D[0]+t*t*(D[1]+t*t*(D[2]+t*t*D[3]))))
        def derivative(t):return 1+3*D[0]*t*t+5*D[1]*t**4+7*D[2]*t**6+9*D[3]*t**8
        upper=math.pi/2-1e-6
        if any(derivative(t)<=0 for t in np.linspace(0,upper,65)) or rd>distorted(upper):return None,'projection_undeclared_or_unsupported'
        lo,hi=0.,upper
        for _ in range(48):
            mid=(lo+hi)/2
            if distorted(mid)<rd:lo=mid
            else:hi=mid
        ratio=math.tan((lo+hi)/2)/rd if rd else 1.
        direction=np.array([xd*ratio,yd*ratio,1.])
        projected=project_camera_point(direction,K,calibration.image_size,calibration)
        if projected.state!='in' or projected.pixel is None or np.linalg.norm(np.array(projected.pixel)-uv)>1e-5:
            return None,'projection_undeclared_or_unsupported'
        if depth.depth_device_frame is not None:
            # One pixel, at most four selected objects per capture, at most once
            # per second per native key. Never full-frame D2H or model inference.
            stats=depth.depth_device_frame.sample_roi_stats(u,v,1,1,128)
            z=stats.get('depth_center')
        elif depth.depth_map is not None and depth.valid_mask is not None and depth.valid_mask[v,u]:
            z=float(depth.depth_map[v,u])
        else:return None,'original_depth_unavailable'
        if type(z) not in (int,float) or not math.isfinite(z) or not .05<=z<=100:
            return None,'depth_sample_invalid'
        E=np.asarray(calibration.world_extrinsics_col_major,dtype=float).reshape(4,4,order='F')
        position=(np.linalg.inv(E)@np.r_[direction*z,1.])[:3]
        if not np.isfinite(position).all() or np.max(np.abs(position))>1000:return None,'depth_sample_invalid'
        current=self.calibration(capture.source_id)
        if current is not calibration and (current is None or current.frame_contract!=frame
            or current.camera_calibration_sha256!=calibration.camera_calibration_sha256
            or current.world_alignment_sha256!=calibration.world_alignment_sha256):return None,'calibration_replaced'
        return ObjectSpatial(frame_id=capture.frame_id,observed_at_us=capture.temporal[0],media_pts_ns=capture.temporal[3],position_m=tuple(position),pixel=(u,v),depth_m=z,
            world_frame_revision=frame.target_frame.revision,frame_transform_sha256=frame.transform_sha256,
            camera_calibration_sha256=calibration.camera_calibration_sha256,
            world_alignment_sha256=calibration.world_alignment_sha256,model=depth.model_name), 'measured_surface'
