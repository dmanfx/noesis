"""Bounded forward model math; never optical visibility or rectified-image inference.

The declared OpenCV fisheye image domain uses its supplied K and ordered D in
that calibration's distorted pixel plane. No image is acquired or rectified.
The formula is checked against installed cv2.fisheye.projectPoints.
"""
from __future__ import annotations
import math
import numpy as np
from .contracts.view_coverage import ViewProjection


def project_camera_point(camera_point, K, image_size, metadata):
    basis,domain='unknown','unknown'
    depth=None
    def result(state,reason,pixel=None,applied=False):
        return ViewProjection(basis=basis,image_domain=domain,distortion_applied=applied,
            state=state,reason=reason,pixel=pixel,camera_depth_m=depth,optical_visibility='unknown')
    point=np.asarray(camera_point,dtype=np.float64)
    if point.shape!=(3,) or not np.isfinite(point).all(): return result('unknown','nonfinite_projection')
    depth=float(point[2])
    status=getattr(metadata,'projection_metadata_status','undeclared')
    model=getattr(metadata,'projection_model',None)
    distortion=getattr(metadata,'distortion_model',None)
    coefficients=getattr(metadata,'distortion_coefficients',None)
    names=getattr(metadata,'distortion_coefficient_names',None)
    if status=='undeclared' and all(value is None for value in (model,distortion,coefficients,names)):
        basis,domain='mathematical_linear_k_only','linear_k_image_plane'
    elif status=='declared' and model=='rectified_pinhole' and distortion=='none':
        if (not isinstance(coefficients,(tuple,list)) or len(coefficients)!=3
            or any(type(value) not in (int,float) or value!=0 for value in coefficients)
            or names is not None and tuple(names)!=('k1','k2','k3')):
            return result('unknown','invalid_projection_metadata')
        basis,domain='mathematical_linear_k_only','linear_k_image_plane'
    elif status!='declared': return result('unknown','invalid_projection_metadata')
    elif model!='opencv_fisheye' or distortion!='fisheye': return result('unknown','unsupported_projection_model')
    elif (not isinstance(coefficients,(tuple,list)) or len(coefficients)!=4
        or any(type(value) not in (int,float) or not math.isfinite(value) for value in coefficients)):
        return result('unknown','invalid_projection_metadata')
    elif names is not None and tuple(names)!=('k1','k2','k3','k4'):
        return result('unknown','unsupported_coefficient_order')
    else: basis,domain='declared_opencv_fisheye','declared_distorted_calibration_image'
    matrix=np.asarray(K,dtype=np.float64)
    if (matrix.shape!=(3,3) or not np.isfinite(matrix).all() or matrix[0,0]<=0 or matrix[1,1]<=0
        or not np.array_equal(matrix[2],[0.,0.,1.]) or len(image_size)!=2
        or any(type(value) is not int or not 1<=value<=32768 for value in image_size)
        or basis=='declared_opencv_fisheye' and matrix[1,0]!=0):
        basis,domain='unknown','unknown'
        return result('unknown','unsupported_projection_domain')
    if depth<=0: return result('out','behind_camera')
    try:
        x,y=float(point[0])/depth,float(point[1])/depth
        applied=False
        if basis=='declared_opencv_fisheye':
            radius=math.hypot(x,y);theta=math.atan(radius);theta2=theta*theta
            k1,k2,k3,k4=coefficients
            distorted=theta*(1+theta2*(k1+theta2*(k2+theta2*(k3+theta2*k4))))
            ratio=distorted/radius if radius else 1.
            x,y=x*ratio,y*ratio
            u=matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]
            v=matrix[1,1]*y+matrix[1,2]
            applied=True
        else:
            image=matrix@np.array([x,y,1.]);u,v=image[0],image[1]
        if not math.isfinite(u) or not math.isfinite(v): return result('unknown','nonfinite_projection')
        inside=0<=u<image_size[0] and 0<=v<image_size[1]
        return result('in' if inside else 'out','inside_image' if inside else 'outside_image',(float(u),float(v)),applied)
    except (ArithmeticError,ValueError): return result('unknown','nonfinite_projection')
