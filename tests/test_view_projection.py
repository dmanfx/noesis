"""Independent installed OpenCV CPU oracle for the declared lens projection."""
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from noesis_core.view_projection import project_camera_point

K=np.array([[400.,3.,320.],[0.,410.,240.],[0.,0.,1.]])
SIZE=(640,480)
def metadata(**changes):
    return SimpleNamespace(**{'projection_metadata_status':'declared','projection_model':'opencv_fisheye',
        'distortion_model':'fisheye','distortion_coefficients':(-.041323572296878525,-.07257812805006623,.06665786839070417,-.02667936264765735),
        'distortion_coefficient_names':None,**changes})

@pytest.mark.parametrize('coefficients',[(0.,0.,0.,0.),(-.041323572296878525,-.07257812805006623,.06665786839070417,-.02667936264765735)])
def test_fisheye_forward_pixels_match_installed_opencv_zero_and_nonzero(coefficients):
    source=metadata(distortion_coefficients=coefficients)
    points=np.array([[[0.,0.,2.],[.4,-.2,2.],[10.,1.,1.]]],dtype=np.float64)
    expected,_=cv2.fisheye.projectPoints(points,np.zeros(3),np.zeros(3),K,np.asarray(coefficients),alpha=K[0,1]/K[0,0])
    for point,oracle in zip(points[0],expected[0]):
        row=project_camera_point(point,K,SIZE,source)
        np.testing.assert_allclose(row.pixel,oracle,atol=1e-10,rtol=0)
        assert row.basis=='declared_opencv_fisheye' and row.image_domain=='declared_distorted_calibration_image'
        assert row.distortion_applied is True and row.optical_visibility=='unknown'
        assert row.camera_depth_m==point[2]
    assert project_camera_point(points[0,-1],K,SIZE,source).state=='out'
    named=metadata(distortion_coefficients=coefficients,distortion_coefficient_names=('k1','k2','k3','k4'))
    assert project_camera_point(points[0,1],K,SIZE,named)==project_camera_point(points[0,1],K,SIZE,source)


def test_behind_nonfinite_invalid_unsupported_order_domain_and_undeclared_are_explicit():
    row=project_camera_point([0.,1.,-1.],K,SIZE,metadata())
    assert row.state=='out' and row.reason=='behind_camera' and row.pixel is None and row.distortion_applied is False
    assert row.basis=='declared_opencv_fisheye' and row.camera_depth_m==-1.
    row=project_camera_point([float('nan'),0.,1.],K,SIZE,metadata())
    assert row.reason=='nonfinite_projection' and row.camera_depth_m is None and row.basis=='unknown'
    for changes,reason in (({'projection_model':'pinhole'},'unsupported_projection_model'),
            ({'distortion_coefficients':(0.,0.,0.)},'invalid_projection_metadata'),
            ({'distortion_coefficients':(0.,float('inf'),0.,0.)},'invalid_projection_metadata'),
            ({'distortion_coefficient_names':('k2','k1','k3','k4')},'unsupported_coefficient_order'),
            ({'projection_metadata_status':'invalid'},'invalid_projection_metadata')):
        row=project_camera_point([.4,.2,1.],K,SIZE,metadata(**changes))
        assert row.state=='unknown' and row.reason==reason and row.pixel is None
        assert row.basis==row.image_domain=='unknown' and row.distortion_applied is False
    bad=K.copy();bad[1,0]=1.
    assert project_camera_point([.4,.2,1.],bad,SIZE,metadata()).reason=='unsupported_projection_domain'
    undeclared=metadata(projection_metadata_status='undeclared',projection_model=None,distortion_model=None,distortion_coefficients=None)
    row=project_camera_point([.4,.2,1.],K,SIZE,undeclared)
    assert row.basis=='mathematical_linear_k_only' and row.image_domain=='linear_k_image_plane' and row.distortion_applied is False
    np.testing.assert_allclose(row.pixel,(K@np.array([.4,.2,1.]))[:2])


def test_explicit_rectified_pinhole_uses_its_linear_intrinsics_without_raw_lens_distortion():
    source=metadata(projection_model='rectified_pinhole',distortion_model='none',
        distortion_coefficients=(0.,0.,0.),distortion_coefficient_names=('k1','k2','k3'))
    row=project_camera_point([.4,.2,1.],K,SIZE,source)
    assert row.basis=='mathematical_linear_k_only' and row.state=='in'
    np.testing.assert_allclose(row.pixel,(K@np.array([.4,.2,1.]))[:2])
    assert row.distortion_applied is False and row.optical_visibility=='unknown'
    source.distortion_coefficients=(.1,0.,0.)
    assert project_camera_point([.4,.2,1.],K,SIZE,source).reason=='invalid_projection_metadata'
