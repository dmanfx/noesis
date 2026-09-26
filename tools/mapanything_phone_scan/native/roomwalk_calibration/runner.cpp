// CPU-only ChArUco observation adapter. Basalt remains the optimization engine.
#include <basalt/optimization/spline_optimize.h>
#include <nlohmann/json.hpp>
#include <tbb/global_control.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

using json = nlohmann::json;
using Opt = basalt::SplineOptimization<5, double>;
using Clock = std::chrono::steady_clock;
constexpr const char* SOURCE = "6d8637b9d68ea18a1a63c1baa72818779e156932";

static void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
static double number(const json& x) {
  require(x.is_number() && !x.is_boolean(), "expected numeric value");
  double v = x.get<double>();
  require(std::isfinite(v), "nonfinite numeric value");
  return v;
}
static int64_t stamp(const json& x) {
  require(x.is_number_integer() && !x.is_boolean(), "timestamp must be integer ns");
  int64_t t = x.get<int64_t>();
  require(t >= 0 && t < INT64_MAX / 2, "timestamp outside supported range");
  return t;
}
template<int R, int C> static Eigen::Matrix<double,R,C> matrix(const json& a) {
  require(a.is_array() && a.size() == R, "matrix row count");
  Eigen::Matrix<double,R,C> m;
  for (int r=0;r<R;++r) {
    require(a[r].is_array() && a[r].size()==C, "matrix column count");
    for(int c=0;c<C;++c) m(r,c)=number(a[r][c]);
  }
  return m;
}
static Eigen::Vector3d vec3(const json& a) {
  require(a.is_array() && a.size()==3, "vector size");
  return {number(a[0]),number(a[1]),number(a[2])};
}
static Sophus::SE3d transform(const json& a) {
  auto m=matrix<4,4>(a);
  auto R=m.block<3,3>(0,0).eval();
  require((R.transpose()*R-Eigen::Matrix3d::Identity()).norm()<1e-6 &&
      std::abs(R.determinant()-1)<1e-6 &&
      (m.row(3)-Eigen::RowVector4d(0,0,0,1)).norm()<1e-9, "invalid rigid transform");
  return {R,m.block<3,1>(0,3)};
}
template<class Derived> static json array(const Eigen::MatrixBase<Derived>& a) {
  json rows=json::array();
  for(int r=0;r<a.rows();++r) {
    json row=json::array();
    for(int c=0;c<a.cols();++c) {
      require(std::isfinite(a(r,c)), "nonfinite solver output");
      row.push_back(a(r,c));
    }
    rows.push_back(row);
  }
  return rows;
}
static json vector(const Eigen::Vector3d& v) {
  require(v.allFinite(), "nonfinite vector output");
  return {v[0],v[1],v[2]};
}

static json run(const json& in) {
  require(in.at("schema")=="roomwalk.basalt_input.v1", "wrong input schema");
  require(in.at("target_type")=="charuco", "target must be actual ChArUco observations");
  const bool visual_only=in.value("visual_only",false);
  int iterations=in.value("max_iterations",60);
  double timeout=number(in.at("timeout_s"));
  int64_t dt=stamp(in.at("knot_spacing_ns"));
  require(iterations>=1 && iterations<=150 && timeout>=1 && timeout<=1800,
      "invalid computational limits");
  require(dt>=20000000 && dt<=100000000, "knot spacing outside bounds");
  const auto start=Clock::now();
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism,2);
  Opt opt(dt);
  opt.resetCalib(1,{"pinhole"});
  opt.calib->intrinsics[0]=basalt::GenericCamera<double>::fromString("pinhole");
  auto k=matrix<3,3>(in.at("K"));
  require(k(0,0)>0 && k(1,1)>0 && std::abs(k(0,1))<1e-12, "invalid pinhole K");
  opt.calib->intrinsics[0].setFromInit(Eigen::Vector4d(k(0,0),k(1,1),k(0,2),k(1,2)));
  opt.calib->resolution.emplace_back(in.at("resolution")[0].get<int>(),in.at("resolution")[1].get<int>());
  opt.calib->T_i_c[0]=transform(in.at("initial_T_imu_camera"));
  opt.calib->cam_time_offset_ns=in.at("initial_cam_time_offset_ns").get<int64_t>();
  require(std::abs(opt.calib->cam_time_offset_ns)<=100000000, "seed offset exceeds 100ms");
  require(in.at("weights").at("provenance").is_string(), "weights require provenance");
  double gyro_sigma=number(in.at("weights").at("gyro_sample_sigma_rad_s"));
  double accel_sigma=number(in.at("weights").at("accel_sample_sigma_m_s2"));
  require(gyro_sigma>0 && accel_sigma>0, "nonpositive residual weights");
  // Use an explicitly declared one-Hz normalization so Basalt's conversion
  // yields exactly the requested discrete residual weights, independently of
  // the TWO actual acquisition rates. This is NOT an IMU noise measurement.
  opt.calib->imu_update_rate=1;
  opt.calib->gyro_noise_std.setConstant(gyro_sigma);
  opt.calib->accel_noise_std.setConstant(accel_sigma);
  opt.calib->gyro_bias_std.setZero();
  opt.calib->accel_bias_std.setZero();
  Eigen::aligned_vector<Eigen::Vector4d> objects;
  require(in.at("object_points").size()>=12 && in.at("object_points").size()<=1000,"object point count");
  for(const auto& p:in.at("object_points")) {
    Eigen::Vector3d v=vec3(p);
    require(v.norm()<10, "target outside size limit");
    objects.emplace_back(v[0],v[1],v[2],1);
  }
  opt.setAprilgridCorners3d(objects);
  int64_t lower=INT64_MAX,upper=0;
  for(const auto& name:{"accel","gyro"}) {
    if(visual_only) continue;
    const auto& rows=in.at(name);
    require(rows.size()>=100 && rows.size()<=200000,"IMU row count outside bounds");
    int64_t previous=-1;
    for(const auto& row:rows) {
      int64_t t=stamp(row.at("timestamp_ns"));
      require(t>previous,"IMU times not strictly increasing");
      if(previous>=0) require(t-previous<=100000000,"IMU gap exceeds 100ms");
      previous=t;
      auto v=vec3(row.at("xyz"));
      require(v.norm()<1000,"IMU magnitude exceeds input bound");
      if(std::string(name)=="accel") opt.addAccelMeasurement(t,v);
      else opt.addGyroMeasurement(t,v);
    }
    lower=std::min(lower,stamp(rows.front().at("timestamp_ns")));
    upper=std::max(upper,stamp(rows.back().at("timestamp_ns")));
  }
  const auto& frames=in.at("frames");
  require(frames.size()>=12 && frames.size()<=1500,"camera observation count");
  if(visual_only) {
    require(!in.contains("accel") && !in.contains("gyro"),"visual holdout cannot consume IMU");
    lower=stamp(frames.front().at("timestamp_ns"))-200000000;
    upper=stamp(frames.back().at("timestamp_ns"))+200000000;
    // Endpoint padding is initialization-only, like the per-frame PnP poses;
    // all pose residuals are removed in the final visual-only corner phase.
    opt.addPoseMeasurement(lower,transform(frames.front().at("T_target_camera")));
    opt.addPoseMeasurement(upper,transform(frames.back().at("T_target_camera")));
  }
  require(upper-lower<=120000000000LL && (upper-lower)/dt<=4000,"solver span/knot limit");
  int64_t previous=-1;
  for(const auto& f:frames) {
    int64_t t=stamp(f.at("timestamp_ns"));
    require(t>previous && t>lower+110000000 && t<upper-110000000,
        "camera timestamps require ordered 110ms IMU brackets");
    previous=t;
    const auto& ids=f.at("ids"); const auto& px=f.at("pixels");
    require(ids.size()>=12 && ids.size()==px.size() && ids.size()<=objects.size(),"corner count mismatch");
    require(f.contains("corner_timestamps_ns") && f.at("corner_timestamps_ns").size()==ids.size(),"explicit corner timestamps required");
    Eigen::aligned_vector<Eigen::Vector2d> positions;
    std::vector<int> indices;
    std::set<int> unique;
    for(size_t j=0;j<ids.size();++j) {
      require(ids[j].is_number_integer(),"corner ID must be integer");
      int id=ids[j].get<int>();
      require(id>=0 && id<int(objects.size()) && unique.insert(id).second,"invalid/duplicate corner ID");
      require(px[j].is_array() && px[j].size()==2,"pixel shape");
      positions.emplace_back(number(px[j][0]),number(px[j][1])); indices.push_back(id);
      int64_t ct=stamp(f.at("corner_timestamps_ns")[j]);
      require(std::abs(ct-t)<=100000000 && ct>lower+100000000 && ct<upper-100000000,"corner timestamp outside frame/IMU bracket");
      // A one-corner measurement evaluates the actual upstream spline at that
      // point's exposure midpoint, including the optimized camera/IMU offset.
      Eigen::aligned_vector<Eigen::Vector2d> one{positions.back()};
      opt.addAprilgridMeasurement(ct,0,one,std::vector<int>{id});
    }
    opt.addPoseMeasurement(t+opt.calib->cam_time_offset_ns,
        transform(f.at("T_target_camera"))*opt.calib->T_i_c[0].inverse());
  }
  opt.setG(vec3(in.at("initial_gravity_target")));
  require(opt.getG().norm()>5 && opt.getG().norm()<15,"invalid gravity seed");
  opt.init();
  json trace=json::array();
  bool converged=false;
  // Bootstrap the spline using visual poses, then remove those correlated
  // pseudo-observations: final estimation uses corner and raw IMU residuals.
  const bool refine_scale=in.value("refine_imu_scale",true);
  for(int phase=0;phase<(visual_only?2:3);++phase) {
    converged=false;
    Eigen::aligned_vector<Eigen::Vector2d> previous_pixels;
    double previous_energy=std::numeric_limits<double>::infinity();
    int stable_visual_steps=0;
    for(int i=0;i<iterations;++i) {
      require(std::chrono::duration<double>(Clock::now()-start).count()<timeout,"solver deadline exceeded");
      double energy=0,reproj=0; int count=0;
      converged=opt.optimize(false,phase==0,phase!=0,phase==2,
          phase==2 && refine_scale,false,4.0,1e-7,energy,count,reproj,false);
      require(std::isfinite(energy) && std::isfinite(reproj) && count>=0,"nonfinite objective");
      require(opt.calib->T_i_c[0].matrix().allFinite() && opt.getG().allFinite(),"nonfinite state");
      require(opt.calib->T_i_c[0].translation().norm()<=0.5 &&
          std::abs(opt.calib->cam_time_offset_ns)<=100000000,"calibration exceeded physical experiment bounds");
      double observable_change=0;
      if(visual_only && phase==1) {
        // A visual-only camera trajectory cannot identify the arbitrary split
        // into spline body frame and T_i_c. Test convergence of OBSERVABLE
        // row-time image predictions, not motion in this internal gauge.
        Eigen::aligned_vector<Eigen::Vector2d> current_pixels;
        for(const auto& f:frames) for(size_t j=0;j<f.at("ids").size();++j) {
          int64_t ct=stamp(f.at("corner_timestamps_ns")[j]);
          auto pose=opt.getT_w_i(ct)*opt.calib->T_i_c[0];
          Eigen::Vector3d p=pose.inverse()*objects[f.at("ids")[j].get<int>()].head<3>();
          require(p.allFinite() && p.z()>0,"visual prediction has invalid depth");
          current_pixels.emplace_back(k(0,0)*p.x()/p.z()+k(0,2),k(1,1)*p.y()/p.z()+k(1,2));
          size_t index=current_pixels.size()-1;
          if(previous_pixels.size()>index) observable_change=std::max(observable_change,(current_pixels.back()-previous_pixels[index]).norm());
        }
        bool stable=!previous_pixels.empty() && observable_change<1e-5 && std::abs(energy-previous_energy)/std::max(1,count)<1e-12;
        stable_visual_steps=stable?stable_visual_steps+1:0;
        if(stable_visual_steps>=3) converged=true;
        previous_pixels=std::move(current_pixels); previous_energy=energy;
      }
      trace.push_back({{"phase",phase},{"iteration",i},{"energy",energy},
          {"reprojection_sum_px",reproj},{"point_count",count},{"converged",converged},
          {"observable_pixel_change",observable_change},{"stable_visual_steps",stable_visual_steps}});
      std::cerr<<"phase="<<phase<<" iteration="<<i<<" energy="<<energy<<"\n";
      if(converged) break;
    }
  }
  Eigen::Vector3d ab,gb; Eigen::Matrix3d as,gs;
  opt.getAccelBias().getBiasAndScale(ab,as);
  opt.getGyroBias().getBiasAndScale(gb,gs);
  json predictions=json::array();
  json corner_predictions=json::array();
  for(const auto& f:frames) {
    int64_t t=stamp(f.at("timestamp_ns"))+opt.getCamTimeOffsetNs();
    auto tc=opt.getT_w_i(t)*opt.calib->T_i_c[0];
    predictions.push_back({{"timestamp_ns",f.at("timestamp_ns")},{"T_target_camera",array(tc.matrix())}});
    json pixels=json::array();
    for(size_t j=0;j<f.at("ids").size();++j) {
      int64_t ct=stamp(f.at("corner_timestamps_ns")[j])+opt.getCamTimeOffsetNs();
      auto pose=opt.getT_w_i(ct)*opt.calib->T_i_c[0];
      Eigen::Vector3d p=pose.inverse()*objects[f.at("ids")[j].get<int>()].head<3>();
      require(p.allFinite() && p.z()>0,"predicted target is behind camera");
      pixels.push_back({k(0,0)*p.x()/p.z()+k(0,2),k(1,1)*p.y()/p.z()+k(1,2)});
    }
    corner_predictions.push_back({{"frame_index",f.at("frame_index")},{"pixels",pixels}});
  }
  json trajectory=json::array();
  if(visual_only) {
    for(int64_t t=stamp(frames.front().at("timestamp_ns"));t<=stamp(frames.back().at("timestamp_ns"));t+=5000000) {
      auto tc=opt.getT_w_i(t)*opt.calib->T_i_c[0];
      trajectory.push_back({{"timestamp_ns",t},{"T_target_camera",array(tc.matrix())}});
    }
  }
  return {{"schema","roomwalk.basalt_result.v1"},{"basalt_commit",SOURCE},
      {"target_type","charuco"},{"target_geometry_unchanged",true},
      {"status",converged?"converged":"iteration_limit"},{"trace",trace},
      {"T_imu_camera",array(opt.calib->T_i_c[0].matrix())},
      {"cam_time_offset_ns",opt.getCamTimeOffsetNs()},
      {"imu_to_camera_offset_ns",-opt.getCamTimeOffsetNs()},
      {"gravity_target",vector(opt.getG())},{"predictions",predictions},
      {"corner_predictions",corner_predictions},
      {"visual_only",visual_only},{"visual_trajectory",trajectory},
      {"convergence_criterion",visual_only?"upstream_state_step_or_three_observable_pixel_steps_below_1e-5_and_energy_per_point_change_below_1e-12":"upstream_state_step_below_1e-7"},
      {"point_timing_model",in.at("point_timing_model")},
      {"point_specific_spline_evaluation",true},
      {"imu_corrections",{{"equation","corrected = matrix * raw - bias"},
          {"accelerometer_matrix",array(Eigen::Matrix3d::Identity()+as)},
          {"gyroscope_matrix",array(Eigen::Matrix3d::Identity()+gs)},
          {"accelerometer_bias",vector(ab)},{"gyroscope_bias",vector(gb)},
          {"accelerometer_parameters",array(opt.getAccelBias().getParam())},
          {"gyroscope_parameters",array(opt.getGyroBias().getParam())}}},
      {"fixed_camera",true},{"imu_scale_refined",refine_scale},
      {"weights",in.at("weights")},{"weight_normalization_rate_hz",1},
      {"covariance",nullptr},{"covariance_status","not_estimated"},
      {"imu_noise_calibrated",false},{"accepted_for_metric_vio",false}};
}

int main(int argc,char** argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--version") {
      std::cout<<"roomwalk.basalt_adapter.v2 "<<SOURCE<<"\n"; return 0;
    }
    require(argc==3,"usage: roomwalk_calibrate_imu input.json output.json");
    require(std::filesystem::file_size(argv[1])<=64*1024*1024,"input exceeds 64MiB");
    require(!std::filesystem::exists(argv[2]),"refusing to overwrite result");
    std::ifstream input(argv[1]); json in; input>>in;
    json out=run(in);
    std::ofstream output(argv[2]); output<<out.dump(2)<<"\n";
    require(output.good(),"could not write result");
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 2; }
}
