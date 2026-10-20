#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <array>
#include <vector>
#include <map>
#include <stdexcept>
#include <charconv>
#include <string>
#include <sstream>
#include <iomanip>
#include <locale>

// Kept in this attested translation unit. The CPU fixture includes the SAME
// helper under a private compile-only macro; production has no alternate path.
namespace noesis {
namespace pose {

struct CropGeometry {
  double original_left, original_top, original_width, original_height;
  std::uint32_t model_width, model_height;
  std::uint32_t source_left, source_top, source_width, source_height;
  std::uint32_t destination_width, destination_height, offset_left, offset_top;
  double ratio_x, ratio_y;
};

struct Keypoint {
  double roi_x, roi_y, absolute_x, absolute_y;
  float confidence;
};

struct ContextCandidateDiagnostic {
  int row_index=-1;
  double score=0, class_id=0;
  std::optional<double> iou;
  std::optional<std::array<double,4>> bbox_xyxy;
  std::array<std::optional<std::array<double,3>>,4> torso_unclipped;
};

// Optional fixed-size same-pass observer; no tensor retention or matching input.
struct PoseExtractionDiagnostic {
  unsigned tensor_count=0;
  bool context_present=false, context_valid=false, shape_valid=false;
  int rows=0, cols=0, qualifying_rows_seen=-1;
  unsigned finite_rows_seen=0,padding_rows_seen=0,score_pass_rows=0,class_pass_rows=0,
    bbox_pass_rows=0,center_pass_rows=0,iou_pass_rows=0;
  std::optional<double> max_score,max_iou;
  unsigned pre_iou_candidates_seen=0;
  bool pre_iou_scan_complete=false;
  std::optional<int> pre_iou_invalid_row_index;
  std::optional<std::array<double,4>> pre_iou_parent_bbox_xywh;
  std::array<std::optional<ContextCandidateDiagnostic>,2> pre_iou_candidates;
  const char* result="tensor_missing";
};

inline std::optional<CropGeometry> make_crop_geometry(
    double left, double top, double width, double height,
    int model_width, int model_height, bool letterbox) {
  // Validate BEFORE unsigned casts, even-rounding additions or SDK-style
  // products. Malformed rectangles must not invoke undefined conversions.
  constexpr auto umax = std::numeric_limits<std::uint32_t>::max();
  if (!std::isfinite(left) || !std::isfinite(top) ||
      !std::isfinite(width) || !std::isfinite(height) ||
      left < 0.0 || top < 0.0 || left >= static_cast<double>(umax) ||
      top >= static_cast<double>(umax) || width <= 0.0 || height <= 0.0 ||
      width > static_cast<double>(umax) || height > static_cast<double>(umax) ||
      model_width <= 0 || model_height <= 0) {
    return std::nullopt;
  }
  const auto source_left = (static_cast<std::uint32_t>(left) + 1U) & ~1U;
  const auto source_top = (static_cast<std::uint32_t>(top) + 1U) & ~1U;
  const auto source_width = static_cast<std::uint32_t>(width) & ~1U;
  const auto source_height = static_cast<std::uint32_t>(height) & ~1U;
  if (source_width == 0U || source_height == 0U ||
      static_cast<std::uint64_t>(source_left) + source_width > umax ||
      static_cast<std::uint64_t>(source_top) + source_height > umax) {
    return std::nullopt;
  }
  const auto mw = static_cast<std::uint32_t>(model_width);
  const auto mh = static_cast<std::uint32_t>(model_height);
  std::uint32_t destination_width = mw, destination_height = mh;
  std::uint32_t offset_left = 0U, offset_top = 0U;
  if (letterbox) {
    // gstnvinfer.cpp get_converted_buffer computes both products as guint.
    // Reject overflowing inputs rather than emulating wrapped SDK arithmetic.
    const auto height_product = static_cast<std::uint64_t>(mw) * source_height;
    const auto width_product = static_cast<std::uint64_t>(mh) * source_width;
    if (height_product > umax || width_product > umax) return std::nullopt;
    const double hdest = static_cast<double>(height_product) / source_width;
    const double wdest = static_cast<double>(width_product) / source_height;
    if (hdest <= mh) {
      destination_height = static_cast<std::uint32_t>(hdest);
    } else {
      destination_width = static_cast<std::uint32_t>(wdest);
    }
    offset_left = (mw - destination_width) / 2U;
    offset_top = (mh - destination_height) / 2U;
  }
  if (destination_width == 0U || destination_height == 0U) return std::nullopt;
  // Exact installed DS9.1 integer source/destination rectangles and symmetric
  // offsets; a truncated nonlimiting destination makes these ratios DIFFERENT.
  return CropGeometry{left, top, width, height, mw, mh, source_left, source_top,
                      source_width, source_height, destination_width,
                      destination_height, offset_left, offset_top,
                      static_cast<double>(destination_width) / source_width,
                      static_cast<double>(destination_height) / source_height};
}

inline std::optional<Keypoint> map_keypoint(
    const CropGeometry& crop, double x, double y, float confidence,
    bool normalized) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(confidence)) {
    return std::nullopt;
  }
  if (normalized) {
    x *= crop.model_width;
    y *= crop.model_height;
  }
  if (!std::isfinite(x) || !std::isfinite(y)) return std::nullopt;
  const double absolute_x = crop.source_left + (x - crop.offset_left) / crop.ratio_x;
  const double absolute_y = crop.source_top + (y - crop.offset_top) / crop.ratio_y;
  if (!std::isfinite(absolute_x) || !std::isfinite(absolute_y)) return std::nullopt;
  // ROI is ORIGINAL object-box relative, not even-crop relative. Existing
  // cache/telemetry consumers depend on this convention and original-box clip.
  const double roi_x = std::clamp(absolute_x - crop.original_left,
                                  0.0, crop.original_width);
  const double roi_y = std::clamp(absolute_y - crop.original_top,
                                  0.0, crop.original_height);
  return Keypoint{roi_x, roi_y, crop.original_left + roi_x,
                  crop.original_top + roi_y, confidence};
}

inline std::optional<CropGeometry> context_crop(double x, double y, double w, double h,
                                               unsigned iw, unsigned ih) {
  if (!std::isfinite(x+y+w+h) || w < 16 || h < 16 || std::min(w,h) >= 128 || w/h > .65 ||
      x < 0 || y < 0 || x+w > iw || y+h > ih) return std::nullopt;
  const double l=std::max(0.,x-.30*w), t=std::max(0.,y-.30*h);
  auto c=make_crop_geometry(l,t,std::min(double(iw),x+1.30*w)-l,
                            std::min(double(ih),y+1.30*h)-t,640,640,true);
  if (!c || c->source_left+c->source_width>iw || c->source_top+c->source_height>ih)
    return std::nullopt;
  c->original_left=x; c->original_top=y; c->original_width=w; c->original_height=h;
  return c;
}

inline void observe_context_candidate(const CropGeometry& c,const std::array<float,57>& r,
    int row_index,const std::array<double,4>& bbox,std::optional<double> iou,
    PoseExtractionDiagnostic& diagnostic) {
  const auto index=diagnostic.pre_iou_candidates_seen++;
  if(index>=diagnostic.pre_iou_candidates.size()) return;
  ContextCandidateDiagnostic value;value.row_index=row_index;value.score=r[4];value.class_id=r[5];
  if(iou && std::isfinite(*iou)) value.iou=iou;
  if(std::all_of(bbox.begin(),bbox.end(),[](double v){return std::isfinite(v);})) value.bbox_xyxy=bbox;
  constexpr std::array<int,4> joints{5,6,11,12};
  for(std::size_t i=0;i<joints.size();++i) {
    const auto base=6+3*joints[i];
    const std::array<double,3> point{c.source_left+(r[base]-c.offset_left)/c.ratio_x,
        c.source_top+(r[base+1]-c.offset_top)/c.ratio_y,r[base+2]};
    if(std::all_of(point.begin(),point.end(),[](double v){return std::isfinite(v);}))
      value.torso_unclipped[i]=point;  // Deliberately BEFORE original-box clipping.
  }
  diagnostic.pre_iou_candidates[index]=value;
}

inline bool strict_context_torso_contained(const CropGeometry& c,const std::array<float,57>& r,
    double l,double t,double a,double b) {
  if (!std::isfinite(l+t+a+b) || l<c.original_left || t<c.original_top ||
      a>c.original_left+c.original_width || b>c.original_top+c.original_height) return false;
  // Existing body/anatomy confidence floor, not a lower score/IoU threshold.
  for(int joint:{5,6,11,12}) {
    const auto base=6+3*joint;
    const double x=c.source_left+(r[base]-c.offset_left)/c.ratio_x;
    const double y=c.source_top+(r[base+1]-c.offset_top)/c.ratio_y;
    const float confidence=r[base+2];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(confidence) ||
        confidence<.35f || confidence>1.f || x<l || x>a || y<t || y>b ||
        x<c.original_left || x>c.original_left+c.original_width ||
        y<c.original_top || y>c.original_top+c.original_height) return false;
  }
  return true;
}

inline bool context_row_matches(const CropGeometry& c, const std::array<float,57>& r,
                                PoseExtractionDiagnostic* diagnostic=nullptr,int row_index=-1,
                                bool* strict_contained=nullptr) {
  if(strict_contained) *strict_contained=false;
  bool padding=true;
  for (float v:r) {
    if (!std::isfinite(v)) return false;
    if(diagnostic && v!=0.f) padding=false;
  }
  if(diagnostic) {
    ++diagnostic->finite_rows_seen;
    if(padding) ++diagnostic->padding_rows_seen;
    if(!diagnostic->max_score || r[4]>*diagnostic->max_score) diagnostic->max_score=r[4];
  }
  if (r[4]<.25f) return false;
  if(diagnostic) ++diagnostic->score_pass_rows;
  if (r[5]!=0.f) return false;
  if(diagnostic) ++diagnostic->class_pass_rows;
  if (r[2]<=r[0] || r[3]<=r[1]) return false;
  if(diagnostic) ++diagnostic->bbox_pass_rows;
  const double l=c.source_left+(r[0]-c.offset_left)/c.ratio_x;
  const double t=c.source_top+(r[1]-c.offset_top)/c.ratio_y;
  const double b=c.source_top+(r[3]-c.offset_top)/c.ratio_y;
  const double a=c.source_left+(r[2]-c.offset_left)/c.ratio_x;
  const double cx=(l+a)/2, cy=(t+b)/2;
  if (cx<c.original_left || cx>c.original_left+c.original_width ||
      cy<c.original_top || cy>c.original_top+c.original_height) return false;
  if(diagnostic) ++diagnostic->center_pass_rows;
  const double intersection=std::max(0.,std::min(a,c.original_left+c.original_width)-std::max(l,c.original_left))*
      std::max(0.,std::min(b,c.original_top+c.original_height)-std::max(t,c.original_top));
  const double total=(a-l)*(b-t)+c.original_width*c.original_height-intersection;
  if(!(total>0)) {
    if(diagnostic) observe_context_candidate(c,r,row_index,{l,t,a,b},std::nullopt,*diagnostic);
    return false;
  }
  const double iou=intersection/total;
  if(diagnostic) observe_context_candidate(c,r,row_index,{l,t,a,b},iou,*diagnostic);
  if(diagnostic && std::isfinite(iou) &&
      (!diagnostic->max_iou || iou>*diagnostic->max_iou)) diagnostic->max_iou=iou;
  if(!(iou>=.5)) {
    if(strict_contained) *strict_contained=strict_context_torso_contained(c,r,l,t,a,b);
    return false;
  }
  if(diagnostic) ++diagnostic->iou_pass_rows;
  return true;
}

inline bool extra_crop_supported(const std::array<double,4>& box,const CropGeometry& crop);
inline bool same_complete_context_body(const CropGeometry& crop,
    const std::array<float,57>& a,const std::array<float,57>& b);

inline bool current_parent_box_valid(const std::array<double,4>& box) {
  return std::all_of(box.begin(),box.end(),[](double v){return std::isfinite(v);}) &&
    box[0]>=0 && box[1]>=0 && box[2]>0 && box[3]>0 &&
    box[0]+box[2]<=1920 && box[1]+box[3]<=1080;
}

// Positive CURRENT full-body lateral correspondence, never a nearest/strongest
// body choice. Preserve full NN vertical enclosure: landmark containment alone
// cannot assign a nested child's body to an enclosing adult. The caller supplies
// ALL actual PGIE parents; missing/incomplete rosters cannot enable this route.
inline bool full_body_context_owner(const CropGeometry& c,const std::array<float,57>& row,
    const std::vector<std::array<double,4>>* current_parents) {
  if(!current_parents || current_parents->empty() || current_parents->size()>128 ||
     c.model_width!=640 || c.model_height!=640 || !std::isfinite(c.ratio_x) ||
     !std::isfinite(c.ratio_y) || c.ratio_x<=0 || c.ratio_y<=0 ||
     !std::all_of(row.begin(),row.end(),[](float v){return std::isfinite(v);}) ||
     row[4]<.25f || row[5]!=0.f || row[2]<=row[0] || row[3]<=row[1])return false;
  const std::array<double,4> parent{c.original_left,c.original_top,c.original_width,c.original_height};
  if(!current_parent_box_valid(parent))return false;
  const double l=c.source_left+(row[0]-c.offset_left)/c.ratio_x;
  const double t=c.source_top+(row[1]-c.offset_top)/c.ratio_y;
  const double r=c.source_left+(row[2]-c.offset_left)/c.ratio_x;
  const double b=c.source_top+(row[3]-c.offset_top)/c.ratio_y;
  if(t<parent[1] || b>parent[1]+parent[3] ||
     !extra_crop_supported({l,t,r-l,b-t},c) || (l+r)/2<parent[0] ||
     (l+r)/2>parent[0]+parent[2] || (t+b)/2<parent[1] || (t+b)/2>parent[1]+parent[3])return false;
  std::array<std::array<double,2>,17> points{};
  for(int k=0;k<17;++k){
    points[k]={c.source_left+(row[6+k*3]-c.offset_left)/c.ratio_x,
               c.source_top+(row[7+k*3]-c.offset_top)/c.ratio_y};
    if(row[8+k*3]<0 || row[8+k*3]>1)return false;
    if(row[8+k*3]>=.35f && (points[k][0]<l || points[k][0]>r ||
       points[k][1]<t || points[k][1]>b))return false;
  }
  for(int k:{5,6,11,12,13,14,15,16})if(row[8+k*3]<.35f)return false;
  // Same half-model-pixel resolution and observed head-triplet cells used by
  // the existing anatomical predicate; confidence alone is not a resolved head.
  const double grid=std::hypot(.5/c.ratio_x,.5/c.ratio_y);
  bool resolved_head=false;
  for(const auto& side:std::array<std::array<int,3>,2>{{{{0,1,3}},{{0,2,4}}}}){
    bool reliable=true;for(int k:side)if(row[8+k*3]<.35f)reliable=false;
    if(!reliable)continue;
    double radius=std::numeric_limits<double>::infinity();
    for(unsigned i=0;i<3;++i)for(unsigned j=i+1;j<3;++j)
      radius=std::min(radius,.5*std::hypot(points[side[i]][0]-points[side[j]][0],
                                        points[side[i]][1]-points[side[j]][1]));
    if(radius>grid)resolved_head=true;
  }
  if(!resolved_head)return false;
  unsigned owners=0;bool own_parent=false;
  for(const auto& other:*current_parents){
    if(!current_parent_box_valid(other))return false;
    bool contains=true;
    for(int k=0;k<17;++k)if(row[8+k*3]>=.35f &&
        (points[k][0]<other[0] || points[k][0]>other[0]+other[2] ||
         points[k][1]<other[1] || points[k][1]>other[1]+other[3]))contains=false;
    if(contains){++owners;if(other==parent)own_parent=true;}
  }
  return owners==1 && own_parent;
}

template <typename Reader>
inline int unique_context_row(const CropGeometry& crop,int rows,int cols,Reader read,
                             PoseExtractionDiagnostic* diagnostic=nullptr,
                             const std::vector<std::array<double,4>>* current_parents=nullptr) {
  if(diagnostic) {
    diagnostic->qualifying_rows_seen=0;diagnostic->result="context_shape_invalid";
    diagnostic->finite_rows_seen=diagnostic->padding_rows_seen=diagnostic->score_pass_rows=
      diagnostic->class_pass_rows=diagnostic->bbox_pass_rows=diagnostic->center_pass_rows=
      diagnostic->iou_pass_rows=0;
    diagnostic->max_score.reset();diagnostic->max_iou.reset();
    diagnostic->pre_iou_candidates_seen=0;diagnostic->pre_iou_scan_complete=false;
    diagnostic->pre_iou_invalid_row_index.reset();diagnostic->pre_iou_candidates={};
    diagnostic->pre_iou_parent_bbox_xywh.reset();
    const std::array<double,4> parent{crop.original_left,crop.original_top,crop.original_width,crop.original_height};
    if(std::all_of(parent.begin(),parent.end(),[](double v){return std::isfinite(v);}))
      diagnostic->pre_iou_parent_bbox_xywh=parent;
  }
  if(rows<=0 || rows>300 || cols!=57) return -1;
  if(diagnostic) diagnostic->result="context_no_matching_row";
  int accepted=-1,contained=-1,complete=-1,first_complete=-1,full_body=-1;
  unsigned contained_count=0,iou_count=0,complete_count=0,full_body_count=0;
  std::array<std::array<float,57>,2> complete_rows{};
  for(int row=0;row<rows;++row) {
    std::array<float,57> values{};
    for(int col=0;col<57;++col) values[col]=read(row,col);
    if(!std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);})) {
      if(diagnostic) {diagnostic->result="context_nonfinite_row";diagnostic->pre_iou_invalid_row_index=row;}
      return -1;
    }
    bool strict_contained=false;
    const bool iou_match=context_row_matches(crop,values,diagnostic,row,&strict_contained);
    if(strict_contained) {++contained_count;contained=row;}
    if(!iou_match){
      if(!strict_contained && full_body_context_owner(crop,values,current_parents)){
        ++full_body_count;full_body=row;}
      continue;
    }
    if(diagnostic) ++diagnostic->qualifying_rows_seen;
    ++iou_count;accepted=row;
    const double l=crop.source_left+(values[0]-crop.offset_left)/crop.ratio_x;
    const double t=crop.source_top+(values[1]-crop.offset_top)/crop.ratio_y;
    const double r=crop.source_left+(values[2]-crop.offset_left)/crop.ratio_x;
    const double b=crop.source_top+(values[3]-crop.offset_top)/crop.ratio_y;
    if(extra_crop_supported({l,t,r-l,b-t},crop)){
      if(complete_count<complete_rows.size())complete_rows[complete_count]=values;
      if(complete_count==0)first_complete=row;
      ++complete_count;complete=row;}
  }
  // Only an ambiguous IoU set uses this refinement. A censored inferred body
  // cannot veto its sole fully RGB-supported competitor; unproven multiple or
  // zero complete bodies remain unresolved. No score/IoU ranking or ownership subtraction.
  if(iou_count>1){accepted=complete_count==1?complete:-1;
    // Exactly two positively proven representations may share this parent's
    // single slot. Preserve original row order and exact row geometry; never
    // use score, transitivity or cross-crop ownership to choose a body.
    if(complete_count==2 && same_complete_context_body(crop,complete_rows[0],complete_rows[1]))
      accepted=first_complete;
    if(diagnostic && accepted<0)diagnostic->result="context_ambiguous_rows";}
  // Task-only contextual crop: no old IoU match may be displaced. A NEW
  // parent-correspondence witness is usable only after the full finite scan.
  if(iou_count==0 && contained_count==1) accepted=contained;
  if(iou_count==0 && contained_count==0 && full_body_count==1) accepted=full_body;
  if(diagnostic) {diagnostic->pre_iou_scan_complete=true;
    if(accepted>=0) diagnostic->result="row_selected";}
  return accepted;
}

// Scalar-only decisions shared with the CPU fixture. A hint is a crop of the
// CURRENT image, never a detection, pose, timestamp renewal or tracker identity.
using Box = std::array<double,4>;  // source-pixel left/top/width/height
inline double overlap(const Box& a,const Box& b) {
  const double w=std::max(0.,std::min(a[0]+a[2],b[0]+b[2])-std::max(a[0],b[0]));
  const double h=std::max(0.,std::min(a[1]+a[3],b[1]+b[3])-std::max(a[1],b[1]));
  const double area=a[2]*a[3]+b[2]*b[3]-w*h;
  return area>0?w*h/area:0.;
}
struct CurrentCandidate {
  Box box;
  std::array<float,51> joints; // original absolute source pixels, not clipped to parent
  float score;
  int row;
};
inline bool extra_crop_supported(const Box& box,const CropGeometry& crop) {
  // Additional births require complete RGB support, not a box inferred into
  // padding or censored at an artificial crop cut. Whole-camera edges retain
  // the existing inclusive image-boundary policy. Parent rows bypass this.
  for(double v:box)if(!std::isfinite(v))return false;
  const double right=box[0]+box[2],bottom=box[1]+box[3];
  const auto crop_right=std::uint64_t(crop.source_left)+crop.source_width;
  const auto crop_bottom=std::uint64_t(crop.source_top)+crop.source_height;
  if(!crop.source_width || !crop.source_height || crop_right>1920 || crop_bottom>1080 ||
     box[2]<=0 || box[3]<=0 || !std::isfinite(right) || !std::isfinite(bottom) ||
     box[0]<0 || box[1]<0 || right>1920 || bottom>1080)return false;
  return (crop.source_left==0 || box[0]>crop.source_left) &&
    (crop.source_top==0 || box[1]>crop.source_top) &&
    (crop_right==1920 || right<crop_right) &&
    (crop_bottom==1080 || bottom<crop_bottom);
}
template<typename Reader>
inline std::vector<CurrentCandidate> current_candidates(const CropGeometry& crop,int rows,int cols,Reader read) {
  if(rows<=0 || rows>300 || cols!=57) throw std::runtime_error("pose ingress tensor shape");
  std::vector<CurrentCandidate> out;
  for(int n=0;n<rows;++n) {
    std::array<float,57> r{};
    for(int c=0;c<57;++c) {r[c]=read(n,c);if(!std::isfinite(r[c]))throw std::runtime_error("pose ingress nonfinite row");}
    if(r[4]<.25f || r[5]!=0.f || r[2]<=r[0] || r[3]<=r[1])continue;
    const double l=crop.source_left+(r[0]-crop.offset_left)/crop.ratio_x;
    const double t=crop.source_top+(r[1]-crop.offset_top)/crop.ratio_y;
    const double right=crop.source_left+(r[2]-crop.offset_left)/crop.ratio_x;
    const double bottom=crop.source_top+(r[3]-crop.offset_top)/crop.ratio_y;
    // Do not create an object from padding or invent clipped detector geometry.
    if(l<0 || t<0 || right>1920 || bottom>1080 || right-l<16 || bottom-t<16)continue;
    CurrentCandidate c{{l,t,right-l,bottom-t},{},r[4],n};
    for(int k=0;k<17;++k) {
      c.joints[k*3]=crop.source_left+(r[6+k*3]-crop.offset_left)/crop.ratio_x;
      c.joints[k*3+1]=crop.source_top+(r[7+k*3]-crop.offset_top)/crop.ratio_y;
      c.joints[k*3+2]=r[8+k*3];
    }
    out.push_back(c);
  }
  return out;
}
inline bool duplicate_box(const Box& box,const std::vector<Box>& prior,double threshold) {
  for(const auto& p:prior)if(overlap(box,p)>=threshold)return true;
  return false;
}
inline bool selected_parent_row(int row,int parent,bool proposal){return !proposal && parent>=0 && row==parent;}
// Same-frame, same UID4/profile/source-raster data ONLY. Confidence is an
// eligibility gate, never a position covariance. The half-spacing cells below
// are disjoint semantic correspondence cells, not fitted model uncertainty.
inline bool same_visible_head(const CurrentCandidate& a,const CropGeometry& ac,
                              const CurrentCandidate& b,const CropGeometry& bc,bool torso_required=true) {
  auto crop_valid=[](const CropGeometry& c){return c.model_width==640 && c.model_height==640 &&
      std::isfinite(c.ratio_x) && std::isfinite(c.ratio_y) && c.ratio_x>0 && c.ratio_y>0;};
  if(!crop_valid(ac) || !crop_valid(bc))return false;
  auto valid=[](const CurrentCandidate& c,int k){const auto* p=&c.joints[k*3];return
      std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
      p[0]>=0 && p[0]<=1920 && p[1]>=0 && p[1]<=1080 && p[2]>=.35f && p[2]<=1;};
  auto distance=[](const CurrentCandidate& x,int i,const CurrentCandidate& y,int j){
    return std::hypot(double(x.joints[i*3])-y.joints[j*3],double(x.joints[i*3+1])-y.joints[j*3+1]);};
  const double grid=std::hypot(.5/ac.ratio_x,.5/ac.ratio_y)+std::hypot(.5/bc.ratio_x,.5/bc.ratio_y);
  // A shared (possibly misassigned) head must not erase a separate reliable
  // torso. Corresponding joints must remain within disjoint shoulder/hip-level
  // cells sized by the shortest observed longitudinal torso bone, BOTH bodies.
  if(torso_required){
    for(int k:{5,6,11,12})if(!valid(a,k) || !valid(b,k))return false;
    const double torso_radius=.5*std::min({distance(a,5,a,11),distance(a,6,a,12),
                                        distance(b,5,b,11),distance(b,6,b,12)});
    if(!(torso_radius>grid))return false;
    for(int k:{5,6,11,12})if(distance(a,k,b,k)>=torso_radius)return false;
  }
  // Projected half model-pixel footprints only resolve correspondence ambiguity;
  // they do NOT explain the much larger crop-dependent torso residuals.
  for(const auto& side:std::array<std::array<int,3>,2>{{{{0,1,3}},{{0,2,4}}}}) {
    bool trusted=true;for(int k:side)if(!valid(a,k) || !valid(b,k))trusted=false;
    if(!trusted)continue;
    double radius=std::numeric_limits<double>::infinity();
    for(unsigned i=0;i<3;++i)for(unsigned j=i+1;j<3;++j)
      radius=std::min(radius,.5*std::min(distance(a,side[i],a,side[j]),distance(b,side[i],b,side[j])));
    if(!(radius>grid))continue; // unresolved/degenerate head landmarks
    bool matched=true;
    for(int k:side){const auto d=distance(a,k,b,k);if(d>=radius){matched=false;break;}
      for(int other:side)if(other!=k &&
        (distance(a,k,b,other)-d<=2*grid || distance(b,k,a,other)-d<=2*grid))matched=false;
    }
    if(matched)return true;
  }
  return false;
}
inline bool same_complete_context_body(const CropGeometry& crop,
    const std::array<float,57>& a,const std::array<float,57>& b) {
  // Inputs are the two already-read current rows, not another tensor access.
  auto candidate=[&](const std::array<float,57>& row){
    const double l=crop.source_left+(row[0]-crop.offset_left)/crop.ratio_x;
    const double t=crop.source_top+(row[1]-crop.offset_top)/crop.ratio_y;
    const double r=crop.source_left+(row[2]-crop.offset_left)/crop.ratio_x;
    const double b=crop.source_top+(row[3]-crop.offset_top)/crop.ratio_y;
    CurrentCandidate c{{l,t,r-l,b-t},{},row[4],-1};
    for(int k=0;k<17;++k){
      c.joints[k*3]=crop.source_left+(row[6+k*3]-crop.offset_left)/crop.ratio_x;
      c.joints[k*3+1]=crop.source_top+(row[7+k*3]-crop.offset_top)/crop.ratio_y;
      c.joints[k*3+2]=row[8+k*3];}
    return c;
  };
  const auto x=candidate(a),y=candidate(b);
  return same_visible_head(x,crop,y,crop) && same_visible_head(y,crop,x,crop);
}
template<typename References>
inline bool unique_visible_head(const CurrentCandidate& a,const CropGeometry& crop,const References& refs) {
  unsigned matches=0;bool compatible=false;
  // Competing supported heads remain ambiguous EVEN if the other torso is
  // missing/different: do not resolve a head misassignment by assuming a body.
  for(const auto* ref:refs)if(same_visible_head(a,crop,ref->c,ref->crop,false)){
    if(++matches>1)return false;compatible=same_visible_head(a,crop,ref->c,ref->crop);}
  return matches==1 && compatible;
}
// Optional, owned scalar inspection only. Nothing below is a matching input.
struct IngressInspection {
  struct Record {
    unsigned source=0,slot=0,crop_id=0,origin=0;
    int frame=0,parent_row=-1,disposition=0; // 0 unselected,1 parent,2 IoU suppressed,3 torso rejected,4 promoted,5 anatomy suppressed
    std::uint64_t pts=0,generation=0,raw=0;
    bool proposal=false,bound=false;
    CropGeometry crop{};
    CurrentCandidate candidate{}; // UNCLIPPED current NN source coordinates
    Box tracked_box{};
    std::array<float,51> attached_joints{}; // exact receipt after parent clipping
  };
  static constexpr unsigned capacity=64,byte_cap=65536;
  static_assert(sizeof(Record)*capacity<byte_cap,"inspection scalar storage bound");
  std::array<Record,capacity> records{};
  std::array<int,3> frames{};
  std::array<unsigned,3> seen{};
  unsigned source=0,frame_count=0,size=0,overflow=0,binding_mismatch=0;
  bool enabled=false,invalid_config=false;
  explicit IngressInspection(const char* setting=nullptr) noexcept {
    if(!setting || !*setting)return;
    const char* p=setting;const char* end=p;while(*end && end-p<80)++end;
    auto number=[&](int& out,char delimiter){auto r=std::from_chars(p,end,out);
      if(r.ec!=std::errc{} || r.ptr==p || (delimiter && (r.ptr==end || *r.ptr!=delimiter)))return false;
      p=r.ptr+(delimiter?1:0);return true;};
    int s=0;
    if(end-p>=80 || !number(s,':') || s<0 || s>=16){invalid_config=true;return;}
    while(p<end && frame_count<frames.size()) {
      auto r=std::from_chars(p,end,frames[frame_count]);
      if(r.ec!=std::errc{} || r.ptr==p || frames[frame_count]<0 ||
         (frame_count && frames[frame_count]<=frames[frame_count-1])){invalid_config=true;return;}
      ++frame_count;p=r.ptr;
      if(p==end)break;
      if(*p++!=',' || p==end){invalid_config=true;return;}
    }
    if(p!=end || frame_count<2){invalid_config=true;return;}
    source=unsigned(s);enabled=true;
  }
  bool selected(unsigned s,int f)const noexcept{
    return enabled && s==source && std::find(frames.begin(),frames.begin()+frame_count,f)!=frames.begin()+frame_count;}
  void mark(unsigned s,int f)noexcept{if(selected(s,f))for(unsigned i=0;i<frame_count;++i)if(frames[i]==f)++seen[i];}
  unsigned add(unsigned s,int f,std::uint64_t pts,unsigned slot,std::uint64_t generation,
      unsigned crop_id,bool proposal,int parent,const CropGeometry& crop,const CurrentCandidate& c)noexcept {
    if(!selected(s,f))return 0;
    if(size==capacity){++overflow;return 0;}
    auto& r=records[size];r.source=s;r.frame=f;r.pts=pts;r.slot=slot;r.generation=generation;
    r.crop_id=crop_id;r.proposal=proposal;r.parent_row=parent;r.crop=crop;r.candidate=c;
    return ++size;
  }
  void disposition(unsigned token,int value)noexcept{if(token && token<=size)records[token-1].disposition=value;}
  void bind(unsigned token,unsigned s,int f,std::uint64_t pts,unsigned slot,std::uint64_t generation,
      unsigned origin,std::uint64_t raw,const Box& box,const CurrentCandidate& attached)noexcept {
    if(!token)return;
    if(token>size){++binding_mismatch;return;}
    auto& r=records[token-1];
    if(r.bound || r.source!=s || r.frame!=f || r.pts!=pts || r.slot!=slot || r.generation!=generation ||
       r.candidate.row!=attached.row || r.candidate.box!=attached.box || r.candidate.score!=attached.score ||
       (origin!=1 && origin!=4) || raw==UINT64_MAX){++binding_mismatch;return;}
    r.bound=true;r.origin=origin;r.raw=raw;r.tracked_box=box;r.attached_joints=attached.joints;
  }
  std::string json()const { // called ONLY by shutdown snapshot, never a callback
    std::ostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(9);
    out<<"{\"enabled\":"<<(enabled?"true":"false")<<",\"invalid_config\":"<<(invalid_config?"true":"false")
       <<",\"source\":"<<source<<",\"frames\":[";
    for(unsigned i=0;i<frame_count;++i){if(i)out<<',';out<<frames[i];}out<<"],\"seen\":[";
    for(unsigned i=0;i<frame_count;++i){if(i)out<<',';out<<seen[i];}
    out<<"],\"frame_candidate_counts\":[";
    for(unsigned f=0;f<frame_count;++f){unsigned count=0;for(unsigned i=0;i<size;++i)if(records[i].frame==frames[f])++count;
      if(f)out<<',';out<<count;}
    out<<"],\"overflow\":"<<overflow<<",\"binding_mismatch\":"<<binding_mismatch
       <<",\"coordinate_frame\":\"current_dewarped_source_pixels_1920x1080\",\"records\":[";
    bool first=true;unsigned omitted=0;
    for(unsigned i=0;i<size;++i){const auto& r=records[i];std::ostringstream row;row.imbue(std::locale::classic());row<<std::setprecision(9);
      auto array=[&](const auto& a){row<<'[';bool sep=false;for(auto v:a){if(sep)row<<',';sep=true;row<<v;}row<<']';};
      row<<"{\"source\":"<<r.source<<",\"frame\":"<<r.frame<<",\"pts\":"<<r.pts<<",\"slot\":"<<r.slot
         <<",\"generation\":"<<r.generation<<",\"crop_id\":"<<r.crop_id<<",\"crop_origin_uid\":"<<(r.proposal?0x4e50524fU:1U)<<",\"proposal\":"<<(r.proposal?"true":"false")
         <<",\"row\":"<<r.candidate.row<<",\"parent_row\":"<<r.parent_row<<",\"disposition\":"<<r.disposition
         <<",\"score\":"<<r.candidate.score<<",\"nn_bbox_xywh\":";array(r.candidate.box);
      row<<",\"unclipped_joints_xyc\":";array(r.candidate.joints);
      row<<",\"parent_bbox_xywh\":";array(Box{r.crop.original_left,r.crop.original_top,r.crop.original_width,r.crop.original_height});
      row<<",\"crop_mapping\":";array(std::array<double,12>{double(r.crop.source_left),double(r.crop.source_top),double(r.crop.source_width),double(r.crop.source_height),double(r.crop.model_width),double(r.crop.model_height),double(r.crop.destination_width),double(r.crop.destination_height),double(r.crop.offset_left),double(r.crop.offset_top),r.crop.ratio_x,r.crop.ratio_y});
      row<<",\"tracker_bound\":"<<(r.bound?"true":"false");
      if(r.bound){row<<",\"raw_id\":"<<r.raw<<",\"origin_uid\":"<<r.origin<<",\"tracked_bbox_xywh\":";array(r.tracked_box);
        row<<",\"attached_joints_xyc\":";array(r.attached_joints);}
      row<<'}';const auto text=row.str();
      if(out.tellp()+std::streamoff(text.size()+128)>byte_cap){++omitted;continue;}
      if(!first)out<<',';first=false;out<<text;
    }
    unsigned missing=0;for(unsigned i=0;i<frame_count;++i)if(enabled && !seen[i])++missing;
    out<<"],\"byte_omitted\":"<<omitted<<",\"missing_frames\":"<<missing<<'}';
    return out.str();
  }
};
struct HintSnapshot {std::int64_t frame;std::uint64_t pts,generation;std::vector<Box> boxes;};
class HintBank {
 public:
  unsigned targets,frame_limit;
  std::uint64_t generation=0;
  std::map<unsigned,std::uint64_t> source_generations;
  std::map<unsigned,HintSnapshot> snapshots;
  HintBank(unsigned count,unsigned frames):targets(count),frame_limit(frames) {
    if(!count || count>128 || !frames || frames>10000)throw std::runtime_error("pose hint limits");
  }
  std::uint64_t epoch(unsigned source)const {auto i=source_generations.find(source);return (generation<<32)+(i==source_generations.end()?0:i->second);}
  void reset(){snapshots.clear();source_generations.clear();++generation;}
  void reset(unsigned source){snapshots.erase(source);++source_generations[source];}
  std::vector<Box> hints(unsigned source,std::int64_t frame,std::uint64_t pts) {
    auto i=snapshots.find(source);if(i==snapshots.end())return {};
    const auto& s=i->second;
    if(frame<=s.frame || pts<=s.pts) {reset(source);return {};}
    if(std::uint64_t(frame-s.frame)>frame_limit)return {};
    return s.boxes;
  }
  void observe(unsigned source,std::int64_t frame,std::uint64_t pts,std::vector<Box> boxes) {
    if(source>=16 || boxes.size()>targets)throw std::runtime_error("pose hint source/target bound");
    snapshots[source]={frame,pts,epoch(source),std::move(boxes)};
  }
};

}  // namespace pose
}  // namespace noesis

#ifndef NOESIS_POSE_CROP_GEOMETRY_TEST
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <atomic>
#include <mutex>

#include <cuda_runtime_api.h>

#include "metadata.hpp"
#include "nvdsmeta.h"
#include "tensor.hpp"
#include "buffer_probe.hpp"
#include "element.hpp"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "gst-nvevent.h"
#include "nvdspreprocess_meta.h"

namespace py = pybind11;

namespace {

constexpr unsigned kContextObjects=128, kContextFrames=3;
struct RoiContext {
  noesis::pose::CropGeometry crop;
  guint source; gint frame; guint gie;
  guint64 pts, object;
  bool restored=false;
};
int roi_context_type() {
  static int type=nvds_get_user_meta_type(const_cast<gchar*>("NOESIS.POSE_ROI_CONTEXT.V1"));
  return type;
}
void* roi_context_copy(void* meta,void*) {
  deepstream::UserMetadata m(meta);
  auto* p=static_cast<RoiContext*>(m.userData());
  return p ? new RoiContext(*p) : nullptr;
}
void roi_context_release(void* meta,void*) {
  deepstream::UserMetadata m(meta); delete static_cast<RoiContext*>(m.userData());
}
RoiContext* roi_context(const deepstream::ObjectMetadata& object) {
  RoiContext* result=nullptr; unsigned count=0;
  object.iterate([&](const deepstream::UserMetadata& m){++count; result=static_cast<RoiContext*>(m.userData());},roi_context_type());
  if (count>1) throw std::runtime_error("duplicate pose crop context");
  return result;
}
struct CurrentPose {
  guint source,slot,origin;gint frame;guint64 pts,generation;
  noesis::pose::CurrentCandidate value;
  bool tracker_bound=false;
  unsigned inspection_token=0;
};
int current_pose_type(){static int t=nvds_get_user_meta_type(const_cast<gchar*>("NOESIS.CURRENT_POSE_INGRESS.V1"));return t;}
void* current_pose_copy(void* meta,void*) {deepstream::UserMetadata m(meta);auto* p=static_cast<CurrentPose*>(m.userData());return p?new CurrentPose(*p):nullptr;}
void current_pose_free(void* meta,void*) {deepstream::UserMetadata m(meta);delete static_cast<CurrentPose*>(m.userData());}
CurrentPose* current_pose(const deepstream::ObjectMetadata& object) {
  CurrentPose* result=nullptr;unsigned count=0;
  object.iterate([&](const deepstream::UserMetadata& m){++count;result=static_cast<CurrentPose*>(m.userData());},current_pose_type());
  if(count>1)throw std::runtime_error("duplicate current pose receipt");return result;
}
void attach_current_pose(NvDsBatchMeta* batch,NvDsFrameMeta* frame,NvDsObjectMeta* object,
                         const noesis::pose::CurrentCandidate& value,guint origin,guint64 generation,unsigned inspection_token=0) {
  deepstream::ObjectMetadata wrapped(object);
  if(current_pose(wrapped))throw std::runtime_error("current pose receipt already attached");
  auto* u=nvds_acquire_user_meta_from_pool(batch);if(!u)throw std::runtime_error("current pose receipt pool exhausted");
  u->user_meta_data=new CurrentPose{frame->source_id,frame->batch_id,origin,frame->frame_num,frame->buf_pts,generation,value,false,inspection_token};
  u->base_meta.meta_type=static_cast<NvDsMetaType>(current_pose_type());
  u->base_meta.copy_func=current_pose_copy;u->base_meta.release_func=current_pose_free;
  nvds_add_user_meta_to_obj(object,u);
}
unsigned meta_count(NvDsMetaList* list) {
  unsigned n=0; for(;list;list=list->next) if(++n>1024) throw std::runtime_error("metadata list bound");
  return n;
}
bool same_pose_object_key(const NvDsObjectMeta& a,const NvDsObjectMeta& b) {
  return a.object_id==b.object_id && a.unique_component_id==b.unique_component_id && a.class_id==b.class_id &&
      (a.object_id!=UNTRACKED_OBJECT_ID ||
       (a.rect_params.left==b.rect_params.left && a.rect_params.top==b.rect_params.top &&
        a.rect_params.width==b.rect_params.width && a.rect_params.height==b.rect_params.height));
}
void validate_pose_object_copies(const NvDsFrameMeta& source,const NvDsFrameMeta& copy,unsigned& objects) {
  // Installed SDK prepends copied objects: list position is NOT object identity.
  // Frame source/frame/FULL PTS/slot already checked by the caller. Scan only
  // this bounded frame, retain no SDK pointers beyond this callback.
  for(auto* u=source.obj_meta_list;u;u=u->next) {
    const auto* p=static_cast<const NvDsObjectMeta*>(u->data);
    const NvDsObjectMeta* q=nullptr;unsigned matches=0;
    for(auto* v=copy.obj_meta_list;v;v=v->next) {
      const auto* candidate=static_cast<const NvDsObjectMeta*>(v->data);
      if(!candidate)throw std::runtime_error("pose copied object null");
      if(same_pose_object_key(*p,*candidate)){q=candidate;++matches;}
    }
    const auto binding=" source="+std::to_string(source.source_id)+" frame="+std::to_string(source.frame_num)+
      " pts="+std::to_string(source.buf_pts)+" object="+std::to_string(p->object_id)+
      " component="+std::to_string(p->unique_component_id)+" class="+std::to_string(p->class_id);
    if(++objects>kContextObjects || matches!=1)
      throw std::runtime_error("pose object copy key ambiguous/missing: matches="+std::to_string(matches)+binding);
    const bool same_rect=std::memcmp(&p->rect_params,&q->rect_params,sizeof(p->rect_params))==0;
    const auto source_users=meta_count(p->obj_user_meta_list),copy_users=meta_count(q->obj_user_meta_list);
    if(p==q || !same_rect || source_users!=copy_users)
      throw std::runtime_error("pose object copy mismatch: alias="+std::to_string(p==q)+
        " rect_equal="+std::to_string(same_rect)+" source_users="+std::to_string(source_users)+
        " copy_users="+std::to_string(copy_users)+binding);
  }
}
// Source stays referenced during all alias checks. gst_buffer_copy shares GstMemory
// (NVMM surface), while the registered SDK batch-meta callback owns metadata copies.
// gstnvdspreprocess.cpp1422..1424 deliberately has NO copy callback. Its
// release callback owns separate tensor/scaling pools; never shallow-share it.
// Only the explicit post-PGIE callsite may omit a precursor targeted SOLELY at
// already-upstream UID1. The ordinary post-tracker route remains exact/strict.
bool consumed_pgie_precursor(const NvDsUserMeta& user,const NvDsBatchMeta& batch,bool after_pgie) {
  if(!after_pgie || user.base_meta.meta_type!=NVDS_PREPROCESS_BATCH_META || user.base_meta.copy_func)return false;
  if(!user.user_meta_data || !user.base_meta.release_func || user.base_meta.batch_meta!=&batch)
    throw std::runtime_error("pose PGIE precursor ownership invalid");
  const auto& p=*static_cast<const GstNvDsPreProcessBatchMeta*>(user.user_meta_data);
  if(p.target_unique_ids.size()!=1 || p.target_unique_ids[0]!=1)
    throw std::runtime_error("pose unconsumed preprocess target");
  return true;
}
unsigned expected_pose_batch_users(const NvDsBatchMeta& batch,bool after_pgie) {
  const auto total=meta_count(batch.batch_user_meta_list);unsigned omitted=0;
  for(auto* n=batch.batch_user_meta_list;n;n=n->next){
    if(!n->data)throw std::runtime_error("pose batch userdata null");
    if(consumed_pgie_precursor(*static_cast<NvDsUserMeta*>(n->data),batch,after_pgie))
      if(++omitted>kContextFrames)throw std::runtime_error("pose PGIE precursor count bound");
  }
  return total-omitted;
}
void validate_pose_batch_users(const NvDsBatchMeta& source,const NvDsBatchMeta& copy,bool after_pgie) {
  if(meta_count(copy.batch_user_meta_list)!=expected_pose_batch_users(source,after_pgie))
    throw std::runtime_error("pose batch userdata copy loss");
  // Counts alone cannot hide one lost type replaced by a different type.
  std::map<int,unsigned> expected,actual;
  for(auto* n=source.batch_user_meta_list;n;n=n->next){auto* u=static_cast<NvDsUserMeta*>(n->data);
    if(!consumed_pgie_precursor(*u,source,after_pgie))++expected[int(u->base_meta.meta_type)];}
  for(auto* n=copy.batch_user_meta_list;n;n=n->next){auto* u=static_cast<NvDsUserMeta*>(n->data);
    if(!u)throw std::runtime_error("pose cloned batch userdata null");
    ++actual[int(u->base_meta.meta_type)];}
  if(actual!=expected)throw std::runtime_error("pose batch userdata type loss/substitution");
}
GstBuffer* private_pose_buffer(GstBuffer* source,bool after_pgie=false) {
  auto* old=gst_buffer_get_nvds_batch_meta(source);
  if(!old || old->max_frames_in_batch==0 || old->max_frames_in_batch>kContextFrames ||
     old->num_frames_in_batch>old->max_frames_in_batch || old->base_meta.batch_meta!=old ||
     old->base_meta.copy_func!=nvds_batch_meta_copy_func ||
     old->base_meta.release_func!=nvds_batch_meta_release_func)
    throw std::runtime_error("pose batch capacity/SDK ownership invalid before copy");
  gpointer iterator=nullptr;GstMeta* gst_meta=nullptr;unsigned metas=0,batch_records=0;
  while((gst_meta=gst_buffer_iterate_meta(source,&iterator))) {
    if(++metas>32) throw std::runtime_error("pose GstMeta bound");
    if(gst_meta->info->api!=NVDS_META_API_TYPE) continue;
    auto* nvds=reinterpret_cast<NvDsMeta*>(gst_meta);
    if(nvds->meta_type!=NVDS_BATCH_GST_META || nvds->meta_data!=old) continue;
    ++batch_records;
    if(nvds->copyfunc!=nvds_batch_meta_copy_func || nvds->freefunc!=nvds_batch_meta_release_func)
      throw std::runtime_error("pose actual Gst batch copy/release contract mismatch");
  }
  if(batch_records!=1) throw std::runtime_error("pose actual Gst batch metadata ownership ambiguous");
  // Public NvDsBatchMeta::meta_mutex is a GRecMutex. Never wait on another
  // branch in this optional hot path; copying without its ownership is unsafe.
  if(!g_rec_mutex_trylock(&old->meta_mutex)) throw std::runtime_error("pose metadata copy busy");
  struct Unlock {GRecMutex* lock;~Unlock(){g_rec_mutex_unlock(lock);}} unlock{&old->meta_mutex};
  const auto expected_batch_users=expected_pose_batch_users(*old,after_pgie);
  const auto source_frame_count=meta_count(old->frame_meta_list);
  if(source_frame_count>kContextFrames)
    throw std::runtime_error("pose SOURCE frame metadata bound: count="+std::to_string(source_frame_count));
  unsigned source_objects=0;
  for(auto* f=old->frame_meta_list;f;f=f->next) {
    // Bound the SOURCE before SDK callbacks can allocate any amplified copy.
    if(!f->data)
      throw std::runtime_error("pose SOURCE frame metadata bound/null");
    const auto* frame=static_cast<NvDsFrameMeta*>(f->data);
    meta_count(frame->frame_user_meta_list);
    for(auto* o=frame->obj_meta_list;o;o=o->next) {
      if(++source_objects>kContextObjects || !o->data)
        throw std::runtime_error("pose SOURCE object bound/null: count="+std::to_string(source_objects));
      const auto* object=static_cast<NvDsObjectMeta*>(o->data);
      meta_count(object->obj_user_meta_list);
      for(auto* previous=frame->obj_meta_list;previous!=o;previous=previous->next)
        if(same_pose_object_key(*object,*static_cast<NvDsObjectMeta*>(previous->data)))
          throw std::runtime_error("pose SOURCE duplicate object key: source="+std::to_string(frame->source_id)+
            " frame="+std::to_string(frame->frame_num)+" pts="+std::to_string(frame->buf_pts)+
            " object="+std::to_string(object->object_id)+" component="+std::to_string(object->unique_component_id)+
            " class="+std::to_string(object->class_id));
    }
  }
  struct SourceRaster {guint width,height,source,slot;gint frame;guint64 pts;};
  std::array<SourceRaster,kContextFrames> source_rasters{};unsigned source_frames=0;
  {
    // Same HEADER-only GstBuffer map used by installed gstnvinfer.cpp2419..2438.
    // Never map/sync/read pixels, dataPtr or mappedAddr. Unmap before SDK copy.
    struct HeaderMap {
      GstBuffer* buffer; GstMapInfo info=GST_MAP_INFO_INIT; bool mapped=false;
      explicit HeaderMap(GstBuffer* b):buffer(b){mapped=gst_buffer_map(b,&info,GST_MAP_READ);}
      ~HeaderMap(){if(mapped)gst_buffer_unmap(buffer,&info);}
    } header(source);
    if(!header.mapped || !header.info.data || header.info.size<sizeof(NvBufSurface))
      throw std::runtime_error("pose surface HEADER unavailable/short: mapped="+
        std::to_string(header.mapped)+" bytes="+std::to_string(header.info.size));
    const auto* surface=reinterpret_cast<const NvBufSurface*>(header.info.data);
    if(surface->batchSize==0 || surface->batchSize>kContextFrames || surface->numFilled==0 ||
       surface->numFilled>surface->batchSize || !surface->surfaceList)
      throw std::runtime_error("pose surface HEADER invalid: batchSize="+std::to_string(surface->batchSize)+
        " numFilled="+std::to_string(surface->numFilled)+" surfaceList_present="+
        std::to_string(surface->surfaceList!=nullptr)+" gpuId="+std::to_string(surface->gpuId)+
        " memType="+std::to_string(surface->memType));
    std::array<bool,kContextFrames> seen_slots{};
    for(auto* f=old->frame_meta_list;f;f=f->next) {
      auto* frame=static_cast<NvDsFrameMeta*>(f->data);
      if(!frame || source_frames>=kContextFrames)
        throw std::runtime_error("pose SOURCE frame list invalid/bound: count="+std::to_string(source_frames));
      const auto binding=" source="+std::to_string(frame->source_id)+" frame="+std::to_string(frame->frame_num)+
        " pts="+std::to_string(frame->buf_pts)+" batch_id="+std::to_string(frame->batch_id);
      if(frame->batch_id>=surface->numFilled || frame->batch_id>=surface->batchSize)
        throw std::runtime_error("pose SOURCE batch_id outside filled HEADER: numFilled="+
          std::to_string(surface->numFilled)+binding);
      if(seen_slots[frame->batch_id] || frame->frame_num<0 || frame->buf_pts==GST_CLOCK_TIME_NONE)
        throw std::runtime_error("pose SOURCE original/duplicate slot invalid:"+binding);
      seen_slots[frame->batch_id]=true;
      const auto& raster=surface->surfaceList[frame->batch_id];
      if(raster.width!=1920 || raster.height!=1080 ||
         (frame->pipeline_width!=0 && frame->pipeline_width!=raster.width) ||
         (frame->pipeline_height!=0 && frame->pipeline_height!=raster.height))
        throw std::runtime_error("pose SOURCE HEADER raster conflict: width="+std::to_string(raster.width)+
          " height="+std::to_string(raster.height)+" pipeline_width="+std::to_string(frame->pipeline_width)+
          " pipeline_height="+std::to_string(frame->pipeline_height)+binding);
      source_rasters[source_frames++]={raster.width,raster.height,frame->source_id,frame->batch_id,
                                      frame->frame_num,frame->buf_pts};
    }
    if(source_frames>surface->numFilled)
      throw std::runtime_error("pose SOURCE frames exceed HEADER numFilled: frames="+std::to_string(source_frames)+
        " numFilled="+std::to_string(surface->numFilled));
  }
  if(source_frames==0 || source_frames!=old->num_frames_in_batch)
    throw std::runtime_error("pose SOURCE frame count mismatch: list="+std::to_string(source_frames)+
      " metadata="+std::to_string(old->num_frames_in_batch));
  GstBuffer* copy=gst_buffer_copy(source);
  if(!copy) throw std::runtime_error("pose buffer copy failed");
  try {
    auto* next=gst_buffer_get_nvds_batch_meta(copy);
    if(!next || next==old || next->max_frames_in_batch!=old->max_frames_in_batch ||
       next->num_frames_in_batch!=old->num_frames_in_batch ||
       meta_count(next->frame_meta_list)!=meta_count(old->frame_meta_list) ||
       meta_count(next->batch_user_meta_list)!=expected_batch_users ||
       gst_buffer_n_memory(copy)!=gst_buffer_n_memory(source)) throw std::runtime_error("pose metadata copy alias/loss");
    validate_pose_batch_users(*old,*next,after_pgie);
    for(guint n=0;n<gst_buffer_n_memory(source);++n)
      if(gst_buffer_peek_memory(copy,n)!=gst_buffer_peek_memory(source,n)) throw std::runtime_error("pose surface was copied");
    auto* a=old->frame_meta_list; auto* b=next->frame_meta_list;
    unsigned objects=0,frame_index=0;
    for(;a&&b;a=a->next,b=b->next) {
      auto* x=static_cast<NvDsFrameMeta*>(a->data); auto* y=static_cast<NvDsFrameMeta*>(b->data);
      if(x==y || x->frame_num!=y->frame_num || x->source_id!=y->source_id || x->buf_pts!=y->buf_pts || x->batch_id!=y->batch_id ||
         meta_count(x->obj_meta_list)!=meta_count(y->obj_meta_list) ||
         meta_count(x->frame_user_meta_list)!=meta_count(y->frame_user_meta_list)) throw std::runtime_error("pose frame copy mismatch");
      const auto& measured=source_rasters[frame_index++];
      if(y->source_id!=measured.source || y->frame_num!=measured.frame || y->buf_pts!=measured.pts || y->batch_id!=measured.slot ||
         (y->pipeline_width!=0 && y->pipeline_width!=measured.width) ||
         (y->pipeline_height!=0 && y->pipeline_height!=measured.height))
        throw std::runtime_error("pose copied source raster/original conflict");
      // Installed nvds_copy_frame_meta omits these appended public scalar fields.
      // Carry ONLY this current buffer HEADER's qualified values, never a default.
      y->pipeline_width=measured.width; y->pipeline_height=measured.height;
      if(y->pipeline_width!=measured.width || y->pipeline_height!=measured.height)
        throw std::runtime_error("pose copied source raster preservation mismatch");
      validate_pose_object_copies(*x,*y,objects);
    }
    return copy;
  } catch(...) {gst_buffer_unref(copy);throw;}
}

class RoiProbeState : public std::enable_shared_from_this<RoiProbeState> {
  GstPad* sink_=nullptr; GstPad* src_=nullptr; gulong pre_=0,post_=0;
  std::mutex close_mutex_;
 public:
  struct ProbeContext {std::shared_ptr<RoiProbeState> owner; bool before;};
  std::atomic<bool> closed{false};
  std::atomic<guint64> copied{0},expanded{0},restored{0},failures{0},elapsed_ns{0};
  static void release(gpointer data) { delete static_cast<ProbeContext*>(data); }
  static GstPadProbeReturn callback(GstPad* pad,GstPadProbeInfo* info,gpointer data) {
    const auto context=*static_cast<ProbeContext*>(data);
    auto self=context.owner;
    const auto start=g_get_monotonic_time();
    try {
      if(self->closed) throw std::runtime_error("pose context closed");
      auto* buffer=GST_PAD_PROBE_INFO_BUFFER(info);
      if(context.before) {
        auto* copy=private_pose_buffer(buffer);
        GST_PAD_PROBE_INFO_DATA(info)=copy; gst_buffer_unref(buffer);
        self->prepare(copy);
      } else self->restore(buffer);
      self->elapsed_ns+=guint64(g_get_monotonic_time()-start)*1000;
      return GST_PAD_PROBE_OK;
    } catch(const std::exception& e) {
      ++self->failures;
      auto* parent=gst_pad_get_parent_element(pad);
      if(parent) {GST_ELEMENT_ERROR(parent,STREAM,FAILED,("pose ROI context integrity failure"),("%s",e.what()));gst_object_unref(parent);}
      return GST_PAD_PROBE_DROP; // Paired with a real terminal bus error, never silent fallback.
    }
  }
  void prepare(GstBuffer* buffer) {
    auto* raw=gst_buffer_get_nvds_batch_meta(buffer); deepstream::BatchMetadata batch(raw);
    ++copied;
    for(auto* f=raw->frame_meta_list;f;f=f->next) {
      auto* frame=static_cast<NvDsFrameMeta*>(f->data);
      if(frame->pipeline_width!=1920 || frame->pipeline_height!=1080 || frame->frame_num<0 ||
         frame->buf_pts==GST_CLOCK_TIME_NONE) throw std::runtime_error("pose context raster/original mismatch");
      for(auto* o=frame->obj_meta_list;o;o=o->next) {
        auto* object=static_cast<NvDsObjectMeta*>(o->data);
        deepstream::ObjectMetadata wrapper(object);
        if(roi_context(wrapper)) throw std::runtime_error("repeated context preparation");
        const auto& r=object->rect_params;
        if(object->class_id!=0 || object->unique_component_id!=1 || r.width<16 || r.height<16 ||
           std::min(r.width,r.height)>=128 || double(r.width)/r.height>.65) continue;
        auto geometry=noesis::pose::context_crop(r.left,r.top,r.width,r.height,1920,1080);
        if(!geometry) throw std::runtime_error("invalid expanded pose crop");
        deepstream::UserMetadata meta(nullptr);
        if(!batch.acquire(meta)) throw std::runtime_error("pose context metadata pool exhausted");
        auto context=std::make_unique<RoiContext>(RoiContext{*geometry,frame->source_id,frame->frame_num,4,frame->buf_pts,object->object_id,false});
        meta.setMetaType(roi_context_type()); meta.setUserData(context.get(),roi_context_copy,roi_context_release);
        wrapper.append(meta); context.release();
        auto expanded_rect=r;
        expanded_rect.left=geometry->source_left; expanded_rect.top=geometry->source_top;
        expanded_rect.width=geometry->source_width; expanded_rect.height=geometry->source_height;
        wrapper.setRectParams(expanded_rect); ++expanded;
      }
    }
  }
  void restore(GstBuffer* buffer) {
    auto* batch=gst_buffer_get_nvds_batch_meta(buffer);
    if(!batch) throw std::runtime_error("pose context restore missing batch");
    for(auto* f=batch->frame_meta_list;f;f=f->next) {
      auto* frame=static_cast<NvDsFrameMeta*>(f->data);
      for(auto* o=frame->obj_meta_list;o;o=o->next) {
        auto* object=static_cast<NvDsObjectMeta*>(o->data); deepstream::ObjectMetadata wrapper(object);
        auto* c=roi_context(wrapper); if(!c) continue;
        if(c->restored || c->source!=frame->source_id || c->frame!=frame->frame_num ||
           c->pts!=frame->buf_pts || c->object!=object->object_id || c->gie!=4 ||
           object->class_id!=0 || object->unique_component_id!=1) throw std::runtime_error("pose context original binding mismatch");
        auto r=wrapper.rectParams();r.left=c->crop.original_left;r.top=c->crop.original_top;
        r.width=c->crop.original_width;r.height=c->crop.original_height;
        wrapper.setRectParams(r); c->restored=true; ++restored;
      }
    }
  }
  void attach(GstElement* element) {
    sink_=gst_element_get_static_pad(element,"sink"); src_=gst_element_get_static_pad(element,"src");
    if(!sink_ || !src_) throw std::runtime_error("pose context actual pads unavailable");
    pre_=gst_pad_add_probe(sink_,GST_PAD_PROBE_TYPE_BUFFER,callback,new ProbeContext{shared_from_this(),true},release);
    post_=gst_pad_add_probe(src_,GST_PAD_PROBE_TYPE_BUFFER,callback,new ProbeContext{shared_from_this(),false},release);
    if(!pre_ || !post_) throw std::runtime_error("pose context probe attach failed");
  }
  void close() {
    GstPad *sink,*src;gulong pre,post;
    {std::lock_guard<std::mutex> lock(close_mutex_);if(closed.exchange(true))return;
      sink=sink_;src=src_;pre=pre_;post=post_;sink_=src_=nullptr;pre_=post_=0;}
    if(sink){if(pre)gst_pad_remove_probe(sink,pre);gst_object_unref(sink);}
    if(src){if(post)gst_pad_remove_probe(src,post);gst_object_unref(src);}
  }
  ~RoiProbeState(){close();}
};

class PoseRoiContextController {
  std::shared_ptr<RoiProbeState> state_=std::make_shared<RoiProbeState>();
 public:
  PoseRoiContextController(deepstream::CustomObject& anchor,const std::string& target) {
    auto* probe=dynamic_cast<deepstream::BufferProbe*>(&anchor);
    auto* element=probe?probe->getTarget():nullptr;
    if(!element || anchor.getName()!="pose_roi_context_anchor" || element->getName()!=target ||
       element->getProbe(anchor.getName())!=probe || !GST_IS_ELEMENT(element->getGObject()))
      throw std::runtime_error("pose context attached SDK anchor mismatch");
    try {state_->attach(GST_ELEMENT(element->getGObject()));} catch(...){state_->close();throw;}
  }
  void close(){state_->close();}
  py::dict snapshot() const {py::dict d;d["copied"]=state_->copied.load();d["expanded"]=state_->expanded.load();
    d["restored"]=state_->restored.load();d["failures"]=state_->failures.load();d["elapsed_ns"]=state_->elapsed_ns.load();return d;}
  ~PoseRoiContextController(){close();}
};

struct JsonPayload {
  std::string json;
};

bool pose_meta_ext_debug_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char* raw = std::getenv("NOESIS_POSE_META_EXT_DEBUG");
    std::string s = raw ? std::string(raw) : std::string();
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    enabled = (!s.empty() && s != "0" && s != "false" && s != "no" && s != "off") ? 1 : 0;
  }
  return enabled == 1;
}

size_t pose_meta_max_payload_bytes() {
  static bool initialized = false;
  static size_t max_bytes = 65536U;
  if (!initialized) {
    initialized = true;
    const char* raw = std::getenv("NOESIS_POSE_META_MAX_JSON_BYTES");
    if (raw && *raw) {
      try {
        const unsigned long parsed = std::stoul(std::string(raw));
        if (parsed >= 1024UL) {
          max_bytes = static_cast<size_t>(parsed);
        }
      } catch (...) {
      }
    }
  }
  return max_bytes;
}

bool pose_payload_within_limit(const std::string& payload_json) {
  const size_t max_bytes = pose_meta_max_payload_bytes();
  if (payload_json.size() <= max_bytes) {
    return true;
  }
  if (pose_meta_ext_debug_enabled()) {
    std::fprintf(
        stderr,
        "[noesis_pose_meta_ext] payload too large: size=%zu max=%zu\n",
        payload_json.size(),
        max_bytes);
  }
  return false;
}

int pose_meta_type() {
  static const int type = static_cast<int>(
      nvds_get_user_meta_type(const_cast<gchar*>("NOESIS.POSE_FEATURES")));
  return type;
}

void* pose_meta_copy(void* opaque_meta, void* /*user_data*/) {
  if (opaque_meta == nullptr) return nullptr;
  // Wrap the callback's opaque metadata through the public DS9 API instead of
  // depending on the underlying metadata structure layout.
  deepstream::UserMetadata user_meta(opaque_meta);
  auto* source = static_cast<JsonPayload*>(user_meta.userData());
  return source != nullptr ? static_cast<void*>(new JsonPayload(*source)) : nullptr;
}

void pose_meta_release(void* opaque_meta, void* /*user_data*/) {
  if (opaque_meta == nullptr) return;
  deepstream::UserMetadata user_meta(opaque_meta);
  delete static_cast<JsonPayload*>(user_meta.userData());
}

template <typename OwnerMetadata>
bool attach_pose_json(
    deepstream::BatchMetadata& batch_meta,
    OwnerMetadata& owner_meta,
    const std::string& payload_json,
    bool replace_existing) {
  if (!pose_payload_within_limit(payload_json)) {
    return false;
  }

  const int meta_type = pose_meta_type();
  if (replace_existing) {
    // DS9 exposes public append but not removal. Updating our owned payload in
    // place preserves one logical record without reaching into SDK internals.
    bool found = false;
    bool updated = false;
    owner_meta.iterate(
        [&](const deepstream::UserMetadata& user_meta) {
          if (found) return;
          found = true;
          auto* payload = static_cast<JsonPayload*>(user_meta.userData());
          if (payload != nullptr) {
            payload->json = payload_json;
            updated = true;
          }
        },
        meta_type);
    if (found) {
      return updated;
    }
  }

  deepstream::UserMetadata user_meta(nullptr);
  if (!batch_meta.acquire(user_meta)) {
    return false;
  }
  auto payload = std::make_unique<JsonPayload>(JsonPayload{payload_json});
  user_meta.setMetaType(meta_type);
  user_meta.setUserData(payload.get(), pose_meta_copy, pose_meta_release);
  payload.release();
  owner_meta.append(user_meta);
  return true;
}

void throw_on_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(unsigned int target_device) {
    throw_on_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice failed");
    if (previous_device_ != static_cast<int>(target_device)) {
      throw_on_cuda(cudaSetDevice(static_cast<int>(target_device)), "cudaSetDevice failed");
      restore_ = true;
    }
  }

  ~CudaDeviceGuard() {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

 private:
  int previous_device_ = 0;
  bool restore_ = false;
};

class TensorLayerMap {
 public:
  // DS9 getLayers() allocates observer Tensor wrappers. Own the wrappers for
  // this synchronous probe call; the inference buffers themselves remain SDK-owned.
  explicit TensorLayerMap(deepstream::TensorOutputUserMetadata& tensor_meta)
      : layers_(tensor_meta.getLayers()) {}

  ~TensorLayerMap() {
    std::unordered_set<deepstream::Tensor*> released;
    for (const auto& entry : layers_) {
      if (entry.second != nullptr && released.insert(entry.second).second) {
        delete entry.second;
      }
    }
  }

  TensorLayerMap(const TensorLayerMap&) = delete;
  TensorLayerMap& operator=(const TensorLayerMap&) = delete;

  const std::unordered_map<std::string, deepstream::Tensor*>& get() const { return layers_; }

 private:
  std::unordered_map<std::string, deepstream::Tensor*> layers_;
};

std::optional<uint64_t> tensor_num_elements(const deepstream::Tensor& tensor) {
  const deepstream::TensorShape shape = tensor.shape();
  if (shape.empty() || shape.size() != static_cast<size_t>(tensor.rank())) {
    return std::nullopt;
  }
  uint64_t count = 1ULL;
  for (uint64_t dim : shape) {
    if (dim == 0ULL || count > std::numeric_limits<uint64_t>::max() / dim) {
      return std::nullopt;
    }
    count *= dim;
  }
  return count;
}

bool tensor_is_contiguous(const deepstream::Tensor& tensor) {
  const deepstream::TensorShape shape = tensor.shape();
  if (shape.empty() || shape.size() != static_cast<size_t>(tensor.rank())) {
    return false;
  }
  uint64_t expected_stride = 1ULL;
  for (size_t offset = 0; offset < shape.size(); ++offset) {
    const size_t axis = shape.size() - 1U - offset;
    if (tensor.stride(static_cast<unsigned int>(axis)) != expected_stride) {
      return false;
    }
    if (shape[axis] == 0ULL ||
        expected_stride > std::numeric_limits<uint64_t>::max() / shape[axis]) {
      return false;
    }
    expected_stride *= shape[axis];
  }
  return true;
}

std::string tensor_shape_string(const deepstream::Tensor& tensor) {
  const deepstream::TensorShape shape = tensor.shape();
  std::string rendered = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    rendered += std::to_string(shape[i]);
    if (i + 1U < shape.size()) rendered += ",";
  }
  rendered += "]";
  return rendered;
}

std::vector<float> copy_float32_tensor_to_host(const deepstream::Tensor& tensor) {
  if (tensor.dtype() != deepstream::Tensor::FLOAT || tensor.bits() != 32U) {
    throw std::runtime_error("Pose tensor must be FLOAT32");
  }
  if (!tensor_is_contiguous(tensor)) {
    throw std::runtime_error("Pose tensor must be contiguous");
  }
  const auto count_opt = tensor_num_elements(tensor);
  if (!count_opt.has_value() || *count_opt == 0ULL ||
      *count_opt > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
    throw std::runtime_error("Pose tensor shape is invalid");
  }
  const size_t count = static_cast<size_t>(*count_opt);
  const uint64_t expected_bytes = static_cast<uint64_t>(count) * sizeof(float);
  if (tensor.size() != expected_bytes || tensor.data() == nullptr) {
    throw std::runtime_error("Pose tensor storage does not match its public shape");
  }

  std::vector<float> values(count);
  switch (tensor.deviceType()) {
    case deepstream::Tensor::CPU:
      std::memcpy(values.data(), tensor.data(), static_cast<size_t>(expected_bytes));
      break;
    case deepstream::Tensor::GPU: {
      CudaDeviceGuard guard(tensor.deviceId());
      throw_on_cuda(
          cudaMemcpy(
              values.data(),
              tensor.data(),
              static_cast<size_t>(expected_bytes),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy pose tensor failed");
      break;
    }
    default:
      throw std::runtime_error("Pose tensor has no supported device");
  }
  return values;
}

struct PoseMatrixAccessor {
  int rows = 0;
  int cols = 0;
  std::function<float(int, int)> at;
};

constexpr size_t kRFDETRQueryCount = 100U;
constexpr size_t kRFDETRClassCount = 2U;
constexpr size_t kRFDETRKeypointSlots = 34U;
constexpr size_t kRFDETRPersonKeypointOffset = 17U;
constexpr size_t kRFDETRPersonKeypointCount = 17U;
constexpr size_t kRFDETRKeypointValues = 8U;
constexpr float kRFDETRNetworkWidth = 576.0f;
constexpr float kRFDETRNetworkHeight = 576.0f;
constexpr double kRFDETRKeypointTraceAlpha = 0.2;
constexpr double kRFDETRMinimumLogWeightArgument = 1.0e-12;

struct RFDETRKeypointTensors {
  std::vector<float> boxes;
  std::vector<float> logits;
  std::vector<float> keypoints;
};

struct RFDETRObjectCandidate {
  size_t frame_object_index = 0U;
  unsigned long object_id = 0UL;
  NvOSD_RectParams rect{};
};

struct RFDETRQueryCandidate {
  size_t query_index = 0U;
  float base_score = 0.0f;
  float score = 0.0f;
  float left = 0.0f;
  float top = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
};

float clipf(float value, float lower, float upper);

bool tensor_has_shape(
    const deepstream::Tensor& tensor,
    const std::vector<uint64_t>& expected) {
  deepstream::TensorShape shape = tensor.shape();
  if (shape.size() == expected.size() + 1U && shape.front() == 1U) {
    shape.erase(shape.begin());
  }
  return shape == expected;
}

deepstream::Tensor* require_named_tensor(
    const std::unordered_map<std::string, deepstream::Tensor*>& layers,
    const char* name,
    const std::vector<uint64_t>& expected_shape) {
  const auto found = layers.find(name);
  if (found == layers.end() || found->second == nullptr) {
    throw std::runtime_error(
        std::string("RF-DETR keypoint tensor meta is missing layer ") + name);
  }
  if (!tensor_has_shape(*found->second, expected_shape)) {
    throw std::runtime_error(
        std::string("RF-DETR keypoint tensor ") + name +
        " has unexpected shape " + tensor_shape_string(*found->second));
  }
  return found->second;
}

RFDETRKeypointTensors read_rfdetr_keypoint_tensors(
    const deepstream::FrameMetadata& frame_meta,
    int gie_id) {
  std::optional<RFDETRKeypointTensors> decoded;
  unsigned int matching_meta_count = 0U;
  std::optional<std::string> decode_error;
  frame_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        deepstream::TensorOutputUserMetadata tensor_meta(user_meta);
        if (!tensor_meta ||
            tensor_meta.uniqueId() != static_cast<unsigned int>(gie_id)) {
          return;
        }
        ++matching_meta_count;
        if (decoded.has_value() || decode_error.has_value()) return;
        try {
          TensorLayerMap owned_layers(tensor_meta);
          const auto& layers = owned_layers.get();
          deepstream::Tensor* boxes = require_named_tensor(
              layers, "dets", {kRFDETRQueryCount, 4U});
          deepstream::Tensor* logits = require_named_tensor(
              layers, "labels", {kRFDETRQueryCount, kRFDETRClassCount});
          deepstream::Tensor* keypoints = require_named_tensor(
              layers,
              "keypoints",
              {
                  kRFDETRQueryCount,
                  kRFDETRKeypointSlots,
                  kRFDETRKeypointValues,
              });
          decoded = RFDETRKeypointTensors{
              copy_float32_tensor_to_host(*boxes),
              copy_float32_tensor_to_host(*logits),
              copy_float32_tensor_to_host(*keypoints),
          };
        } catch (const std::exception& error) {
          decode_error = error.what();
        }
      },
      NVDSINFER_TENSOR_OUTPUT_META);

  if (matching_meta_count == 0U) {
    throw std::runtime_error(
        "RF-DETR keypoint PGIE frame tensor meta is missing");
  }
  if (matching_meta_count != 1U) {
    throw std::runtime_error(
        "RF-DETR keypoint PGIE requires exactly one frame tensor meta record");
  }
  if (decode_error.has_value()) {
    throw std::runtime_error(*decode_error);
  }
  if (!decoded.has_value() ||
      decoded->boxes.size() != kRFDETRQueryCount * 4U ||
      decoded->logits.size() != kRFDETRQueryCount * kRFDETRClassCount ||
      decoded->keypoints.size() !=
          kRFDETRQueryCount * kRFDETRKeypointSlots *
              kRFDETRKeypointValues) {
    throw std::runtime_error(
        "RF-DETR keypoint PGIE tensor storage contract is invalid");
  }
  return std::move(*decoded);
}

float sigmoid_stable(float value) {
  if (value >= 0.0f) {
    const float z = std::exp(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(value);
  return z / (1.0f + z);
}

double log_add_exp_stable(double left, double right) {
  if (left == -std::numeric_limits<double>::infinity()) return right;
  if (right == -std::numeric_limits<double>::infinity()) return left;
  const double maximum = std::max(left, right);
  return maximum + std::log1p(std::exp(-std::abs(left - right)));
}

float rfdetr_official_fused_person_score(
    const RFDETRKeypointTensors& tensors,
    size_t query_index,
    float base_score) {
  const size_t query_base =
      query_index * kRFDETRKeypointSlots * kRFDETRKeypointValues;
  double weighted_log_trace_sum =
      -std::numeric_limits<double>::infinity();
  double log_weight_sum = -std::numeric_limits<double>::infinity();
  for (size_t keypoint = 0U;
       keypoint < kRFDETRPersonKeypointCount;
       ++keypoint) {
    const size_t offset =
        query_base +
        (kRFDETRPersonKeypointOffset + keypoint) *
            kRFDETRKeypointValues;
    const double findable_logit =
        static_cast<double>(tensors.keypoints[offset + 2U]);
    const double log_l11 =
        static_cast<double>(tensors.keypoints[offset + 4U]);
    const double l21 =
        static_cast<double>(tensors.keypoints[offset + 5U]);
    const double log_l22 =
        static_cast<double>(tensors.keypoints[offset + 6U]);
    if (!std::isfinite(findable_logit) || !std::isfinite(log_l11) ||
        !std::isfinite(l21) || !std::isfinite(log_l22)) {
      throw std::runtime_error(
          "RF-DETR keypoint output contains non-finite score components");
    }

    const double findable = static_cast<double>(
        sigmoid_stable(static_cast<float>(findable_logit)));
    const double log_weight = std::log(
        std::max(findable, kRFDETRMinimumLogWeightArgument));
    const double log_t1 = -2.0 * log_l11;
    const double log_t2 = -2.0 * log_l22;
    const double log_t3 =
        2.0 * std::log(std::max(
                  std::abs(l21), kRFDETRMinimumLogWeightArgument)) +
        log_t1 + log_t2;
    const double log_trace = log_add_exp_stable(
        log_add_exp_stable(log_t1, log_t2), log_t3);
    weighted_log_trace_sum = log_add_exp_stable(
        weighted_log_trace_sum, log_trace + log_weight);
    log_weight_sum =
        log_add_exp_stable(log_weight_sum, log_weight);
  }

  const double log_mean_trace =
      weighted_log_trace_sum - log_weight_sum;
  const double fused =
      static_cast<double>(base_score) *
      std::exp(-kRFDETRKeypointTraceAlpha * log_mean_trace);
  const float fused_score = static_cast<float>(fused);
  if (!std::isfinite(log_mean_trace) || !std::isfinite(fused) ||
      !std::isfinite(fused_score) || fused_score < 0.0f) {
    throw std::runtime_error(
        "RF-DETR keypoint uncertainty-fused person score is invalid");
  }
  return fused_score;
}

float rect_iou(
    float left_a,
    float top_a,
    float width_a,
    float height_a,
    float left_b,
    float top_b,
    float width_b,
    float height_b) {
  if (!(width_a > 0.0f && height_a > 0.0f &&
        width_b > 0.0f && height_b > 0.0f)) {
    return 0.0f;
  }
  const float right_a = left_a + width_a;
  const float bottom_a = top_a + height_a;
  const float right_b = left_b + width_b;
  const float bottom_b = top_b + height_b;
  const float intersection_width =
      std::max(0.0f, std::min(right_a, right_b) - std::max(left_a, left_b));
  const float intersection_height =
      std::max(0.0f, std::min(bottom_a, bottom_b) - std::max(top_a, top_b));
  const float intersection = intersection_width * intersection_height;
  const float union_area =
      width_a * height_a + width_b * height_b - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

std::vector<RFDETRObjectCandidate> collect_rfdetr_person_objects(
    const deepstream::FrameMetadata& frame_meta) {
  std::vector<RFDETRObjectCandidate> objects;
  size_t object_index = 0U;
  frame_meta.iterate(
      [&](const deepstream::ObjectMetadata& object_meta) {
        if (object_meta.classId() == 0U) {
          const NvOSD_RectParams& rect = object_meta.rectParams();
          if (std::isfinite(rect.left) && std::isfinite(rect.top) &&
              std::isfinite(rect.width) && std::isfinite(rect.height) &&
              rect.width > 0.0f && rect.height > 0.0f) {
            objects.push_back(RFDETRObjectCandidate{
                object_index,
                object_meta.objectId(),
                rect,
            });
          }
        }
        ++object_index;
      });
  return objects;
}

std::vector<RFDETRQueryCandidate> collect_rfdetr_person_queries(
    const RFDETRKeypointTensors& tensors,
    float frame_width,
    float frame_height,
    float score_threshold) {
  std::vector<RFDETRQueryCandidate> queries;
  queries.reserve(kRFDETRQueryCount);
  for (size_t query = 0U; query < kRFDETRQueryCount; ++query) {
    const float person_logit =
        tensors.logits[query * kRFDETRClassCount + 1U];
    if (!std::isfinite(person_logit)) {
      throw std::runtime_error(
          "RF-DETR keypoint labels contain a non-finite person logit");
    }
    const float base_score = sigmoid_stable(person_logit);
    const float score = rfdetr_official_fused_person_score(
        tensors, query, base_score);
    if (score < score_threshold) continue;
    const float* box = tensors.boxes.data() + query * 4U;
    const float cx = box[0];
    const float cy = box[1];
    const float width = box[2];
    const float height = box[3];
    if (!std::isfinite(cx) || !std::isfinite(cy) ||
        !std::isfinite(width) || !std::isfinite(height) ||
        width <= 0.0f || height <= 0.0f) {
      throw std::runtime_error(
          "RF-DETR keypoint dets contain an invalid normalized cxcywh row");
    }
    if (width * kRFDETRNetworkWidth < 1.0f ||
        height * kRFDETRNetworkHeight < 1.0f) {
      continue;
    }
    const float x1 = clipf((cx - width * 0.5f) * frame_width, 0.0f, frame_width);
    const float y1 = clipf((cy - height * 0.5f) * frame_height, 0.0f, frame_height);
    const float x2 = clipf((cx + width * 0.5f) * frame_width, 0.0f, frame_width);
    const float y2 = clipf((cy + height * 0.5f) * frame_height, 0.0f, frame_height);
    if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;
    queries.push_back(RFDETRQueryCandidate{
        query,
        base_score,
        score,
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    });
  }
  return queries;
}

py::dict decode_rfdetr_keypoint_query(
    const RFDETRKeypointTensors& tensors,
    const RFDETRQueryCandidate& query,
    const RFDETRObjectCandidate& object,
    float frame_width,
    float frame_height,
    float iou) {
  py::list keypoints_roi;
  py::list keypoints_abs;
  const size_t query_base =
      query.query_index * kRFDETRKeypointSlots * kRFDETRKeypointValues;
  for (size_t keypoint = 0U;
       keypoint < kRFDETRPersonKeypointCount;
       ++keypoint) {
    const size_t offset =
        query_base +
        (kRFDETRPersonKeypointOffset + keypoint) *
            kRFDETRKeypointValues;
    const float raw_x = tensors.keypoints[offset];
    const float raw_y = tensors.keypoints[offset + 1U];
    const float raw_confidence = tensors.keypoints[offset + 2U];
    if (!std::isfinite(raw_x) || !std::isfinite(raw_y) ||
        !std::isfinite(raw_confidence)) {
      throw std::runtime_error(
          "RF-DETR keypoint output contains non-finite active values");
    }
    const float absolute_x = clipf(raw_x * frame_width, 0.0f, frame_width);
    const float absolute_y = clipf(raw_y * frame_height, 0.0f, frame_height);
    const float roi_x = clipf(
        absolute_x - object.rect.left, 0.0f, object.rect.width);
    const float roi_y = clipf(
        absolute_y - object.rect.top, 0.0f, object.rect.height);
    const float confidence = sigmoid_stable(raw_confidence);
    keypoints_roi.append(py::make_tuple(roi_x, roi_y, confidence));
    keypoints_abs.append(
        py::make_tuple(absolute_x, absolute_y, confidence));
  }

  py::dict payload;
  payload["object_index"] = object.frame_object_index;
  payload["object_id"] = object.object_id;
  payload["query_index"] = query.query_index;
  payload["match_iou"] = iou;
  payload["base_score"] = query.base_score;
  payload["score"] = query.score;
  payload["bbox"] = py::make_tuple(
      object.rect.left,
      object.rect.top,
      object.rect.width,
      object.rect.height);
  payload["keypoints_roi"] = std::move(keypoints_roi);
  payload["keypoints_abs"] = std::move(keypoints_abs);
  return payload;
}

py::dict extract_rfdetr_keypoint_matches_impl(
    const deepstream::FrameMetadata& frame_meta,
    int gie_id,
    float score_threshold,
    float min_iou,
    float ambiguity_margin) {
  if (gie_id <= 0) {
    throw std::invalid_argument("RF-DETR keypoint gie_id must be positive");
  }
  if (!std::isfinite(score_threshold) ||
      score_threshold < 0.0f || score_threshold > 1.0f) {
    throw std::invalid_argument(
        "RF-DETR keypoint score threshold must be in [0,1]");
  }
  if (!std::isfinite(min_iou) || min_iou <= 0.0f || min_iou > 1.0f) {
    throw std::invalid_argument(
        "RF-DETR keypoint match IoU must be in (0,1]");
  }
  if (!std::isfinite(ambiguity_margin) ||
      ambiguity_margin < 0.0f || ambiguity_margin >= 1.0f) {
    throw std::invalid_argument(
        "RF-DETR keypoint ambiguity margin must be in [0,1)");
  }

  const unsigned int pipeline_width = frame_meta.pipelineWidth();
  const unsigned int pipeline_height = frame_meta.pipelineHeight();
  if (pipeline_width == 0U || pipeline_height == 0U) {
    throw std::runtime_error(
        "RF-DETR keypoint frame has no pipeline coordinate dimensions");
  }
  const float frame_width = static_cast<float>(pipeline_width);
  const float frame_height = static_cast<float>(pipeline_height);
  const RFDETRKeypointTensors tensors =
      read_rfdetr_keypoint_tensors(frame_meta, gie_id);
  const std::vector<RFDETRObjectCandidate> objects =
      collect_rfdetr_person_objects(frame_meta);
  const std::vector<RFDETRQueryCandidate> queries =
      collect_rfdetr_person_queries(
          tensors, frame_width, frame_height, score_threshold);

  std::vector<std::vector<float>> ious(
      objects.size(), std::vector<float>(queries.size(), 0.0f));
  for (size_t object_index = 0U;
       object_index < objects.size();
       ++object_index) {
    const auto& object = objects[object_index];
    for (size_t query_index = 0U;
         query_index < queries.size();
         ++query_index) {
      const auto& query = queries[query_index];
      ious[object_index][query_index] = rect_iou(
          object.rect.left,
          object.rect.top,
          object.rect.width,
          object.rect.height,
          query.left,
          query.top,
          query.width,
          query.height);
    }
  }

  py::list matches;
  size_t unmatched_objects = 0U;
  size_t ambiguous_objects = 0U;
  for (size_t object_index = 0U;
       object_index < objects.size();
       ++object_index) {
    if (queries.empty()) {
      ++unmatched_objects;
      continue;
    }
    size_t best_query = 0U;
    float best_iou = -1.0f;
    float second_query_iou = -1.0f;
    for (size_t query_index = 0U;
         query_index < queries.size();
         ++query_index) {
      const float iou = ious[object_index][query_index];
      if (iou > best_iou) {
        second_query_iou = best_iou;
        best_iou = iou;
        best_query = query_index;
      } else if (iou > second_query_iou) {
        second_query_iou = iou;
      }
    }
    if (best_iou < min_iou) {
      ++unmatched_objects;
      continue;
    }

    size_t best_object_for_query = 0U;
    float best_object_iou = -1.0f;
    float second_object_iou = -1.0f;
    for (size_t other_object = 0U;
         other_object < objects.size();
         ++other_object) {
      const float iou = ious[other_object][best_query];
      if (iou > best_object_iou) {
        second_object_iou = best_object_iou;
        best_object_iou = iou;
        best_object_for_query = other_object;
      } else if (iou > second_object_iou) {
        second_object_iou = iou;
      }
    }
    const float query_margin =
        best_iou - std::max(0.0f, second_query_iou);
    const float object_margin =
        best_object_iou - std::max(0.0f, second_object_iou);
    if (best_object_for_query != object_index ||
        query_margin < ambiguity_margin ||
        object_margin < ambiguity_margin) {
      ++ambiguous_objects;
      continue;
    }
    matches.append(decode_rfdetr_keypoint_query(
        tensors,
        queries[best_query],
        objects[object_index],
        frame_width,
        frame_height,
        best_iou));
  }

  py::dict diagnostics;
  diagnostics["person_objects"] = objects.size();
  diagnostics["person_queries"] = queries.size();
  diagnostics["matched_objects"] = py::len(matches);
  diagnostics["unmatched_objects"] = unmatched_objects;
  diagnostics["ambiguous_objects"] = ambiguous_objects;
  diagnostics["score_threshold"] = score_threshold;
  diagnostics["min_iou"] = min_iou;
  diagnostics["ambiguity_margin"] = ambiguity_margin;
  diagnostics["pipeline_width"] = pipeline_width;
  diagnostics["pipeline_height"] = pipeline_height;

  py::dict result;
  result["matches"] = std::move(matches);
  result["diagnostics"] = std::move(diagnostics);
  return result;
}

std::optional<PoseMatrixAccessor> build_pose_accessor(
    const deepstream::Tensor& tensor,
    const std::vector<float>& values) {
  const int min_cols = 56;
  const int max_cols = 128;
  const deepstream::TensorShape shape = tensor.shape();
  if (shape.empty()) return std::nullopt;

  int axis = -1;
  int best_distance = std::numeric_limits<int>::max();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > static_cast<uint64_t>(std::numeric_limits<int>::max())) continue;
    const int dim = static_cast<int>(shape[i]);
    if (dim < min_cols || dim > max_cols) continue;
    const int distance = std::abs(dim - 57);
    if (distance < best_distance) {
      best_distance = distance;
      axis = static_cast<int>(i);
    }
  }
  if (axis < 0) return std::nullopt;

  uint64_t rows_u64 = 1ULL;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int>(i) == axis) continue;
    if (shape[i] == 0ULL || rows_u64 > std::numeric_limits<uint64_t>::max() / shape[i]) {
      return std::nullopt;
    }
    rows_u64 *= shape[i];
  }
  if (rows_u64 == 0ULL || rows_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  const int rows = static_cast<int>(rows_u64);
  const int cols = static_cast<int>(shape[static_cast<size_t>(axis)]);

  std::vector<uint64_t> strides(shape.size(), 1ULL);
  for (size_t offset = 1; offset < shape.size(); ++offset) {
    const size_t i = shape.size() - 1U - offset;
    strides[i] = strides[i + 1U] * shape[i + 1U];
  }
  std::vector<int> row_axes;
  row_axes.reserve(shape.size() - 1U);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int>(i) != axis) row_axes.push_back(static_cast<int>(i));
  }

  const float* base = values.data();
  return PoseMatrixAccessor{
      rows,
      cols,
      [base, shape, strides, row_axes, axis](int row, int col) -> float {
        uint64_t index = static_cast<uint64_t>(col) * strides[static_cast<size_t>(axis)];
        int remaining = row;
        for (int j = static_cast<int>(row_axes.size()) - 1; j >= 0; --j) {
          const int current_axis = row_axes[static_cast<size_t>(j)];
          const int dim = static_cast<int>(shape[static_cast<size_t>(current_axis)]);
          const int coordinate = dim > 0 ? remaining % dim : 0;
          remaining = dim > 0 ? remaining / dim : remaining;
          index += static_cast<uint64_t>(coordinate) *
                   strides[static_cast<size_t>(current_axis)];
        }
        return base[index];
      }};
}

std::pair<std::string, deepstream::Tensor*> select_pose_layer(
    const std::unordered_map<std::string, deepstream::Tensor*>& layers) {
  const auto named = layers.find("output0");
  if (named == layers.end()) {
    return {"", nullptr};
  }
  return {named->first, named->second};
}

bool extract_best_pose_row(
    deepstream::TensorOutputUserMetadata& tensor_meta,
    float score_threshold,
    std::vector<float>& row_out,
    const RoiContext* context=nullptr,
    noesis::pose::PoseExtractionDiagnostic* diagnostic=nullptr) {
  TensorLayerMap owned_layers(tensor_meta);
  const auto selected = select_pose_layer(owned_layers.get());
  if (selected.second == nullptr) {
    if(diagnostic) diagnostic->result="output0_missing";
    return false;
  }

  std::vector<float> values = copy_float32_tensor_to_host(*selected.second);
  if (pose_meta_ext_debug_enabled()) {
    std::fprintf(
        stderr,
        "[noesis_pose_meta_ext] uid=%u layer=%s dtype=%d bits=%u device=%d shape=%s score_th=%.3f\n",
        tensor_meta.uniqueId(),
        selected.first.c_str(),
        static_cast<int>(selected.second->dtype()),
        selected.second->bits(),
        static_cast<int>(selected.second->deviceType()),
        tensor_shape_string(*selected.second).c_str(),
        score_threshold);
  }

  auto accessor_opt = build_pose_accessor(*selected.second, values);
  if(diagnostic) diagnostic->result="shape_invalid";
  if (!accessor_opt.has_value()) return false;
  const PoseMatrixAccessor accessor = *accessor_opt;
  if(diagnostic) {diagnostic->rows=accessor.rows;diagnostic->cols=accessor.cols;}
  if (accessor.rows <= 0 || accessor.cols < 56) return false;
  if(diagnostic) diagnostic->shape_valid=true;

  if(context) {
    if(!context->restored || accessor.cols!=57 || accessor.rows>300 || score_threshold!=.25f) {
      if(diagnostic) {diagnostic->shape_valid=false;diagnostic->result="context_shape_invalid";}
      return false;
    }
    const int accepted=noesis::pose::unique_context_row(context->crop,accessor.rows,accessor.cols,
        [&](int row,int col){return accessor.at(row,col);},diagnostic);
    if(accepted<0) return false;
    row_out.resize(57);
    for(int col=0;col<57;++col) row_out[col]=accessor.at(accepted,col);
    return true;
  }

  float best_score = -std::numeric_limits<float>::infinity();
  int best_row = -1;
  for (int row = 0; row < accessor.rows; ++row) {
    const float score = accessor.at(row, 4);
    if (!std::isfinite(score)) continue;
    if (score > best_score) {
      best_score = score;
      best_row = row;
    }
  }
  if (best_row < 0 || best_score < score_threshold) {
    if(diagnostic) diagnostic->result="score_below_threshold";
    if (pose_meta_ext_debug_enabled()) {
      std::fprintf(
          stderr,
          "[noesis_pose_meta_ext] no row over threshold: best=%.4f threshold=%.4f rows=%d cols=%d\n",
          best_score,
          score_threshold,
          accessor.rows,
          accessor.cols);
    }
    return false;
  }

  row_out.resize(static_cast<size_t>(accessor.cols));
  if(diagnostic) diagnostic->result="row_selected";
  for (int col = 0; col < accessor.cols; ++col) {
    row_out[static_cast<size_t>(col)] = accessor.at(best_row, col);
  }
  return true;
}

float clipf(float value, float lower, float upper) {
  if (value < lower) return lower;
  if (value > upper) return upper;
  return value;
}

std::optional<py::dict> decode_pose_keypoints_payload(
    const deepstream::ObjectMetadata& obj_meta,
    int gie_id,
    int model_w,
    int model_h,
    float score_threshold,
    bool letterbox,
    noesis::pose::PoseExtractionDiagnostic* diagnostic=nullptr) {
  if(const auto* owned=current_pose(obj_meta)) {
    if(!owned->tracker_bound || gie_id!=4 || model_w!=640 || model_h!=640 || score_threshold!=.25f || !letterbox ||
       (owned->origin!=1 && owned->origin!=4) || obj_meta.uniqueComponentId()!=owned->origin)
      throw std::runtime_error("current pose original/profile/tracker binding mismatch");
    py::list absolute,relative;const auto& r=obj_meta.rectParams();
    for(unsigned k=0;k<17;++k){const auto& j=owned->value.joints;
      absolute.append(py::make_tuple(j[k*3],j[k*3+1],j[k*3+2]));
      relative.append(py::make_tuple(j[k*3]-r.left,j[k*3+1]-r.top,j[k*3+2]));}
    py::dict p;p["score"]=owned->value.score;p["keypoints_abs"]=absolute;p["keypoints_roi"]=relative;
    if(diagnostic){diagnostic->result="owned_current_ingress";diagnostic->shape_valid=true;diagnostic->rows=1;diagnostic->cols=57;}
    return p;
  }
  const NvOSD_RectParams& rect = obj_meta.rectParams();
  const auto* context=roi_context(obj_meta);
  if(diagnostic) {diagnostic->context_present=context!=nullptr;diagnostic->result="context_binding_invalid";}
  if(context && (!context->restored || context->gie!=guint(gie_id) || model_w!=640 || model_h!=640 ||
      !letterbox || context->object!=obj_meta.objectId() ||
      rect.left!=context->crop.original_left || rect.top!=context->crop.original_top ||
      rect.width!=context->crop.original_width || rect.height!=context->crop.original_height)) return std::nullopt;
  if(diagnostic) {diagnostic->context_valid=true;diagnostic->result="crop_invalid";}
  const auto geometry = context ? std::optional<noesis::pose::CropGeometry>(context->crop) : noesis::pose::make_crop_geometry(
      rect.left, rect.top, rect.width, rect.height, model_w, model_h, letterbox);
  if (!geometry.has_value()) return std::nullopt;
  if(diagnostic) diagnostic->result="tensor_missing";

  std::vector<float> best_row;
  std::optional<std::string> matched_error;
  unsigned matching_tensors=0;
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (!context && !best_row.empty()) return;
        deepstream::TensorOutputUserMetadata tensor_meta(user_meta);
        if (!tensor_meta || tensor_meta.uniqueId() != static_cast<unsigned int>(gie_id)) {
          return;
        }
        ++matching_tensors;
        if(diagnostic) diagnostic->tensor_count=matching_tensors;
        try {
          (void)extract_best_pose_row(tensor_meta, score_threshold, best_row,context,diagnostic);
        } catch (const std::exception& exc) {
          matched_error = exc.what();
          if(diagnostic) diagnostic->result="tensor_exception";
        }
      },
      NVDSINFER_TENSOR_OUTPUT_META);
  if(context && matching_tensors!=1) {
    if(diagnostic) diagnostic->result=matching_tensors==0?"tensor_missing":"tensor_count_invalid";
    return std::nullopt;
  }
  if (best_row.empty()) {
    if (matched_error.has_value() && pose_meta_ext_debug_enabled()) {
      std::fprintf(stderr, "[noesis_pose_meta_ext] %s\n", matched_error->c_str());
    }
    return std::nullopt;
  }

  constexpr size_t pose_values = 17U * 3U;
  if (best_row.size() < pose_values + 5U) {
    if(diagnostic) diagnostic->result="row_width_invalid";
    return std::nullopt;
  }

  const float score = best_row[4];
  const size_t key_start = best_row.size() - pose_values;

  float max_xywh = 0.0f;
  for (size_t i = 0; i < 4U && i < best_row.size(); ++i) {
    max_xywh = std::max(max_xywh, std::fabs(best_row[i]));
  }
  const bool normalized = context ? false : max_xywh <= 2.0f;

  py::list keypoints_roi;
  py::list keypoints_abs;
  for (size_t keypoint = 0; keypoint < 17U; ++keypoint) {
    const float x = best_row[key_start + keypoint * 3U];
    const float y = best_row[key_start + keypoint * 3U + 1U];
    const float confidence = best_row[key_start + keypoint * 3U + 2U];

    const auto point = noesis::pose::map_keypoint(
        *geometry, x, y, confidence, normalized);
    if (!point.has_value()) {
      if(diagnostic) diagnostic->result="keypoint_mapping_invalid";
      return std::nullopt;
    }
    keypoints_roi.append(py::make_tuple(point->roi_x, point->roi_y, point->confidence));
    keypoints_abs.append(py::make_tuple(
        point->absolute_x, point->absolute_y, point->confidence));
  }

  py::dict payload;
  payload["score"] = score;
  payload["keypoints_roi"] = std::move(keypoints_roi);
  payload["keypoints_abs"] = std::move(keypoints_abs);
  if(diagnostic) diagnostic->result="decoded";
  return payload;
}

// One existing SGIE, before exclusion/NvDCF. Temporary proposals have their own
// origin, no detection confidence and no tracker identity; only CURRENT NN rows
// become UID4 detections. No SDK pointer, surface or tensor is cached here.
class SinglePassPoseState : public std::enable_shared_from_this<SinglePassPoseState> {
  static constexpr guint proposal_uid=0x4e50524fU;
  struct CacheEntry {guint origin;gint frame;};
  struct Pad {GstPad* pad;gulong id;};
  struct Context {std::shared_ptr<SinglePassPoseState> self;unsigned stage;};
  std::vector<Pad> pads_;
  std::mutex state_mutex_;
  noesis::pose::HintBank hints_;
  noesis::pose::IngressInspection inspection_;
  std::mutex inspection_mutex_;
  std::atomic<unsigned> inspection_busy_{0};
  std::map<std::pair<guint,guint64>,CacheEntry> origins_;
  double duplicate_iou_;
  std::atomic<bool> closed_{false};
  std::atomic<guint64> fresh_{0},proposals_{0},promoted_{0},suppressed_{0};
  static noesis::pose::Box box(const NvDsObjectMeta& o){const auto& r=o.rect_params;return {r.left,r.top,r.width,r.height};}
  static void release(gpointer p){delete static_cast<Context*>(p);}
  static GstPadProbeReturn callback(GstPad* pad,GstPadProbeInfo* info,gpointer data) {
    auto c=*static_cast<Context*>(data);
    try {
      if(GST_PAD_PROBE_INFO_TYPE(info)&GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
        auto* e=GST_PAD_PROBE_INFO_EVENT(info);const auto type=GST_EVENT_TYPE(e);
        guint source=0;
        if(type==GstEventType(GST_NVEVENT_STREAM_RESET)){gst_nvevent_parse_stream_reset(e,&source);c.self->reset_source(source);}
        else if(type==GstEventType(GST_NVEVENT_PAD_DELETED)){gst_nvevent_parse_pad_deleted(e,&source);c.self->reset_source(source);}
        else if(type==GstEventType(GST_NVEVENT_STREAM_EOS)){gst_nvevent_parse_stream_eos(e,&source);c.self->reset_source(source);}
        else if(type==GstEventType(GST_NVEVENT_STREAM_START)){gchar* id=nullptr;gst_nvevent_parse_stream_start(e,&source,&id);g_free(id);c.self->reset_source(source);}
        else if(type==GST_EVENT_STREAM_START || type==GST_EVENT_SEGMENT || type==GST_EVENT_FLUSH_STOP || type==GST_EVENT_EOS)c.self->reset_all();
        return GST_PAD_PROBE_OK;
      }
      auto* b=GST_PAD_PROBE_INFO_BUFFER(info);
      if(c.stage==0){auto* copy=private_pose_buffer(b,true);GST_PAD_PROBE_INFO_DATA(info)=copy;gst_buffer_unref(b);c.self->prepare(copy);}
      else if(c.stage==1)c.self->ingress(b);
      else c.self->observe(b);
      return GST_PAD_PROBE_OK;
    }catch(const std::exception& e){auto* p=gst_pad_get_parent_element(pad);
      if(p){GST_ELEMENT_ERROR(p,STREAM,FAILED,("single-pass pose ingress integrity failure"),("%s",e.what()));gst_object_unref(p);}
      return GST_PAD_PROBE_DROP;}
  }
  void install(GstElement* element,const char* name,unsigned stage,bool events=false) {
    auto* pad=gst_element_get_static_pad(element,name);if(!pad)throw std::runtime_error("single-pass pose static pad missing");
    auto type=GstPadProbeType(GST_PAD_PROBE_TYPE_BUFFER|(events?GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM:0));
    auto id=gst_pad_add_probe(pad,type,callback,new Context{shared_from_this(),stage},release);
    if(!id){gst_object_unref(pad);throw std::runtime_error("single-pass pose probe attachment failed");}pads_.push_back({pad,id});
  }
  static GstElement* target(deepstream::CustomObject& anchor,const std::string& expected,const char* name) {
    auto* p=dynamic_cast<deepstream::BufferProbe*>(&anchor);auto* e=p?p->getTarget():nullptr;
    if(!e || anchor.getName()!=name || e->getName()!=expected || e->getProbe(name)!=p || !GST_IS_ELEMENT(e->getGObject()))
      throw std::runtime_error("single-pass pose public anchor mismatch");
    return GST_ELEMENT(e->getGObject());
  }
 public:
  SinglePassPoseState(double iou,unsigned count,unsigned shadow):hints_(count,shadow),
      inspection_(std::getenv("NOESIS_POSE_INGRESS_INSPECT")),duplicate_iou_(iou) {
    if(!std::isfinite(iou) || iou<=0 || iou>=1)throw std::runtime_error("selected tracker duplicate threshold invalid");
  }
  void attach(deepstream::CustomObject& pose,const std::string& pose_name,
      deepstream::CustomObject& tracker,const std::string& tracker_name) {
    try{auto* p=target(pose,pose_name,"pose_roi_context_anchor");
      gboolean tensor_input=FALSE;g_object_get(p,"input-tensor-meta",&tensor_input,nullptr);
      if(tensor_input)throw std::runtime_error("pose cannot omit a still-required preprocess tensor");
      auto* t=target(tracker,tracker_name,"pose_tracker_ingress_anchor");
      install(p,"sink",0,true);install(p,"src",1);install(t,"src",2);
    }catch(...){close();throw;}
  }
  void reset_all(){std::lock_guard<std::mutex> lock(state_mutex_);hints_.reset();origins_.clear();}
  void reset_source(unsigned source){std::lock_guard<std::mutex> lock(state_mutex_);hints_.reset(source);
    for(auto i=origins_.begin();i!=origins_.end();)if(i->first.first==source)i=origins_.erase(i);else++i;}
  void prepare(GstBuffer* buffer) {
    auto* batch=gst_buffer_get_nvds_batch_meta(buffer);if(!batch)throw std::runtime_error("pose ingress batch missing");
    for(auto* f=batch->frame_meta_list;f;f=f->next){auto* frame=static_cast<NvDsFrameMeta*>(f->data);
      if(frame->frame_num<0 || frame->buf_pts==GST_CLOCK_TIME_NONE || frame->pipeline_width!=1920 || frame->pipeline_height!=1080)
        throw std::runtime_error("pose ingress original raster/frame unavailable");
      std::vector<noesis::pose::Box> current,prior;
      for(auto* n=frame->obj_meta_list;n;n=n->next){auto* o=static_cast<NvDsObjectMeta*>(n->data);
        if(o->class_id!=0)continue;
        if(o->unique_component_id!=1)throw std::runtime_error("pre-pose object origin is not actual PGIE");
        current.push_back(box(*o));}
      {std::lock_guard<std::mutex> lock(state_mutex_);const auto before=hints_.epoch(frame->source_id);
        prior=hints_.hints(frame->source_id,frame->frame_num,frame->buf_pts);
        if(before!=hints_.epoch(frame->source_id))
          for(auto i=origins_.begin();i!=origins_.end();)if(i->first.first==frame->source_id)i=origins_.erase(i);else++i;}
      for(const auto& b:prior){if(noesis::pose::duplicate_box(b,current,duplicate_iou_))continue;
        if(current.size()>=kContextObjects)throw std::runtime_error("pose inference ROI bound");
        auto* o=nvds_acquire_obj_meta_from_pool(batch);if(!o)throw std::runtime_error("pose proposal pool exhausted");
        o->class_id=0;o->unique_component_id=proposal_uid;o->object_id=UNTRACKED_OBJECT_ID;o->confidence=-.1f;
        o->rect_params.left=b[0];o->rect_params.top=b[1];o->rect_params.width=b[2];o->rect_params.height=b[3];
        nvds_add_obj_meta_to_frame(frame,o,nullptr);current.push_back(b);++proposals_;}
      deepstream::BatchMetadata wrapped_batch(batch);
      for(auto* n=frame->obj_meta_list;n;n=n->next){auto* o=static_cast<NvDsObjectMeta*>(n->data);if(o->class_id!=0)continue;
        const auto b=box(*o);if(b[2]<16 || b[3]<16)continue;
        auto crop=noesis::pose::context_crop(b[0],b[1],b[2],b[3],1920,1080);
        if(!crop)crop=noesis::pose::make_crop_geometry(b[0],b[1],b[2],b[3],640,640,true);
        if(!crop)throw std::runtime_error("pose ingress current crop invalid");
        deepstream::UserMetadata meta(nullptr);if(!wrapped_batch.acquire(meta))throw std::runtime_error("pose crop pool exhausted");
        auto* c=new RoiContext{*crop,frame->source_id,frame->frame_num,4,frame->buf_pts,o->object_id,false};
        meta.setMetaType(roi_context_type());meta.setUserData(c,roi_context_copy,roi_context_release);
        deepstream::ObjectMetadata w(o);w.append(meta);
        o->rect_params.left=crop->source_left;o->rect_params.top=crop->source_top;
        o->rect_params.width=crop->source_width;o->rect_params.height=crop->source_height;}
    }
  }
  void ingress(GstBuffer* buffer) {
    auto* batch=gst_buffer_get_nvds_batch_meta(buffer);if(!batch)throw std::runtime_error("pose output batch missing");
    for(auto* f=batch->frame_meta_list;f;f=f->next){auto* frame=static_cast<NvDsFrameMeta*>(f->data);
      struct Output{noesis::pose::CurrentCandidate c;unsigned inspection_token;noesis::pose::CropGeometry crop;bool parent;};std::vector<Output> outputs;
      std::vector<noesis::pose::Box> detections,known_parent_rows;std::vector<NvDsObjectMeta*> remove;
      // Snapshot the complete CURRENT PGIE roster ONCE, before any crop is
      // restored. Include parents without usable pose/context; hints are not
      // parents. Only owned boxes survive this pass, never SDK object pointers.
      std::vector<noesis::pose::Box> current_parents;current_parents.reserve(kContextObjects);
      bool parent_roster_complete=true;
      for(auto* n=frame->obj_meta_list;n;n=n->next){auto* o=static_cast<NvDsObjectMeta*>(n->data);
        if(o->class_id!=0 || o->unique_component_id==proposal_uid)continue;
        if(o->unique_component_id!=1)throw std::runtime_error("pose output unexpected input origin");
        deepstream::ObjectMetadata original(o);auto* receipt=roi_context(original);
        if(receipt && (receipt->source!=frame->source_id || receipt->frame!=frame->frame_num ||
            receipt->pts!=frame->buf_pts || receipt->restored))
          throw std::runtime_error("pose output current crop receipt mismatch");
        const auto b=receipt?noesis::pose::Box{receipt->crop.original_left,receipt->crop.original_top,
            receipt->crop.original_width,receipt->crop.original_height}:box(*o);
        if(!noesis::pose::current_parent_box_valid(b) || current_parents.size()>=kContextObjects)
          parent_roster_complete=false;
        else current_parents.push_back(b);
      }
      guint64 generation;{std::lock_guard<std::mutex> lock(state_mutex_);generation=hints_.epoch(frame->source_id);}
      // Optional inspection must never wait for snapshot or another stage.
      std::unique_lock<std::mutex> inspection_lock(inspection_mutex_,std::defer_lock);
      const bool inspecting=inspection_.selected(frame->source_id,frame->frame_num) && inspection_lock.try_lock();
      if(inspecting)inspection_.mark(frame->source_id,frame->frame_num);
      else if(inspection_.selected(frame->source_id,frame->frame_num))++inspection_busy_;
      unsigned crop_id=0;
      for(auto* n=frame->obj_meta_list;n;n=n->next){auto* o=static_cast<NvDsObjectMeta*>(n->data);if(o->class_id!=0)continue;
        const auto current_crop=crop_id++;
        const bool proposal=o->unique_component_id==proposal_uid;
        if(!proposal && o->unique_component_id!=1)throw std::runtime_error("pose output unexpected input origin");
        deepstream::ObjectMetadata w(o);auto* ctx=roi_context(w);
        if(proposal)remove.push_back(o);
        if(!ctx){if(!proposal)detections.push_back(box(*o));continue;}
        if(ctx->source!=frame->source_id || ctx->frame!=frame->frame_num || ctx->pts!=frame->buf_pts || ctx->restored)
          throw std::runtime_error("pose output current crop receipt mismatch");
        const auto& c=ctx->crop;o->rect_params.left=c.original_left;o->rect_params.top=c.original_top;
        o->rect_params.width=c.original_width;o->rect_params.height=c.original_height;ctx->restored=true;
        std::vector<noesis::pose::CurrentCandidate> rows;unsigned tensors=0;int parent_row=-1;
        w.iterate([&](const deepstream::UserMetadata& u){deepstream::TensorOutputUserMetadata t(u);
          if(!t || t.uniqueId()!=4)return;++tensors;
          TensorLayerMap map(t);const auto selected=select_pose_layer(map.get());if(!selected.second)return;
          auto values=copy_float32_tensor_to_host(*selected.second);auto a=build_pose_accessor(*selected.second,values);
          if(!a)throw std::runtime_error("pose ingress accessor invalid");
          rows=noesis::pose::current_candidates(c,a->rows,a->cols,[&](int r,int col){return a->at(r,col);});
          if(!proposal)parent_row=noesis::pose::unique_context_row(c,a->rows,a->cols,
              [&](int r,int col){return a->at(r,col);},nullptr,
              parent_roster_complete?&current_parents:nullptr);
        },NVDSINFER_TENSOR_OUTPUT_META);
        if(tensors>1)throw std::runtime_error("pose ingress duplicate UID4 tensors");
        std::array<unsigned,300> tokens{};
        if(inspecting)for(const auto& row:rows)tokens[row.row]=inspection_.add(frame->source_id,frame->frame_num,
            frame->buf_pts,frame->batch_id,generation,current_crop,proposal,parent_row,c,row);
        // Keep normal parent correspondence, not the strongest unrelated body.
        if(!proposal){detections.push_back(box(*o));
          for(auto row:rows)if(row.row==parent_row){
            // Preserve the existing parent's exact clipping/ROI mapping. New
            // child detections retain their own unclipped NN body instead.
            for(unsigned k=0;k<17;++k){row.joints[k*3]=std::clamp<double>(row.joints[k*3],c.original_left,c.original_left+c.original_width);
              row.joints[k*3+1]=std::clamp<double>(row.joints[k*3+1],c.original_top,c.original_top+c.original_height);}
            attach_current_pose(batch,frame,o,row,1,generation,tokens[row.row]);
            if(inspecting)inspection_.disposition(tokens[row.row],1);break;}}
        for(const auto& row:rows){
          // Parent correspondence is stronger than detector-box IoU. Never
          // re-promote its selected row, including a contained small body.
          const bool parent=noesis::pose::selected_parent_row(row.row,parent_row,proposal);
          if(parent)known_parent_rows.push_back(row.box);
          outputs.push_back({row,tokens[row.row],c,parent});}
      }
      // Remove even proposals with absent/invalid NN output. Never give a hint
      // (or its prior raw ID) to the tracker, exclusion, Swin or canonical owner.
      for(auto* o:remove)nvds_remove_obj_meta_from_frame(frame,o);
      std::stable_sort(outputs.begin(),outputs.end(),[](const Output& a,const Output& b){return a.c.score>b.c.score;});
      // Pointers refer ONLY to this owned, completed scalar vector, never SDK
      // wrappers or another frame. Current PGIE parent anatomy is preferred;
      // accepted extra rows can subsequently suppress their own cross-crop copy.
      std::vector<const Output*> anatomy;
      for(const auto& output:outputs)if(output.parent)anatomy.push_back(&output);
      for(const auto& output:outputs){const auto& c=output.c;
        if(output.parent)continue;
        if(!noesis::pose::extra_crop_supported(c.box,output.crop)){
          // 6: crop completeness unproven; retain the exact inspection row.
          if(inspecting)inspection_.disposition(output.inspection_token,6);continue;}
        if(noesis::pose::duplicate_box(c.box,detections,duplicate_iou_) ||
           noesis::pose::duplicate_box(c.box,known_parent_rows,duplicate_iou_)){++suppressed_;if(inspecting)inspection_.disposition(output.inspection_token,2);continue;}
        if(noesis::pose::unique_visible_head(c,output.crop,anatomy)){
          ++suppressed_;if(inspecting)inspection_.disposition(output.inspection_token,5);continue;}
        bool torso=true;for(int k:{5,6,11,12})if(c.joints[k*3+2]<.35f)torso=false;
        if(!torso){if(inspecting)inspection_.disposition(output.inspection_token,3);continue;}
        if(detections.size()>=hints_.targets)throw std::runtime_error("pose detection target bound");
        auto* o=nvds_acquire_obj_meta_from_pool(batch);if(!o)throw std::runtime_error("pose detection pool exhausted");
        o->class_id=0;o->unique_component_id=4;o->object_id=UNTRACKED_OBJECT_ID;o->confidence=c.score;
        o->rect_params.left=c.box[0];o->rect_params.top=c.box[1];o->rect_params.width=c.box[2];o->rect_params.height=c.box[3];
        o->detector_bbox_info.org_bbox_coords={float(c.box[0]),float(c.box[1]),float(c.box[2]),float(c.box[3])};
        g_strlcpy(o->obj_label,"person",MAX_LABEL_SIZE);attach_current_pose(batch,frame,o,c,4,generation,output.inspection_token);
        if(inspecting)inspection_.disposition(output.inspection_token,4);
        nvds_add_obj_meta_to_frame(frame,o,nullptr);detections.push_back(c.box);++promoted_;
        anatomy.push_back(&output);
      }
      if(std::any_of(outputs.begin(),outputs.end(),[](const Output& o){return !o.parent;}))frame->bInferDone=TRUE;
    }
  }
  void observe(GstBuffer* buffer) {
    auto* batch=gst_buffer_get_nvds_batch_meta(buffer);if(!batch)throw std::runtime_error("pose tracker batch missing");
    std::lock_guard<std::mutex> lock(state_mutex_);
    for(auto* f=batch->frame_meta_list;f;f=f->next){auto* frame=static_cast<NvDsFrameMeta*>(f->data);std::vector<noesis::pose::Box> boxes;
      for(auto i=origins_.begin();i!=origins_.end();)if(i->first.first==frame->source_id &&
        frame->frame_num-i->second.frame>gint(hints_.frame_limit))i=origins_.erase(i);else++i;
      for(auto* n=frame->obj_meta_list;n;n=n->next){auto* o=static_cast<NvDsObjectMeta*>(n->data);if(o->class_id!=0)continue;
        if(o->object_id==UNTRACKED_OBJECT_ID || o->unique_component_id==proposal_uid)throw std::runtime_error("proposal/untracked object escaped NvDCF");
        const auto key=std::make_pair(frame->source_id,o->object_id);deepstream::ObjectMetadata w(o);auto* p=current_pose(w);
        if(p){if(p->source!=frame->source_id || p->slot!=frame->batch_id || p->frame!=frame->frame_num ||
          p->pts!=frame->buf_pts || p->generation!=hints_.epoch(frame->source_id) || p->tracker_bound || p->origin!=o->unique_component_id)
            throw std::runtime_error("pose tracker original receipt conflict");
          p->tracker_bound=true;origins_[key]={p->origin,frame->frame_num};++fresh_;
          if(p->inspection_token){std::unique_lock<std::mutex> inspection_lock(inspection_mutex_,std::try_to_lock);
            if(inspection_lock.owns_lock())inspection_.bind(p->inspection_token,p->source,p->frame,p->pts,p->slot,p->generation,p->origin,
                o->object_id,box(*o),p->value);
            else ++inspection_busy_;}
        }else if(o->confidence>=0 && o->unique_component_id==1)origins_[key]={1,frame->frame_num};
        else {auto i=origins_.find(key);if(i==origins_.end())throw std::runtime_error("pose tracker shadow origin unavailable");
          o->unique_component_id=i->second.origin;i->second.frame=frame->frame_num;}
        if(o->unique_component_id!=1 && o->unique_component_id!=4)throw std::runtime_error("unsupported person origin before Swin");
        const auto b=box(*o);if(b[2]>=16 && b[3]>=16)boxes.push_back(b);
      }
      if(origins_.size()>16U*hints_.targets*2)throw std::runtime_error("pose origin ledger bound");
      hints_.observe(frame->source_id,frame->frame_num,frame->buf_pts,std::move(boxes));
    }
  }
  py::dict snapshot(){py::dict d;d["fresh"]=fresh_.load();d["proposals"]=proposals_.load();d["promoted"]=promoted_.load();d["suppressed"]=suppressed_.load();
    try{noesis::pose::IngressInspection copy;{std::lock_guard<std::mutex> lock(inspection_mutex_);copy=inspection_;}
      d["inspection"]=py::module_::import("json").attr("loads")(copy.json());
      d["inspection_busy"]=inspection_busy_.load();
    }catch(...){d["inspection_error"]="snapshot_unavailable";}return d;}
  void close(){if(closed_.exchange(true))return;for(auto p:pads_){gst_pad_remove_probe(p.pad,p.id);gst_object_unref(p.pad);}pads_.clear();reset_all();}
  ~SinglePassPoseState(){close();}
};

class SinglePassPoseController {
  std::shared_ptr<SinglePassPoseState> state_;
 public:
  SinglePassPoseController(deepstream::CustomObject& pose,const std::string& pose_name,
      deepstream::CustomObject& tracker,const std::string& tracker_name,double iou,unsigned count,unsigned shadow)
      :state_(std::make_shared<SinglePassPoseState>(iou,count,shadow)) {
    state_->attach(pose,pose_name,tracker,tracker_name);
  }
  void close(){state_->close();}
  void reset_source(unsigned source){state_->reset_source(source);}
  py::dict snapshot(){return state_->snapshot();}
  ~SinglePassPoseController(){close();}
};

}  // namespace

bool attach_pose_features(
    deepstream::BatchMetadata& batch_meta,
    deepstream::ObjectMetadata& obj_meta,
    const std::string& payload_json,
    bool replace_existing) {
  return attach_pose_json(batch_meta, obj_meta, payload_json, replace_existing);
}

bool attach_pose_features_frame(
    deepstream::BatchMetadata& batch_meta,
    deepstream::FrameMetadata& frame_meta,
    const std::string& payload_json,
    bool replace_existing) {
  return attach_pose_json(batch_meta, frame_meta, payload_json, replace_existing);
}

py::object extract_pose_features(const deepstream::ObjectMetadata& obj_meta) {
  std::optional<std::string> payload;
  obj_meta.iterate(
      [&](const deepstream::UserMetadata& user_meta) {
        if (payload.has_value()) return;
        auto* stored = static_cast<JsonPayload*>(user_meta.userData());
        if (stored != nullptr) payload = stored->json;
      },
      pose_meta_type());
  return payload.has_value() ? py::cast(*payload) : py::none();
}

py::object extract_pose_keypoints(
    const deepstream::ObjectMetadata& obj_meta,
    int gie_id,
    int model_w,
    int model_h,
    float score_threshold,
    bool letterbox,
    py::object diagnostics=py::none()) {
  if(!diagnostics.is_none() && !PyDict_CheckExact(diagnostics.ptr()))
    throw py::type_error("diagnostics must be an exact dict or None");
  noesis::pose::PoseExtractionDiagnostic diagnostic;
  auto* observer=diagnostics.is_none()?nullptr:&diagnostic;
  const auto publish=[&] {
    if(!observer) return;
    auto out=py::reinterpret_borrow<py::dict>(diagnostics);
    out["tensor_count"]=diagnostic.tensor_count;
    out["context_present"]=diagnostic.context_present;
    out["context_valid"]=diagnostic.context_valid;
    out["shape_valid"]=diagnostic.shape_valid;
    out["rows"]=diagnostic.rows;out["cols"]=diagnostic.cols;
    out["qualifying_rows_seen"]=diagnostic.qualifying_rows_seen;
    out["finite_rows_seen"]=diagnostic.finite_rows_seen;
    out["padding_rows_seen"]=diagnostic.padding_rows_seen;
    out["score_pass_rows"]=diagnostic.score_pass_rows;
    out["class_pass_rows"]=diagnostic.class_pass_rows;
    out["bbox_pass_rows"]=diagnostic.bbox_pass_rows;
    out["center_pass_rows"]=diagnostic.center_pass_rows;
    out["iou_pass_rows"]=diagnostic.iou_pass_rows;
    out["max_score"]=diagnostic.max_score?py::cast(*diagnostic.max_score):py::none();
    out["max_iou"]=diagnostic.max_iou?py::cast(*diagnostic.max_iou):py::none();
    out["pre_iou_candidates_seen"]=diagnostic.pre_iou_candidates_seen;
    out["pre_iou_scan_complete"]=diagnostic.pre_iou_scan_complete;
    out["pre_iou_invalid_row_index"]=diagnostic.pre_iou_invalid_row_index?py::cast(*diagnostic.pre_iou_invalid_row_index):py::none();
    out["pre_iou_parent_bbox_xywh"]=diagnostic.pre_iou_parent_bbox_xywh?py::cast(*diagnostic.pre_iou_parent_bbox_xywh):py::none();
    py::list candidates;
    for(const auto& candidate:diagnostic.pre_iou_candidates) {
      if(!candidate) continue;
      py::dict value;value["row_index"]=candidate->row_index;value["score"]=candidate->score;
      value["class_id"]=candidate->class_id;value["iou"]=candidate->iou?py::cast(*candidate->iou):py::none();
      value["bbox_xyxy"]=candidate->bbox_xyxy?py::cast(*candidate->bbox_xyxy):py::none();
      py::list torso;
      for(const auto& point:candidate->torso_unclipped) torso.append(point?py::cast(*point):py::none());
      value["torso_unclipped"]=torso;candidates.append(value);
    }
    out["pre_iou_candidates"]=candidates;
    out["result"]=diagnostic.result;
  };
  std::optional<py::dict> decoded;
  try {
    decoded=decode_pose_keypoints_payload(
        obj_meta, gie_id, model_w, model_h, score_threshold, letterbox,observer);
  } catch(...) {
    diagnostic.result="extraction_exception";publish();throw;
  }
  publish();
  if (!decoded.has_value()) {
    return py::none();
  }
  return std::move(*decoded);
}

#ifndef NOESIS_POSE_CONTEXT_NATIVE_TEST
PYBIND11_MODULE(noesis_pose_meta_ext, m) {
  py::class_<SinglePassPoseController>(m,"SinglePassPoseController")
      .def(py::init<deepstream::CustomObject&,const std::string&,deepstream::CustomObject&,const std::string&,double,unsigned,unsigned>())
      .def("snapshot",&SinglePassPoseController::snapshot).def("close",&SinglePassPoseController::close)
      .def("reset_source",&SinglePassPoseController::reset_source);
  py::class_<PoseRoiContextController>(m,"PoseRoiContextController")
      .def(py::init<deepstream::CustomObject&,const std::string&>())
      .def("snapshot",&PoseRoiContextController::snapshot)
      .def("close",&PoseRoiContextController::close);
  m.doc() =
      "Noesis DS9 public-Service-Maker helper for pose tensors and pose-feature user metadata.";
  m.def(
      "attach_pose_features",
      &attach_pose_features,
      py::arg("batch_meta"),
      py::arg("obj_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.POSE_FEATURES through BatchMetadata/UserMetadata to an object.");
  m.def(
      "attach_pose_features_frame",
      &attach_pose_features_frame,
      py::arg("batch_meta"),
      py::arg("frame_meta"),
      py::arg("payload_json"),
      py::arg("replace_existing") = true,
      "Attach NOESIS.POSE_FEATURES through BatchMetadata/UserMetadata to a frame.");
  m.def(
      "extract_pose_features",
      &extract_pose_features,
      py::arg("obj_meta"),
      "Extract NOESIS.POSE_FEATURES through public ObjectMetadata iteration.");
  m.def(
      "extract_pose_keypoints",
      &extract_pose_keypoints,
      py::arg("obj_meta"),
      py::arg("gie_id"),
      py::arg("model_w"),
      py::arg("model_h"),
      py::arg("score_threshold") = 0.25f,
      py::arg("letterbox") = true,
      py::arg("diagnostics") = py::none(),
      "Extract the best pose row through TensorOutputUserMetadata/Tensor.");
  m.def(
      "extract_rfdetr_keypoint_matches",
      &extract_rfdetr_keypoint_matches_impl,
      py::arg("frame_meta"),
      py::arg("gie_id"),
      py::arg("score_threshold") = 0.4f,
      py::arg("min_iou") = 0.7f,
      py::arg("ambiguity_margin") = 0.05f,
      "Decode frame-owned RF-DETR PGIE tensors and strictly associate "
      "person queries with object metadata.");
  m.def("pose_meta_type", []() { return pose_meta_type(); });
}
#endif
#endif  // NOESIS_POSE_CROP_GEOMETRY_TEST
