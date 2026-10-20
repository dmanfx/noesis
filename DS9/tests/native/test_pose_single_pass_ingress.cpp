#ifdef NOESIS_POSE_INGRESS_SDK_TEST
// ROOT ONLY public SDK/header test: no pipeline, pixel allocation or GPU call.
#define NOESIS_POSE_CONTEXT_NATIVE_TEST 1
#include "../../native/noesis_pose_meta_ext.cpp"
#include "nvdspreprocess_meta.h"
#include <iostream>
static int released=0;
static void precursor_release(void* data,void*) {
  auto* u=static_cast<NvDsUserMeta*>(data);
  delete static_cast<GstNvDsPreProcessBatchMeta*>(u->user_meta_data);++released;
}
static void* precursor_copy(void* data,void*) {
  auto* u=static_cast<NvDsUserMeta*>(data);
  return new GstNvDsPreProcessBatchMeta(*static_cast<GstNvDsPreProcessBatchMeta*>(u->user_meta_data));
}
static bool check_copy(const char* label,std::vector<guint64> targets,bool copyable,bool unknown,bool strict,bool expected) {
  const auto initial_released=released;
  struct Header{NvBufSurface surface;NvBufSurfaceParams slot;};
  auto* h=static_cast<Header*>(g_malloc0(sizeof(Header)));
  h->surface.batchSize=3;h->surface.numFilled=1;h->surface.memType=NVBUF_MEM_CUDA_DEVICE;
  h->surface.surfaceList=&h->slot;h->slot.width=1920;h->slot.height=1080;
  auto* source=gst_buffer_new_wrapped(h,sizeof(Header));auto* b=nvds_create_batch_meta(3);
  b->base_meta.batch_meta=b;b->base_meta.copy_func=nvds_batch_meta_copy_func;
  b->base_meta.release_func=nvds_batch_meta_release_func;b->max_frames_in_batch=3;
  auto* gstmeta=gst_buffer_add_nvds_meta(source,b,nullptr,nvds_batch_meta_copy_func,nvds_batch_meta_release_func);
  gstmeta->meta_type=NVDS_BATCH_GST_META;
  auto* f=nvds_acquire_frame_meta_from_pool(b);f->source_id=2;f->batch_id=0;
  f->frame_num=17;f->buf_pts=566666666;f->pipeline_width=1920;f->pipeline_height=1080;
  nvds_add_frame_meta_to_batch(b,f);
  auto* u=nvds_acquire_user_meta_from_pool(b);auto* p=new GstNvDsPreProcessBatchMeta{};
  p->target_unique_ids=std::move(targets);u->user_meta_data=p;u->base_meta.batch_meta=b;
  u->base_meta.meta_type=unknown?NvDsMetaType(nvds_get_user_meta_type(const_cast<gchar*>("NOESIS.TEST.UNKNOWN_BATCH"))):NVDS_PREPROCESS_BATCH_META;
  u->base_meta.copy_func=copyable?precursor_copy:nullptr;
  u->base_meta.release_func=precursor_release;nvds_add_user_meta_to_batch(b,u);
  auto* plain=gst_buffer_copy(source);auto* copied=gst_buffer_get_nvds_batch_meta(plain);
  std::cout<<"case="<<label<<" source_batch_users="<<meta_count(b->batch_user_meta_list)
    <<" clone_batch_users="<<meta_count(copied->batch_user_meta_list)
    <<" source_type="<<unsigned(u->base_meta.meta_type)<<" no_copy="<<(u->base_meta.copy_func==nullptr)
    <<" source_max="<<b->max_frames_in_batch<<" clone_max="<<copied->max_frames_in_batch
    <<" source_frames="<<b->num_frames_in_batch<<" clone_frames="<<copied->num_frames_in_batch
    <<" source_mem="<<gst_buffer_n_memory(source)<<" clone_mem="<<gst_buffer_n_memory(plain)<<std::endl;
  gst_buffer_unref(plain);bool accepted=false,source_intact=false,clone_survived=false;
  GstBuffer* own=nullptr;
  try {
#ifdef NOESIS_POSE_INGRESS_CONSUMED_PREPROCESS
    own=private_pose_buffer(source,!strict);
#else
    own=private_pose_buffer(source);
#endif
    accepted=true;
  }catch(const std::exception& e){std::cout<<"production_guard="<<e.what()<<std::endl;}
  // The diagnostic plain clone releases its OWN copied payload only.
  source_intact=b->batch_user_meta_list && u->user_meta_data==p && released==initial_released+(copyable?1:0);
  gst_buffer_unref(source);
  if(own){auto* preserved=gst_buffer_get_nvds_batch_meta(own);
    auto* original=static_cast<NvDsFrameMeta*>(preserved->frame_meta_list->data);
    clone_survived=original->source_id==2 && original->frame_num==17 && original->buf_pts==566666666 &&
      original->pipeline_width==1920 && original->pipeline_height==1080 && gst_buffer_n_memory(own)==1;
    gst_buffer_unref(own);}
  const auto expected_releases=(copyable?2:1)+(accepted&&copyable?1:0);
  std::cout<<"accepted="<<accepted<<" expected="<<expected<<" source_intact="<<source_intact
    <<" clone_survived="<<clone_survived<<" released="<<released-initial_released<<std::endl;
  return accepted==expected && source_intact && (!accepted || clone_survived) && released-initial_released==expected_releases;
}
int main() {
  gst_init(nullptr,nullptr);bool ok=true;
#ifdef NOESIS_POSE_INGRESS_CONSUMED_PREPROCESS
  ok=check_copy("consumed-pgie",{1},false,false,false,true)&&ok;
  ok=check_copy("legacy-strict",{1},false,false,true,false)&&ok;
  ok=check_copy("foreign-target",{4},false,false,false,false)&&ok;
  ok=check_copy("shared-target",{1,4},false,false,false,false)&&ok;
  ok=check_copy("missing-target",{},false,false,false,false)&&ok;
  ok=check_copy("copyable-preserved",{1},true,false,false,true)&&ok;
  ok=check_copy("unknown-lost",{1},false,true,false,false)&&ok;
  ok=check_copy("unknown-copyable-preserved",{1},true,true,false,true)&&ok;
#else
  ok=check_copy("before",{1},false,false,true,false);
#endif
  return ok?0:1;
}
#else
// CPU only: include the SAME scalar functions used by the native ingress.
#define NOESIS_POSE_CROP_GEOMETRY_TEST 1
#include "../../native/noesis_pose_meta_ext.cpp"
#include <iostream>
#include <cassert>
using namespace noesis::pose;
static void full_body_parent_case() {
  // R101 Kitchen383: exact current PGIE parent/crop and retained NN row.
  auto crop=make_crop_geometry(846,370,62,186,640,640,true);assert(crop);
  crop->original_left=858;crop->original_top=404.5;
  crop->original_width=39.5;crop->original_height=116.5;
  std::array<std::array<float,57>,300> rows{};
  const Box nn{857.024648,405.274609,28.5622066,78.7957031};
  const std::array<float,51> joints{
    867.976501f,416.717957f,.981445312f,870.159607f,415.010559f,.9921875f,
    866.812195f,414.974213f,.821289062f,875.908447f,415.337494f,.969726562f,
    866.448364f,415.1922f,.0594177246f,879.910828f,424.492188f,.994140625f,
    864.483582f,423.838287f,.993164062f,881.802795f,438.587494f,.978515625f,
    860.044617f,435.681244f,.924316406f,881.511719f,449.485931f,.961425781f,
    860.554016f,447.088287f,.858886719f,875.981201f,448.614075f,.989746094f,
    866.448364f,448.323425f,.990234375f,876.054016f,463.145325f,.971191406f,
    868.413147f,462.782043f,.977050781f,875.981201f,478.403137f,.942871094f,
    868.922546f,477.967194f,.956054688f};
  auto put=[&](int n,const Box& b,const std::array<float,51>& points){auto& row=rows[n];
    row[0]=(b[0]-crop->source_left)*crop->ratio_x+crop->offset_left;
    row[1]=(b[1]-crop->source_top)*crop->ratio_y+crop->offset_top;
    row[2]=(b[0]+b[2]-crop->source_left)*crop->ratio_x+crop->offset_left;
    row[3]=(b[1]+b[3]-crop->source_top)*crop->ratio_y+crop->offset_top;row[4]=.80078125f;
    for(int k=0;k<17;++k){row[6+k*3]=(points[k*3]-crop->source_left)*crop->ratio_x+crop->offset_left;
      row[7+k*3]=(points[k*3+1]-crop->source_top)*crop->ratio_y+crop->offset_top;
      row[8+k*3]=points[k*3+2];}};
  put(0,nn,joints);
  auto read=[&](int r,int c){return rows[r][c];};
  std::vector<Box> parents{{858,404.5,39.5,116.5},{1094,357.5,53,126.5},
    {1194,377.5,34,78.5},{1221,387,38,84.5},{1288,353.5,51,122.5}};
  auto selected=[&](){return unique_context_row(*crop,300,57,read,nullptr,&parents);};
  assert(unique_context_row(*crop,300,57,read)==-1); // default/old383 route unchanged
  assert(selected()==0 && full_body_context_owner(*crop,rows[0],&parents));
  const auto original_rows=rows;const auto original_parents=parents;
  // Every actual PGIE parent participates in uniqueness, not merely earlier
  // crops, the inferred NN rectangle, or the current parent alone.
  std::reverse(parents.begin(),parents.end());assert(selected()==0);parents=original_parents;
  parents.erase(parents.begin());assert(selected()==-1);parents=original_parents;
  parents.push_back(parents[0]);assert(selected()==-1);parents=original_parents;
  parents.push_back({859,405,26,80});assert(selected()==-1);parents=original_parents;
  parents.push_back({0,0,std::numeric_limits<double>::quiet_NaN(),2});assert(selected()==-1);
  parents=std::vector<Box>(129,original_parents[0]);assert(selected()==-1);parents={};assert(selected()==-1);
  parents=original_parents;
  rows[1]=rows[0];assert(selected()==-1);rows=original_rows; // two full witnesses
  // No positive witness may settle an OLD ambiguous contained or IoU set.
  put(1,{860,405.5,25,79},joints);assert(selected()==1);
  put(2,{860,405.5,25,79},joints);
  assert(selected()==-1);rows=original_rows;
  put(1,parents[0],{});assert(selected()==1);
  put(2,parents[0],{});assert(selected()==-1);rows=original_rows;
  for(int k:{5,6,11,12,13,14,15,16}){rows[0][8+k*3]=.349f;assert(selected()==-1);rows=original_rows;}
  rows[0][8]=.349f;assert(selected()==-1);rows=original_rows; // no resolved head triplet
  for(int k=0;k<5;++k){rows[0][6+k*3]=rows[0][6];rows[0][7+k*3]=rows[0][7];}
  assert(selected()==-1);rows=original_rows;
  // A reliable wrist outside either observed box also defeats correspondence.
  rows[0][6+7*3]=(857.5-crop->source_left)*crop->ratio_x+crop->offset_left;
  assert(selected()==-1);rows=original_rows;
  rows[0][6+7*3]=(886-crop->source_left)*crop->ratio_x+crop->offset_left;
  assert(selected()==-1);rows=original_rows;
  rows[0][8+7*3]=1.01f;assert(selected()==-1);rows=original_rows;
  rows[0][4]=.249f;assert(selected()==-1);rows=original_rows;
  rows[0][5]=1;assert(selected()==-1);rows=original_rows;
  put(0,{846,405.274609,39.5868546,78.7957031},joints);assert(selected()==-1);rows=original_rows;
  put(0,{857.024648,404.499,28.5622066,79.5713121},joints);
  assert(!full_body_context_owner(*crop,rows[0],&parents));rows=original_rows;
  put(0,{857.024648,405.274609,28.5622066,115.726391},joints);
  assert(!full_body_context_owner(*crop,rows[0],&parents));rows=original_rows;
  rows[299][56]=std::numeric_limits<float>::quiet_NaN();assert(selected()==-1);rows=original_rows;
  int reads=0;assert(unique_context_row(*crop,300,57,[&](int r,int c){++reads;return rows[r][c];},
    nullptr,&parents)==0 && reads==300*57); // finite tail scanned once, no tensor reread
  // Literal378 and384 legacy IoU winners remain unchanged even without a roster.
  for(int frame:{378,384}){
    crop=frame==378?make_crop_geometry(846,370,62,184,640,640,true):
                    make_crop_geometry(848,382,56,128,640,640,true);assert(crop);
    const Box parent=frame==378?Box{858,405,39.5,116}:Box{858.5,405,35,80};
    crop->original_left=parent[0];crop->original_top=parent[1];
    crop->original_width=parent[2];crop->original_height=parent[3];rows={};
    put(0,frame==378?Box{858.039535,405.901563,31.6488372,78.8109375}:
                       Box{857.25,405.1,27.85,79.5},{});
    assert(unique_context_row(*crop,300,57,read)==0);
  }
  // Literal R79 child raw24 came from proposal crop0, NOT a PGIE child parent.
  // All reliable landmarks, including knees/feet, fit the mother's PGIE box.
  // Retain full NN vertical extent to reject this nested-child misattachment.
  crop=make_crop_geometry(456,392,76,198,640,640,true);assert(crop);
  const Box child_parent{470.889801,428.815613,47.8489799,124.214066};
  crop->original_left=child_parent[0];crop->original_top=child_parent[1];
  crop->original_width=child_parent[2];crop->original_height=child_parent[3];
  const Box child{469.493878,428.274219,49.8653061,125.219531};
  const std::array<float,51> child_joints{
    476.938782f,448.963684f,.928222656f,479.653076f,446.217957f,.964355469f,
    477.481628f,446.101959f,.166992188f,491.20816f,447.300781f,.923828125f,
    483.530609f,446.643372f,.00351142883f,504.624481f,460.951965f,.961914062f,
    484.771423f,460.255859f,.805175781f,504.857147f,486.5914f,.806152344f,
    474.534698f,479.08905f,.293212891f,488.726532f,478.625f,.717773438f,
    483.995911f,476.8461f,.493408203f,501.444885f,499.662506f,.866210938f,
    489.269379f,498.734375f,.729492188f,501.057129f,518.224976f,.803710938f,
    489.502045f,516.368774f,.587402344f,501.755096f,542.897644f,.690917969f,
    493.224487f,541.041382f,.541503906f};
  rows={};put(0,child,child_joints);
  assert(current_candidates(*crop,300,57,read).size()==1 && extra_crop_supported(child,*crop));
  const Box actual_mother_parent{450.5,355,85.5,194.5};
  crop->original_left=actual_mother_parent[0];crop->original_top=actual_mother_parent[1];
  crop->original_width=actual_mother_parent[2];crop->original_height=actual_mother_parent[3];
  parents={actual_mother_parent};assert(!full_body_context_owner(*crop,rows[0],&parents));
  // The actual PGIE mother's OWN crop also contains a second child NN row.
  crop=make_crop_geometry(424,296,136,310,640,640,true);assert(crop);
  crop->original_left=actual_mother_parent[0];crop->original_top=actual_mother_parent[1];
  crop->original_width=actual_mother_parent[2];crop->original_height=actual_mother_parent[3];
  const std::array<float,51> mother_crop_child_joints{
    479.128571f,451.484375f,.549316406f,480.100006f,448.941406f,.723144531f,
    480.100006f,448.457031f,.129394531f,487.385712f,450.03125f,.881347656f,
    490.785706f,448.820312f,.110290527f,491.757141f,462.746094f,.967285156f,
    498.921417f,460.203125f,.915527344f,485.807129f,480.546875f,.897460938f,
    491.635712f,476.308594f,.485595703f,481.192871f,479.820312f,.822753906f,
    483.257141f,477.398438f,.382080078f,497.707153f,492.777344f,.942382812f,
    503.414276f,491.324219f,.9140625f,497.950012f,513.484375f,.916015625f,
    502.928558f,513.f,.889648438f,497.950012f,540.972656f,.882324219f,
    501.592865f,538.429688f,.876464844f};
  rows={};put(0,{470.507143,432.109375,48.5714286,120.125},mother_crop_child_joints);
  assert(!full_body_context_owner(*crop,rows[0],&parents));
  parents={{731.5,615,195.5,260},{554.5,330.5,102.5,242},actual_mother_parent};
  assert(!full_body_context_owner(*crop,rows[0],&parents));
  std::cout<<"literal383 full-body / 378+384 / unique roster / anatomy / R79 / full scan PASS\n";
}
static void parent_crop_ambiguity_case() {
  auto crop=make_crop_geometry(436,288,84,348,640,640,true);assert(crop);
  crop->original_left=452.5;crop->original_top=352.5;
  crop->original_width=52.5;crop->original_height=218;
  std::array<std::array<float,57>,300> rows{};
  auto put=[&](int n,std::array<float,4> bbox,float score){
    std::copy(bbox.begin(),bbox.end(),rows[n].begin());rows[n][4]=score;};
  // R92 F705 crop3 exact float32 NN endpoints, selected PGIE adult23.
  put(0,{293.25f,109.875f,397.5f,559.5f},.657714844f);
  put(1,{271.f,120.375f,363.f,493.75f},.301513672f);
  auto read=[&](int r,int c){return rows[r][c];};
  PoseExtractionDiagnostic diagnostic;
  assert(unique_context_row(*crop,300,57,read,&diagnostic)==1);
  assert(diagnostic.qualifying_rows_seen==2 && diagnostic.pre_iou_scan_complete);
  assert(std::string(diagnostic.result)=="row_selected");
  // A legacy single IoU winner remains usable even if the inferred box is
  // censored at the crop cut; the NEW rule only resolves ambiguous parents.
  rows[1]={};assert(unique_context_row(*crop,300,57,read)==0);
  rows[1]=rows[0];assert(unique_context_row(*crop,300,57,read)==-1); // zero complete
  rows[0]=rows[1]={};
  put(0,{271.f,120.375f,363.f,493.75f},.9f);rows[1]=rows[0];rows[1][4]=.25f;
  assert(unique_context_row(*crop,300,57,read)==-1); // two complete, scores irrelevant
  // A malformed row after two IoU candidates must still fail the full scan.
  rows[299][0]=std::numeric_limits<float>::quiet_NaN();
  assert(unique_context_row(*crop,300,57,read)==-1);rows[299]={};
  // Literal R92 F708 crop3: TWO complete candidates cannot be disambiguated.
  crop=make_crop_geometry(438,296,74,298,640,640,true);assert(crop);
  crop->original_left=452.75;crop->original_top=352.5;
  crop->original_width=46.5;crop->original_height=187;
  rows={};
  put(0,{274.5f,120.375f,388.25f,542.f},.336425781f);
  put(1,{298.5f,115.875f,399.f,546.f},.320068359f);
  put(2,{268.5f,119.875f,372.75f,536.f},.281494141f);
  assert(unique_context_row(*crop,300,57,read,&diagnostic)==-1);
  assert(diagnostic.qualifying_rows_seen==3 && diagnostic.pre_iou_scan_complete);
  assert(std::string(diagnostic.result)=="context_ambiguous_rows");
  // R93 F708 actual retained UNCLIPPED source joints (row0,row2), recast
  // through this exact crop. The previous bbox-only case deliberately had no
  // positive head/body evidence and MUST remain ambiguous.
  const std::array<std::array<float,51>,2> actual708{{
    {{469.262665f,375.56366f,0.836425781f,472.18988f,370.965637f,0.847167969f,466.803802f,371.72226f,0.580078125f,479.215179f,370.849213f,0.73046875f,470.082275f,372.944519f,0.274414062f,480.034821f,394.770691f,0.853027344f,474.063293f,395.585541f,0.870117188f,477.22467f,431.264069f,0.299804688f,472.541138f,433.475769f,0.270263672f,478.278473f,462.81015f,0.161376953f,482.02533f,463.042969f,0.134155273f,486.006317f,453.730469f,0.780273438f,483.196198f,455.243744f,0.789550781f,491.860748f,495.520325f,0.604980469f,493.617096f,495.4039f,0.607910156f,494.085449f,532.421082f,0.695800781f,493.265808f,532.770325f,0.705566406f}},
    {{469.965179f,374.923431f,0.942382812f,473.243683f,370.383606f,0.944335938f,466.803802f,371.140228f,0.728515625f,480.386078f,370.267181f,0.7890625f,468.325958f,372.071472f,0.266845703f,480.503174f,394.130463f,0.894042969f,472.424042f,395.527344f,0.946777344f,475.585449f,432.428131f,0.250976562f,470.316467f,432.893738f,0.327148438f,472.072784f,460.947662f,0.128540039f,472.892395f,461.064056f,0.150512695f,484.132904f,454.079681f,0.701171875f,480.971527f,455.476562f,0.751464844f,489.284821f,496.335144f,0.486328125f,488.465179f,495.287506f,0.542480469f,490.572784f,532.537476f,0.634277344f,489.050629f,532.304688f,0.677246094f}}
  }};
  auto joints=[&](int row,const std::array<float,51>& source){for(int k=0;k<17;++k){
    rows[row][6+k*3]=float((source[k*3]-crop->source_left)*crop->ratio_x+crop->offset_left);
    rows[row][7+k*3]=float((source[k*3+1]-crop->source_top)*crop->ratio_y+crop->offset_top);
    rows[row][8+k*3]=source[k*3+2];}};
  joints(0,actual708[0]);joints(2,actual708[1]);
  const auto originals=rows;unsigned reads=0;
  auto counted=[&](int r,int c){++reads;return rows[r][c];};
  assert(unique_context_row(*crop,300,57,counted,&diagnostic)==0 && reads==300*57);
  assert(rows==originals && diagnostic.qualifying_rows_seen==3 && diagnostic.pre_iou_scan_complete);
  // Representative is original row order, NOT score/IoU ranking.
  rows[2][4]=.99f;rows[0][4]=.25f;assert(unique_context_row(*crop,300,57,read)==0);rows=originals;
  // Positive duplicate proof is mandatory in both head AND torso.
  for(int k:{0,1,2,3,4})rows[2][6+k*3]+=float(50*crop->ratio_x);
  assert(unique_context_row(*crop,300,57,read)==-1);rows=originals;
  for(int k:{5,6,11,12})rows[2][7+k*3]+=float(80*crop->ratio_y);
  assert(unique_context_row(*crop,300,57,read)==-1);rows=originals;
  rows[2][8]=.349f;assert(unique_context_row(*crop,300,57,read)==-1);rows=originals;
  rows[1]=rows[0];assert(unique_context_row(*crop,300,57,read)==-1);rows=originals; // >2 complete
  rows[299][0]=std::numeric_limits<float>::quiet_NaN();
  assert(unique_context_row(*crop,300,57,read)==-1);rows=originals;
  // No-IoU strict-contained recovery remains the old path, not a tie breaker.
  rows={};crop=make_crop_geometry(100,100,200,300,640,640,true);assert(crop);
  const Box small{160,180,60,120};
  auto nn=[&](double x,double y){return std::array<float,2>{
    float((x-crop->source_left)*crop->ratio_x+crop->offset_left),
    float((y-crop->source_top)*crop->ratio_y+crop->offset_top)};};
  auto a=nn(small[0],small[1]),b=nn(small[0]+small[2],small[1]+small[3]);
  put(0,{a[0],a[1],b[0],b[1]},.9f);
  for(int k:{5,6,11,12}){auto p=nn(170+(k%2)*20,200+(k>6)*60);
    rows[0][6+k*3]=p[0];rows[0][7+k*3]=p[1];rows[0][8+k*3]=.9f;}
  assert(unique_context_row(*crop,300,57,read)==0);
  rows[0][8+5*3]=.349f;assert(unique_context_row(*crop,300,57,read)==-1);
  std::cout<<"parent crop complete705 / ambiguous708 / legacy single / contained / finite PASS\n";
}
static void crop_support_case() {
  // Literal R89 quantized NN endpoints. Recompute integer crop scaling, never
  // add the rounded source-space JSON left+width (F209 is EXACTLY at538).
  auto decode=[](int l,int t,int w,int h,std::array<float,4> nn,float score){
    auto crop=make_crop_geometry(l,t,w,h,640,640,true);assert(crop);
    // Only literal bbox/score are relevant here. Do not invent reliable joints;
    // anatomy suppression/eligibility remains covered by its separate fixture.
    std::array<float,57> row{};std::copy(nn.begin(),nn.end(),row.begin());row[4]=score;
    auto rows=current_candidates(*crop,1,57,[&](int,int c){return row[c];});
    assert(rows.size()==1); // old candidate path retains these real NN boxes
    return std::make_pair(rows[0],*crop);
  };
  const auto woman198=decode(482,266,118,360,{213.625f,161.75f,274.5f,489.5f},.466796875f);
  const auto man209=decode(416,300,122,296,{383.5f,99.125f,451.f,529.f},.689453125f);
  const auto man220=decode(410,302,156,296,{414.f,84.375f,488.25f,486.f},.753417969f);
  const auto interior198=decode(424,300,118,298,{365.5f,75.5f,445.25f,567.f},.744628906f);
  assert(woman198.first.box[0]<482);
  assert(man209.first.box[0]+man209.first.box[2]==538.);
  assert(man220.first.box[0]+man220.first.box[2]>566.);
  for(const auto* p:{&woman198,&man209,&man220})assert(!extra_crop_supported(p->first.box,p->second));
  assert(extra_crop_supported(interior198.first.box,interior198.second));
  // Exclusion is birth-only: a selected parent keeps its old attachment route
  // even when the same measured row cannot establish an additional birth.
  assert(selected_parent_row(man209.first.row,man209.first.row,false));
  IngressInspection inspection("2:198,209,220");
  const auto token=inspection.add(2,209,6966666666ULL,0,1,2,true,-1,man209.second,man209.first);
  inspection.disposition(token,6);assert(inspection.size==1 && inspection.records[0].disposition==6);
  auto crop=make_crop_geometry(100,100,200,300,640,640,true);assert(crop);
  const Box inside{101,101,198,298};assert(extra_crop_supported(inside,*crop));
  // Internal cut equality and outside are unproven on EACH side; no margin.
  for(int side=0;side<4;++side)for(double delta:{0.,-1.}){
    auto b=inside;
    if(side==0){b[0]=100+delta;b[2]=299-b[0];}
    if(side==1){b[1]=100+delta;b[3]=399-b[1];}
    if(side==2)b[2]=300-delta-b[0];
    if(side==3)b[3]=400-delta-b[1];
    assert(!extra_crop_supported(b,*crop));
  }
  auto camera=make_crop_geometry(0,0,1920,1080,640,640,true);assert(camera);
  assert(extra_crop_supported(Box{0,0,1920,1080},*camera));
  for(const Box b:{Box{-1,0,1920,1080},Box{0,-1,1920,1080},
                  Box{0,0,1921,1080},Box{0,0,1920,1081}})assert(!extra_crop_supported(b,*camera));
  // Mixed camera/internal edges preserve inclusive support ONLY at camera edges.
  auto mixed=make_crop_geometry(0,100,1920,300,640,640,true);assert(mixed);
  assert(extra_crop_supported(Box{0,101,1920,298},*mixed));
  assert(!extra_crop_supported(Box{0,100,1920,299},*mixed));
  auto invalid=inside;invalid[0]=std::numeric_limits<double>::quiet_NaN();assert(!extra_crop_supported(invalid,*crop));
  invalid=inside;invalid[2]=0;assert(!extra_crop_supported(invalid,*crop));
  auto bad=*crop;bad.source_width=1920;assert(!extra_crop_supported(inside,bad));
  // Existing literal independent/occluded bodies and small Kitchen rows remain
  // interior; crop support neither matches anatomy nor changes parent clipping.
  // Actual R79 F1056 raw24 UID4 retained crop/NN box, not a generic child crop.
  auto child_crop=make_crop_geometry(456,392,76,198,640,640,true);assert(child_crop);
  assert(child_crop->destination_width==245 && child_crop->offset_left==197);
  assert(extra_crop_supported(Box{469.493878,428.274219,49.8653061,125.219531},*child_crop));
  auto k431=make_crop_geometry(846,370,62,188,640,640,true);assert(k431);
  auto k437=make_crop_geometry(846,368,64,188,640,640,true);assert(k437);
  assert(extra_crop_supported(Box{859.0758293838862,404.91953125,
    888.3127962085308-859.0758293838862,484.4890625-404.91953125},*k431));
  assert(extra_crop_supported(Box{859.0506912442396,405.93046875,
    883.9723502304148-859.0506912442396,484.765625-405.93046875},*k437));
  std::cout<<"literal crop cuts / interior / camera edges / parent exclusion PASS\n";
}
static void anatomy_case() {
  // R78 F1153 exact used landmarks: nose, left eye/ear, shoulders and hips.
  // Other joints are absent and are NOT consulted by this conservative rule.
  auto crop=make_crop_geometry(400,330,120,270,640,640,true);assert(crop);
  auto crop2=*crop;crop->ratio_x=1.65384615;crop->ratio_y=1.65803109;
  crop2.ratio_x=2.17346939;crop2.ratio_y=2.17687075;
  CurrentCandidate fragment{{474.104651,353.298047,32.1976744,181.314453},{},.71875f,1};
  CurrentCandidate adult{{473.196009,354.847656,62.0551643,185.932031},{},.92285f,0};
  auto set=[](CurrentCandidate& c,int k,float x,float y,float score){c.joints[k*3]=x;c.joints[k*3+1]=y;c.joints[k*3+2]=score;};
  set(fragment,0,481.51,371.92,.920f);set(fragment,1,483.93,367.77,.946f);set(fragment,3,494.97,367.85,.829f);
  set(fragment,5,506,388.958,.667f);set(fragment,6,483.779,390.767,.901f);
  set(fragment,11,499.953,441.731,.782f);set(fragment,12,485.140,441.128,.882f);
  set(adult,0,480.15,372.30,.936f);set(adult,1,482,368.11,.938f);set(adult,3,493.50,367.42,.947f);
  set(adult,5,516.962,388.095,.996f);set(adult,6,488.092,391.196,.985f);
  set(adult,11,510.981,447.182,.998f);set(adult,12,491.772,445.919,.994f);
  assert(!duplicate_box(fragment.box,{adult.box},.55)); // actual old decision admits it
  assert(same_visible_head(fragment,*crop,adult,crop2));
  struct Ref{CurrentCandidate c;CropGeometry crop;};Ref one{adult,crop2};
  std::vector<const Ref*> refs{&one};assert(unique_visible_head(fragment,*crop,refs));
  // Shared visible head with genuinely different reliable torso is NOT enough.
  auto distinct=fragment;for(int k:{5,6,11,12})distinct.joints[k*3+1]+=80;
  assert(!same_visible_head(distinct,*crop,adult,crop2));
  auto missing=fragment;missing.joints[11*3+2]=.349f;assert(!same_visible_head(missing,*crop,adult,crop2));
  missing=fragment;missing.joints[0*3+2]=.349f;assert(!same_visible_head(missing,*crop,adult,crop2));
  missing=fragment;missing.joints[3*3]=std::numeric_limits<float>::quiet_NaN();
  assert(!same_visible_head(missing,*crop,adult,crop2));
  auto untrusted=*crop;untrusted.ratio_x=0;assert(!same_visible_head(fragment,untrusted,adult,crop2));
  untrusted=*crop;untrusted.model_width=320;assert(!same_visible_head(fragment,untrusted,adult,crop2));
  // Two independently present nearby supported heads are ambiguous, not a merge.
  Ref nearby=one;for(int k:{0,1,3,5,6,11,12})nearby.c.joints[k*3]+=.1f;
  refs.push_back(&nearby);assert(!unique_visible_head(fragment,*crop,refs));
  for(int k:{5,6,11,12})nearby.c.joints[k*3+1]+=80;
  assert(!unique_visible_head(fragment,*crop,refs)); // head ambiguity cannot choose a torso
  // R78 F1056 genuine nested child24/mother27 must remain separate.
  CurrentCandidate child{{469.929,429.086,49.988,124.253},{},.583f,0};
  CurrentCandidate mother{{452.414,354.791,82.329,190.662},{},.753f,0};
  set(child,0,477.05,449.31,.866f);set(child,1,479.52,446.64,.942f);set(child,3,490.98,447.65,.913f);
  set(child,5,503.357,459.792,.940f);set(child,6,487.107,459.018,.800f);
  set(child,11,502.661,493.475,.875f);set(child,12,492.446,492.702,.791f);
  set(mother,0,466.74,379.43,.984f);set(mother,1,469.41,374.29,.989f);set(mother,3,481.56,370.71,.971f);
  set(mother,5,503.414,388.031,.995f);set(mother,6,472.814,393.480,.977f);
  set(mother,11,506.936,448.215,.984f);set(mother,12,485.564,448.941,.957f);
  assert(!same_visible_head(child,*crop,mother,crop2));
  // R78 F1056 row2: weak/inconsistent head and displaced torso stays unresolved.
  auto weak=fragment;set(weak,0,465.65,379.31,.393f);set(weak,1,467.59,377.31,.437f);
  set(weak,3,471.48,378.40,.319f);set(weak,5,475.243,389.969,.636f);set(weak,6,459.336,390.150,.964f);
  set(weak,11,474.636,416.549,.768f);set(weak,12,463.586,416.609,.927f);
  assert(!same_visible_head(weak,*crop,mother,crop2));
  // Mirror permutations are not anatomical correspondence.
  auto mirrored=fragment;for(int k=0;k<3;++k)std::swap(mirrored.joints[1*3+k],mirrored.joints[3*3+k]);
  assert(!same_visible_head(mirrored,*crop,adult,crop2));
  // Actual R79 F1056 native child24 and mother27 retain independent anatomy.
  auto r79_child_crop=make_crop_geometry(456,392,76,198,640,640,true);assert(r79_child_crop);
  CurrentCandidate r79_child{{469.493878,428.274219,49.8653061,125.219531},{},0.658203125f,0};
  set(r79_child,0,476.938782f,448.963684f,0.928222656f);
  set(r79_child,1,479.653076f,446.217957f,0.964355469f);
  set(r79_child,2,477.481628f,446.101959f,0.166992188f);
  set(r79_child,3,491.20816f,447.300781f,0.923828125f);
  set(r79_child,4,483.530609f,446.643372f,0.00351142883f);
  set(r79_child,5,504.624481f,460.951965f,0.961914062f);
  set(r79_child,6,484.771423f,460.255859f,0.805175781f);
  set(r79_child,11,501.444885f,499.662506f,0.866210938f);
  set(r79_child,12,489.269379f,498.734375f,0.729492188f);
  auto r79_mother_crop=make_crop_geometry(424,296,136,310,640,640,true);assert(r79_mother_crop);
  CurrentCandidate r79_mother{{452.414286,354.791016,82.3285714,190.662109},{},0.753417969f,0};
  set(r79_mother,0,466.742859f,379.433594f,0.983886719f);
  set(r79_mother,1,469.414276f,374.287109f,0.988769531f);
  set(r79_mother,2,463.342865f,376.527344f,0.756347656f);
  set(r79_mother,3,481.557129f,370.714844f,0.970703125f);
  set(r79_mother,4,464.557129f,376.285156f,0.0208435059f);
  set(r79_mother,5,503.414276f,388.03125f,0.995117188f);
  set(r79_mother,6,472.814301f,393.480469f,0.9765625f);
  set(r79_mother,11,506.935699f,448.214844f,0.984375f);
  set(r79_mother,12,485.564301f,448.941406f,0.956542969f);
  assert(same_visible_head(r79_child,*r79_child_crop,r79_child,*r79_child_crop));
  assert(!same_visible_head(r79_child,*r79_child_crop,r79_mother,*r79_mother_crop));
  assert(!same_visible_head(r79_mother,*r79_mother_crop,r79_child,*r79_child_crop));
  std::cout<<"literal R78 fragment / nested child / wrong torso / nearby ambiguity / missing provenance PASS\n";
}
static void inspection_case(bool capped) {
  IngressInspection disabled;
  assert(!disabled.enabled && !disabled.selected(2,1056));
  for(const char* bad:{"2:1056", "2:1153,1056", "16:705,1056,1153", "2:705,1056,1153,1200", "2:705,1056,", "2:705,705"}) {
    IngressInspection invalid(bad);assert(!invalid.enabled && invalid.invalid_config);
  }
  IngressInspection d("2:705,1056,1153");assert(d.enabled);
  assert(!d.selected(1,1056) && !d.selected(2,1057));
  auto crop=make_crop_geometry(450,350,90,220,640,640,true);assert(crop);
  CurrentCandidate mother{{451,355,82,163},{},.73584f,1};
  CurrentCandidate child{{475,431,44,121},{},.77197f,0};
  for(unsigned k=0;k<17;++k){mother.joints[k*3]=452+k;mother.joints[k*3+1]=355+k;mother.joints[k*3+2]=.9f;
    child.joints[k*3]=476+k;child.joints[k*3+1]=432+k;child.joints[k*3+2]=.95f;}
  const auto original=child.joints;
  assert(d.add(1,1056,35200000000ULL,2,7,0,false,1,*crop,child)==0);
  d.mark(2,1056);
  const auto a=d.add(2,1056,35200000000ULL,2,7,0,false,1,*crop,mother);
  const auto b=d.add(2,1056,35200000000ULL,2,7,0,false,1,*crop,child);
  d.disposition(a,1);d.disposition(b,4);
  auto attached=mother;attached.joints[0]=460; // retain original separately from exact parent clipping
  d.bind(a,2,1056,35200000000ULL,2,7,1,28,mother.box,attached);
  d.bind(b,2,1056,35200000000ULL,1,7,4,25,child.box,child);assert(!d.records[b-1].bound);
  d.bind(b,2,1057,35200000000ULL,2,7,4,25,child.box,child);assert(!d.records[b-1].bound);
  d.bind(b,2,1056,35200000001ULL,2,7,4,25,child.box,child);assert(!d.records[b-1].bound);
  d.bind(b,2,1056,35200000000ULL,2,8,4,25,child.box,child);assert(!d.records[b-1].bound);
  d.bind(b,2,1056,35200000000ULL,2,7,4,25,child.box,child);assert(d.records[b-1].bound);
  d.bind(b,2,1056,35200000000ULL,2,7,4,25,child.box,child);assert(d.binding_mismatch==5);
  assert(child.joints==original && d.records[b-1].candidate.joints==original);
  assert(!duplicate_box(child.box,{mother.box},.55)); // inspection never decides nested bodies
  if(capped){for(unsigned n=0;n<100;++n){auto huge=child;huge.joints.fill(std::numeric_limits<float>::max());
      d.add(2,1153,38433333333ULL,0,7,n,true,-1,*crop,huge);}
    assert(d.size==64 && d.overflow==38);}
  const auto json=d.json();assert(json.size()<=IngressInspection::byte_cap);std::cout<<json<<'\n';
}
int main(int argc,char** argv) {
  if(argc==2 && std::string(argv[1])=="--full-body-parent"){full_body_parent_case();return 0;}
  if(argc==2 && std::string(argv[1])=="--parent-ambiguity"){parent_crop_ambiguity_case();return 0;}
  if(argc==2 && std::string(argv[1])=="--crop-support"){crop_support_case();return 0;}
  if(argc==2 && std::string(argv[1])=="--anatomy"){anatomy_case();return 0;}
  if(argc==2){inspection_case(std::string(argv[1])=="--inspection-caps");return 0;}
  // Literal R74 source geometry/score counterexample, transformed through the
  // actual crop. Mother remains the parent; the child is a separate NN row.
  auto crop=make_crop_geometry(450,350,90,220,640,640,true);assert(crop);
  std::array<std::array<float,57>,300> rows{};
  auto row=[&](unsigned index,Box box,float score){auto& r=rows[index];
    r[0]=(box[0]-crop->source_left)*crop->ratio_x+crop->offset_left;
    r[1]=(box[1]-crop->source_top)*crop->ratio_y+crop->offset_top;
    r[2]=(box[0]+box[2]-crop->source_left)*crop->ratio_x+crop->offset_left;
    r[3]=(box[1]+box[3]-crop->source_top)*crop->ratio_y+crop->offset_top;r[4]=score;
    for(int k=0;k<17;++k){r[6+k*3]=(box[0]+box[2]*.5-crop->source_left)*crop->ratio_x+crop->offset_left;
      r[7+k*3]=(box[1]+box[3]*.5-crop->source_top)*crop->ratio_y+crop->offset_top;r[8+k*3]=.96f;}};
  const Box child{475.695,431.281,43.619,120.657},mother{451.748,354.805,82.838,162.398};
  row(0,child,.77197f);row(1,mother,.73584f);
  auto read=[&](int r,int c){return rows[r][c];};
  auto out=current_candidates(*crop,300,57,read);assert(out.size()==2);
  std::vector<Box> detections{mother};
  assert(!duplicate_box(out[0].box,detections,.55));
  detections.push_back(out[0].box);
  assert(duplicate_box(out[1].box,detections,.55));
  assert(duplicate_box(out[0].box,detections,.55)); // duplicate crop suppressed
  assert(!duplicate_box(Box{800,400,40,120},detections,.55)); // distinct person
  assert(std::abs(out[0].score-.77197f)<1e-6 && out[0].joints[2]==.96f);
  rows[2][4]=.249f;assert(current_candidates(*crop,300,57,read).size()==2);
  rows[2][4]=.9f;rows[2][5]=1;assert(current_candidates(*crop,300,57,read).size()==2);
  rows[299][0]=std::numeric_limits<float>::quiet_NaN();bool refused=false;
  try{current_candidates(*crop,300,57,read);}catch(const std::runtime_error&){refused=true;}assert(refused);
  rows={};const Box contained{470,370,45,140};row(0,contained,.9f);
  const int known=unique_context_row(*crop,300,57,read);
  assert(known==0 && overlap(contained,Box{450,350,90,220})<.55);
  assert(selected_parent_row(0,known,false)); // exclude independently of PGIE IoU
  assert(!selected_parent_row(1,known,false)); // genuine other row survives
  assert(!selected_parent_row(0,known,true)); // hint is not a detection parent
  assert(duplicate_box(contained,{contained},.55)); // same row from another crop
  HintBank hints(32,90);
  hints.observe(2,1048,34933333333ULL,{mother,child});
  const auto epoch=hints.epoch(2);
  assert(hints.hints(2,1049,34966666666ULL).size()==2); // skipped PGIE: CURRENT-image crops
  assert(hints.hints(1,1049,34966666666ULL).empty());
  hints.reset(0);assert(hints.epoch(2)==epoch);assert(hints.hints(2,1050,35000000000ULL).size()==2);
  assert(hints.hints(2,1139,38000000000ULL).empty()); // bounded old hint, no stale detection
  assert(hints.hints(2,1047,34900000000ULL).empty()); // wrap/reorder clears source
  assert(hints.epoch(2)!=epoch);
  hints.observe(2,5,100,{child});hints.observe(2,6,200,{});assert(hints.hints(2,7,300).empty()); // termination
  hints.observe(2,8,400,{mother});hints.reset(2);assert(hints.hints(2,9,500).empty()); // epoch/reuse
  hints.observe(1,1,1,{child});hints.reset();assert(hints.hints(1,2,2).empty());
  bool capped=false;try{hints.observe(1,1,1,std::vector<Box>(33,child));}catch(...){capped=true;}assert(capped);
  std::cout<<"current NN mother+child/dedup/distinct/hints/skips/epoch/reset/termination/bounds PASS\n";
}
#endif
