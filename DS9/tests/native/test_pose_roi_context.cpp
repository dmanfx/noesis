// SOURCE ONLY fixture: root compiles against the SAME native extension SDK/libs.
// No engine, CUDA operation, frame extraction or production Python import.
#define NOESIS_POSE_CONTEXT_NATIVE_TEST 1
#include "../../native/noesis_pose_meta_ext.cpp"
#include <iostream>
#include <thread>

namespace {
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::atomic<int> payload_alive{0},payload_copies{0};
void* test_copy(void* p,void*){
  auto* m=static_cast<NvDsUserMeta*>(p);++payload_alive;++payload_copies;
  return new int(*static_cast<int*>(m->user_meta_data));
}
void test_free(void* p,void*){
  auto* m=static_cast<NvDsUserMeta*>(p);delete static_cast<int*>(m->user_meta_data);--payload_alive;
}
GstBuffer* make_buffer(float width=30.5f,guint source=1,float height=72.f) {
  // Synthetic HEADER ONLY, deliberately no pixel allocation/GPU evidence.
  struct Header {NvBufSurface surface;NvBufSurfaceParams slots[3];};
  auto* header=static_cast<Header*>(g_malloc0(sizeof(Header)));
  header->surface.batchSize=3;header->surface.numFilled=1;
  header->surface.memType=NVBUF_MEM_CUDA_DEVICE;header->surface.surfaceList=header->slots;
  header->slots[0].width=1920;header->slots[0].height=1080;
  auto* buffer=gst_buffer_new_wrapped(header,sizeof(Header));
  auto* batch=nvds_create_batch_meta(3);
  // Literal installed gstnvstreamdemux.cpp330..335 / gstnvstreammux.cpp1478..1484.
  // nvds_create_batch_meta allocates pools but does NOT set this capacity field.
  batch->base_meta.batch_meta=batch;
  batch->base_meta.copy_func=nvds_batch_meta_copy_func;
  batch->base_meta.release_func=nvds_batch_meta_release_func;
  batch->max_frames_in_batch=3;
  auto* meta=gst_buffer_add_nvds_meta(buffer,batch,nullptr,nvds_batch_meta_copy_func,nvds_batch_meta_release_func);
  check(meta,"attach batch");meta->meta_type=NVDS_BATCH_GST_META;
  auto* frame=nvds_acquire_frame_meta_from_pool(batch);frame->frame_num=235;frame->source_id=source;frame->batch_id=0;
  frame->buf_pts=7841166666ULL;frame->pipeline_width=1920;frame->pipeline_height=1080;
  nvds_add_frame_meta_to_batch(batch,frame);
  auto* object=nvds_acquire_obj_meta_from_pool(batch);object->class_id=0;
  object->unique_component_id=1;object->object_id=6;object->rect_params.left=673;
  object->rect_params.top=421;object->rect_params.width=width;object->rect_params.height=height;
  nvds_add_obj_meta_to_frame(frame,object,nullptr);
  auto* user=nvds_acquire_user_meta_from_pool(batch);user->user_meta_data=new int(73);++payload_alive;
  user->base_meta.meta_type=nvds_get_user_meta_type(const_cast<gchar*>("NOESIS.TEST.POSE_CONTEXT"));
  user->base_meta.copy_func=test_copy;user->base_meta.release_func=test_free;
  nvds_add_user_meta_to_obj(object,user);
  return buffer;
}
NvDsObjectMeta* first_object(GstBuffer* b){return static_cast<NvDsObjectMeta*>(
  static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(b)->frame_meta_list->data)->obj_meta_list->data);}
void multi_object_copy_contract() {
  auto* original=make_buffer();auto* batch=gst_buffer_get_nvds_batch_meta(original);
  auto* frame=static_cast<NvDsFrameMeta*>(batch->frame_meta_list->data);
  auto* second=nvds_acquire_obj_meta_from_pool(batch);second->class_id=0;
  second->unique_component_id=1;second->object_id=9;
  second->rect_params.left=900;second->rect_params.top=300;
  second->rect_params.width=28;second->rect_params.height=80;
  nvds_add_obj_meta_to_frame(frame,second,nullptr);
  auto* user=nvds_acquire_user_meta_from_pool(batch);user->user_meta_data=new int(91);++payload_alive;
  user->base_meta.meta_type=nvds_get_user_meta_type(const_cast<gchar*>("NOESIS.TEST.POSE_CONTEXT"));
  user->base_meta.copy_func=test_copy;user->base_meta.release_func=test_free;
  nvds_add_user_meta_to_obj(second,user);
  auto* sdk=gst_buffer_copy(original);
  std::cout<<"SDK object order source="<<first_object(original)->object_id
           <<" copy="<<first_object(sdk)->object_id<<std::endl;
  check(first_object(original)->object_id==9 && first_object(sdk)->object_id==6,
        "installed SDK prepend/reversal witness changed");gst_buffer_unref(sdk);
  GstBuffer* copy=nullptr;
  try {
    copy=private_pose_buffer(original);
    auto* copied_frame=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(copy)->frame_meta_list->data);
    auto* a=static_cast<NvDsObjectMeta*>(copied_frame->obj_meta_list->data);
    auto* b=static_cast<NvDsObjectMeta*>(copied_frame->obj_meta_list->next->data);
    const auto saved_id=b->object_id;b->object_id=a->object_id;
    unsigned objects=0;bool rejected=false;
    try{validate_pose_object_copies(*frame,*copied_frame,objects);}catch(const std::exception& e){
      rejected=std::string(e.what()).find("key ambiguous/missing")!=std::string::npos;}
    check(rejected,"ambiguous CLONE object keys accepted");b->object_id=saved_id;
    const auto saved_width=a->rect_params.width;a->rect_params.width+=1;objects=0;rejected=false;
    try{validate_pose_object_copies(*frame,*copied_frame,objects);}catch(const std::exception& e){
      rejected=std::string(e.what()).find("rect_equal=0")!=std::string::npos;}
    check(rejected,"same-key changed rectangle accepted");a->rect_params.width=saved_width;
    const auto saved_class=b->class_id;b->class_id=4;objects=0;rejected=false;
    try{validate_pose_object_copies(*frame,*copied_frame,objects);}catch(const std::exception& e){
      rejected=std::string(e.what()).find("key ambiguous/missing")!=std::string::npos;}
    check(rejected,"foreign CLONE component/class key accepted");b->class_id=saved_class;
    RoiProbeState state;state.prepare(copy);state.restore(copy);
    unsigned matched=0;
    for(auto* o=copied_frame->obj_meta_list;o;o=o->next) {
      auto* object=static_cast<NvDsObjectMeta*>(o->data);deepstream::ObjectMetadata wrapped(object);
      check(roi_context(wrapped)&&roi_context(wrapped)->restored,"multi-object crop context lost");
      auto* payload=static_cast<NvDsUserMeta*>(object->obj_user_meta_list->data);
      // Find original test userdata, not the newly added crop context.
      for(auto* m=object->obj_user_meta_list;m;m=m->next){auto* u=static_cast<NvDsUserMeta*>(m->data);
        if(u->base_meta.meta_type==user->base_meta.meta_type)payload=u;}
      if(object->object_id==6)check(object->rect_params.left==673 && object->rect_params.width==30.5f &&
        *static_cast<int*>(payload->user_meta_data)==73,"first object's rect/userdata corrupted");
      else if(object->object_id==9)check(object->rect_params.left==900 && object->rect_params.width==28 &&
        *static_cast<int*>(payload->user_meta_data)==91,"second object's rect/userdata corrupted");
      else check(false,"foreign object admitted");++matched;
    }
    check(matched==2 && first_object(original)->rect_params.left==900,"multi-object count/original changed");
  } catch(...) {if(copy)gst_buffer_unref(copy);gst_buffer_unref(original);throw;}
  gst_buffer_unref(original);check(payload_alive==2,"multi-object owned userdata died early");
  gst_buffer_unref(copy);check(payload_alive==0,"multi-object userdata leaked");
}
void object_key_scope_and_source_duplicates() {
  auto* original=make_buffer();auto* batch=gst_buffer_get_nvds_batch_meta(original);
  auto* frame=static_cast<NvDsFrameMeta*>(batch->frame_meta_list->data);
  auto* duplicate=nvds_acquire_obj_meta_from_pool(batch);duplicate->class_id=0;
  duplicate->unique_component_id=1;duplicate->object_id=6;duplicate->rect_params=first_object(original)->rect_params;
  nvds_add_obj_meta_to_frame(frame,duplicate,nullptr);
  bool rejected=false;const int before=payload_copies;
  try{auto* bad=private_pose_buffer(original);gst_buffer_unref(bad);}catch(const std::exception& e){
    rejected=std::string(e.what()).find("SOURCE duplicate object key")!=std::string::npos;}
  check(rejected && payload_copies==before,"SOURCE duplicate key copied/accepted");gst_buffer_unref(original);
  original=make_buffer();batch=gst_buffer_get_nvds_batch_meta(original);
  frame=nvds_acquire_frame_meta_from_pool(batch);frame->frame_num=236;frame->source_id=2;
  frame->buf_pts=7841167666ULL;frame->batch_id=1;nvds_add_frame_meta_to_batch(batch,frame);
  auto* same_id=nvds_acquire_obj_meta_from_pool(batch);same_id->class_id=0;
  same_id->unique_component_id=1;same_id->object_id=6;same_id->rect_params.left=900;
  same_id->rect_params.top=300;same_id->rect_params.width=28;same_id->rect_params.height=80;
  nvds_add_obj_meta_to_frame(frame,same_id,nullptr);
  GstMapInfo map=GST_MAP_INFO_INIT;check(gst_buffer_map(original,&map,GST_MAP_WRITE),"two-source header map");
  auto* surface=reinterpret_cast<NvBufSurface*>(map.data);surface->numFilled=2;
  surface->surfaceList[1].width=1920;surface->surfaceList[1].height=1080;gst_buffer_unmap(original,&map);
  auto* copy=private_pose_buffer(original);RoiProbeState state;state.prepare(copy);state.restore(copy);
  unsigned checked=0;
  for(auto* f=gst_buffer_get_nvds_batch_meta(copy)->frame_meta_list;f;f=f->next){
    auto* current=static_cast<NvDsFrameMeta*>(f->data);deepstream::ObjectMetadata object(current->obj_meta_list->data);
    auto* context=roi_context(object);
    check(context && context->object==6 && context->source==current->source_id &&
          context->frame==current->frame_num && context->pts==current->buf_pts && context->restored,
          "same ID across independent current frames/sources conflated");++checked;
  }
  check(checked==2,"independent frame/object lost");gst_buffer_unref(original);gst_buffer_unref(copy);
  check(payload_alive==0,"key scope fixture userdata leaked");
}
void metadata_lifetime_and_concurrent_reader() {
  auto* original=make_buffer();auto* source_object=first_object(original);
  auto* sdk_only=gst_buffer_copy(original);
  auto* omitted=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(sdk_only)->frame_meta_list->data);
  // Exact observed installed SDK omission, independently before owner carry-through.
  check(omitted->pipeline_width==0 && omitted->pipeline_height==0 &&
        omitted->frame_num==235 && omitted->buf_pts==7841166666ULL && omitted->source_id==1,
        "installed SDK omission/current original witness changed");
  gst_buffer_unref(sdk_only);
  auto* copy=private_pose_buffer(original);RoiProbeState state;
  auto* preserved=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(copy)->frame_meta_list->data);
  check(preserved->pipeline_width==1920 && preserved->pipeline_height==1080 &&
        preserved->frame_num==235 && preserved->buf_pts==7841166666ULL && preserved->source_id==1,
        "validated source scalar geometry not carried");
  std::atomic<bool> running{true},changed{false};
  std::thread reader([&]{while(running){if(source_object->rect_params.width!=30.5f || source_object->rect_params.left!=673)changed=true;}});
  try{state.prepare(copy);state.restore(copy);}catch(...){running=false;reader.join();throw;}
  running=false;reader.join();check(!changed,"side branch observed context mutation");
  auto* target=first_object(copy);deepstream::ObjectMetadata object(target);
  auto* context=roi_context(object);check(context&&context->restored,"context not restored");
  check(context->pts==7841166666ULL && target->rect_params.width==30.5f,"original/full PTS lost");
  check(source_object->rect_params.width==30.5f,"original changed");
  auto* second=gst_buffer_copy(copy);gst_buffer_unref(copy);
  deepstream::ObjectMetadata second_object(first_object(second));
  auto* second_context=roi_context(second_object);check(second_context&&second_context->restored,"copied owned context died early");
  gst_buffer_unref(original);check(payload_alive==1,"userdata copy/release ownership");
  gst_buffer_unref(second);check(payload_alive==0 && payload_copies>=2,"userdata leak/double release");
}
void defaults_and_binding() {
  for(float width:{128.f,129.f,140.f,15.f}) {
    auto* buffer=make_buffer(width,1,260.f);auto* copy=private_pose_buffer(buffer);RoiProbeState state;
    state.prepare(copy);state.restore(copy);deepstream::ObjectMetadata object(first_object(copy));
    check(!roi_context(object)&&first_object(copy)->rect_params.width==width,"large/ineligible changed");
    gst_buffer_unref(copy);gst_buffer_unref(buffer);
  }
  for(float width:{64.f,127.99f}) {
    auto* buffer=make_buffer(width,1,260.f);auto* copy=private_pose_buffer(buffer);RoiProbeState state;
    state.prepare(copy);state.restore(copy);deepstream::ObjectMetadata object(first_object(copy));
    auto* context=roi_context(object);
    check(context && context->restored && context->crop.original_width==width &&
          first_object(copy)->rect_params.width==width && first_object(buffer)->rect_params.width==width,
          "task-only sub128 context/original restoration boundary failed");
    gst_buffer_unref(copy);gst_buffer_unref(buffer);
  }
  auto* buffer=make_buffer();auto* copy=private_pose_buffer(buffer);RoiProbeState state;state.prepare(copy);
  auto* frame=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(copy)->frame_meta_list->data);
  frame->source_id=2;bool rejected=false;try{state.restore(copy);}catch(const std::exception&){rejected=true;}
  check(rejected,"source collision accepted");frame->source_id=1;frame->buf_pts++;
  rejected=false;try{state.restore(copy);}catch(const std::exception&){rejected=true;}
  check(rejected,"full PTS mismatch accepted");gst_buffer_unref(copy);gst_buffer_unref(buffer);
  auto* missing=gst_buffer_new();rejected=false;try{auto* bad=private_pose_buffer(missing);gst_buffer_unref(bad);}catch(const std::exception&){rejected=true;}
  check(rejected,"missing SDK batch accepted");gst_buffer_unref(missing);
  auto* malformed=make_buffer();auto* raw=gst_buffer_get_nvds_batch_meta(malformed);
  raw->max_frames_in_batch=0;rejected=false;
  try{auto* bad=private_pose_buffer(malformed);gst_buffer_unref(bad);}catch(const std::exception&){rejected=true;}
  check(rejected,"zero capacity reached SDK copy");raw->max_frames_in_batch=3;
  auto* gst_meta=gst_buffer_get_nvds_meta(malformed);auto saved=gst_meta->copyfunc;
  gst_meta->copyfunc=nullptr;rejected=false;
  try{auto* bad=private_pose_buffer(malformed);gst_buffer_unref(bad);}catch(const std::exception&){rejected=true;}
  check(rejected,"unowned Gst copy callback accepted");gst_meta->copyfunc=saved;gst_buffer_unref(malformed);
  for(guint invalid:{1918U,1921U}) {
    auto* source=make_buffer();auto* frame=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(source)->frame_meta_list->data);
    frame->pipeline_width=invalid;rejected=false;
    try{auto* bad=private_pose_buffer(source);gst_buffer_unref(bad);}catch(const std::exception&){rejected=true;}
    check(rejected && frame->pipeline_width==invalid,"unqualified SOURCE raster defaulted/repaired");
    gst_buffer_unref(source);
  }
}
// Each injected defect must fail BEFORE copying and leave SOURCE unchanged.
void surface_header_contract() {
  auto* source=make_buffer();
  auto* frame=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(source)->frame_meta_list->data);
  frame->pipeline_width=0;frame->pipeline_height=0;
  auto* copy=private_pose_buffer(source);
  auto* qualified=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(copy)->frame_meta_list->data);
  check(qualified->pipeline_width==1920 && qualified->pipeline_height==1080 &&
        qualified->source_id==frame->source_id && qualified->frame_num==frame->frame_num &&
        qualified->buf_pts==frame->buf_pts && qualified->batch_id==frame->batch_id,
        "current HEADER did not qualify private raster/current binding");
  check(frame->pipeline_width==0 && frame->pipeline_height==0,"SOURCE optional raster mutated");
  RoiProbeState state;state.prepare(copy);state.restore(copy);gst_buffer_unref(copy);gst_buffer_unref(source);
  // Match installed nvinfer: select THIS frame's batch_id, not surfaceList[0].
  source=make_buffer();frame=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(source)->frame_meta_list->data);
  GstMapInfo slot_map=GST_MAP_INFO_INIT;check(gst_buffer_map(source,&slot_map,GST_MAP_WRITE),"slot fixture map");
  auto* slots=reinterpret_cast<NvBufSurface*>(slot_map.data);slots->numFilled=3;
  slots->surfaceList[2].width=1920;slots->surfaceList[2].height=1080;
  slots->surfaceList[0].width=1;frame->batch_id=2;frame->pipeline_width=0;frame->pipeline_height=0;
  gst_buffer_unmap(source,&slot_map);copy=private_pose_buffer(source);
  qualified=static_cast<NvDsFrameMeta*>(gst_buffer_get_nvds_batch_meta(copy)->frame_meta_list->data);
  check(qualified->batch_id==2 && qualified->pipeline_width==1920 && qualified->pipeline_height==1080,
        "frame batch_id did not select its own HEADER geometry");
  gst_buffer_unref(copy);gst_buffer_unref(source);
  for(int fault=0;fault<11;++fault) {
    source=make_buffer();auto* batch=gst_buffer_get_nvds_batch_meta(source);
    frame=static_cast<NvDsFrameMeta*>(batch->frame_meta_list->data);
    GstMapInfo mapped=GST_MAP_INFO_INIT;check(gst_buffer_map(source,&mapped,GST_MAP_WRITE),"fixture HEADER map");
    auto* surface=reinterpret_cast<NvBufSurface*>(mapped.data);
    const auto saved=*surface;
    if(fault==0)surface->batchSize=0;
    if(fault==1)surface->batchSize=4;
    if(fault==2)surface->numFilled=0;
    if(fault==3)surface->numFilled=4;
    if(fault==4)surface->surfaceList=nullptr;
    if(fault==5)frame->batch_id=1;
    if(fault==6)surface->surfaceList[0].width=1918;
    if(fault==7)frame->pipeline_height=1078;
    if(fault==8)frame->buf_pts=GST_CLOCK_TIME_NONE;
    if(fault==9)frame->frame_num=-1;
    if(fault==10)batch->num_frames_in_batch=2;
    gst_buffer_unmap(source,&mapped);
    const int before=payload_copies;bool refused=false;
    try{auto* bad=private_pose_buffer(source);gst_buffer_unref(bad);}catch(const std::exception& e){
      refused=std::string(e.what()).find("pose SOURCE")!=std::string::npos ||
              std::string(e.what()).find("pose surface HEADER")!=std::string::npos;
    }
    check(refused && payload_copies==before,"malformed HEADER/current binding reached SDK copy");
    // Restore fixture bookkeeping before SDK free (not production repair).
    check(gst_buffer_map(source,&mapped,GST_MAP_WRITE),"fixture HEADER restore map");
    *reinterpret_cast<NvBufSurface*>(mapped.data)=saved;gst_buffer_unmap(source,&mapped);
    frame->batch_id=0;frame->frame_num=235;frame->buf_pts=7841166666ULL;batch->num_frames_in_batch=1;
    gst_buffer_unref(source);
  }
  source=make_buffer();gst_buffer_remove_all_memory(source);gst_buffer_append_memory(source,gst_allocator_alloc(nullptr,4,nullptr));
  bool refused=false;try{auto* bad=private_pose_buffer(source);gst_buffer_unref(bad);}catch(const std::exception& e){
    refused=std::string(e.what()).find("HEADER unavailable/short")!=std::string::npos;}
  check(refused,"short/missing HEADER accepted");gst_buffer_unref(source);
  source=make_buffer();auto* batch=gst_buffer_get_nvds_batch_meta(source);
  frame=static_cast<NvDsFrameMeta*>(batch->frame_meta_list->data);
  for(unsigned n=1;n<=kContextObjects;++n){auto* o=nvds_acquire_obj_meta_from_pool(batch);
    o->object_id=1000+n;o->unique_component_id=1;o->class_id=0;nvds_add_obj_meta_to_frame(frame,o,nullptr);}
  const int before=payload_copies;refused=false;
  try{auto* bad=private_pose_buffer(source);gst_buffer_unref(bad);}catch(const std::exception& e){
    refused=std::string(e.what()).find("SOURCE object bound")!=std::string::npos;}
  check(refused && payload_copies==before,"SOURCE amplified object count copied before bound");gst_buffer_unref(source);
  source=make_buffer();batch=gst_buffer_get_nvds_batch_meta(source);
  auto* duplicate=nvds_acquire_frame_meta_from_pool(batch);duplicate->source_id=2;duplicate->frame_num=236;
  duplicate->buf_pts=7841166667ULL;duplicate->batch_id=0;nvds_add_frame_meta_to_batch(batch,duplicate);
  refused=false;const int before_duplicate=payload_copies;
  try{auto* bad=private_pose_buffer(source);gst_buffer_unref(bad);}catch(const std::exception& e){
    refused=std::string(e.what()).find("duplicate slot")!=std::string::npos;}
  check(refused && payload_copies==before_duplicate,"duplicate frame batch slot copied");gst_buffer_unref(source);
}
std::array<float,57> matching_row(const noesis::pose::CropGeometry& c) {
  std::array<float,57> r{};r[4]=.8f;r[5]=0;
  r[0]=float((c.original_left-c.source_left)*c.ratio_x+c.offset_left);
  r[1]=float((c.original_top-c.source_top)*c.ratio_y+c.offset_top);
  r[2]=float((c.original_left+c.original_width-c.source_left)*c.ratio_x+c.offset_left);
  r[3]=float((c.original_top+c.original_height-c.source_top)*c.ratio_y+c.offset_top);
  return r;
}
void selection_and_geometry() {
  auto c=noesis::pose::context_crop(673,421,30.5,72,1920,1080);check(bool(c),"crop refused");
  auto row=matching_row(*c);check(noesis::pose::context_row_matches(*c,row),"same original not selected");
  auto bad=row;bad[5]=1;check(!noesis::pose::context_row_matches(*c,bad),"wrong class");
  bad=row;bad[4]=.249f;check(!noesis::pose::context_row_matches(*c,bad),"score threshold");
  bad=row;bad[0]+=400;bad[2]+=400;check(!noesis::pose::context_row_matches(*c,bad),"neighbor selected");
  bad=row;bad[7]=std::numeric_limits<float>::quiet_NaN();check(!noesis::pose::context_row_matches(*c,bad),"nonfinite row");
  std::array<std::array<float,57>,3> rows{};rows[1]=row;
  auto read=[&](int r,int col){return rows[r][col];};
  check(noesis::pose::unique_context_row(*c,3,57,read)==1,"actual unique row selector");
  noesis::pose::PoseExtractionDiagnostic diagnostic;
  unsigned reads=0;
  auto counted=[&](int r,int col){++reads;return rows[r][col];};
  const auto plain=noesis::pose::unique_context_row(*c,3,57,counted);
  const auto plain_reads=reads;reads=0;
  check(noesis::pose::unique_context_row(*c,3,57,counted,&diagnostic)==plain &&
        reads==plain_reads && diagnostic.qualifying_rows_seen==1 &&
        std::strcmp(diagnostic.result,"row_selected")==0,
        "optional diagnostic changed selector/result/read count");
  rows[2]=row;
  check(noesis::pose::unique_context_row(*c,3,57,read)==-1,"two matching rows accepted");
  check(noesis::pose::unique_context_row(*c,3,57,read,&diagnostic)==-1 &&
        diagnostic.qualifying_rows_seen==2 &&
        std::strcmp(diagnostic.result,"context_ambiguous_rows")==0,"ambiguity diagnostic");
  rows[2]={};rows[1]={};
  check(noesis::pose::unique_context_row(*c,3,57,read)==-1,"zero matching rows accepted");
  check(noesis::pose::unique_context_row(*c,3,57,read,&diagnostic)==-1 &&
        diagnostic.qualifying_rows_seen==0 &&
        std::strcmp(diagnostic.result,"context_no_matching_row")==0,"no match diagnostic");
  check(noesis::pose::unique_context_row(*c,3,56,read)==-1,"56-column fallback");
  check(noesis::pose::unique_context_row(*c,3,56,read,&diagnostic)==-1 &&
        std::strcmp(diagnostic.result,"context_shape_invalid")==0,"shape diagnostic");
  check(noesis::pose::unique_context_row(*c,3,58,read)==-1,"58-column fallback");
  check(noesis::pose::unique_context_row(*c,301,57,read)==-1,"row bound bypassed");
  rows[1]=row;rows[2][6]=std::numeric_limits<float>::quiet_NaN();
  check(noesis::pose::unique_context_row(*c,3,57,read)==-1,"malformed unrelated output row ignored");
  check(noesis::pose::unique_context_row(*c,3,57,read,&diagnostic)==-1 &&
        std::strcmp(diagnostic.result,"context_nonfinite_row")==0,"nonfinite diagnostic");
  check(bool(noesis::pose::context_crop(0,0,64,160,1920,1080)),"64 short-side context missing");
  check(bool(noesis::pose::context_crop(0,0,127.99,260,1920,1080)),"127.99 short-side context missing");
  check(!noesis::pose::context_crop(0,0,128,260,1920,1080),"128 boundary ROI changed");
  check(!noesis::pose::context_crop(0,0,129,260,1920,1080),"129+ large ROI changed");
  check(!noesis::pose::context_crop(0,0,15,72,1920,1080),"ineligible ROI expanded");
  check(!noesis::pose::context_crop(-1,0,30,72,1920,1080),"invalid original accepted");
}
void extraction_diagnostic_without_tensor() {
  auto* original=make_buffer();auto* copy=private_pose_buffer(original);RoiProbeState state;
  state.prepare(copy);state.restore(copy);deepstream::ObjectMetadata object(first_object(copy));
  noesis::pose::PoseExtractionDiagnostic diagnostic;
  const auto plain=decode_pose_keypoints_payload(object,4,640,640,.25f,true);
  const auto observed=decode_pose_keypoints_payload(object,4,640,640,.25f,true,&diagnostic);
  check(!plain && !observed && diagnostic.context_present && diagnostic.context_valid &&
        diagnostic.tensor_count==0 && !diagnostic.shape_valid &&
        diagnostic.qualifying_rows_seen==-1 && std::strcmp(diagnostic.result,"tensor_missing")==0,
        "no tensor/default parity diagnostic");
  auto* context=roi_context(object);context->restored=false;
  diagnostic={};
  check(!decode_pose_keypoints_payload(object,4,640,640,.25f,true,&diagnostic) &&
        diagnostic.context_present && !diagnostic.context_valid && diagnostic.tensor_count==0 &&
        std::strcmp(diagnostic.result,"context_binding_invalid")==0,"context refusal diagnostic");
  gst_buffer_unref(copy);gst_buffer_unref(original);
}
void context_subpredicate_diagnostic() {
  auto crop=noesis::pose::context_crop(673,421,30.5,72,1920,1080);
  check(bool(crop),"diagnostic crop");const auto good=matching_row(*crop);
  for(int failure=0;failure<6;++failure) {
    std::array<std::array<float,57>,3> rows{};rows[1]=good;
    if(failure==0) rows[1][4]=.249f;
    if(failure==1) rows[1][5]=1;
    if(failure==2) rows[1][2]=rows[1][0];
    if(failure==3) {rows[1][0]+=400;rows[1][2]+=400;}
    if(failure==4) {
      const auto cx=(good[0]+good[2])/2,cy=(good[1]+good[3])/2;
      rows[1][0]=cx-2*(good[2]-good[0]);rows[1][2]=cx+2*(good[2]-good[0]);
      rows[1][1]=cy-2*(good[3]-good[1]);rows[1][3]=cy+2*(good[3]-good[1]);
    }
    unsigned reads=0;auto read=[&](int r,int col){++reads;return rows[r][col];};
    const int plain=noesis::pose::unique_context_row(*crop,3,57,read);
    const auto plain_reads=reads;reads=0;noesis::pose::PoseExtractionDiagnostic d;
    check(noesis::pose::unique_context_row(*crop,3,57,read,&d)==plain && reads==plain_reads,
          "subpredicate diagnostic changed result/accessor reads");
    check(d.finite_rows_seen==3 && d.padding_rows_seen==2 && d.max_score &&
          *d.max_score==rows[1][4],"padding/finite/max-score diagnostic");
    check(d.score_pass_rows==unsigned(failure!=0) &&
          d.class_pass_rows==unsigned(failure!=0 && failure!=1) &&
          d.bbox_pass_rows==unsigned(failure>=3) &&
          d.center_pass_rows==unsigned(failure>=4) &&
          d.iou_pass_rows==unsigned(failure==5),"ordered predicate counts");
    check(d.max_iou.has_value()==(failure>=4),"IoU unavailable stage fabricated");
    check(failure!=4 || (*d.max_iou>=0 && *d.max_iou<.5),"low IoU diagnostic");
    check((plain>=0)==(failure==5),"negative row became accepted");
  }
  // Ambiguity refinement requires the complete finite scan; three complete
  // candidates remain ambiguous and optional observation cannot change that.
  std::array<std::array<float,57>,3> rows{good,good,good};unsigned reads=0;
  noesis::pose::PoseExtractionDiagnostic d;
  check(noesis::pose::unique_context_row(*crop,3,57,[&](int r,int c){++reads;return rows[r][c];},&d)==-1 &&
        reads==3*57 && d.finite_rows_seen==3 && d.qualifying_rows_seen==3,
        "ambiguous selector failed full scan");
}
void pre_iou_candidate_diagnostic() {
  const auto crop=noesis::pose::context_crop(673,421,30.5,72,1920,1080);
  check(bool(crop),"candidate diagnostic crop");const auto good=matching_row(*crop);
  auto low=good;
  const float cx=(good[0]+good[2])/2,cy=(good[1]+good[3])/2;
  low[0]=cx-(good[2]-good[0]);low[2]=cx+(good[2]-good[0]);
  low[1]=cy-(good[3]-good[1]);low[3]=cy+(good[3]-good[1]);
  auto torso=[&](std::array<float,57>& row,bool outside) {
    unsigned i=0;
    for(int joint:{5,6,11,12}) {
      const double x=outside && i==0?crop->original_left-10.:
          crop->original_left+crop->original_width*(i%2?.75:.25);
      const double y=crop->original_top+crop->original_height*(i<2?.25:.75);
      row[6+joint*3]=float((x-crop->source_left)*crop->ratio_x+crop->offset_left);
      row[7+joint*3]=float((y-crop->source_top)*crop->ratio_y+crop->offset_top);
      row[8+joint*3]=.9f;++i;
    }
  };
  std::array<std::array<float,57>,4> rows{low,low,low,low};
  for(auto& row:rows)torso(row,false);torso(rows[1],true);
  unsigned reads=0;auto read=[&](int r,int c){++reads;return rows[r][c];};
  const int plain=noesis::pose::unique_context_row(*crop,4,57,read);const auto plain_reads=reads;
  noesis::pose::PoseExtractionDiagnostic d;reads=0;
  check(noesis::pose::unique_context_row(*crop,4,57,read,&d)==plain && plain==-1 && reads==plain_reads,
        "candidate observer changed rejection/reads");
  check(d.pre_iou_candidates_seen==4 && d.pre_iou_scan_complete &&
        d.pre_iou_candidates[0] && d.pre_iou_candidates[1] && !d.pre_iou_invalid_row_index,
        "candidate retention cap hid total/complete scan");
  check(d.pre_iou_parent_bbox_xywh && (*d.pre_iou_parent_bbox_xywh)[0]==crop->original_left,
        "candidate parent missing");
  const auto& inside=*d.pre_iou_candidates[0];const auto& outside=*d.pre_iou_candidates[1];
  check(inside.row_index==0 && outside.row_index==1 && inside.iou && *inside.iou<.5 && inside.bbox_xyxy,
        "low-IoU same row evidence missing");
  for(const auto& p:inside.torso_unclipped)check(p && (*p)[0]>crop->original_left &&
      (*p)[0]<crop->original_left+crop->original_width && (*p)[2]>.35,"inside torso missing");
  check(outside.torso_unclipped[0] && (*outside.torso_unclipped[0])[0]<crop->original_left-9.9,
        "outside torso was clipped into parent");
  rows[0][6+5*3]=std::numeric_limits<float>::quiet_NaN();reads=0;
  check(noesis::pose::unique_context_row(*crop,4,57,read,&d)==-1 && reads==57 &&
        d.pre_iou_invalid_row_index && *d.pre_iou_invalid_row_index==0 &&
        d.pre_iou_candidates_seen==0 && !d.pre_iou_candidates[0] && !d.pre_iou_scan_complete,
        "nonfinite torso retained/stale candidate or extra reads");
  rows={good,good,good,good};reads=0;
  check(noesis::pose::unique_context_row(*crop,4,57,read,&d)==-1 && reads==4*57 &&
        d.pre_iou_candidates_seen==4 && d.pre_iou_scan_complete,
        "ambiguous full scan lost total/complete truth");
}
void contained_parent_correspondence() {
  std::optional<noesis::pose::CropGeometry> crop;
  std::array<float,57> baseline{};
  auto actual=[&](const std::array<double,4>& parent,const std::array<double,4>& box,
                 const std::array<double,12>& torso,float score,double expected_iou) {
    crop=noesis::pose::context_crop(parent[0],parent[1],parent[2],parent[3],1920,1080);
    check(bool(crop),"actual original context unavailable");std::array<float,57> row{};
    auto mx=[&](double x){return float((x-crop->source_left)*crop->ratio_x+crop->offset_left);};
    auto my=[&](double y){return float((y-crop->source_top)*crop->ratio_y+crop->offset_top);};
    row[0]=mx(box[0]);row[1]=my(box[1]);row[2]=mx(box[2]);row[3]=my(box[3]);row[4]=score;
    unsigned i=0;for(int joint:{5,6,11,12}) {
      row[6+3*joint]=mx(torso[i]);row[7+3*joint]=my(torso[i+1]);row[8+3*joint]=float(torso[i+2]);i+=3;
    }
    // EXACT relevant I157 originals, not a fabricated full pose/ground grant.
    check(!noesis::pose::context_row_matches(*crop,row),"old IoU unexpectedly accepted actual case");
    unsigned reads=0;auto read=[&](int,int col){++reads;return row[col];};
    check(noesis::pose::unique_context_row(*crop,1,57,read)==0 && reads==57,
          "strict contained actual original not selected/single pass");
    reads=0;noesis::pose::PoseExtractionDiagnostic d;
    check(noesis::pose::unique_context_row(*crop,1,57,read,&d)==0 && reads==57 &&
          d.pre_iou_scan_complete && d.qualifying_rows_seen==0 && d.max_iou &&
          std::abs(*d.max_iou-expected_iou)<1e-6,"actual IoU/default observer parity");
    baseline=row;
  };
  actual({858.5,404.5,39.5,118.},
      {859.0758293838862,404.91953125,888.3127962085308,484.4890625},
      {879.5710900473933,424.2703125,.99462890625,864.218009478673,423.79296875,.98876953125,
       875.3104265402843,447.1828125,.99267578125,865.9075829383886,446.7421875,.9912109375},
      .75927734375f,.4991142985215127);
  actual({858.5,404.0000305175781,40.,118.5},
      {859.0506912442396,405.93046875,883.9723502304148,484.765625},
      {878.5898617511521,423.8125,.99462890625,863.6221198156682,424.0328125,.9833984375,
       874.9769585253456,446.6515625,.99462890625,865.7603686635945,446.578125,.99072265625},
      .8056640625f,.4144942785199027);
  for(int fault=0;fault<9;++fault) {
    auto row=baseline;
    if(fault==0)row[6+3*5]=float((crop->original_left-1-crop->source_left)*crop->ratio_x+crop->offset_left);
    if(fault==1)row[6+3*5]=row[0]-1;  // Inside parent, but outside the candidate box.
    if(fault==2)row[8+3*5]=.349f;
    if(fault==3)row[8+3*12]=0.f;
    if(fault==4)row[6+3*11]=std::numeric_limits<float>::quiet_NaN();
    if(fault==5)row[5]=1.f;
    if(fault==6)row[4]=.249f;
    if(fault==7) {row[0]-=100;row[2]+=100;}  // Parent-crossing bbox with torso still inside.
    if(fault==8)row[8+3*5]=1.01f;
    check(noesis::pose::unique_context_row(*crop,1,57,[&](int,int col){return row[col];})==-1,
          "invalid/weak/foreign/clipped contained witness accepted");
  }
  auto boundary=baseline;for(int joint:{5,6,11,12})boundary[8+3*joint]=.35f;
  check(noesis::pose::unique_context_row(*crop,1,57,[&](int,int col){return boundary[col];})==0,
        "existing inclusive anatomy confidence boundary changed");
  std::array<std::array<float,57>,2> rows{baseline,baseline};unsigned reads=0;
  auto read=[&](int r,int col){++reads;return rows[r][col];};
  check(noesis::pose::unique_context_row(*crop,2,57,read)==-1 && reads==2*57,
        "two overlapping contained candidates accepted");
  rows[1]=matching_row(*crop);reads=0;
  check(noesis::pose::unique_context_row(*crop,2,57,read)==1 && reads==2*57,
        "contained witness displaced original IoU route");
  rows[1]={};rows[1][0]=std::numeric_limits<float>::infinity();reads=0;
  check(noesis::pose::unique_context_row(*crop,2,57,read)==-1 && reads==2*57,
        "contained witness survived malformed remainder of scan");
}
void narrow_original_context_contract() {
  struct Case {float width,height;bool eligible;};
  for(const auto v: {Case{94.5f,125.f,false},Case{103.5f,123.f,false},
                    Case{46.5f,98.5f,true},Case{65.f,100.f,true},
                    Case{65.01f,100.f,false},Case{64.99f,100.f,true}}) {
    auto* original=make_buffer(v.width,2,v.height);
    auto* copy=private_pose_buffer(original);RoiProbeState state;
    state.prepare(copy);state.restore(copy);
    auto* target=first_object(copy);deepstream::ObjectMetadata object(target);
    auto* context=roi_context(object);
    check(bool(context)==v.eligible,"narrow-only native eligibility mismatch");
    check(target->rect_params.width==v.width && target->rect_params.height==v.height &&
          first_object(original)->rect_params.width==v.width &&
          first_object(original)->rect_params.height==v.height,"original box changed");
    check(!context || context->restored,"eligible narrow box not restored");
    gst_buffer_unref(copy);gst_buffer_unref(original);
  }
}
struct OrderCheck {unsigned before=0,after=0;};
GstPadProbeReturn see_expanded(GstPad*,GstPadProbeInfo* info,gpointer p){
  auto* v=static_cast<OrderCheck*>(p);auto* object=first_object(GST_PAD_PROBE_INFO_BUFFER(info));
  if(object->rect_params.width>30.5f)++v->before;return GST_PAD_PROBE_OK;}
GstPadProbeReturn see_restored(GstPad*,GstPadProbeInfo* info,gpointer p){
  auto* v=static_cast<OrderCheck*>(p);auto* object=first_object(GST_PAD_PROBE_INFO_BUFFER(info));
  if(object->rect_params.width==30.5f)++v->after;return GST_PAD_PROBE_OK;}
GstFlowReturn consume(GstPad*,GstObject*,GstBuffer* b){gst_buffer_unref(b);return GST_FLOW_OK;}
void actual_pad_order_and_close() {
  auto* element=gst_element_factory_make("identity","pose_fixture");check(element,"identity unavailable");
  auto state=std::make_shared<RoiProbeState>();state->attach(element);
  auto* sink=gst_element_get_static_pad(element,"sink");auto* src=gst_element_get_static_pad(element,"src");
  OrderCheck order;gst_pad_add_probe(sink,GST_PAD_PROBE_TYPE_BUFFER,see_expanded,&order,nullptr);
  gst_pad_add_probe(src,GST_PAD_PROBE_TYPE_BUFFER,see_restored,&order,nullptr);
  auto* input=gst_pad_new("fixture_input",GST_PAD_SRC);auto* output=gst_pad_new("fixture_output",GST_PAD_SINK);
  gst_pad_set_chain_function(output,consume);gst_pad_set_active(input,TRUE);gst_pad_set_active(output,TRUE);
  check(gst_pad_link(input,sink)==GST_PAD_LINK_OK && gst_pad_link(src,output)==GST_PAD_LINK_OK,"pad link");
  gst_element_set_state(element,GST_STATE_PLAYING);
  gst_pad_push_event(input,gst_event_new_stream_start("pose-context-fixture"));
  GstSegment segment;gst_segment_init(&segment,GST_FORMAT_TIME);gst_pad_push_event(input,gst_event_new_segment(&segment));
  auto* original=make_buffer();gst_buffer_ref(original);
  const auto flow=gst_pad_push(input,original);
  check(flow==GST_FLOW_OK && order.before==1 && order.after==1,"pre/restore/consumer ordering");
  check(first_object(original)->rect_params.width==30.5f,"shared original mutated");gst_buffer_unref(original);
  state->close();state->close();check(state->copied==1 && state->expanded==1 && state->restored==1,"bounded counters");
  gst_element_set_state(element,GST_STATE_NULL);gst_pad_unlink(input,sink);gst_pad_unlink(src,output);
  gst_pad_set_active(input,FALSE);gst_pad_set_active(output,FALSE);
  gst_object_unref(input);gst_object_unref(output);gst_object_unref(sink);gst_object_unref(src);gst_object_unref(element);
}
}
int main(int argc,char** argv){gst_init(&argc,&argv);
  if(argc==2 && std::strcmp(argv[1],"--multi-object-only")==0){multi_object_copy_contract();
    std::cout<<"multi-object pose copy fixture PASS\n";return 0;}
  multi_object_copy_contract();object_key_scope_and_source_duplicates();metadata_lifetime_and_concurrent_reader();
  defaults_and_binding();surface_header_contract();selection_and_geometry();extraction_diagnostic_without_tensor();context_subpredicate_diagnostic();pre_iou_candidate_diagnostic();contained_parent_correspondence();narrow_original_context_contract();actual_pad_order_and_close();
  check(payload_alive==0,"remaining user payload");std::cout<<"pose ROI context native CPU fixtures PASS\n";}
