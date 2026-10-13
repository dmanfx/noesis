import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const clone = (v) => JSON.parse(JSON.stringify(v));
const fetchJSON = async (url) => {const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json();};
const tell = (message, error=false) => {$('notice').textContent = message; $('notice').classList.toggle('error', error);};
let model, original, metadata, checks=[], fixtures=[], selected=null, savedRevision, undo=[], redo=[];
let cloud, selectedFixture=0, measuring=false, measurementPoints=[];
const container = $('viewport');
const scene = new THREE.Scene(); scene.background = new THREE.Color('#f8fafb');
const renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true, powerPreference:'low-power'});
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); container.appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(42, 1, .05, 150);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = false;
const semantics = new THREE.Group(), person = new THREE.Group(), measures = new THREE.Group();
scene.add(semantics, person, measures);
const grid = new THREE.GridHelper(14,28,'#bdceda','#dde5eb'); grid.position.y=-.04; scene.add(grid);
const center = new THREE.Vector3(); let extent=8, queued=false, mode='perspective';
const raycaster = new THREE.Raycaster(), pointer=new THREE.Vector2();
raycaster.params.Points.threshold=.025;

function render() {if (queued) return; queued=true; requestAnimationFrame(()=>{queued=false; renderer.render(scene,camera);});}
controls.addEventListener('change', render);
new ResizeObserver(()=>{const w=container.clientWidth,h=container.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); render();}).observe(container);
function dispose(group) {while(group.children.length) {const obj=group.children.pop(); obj.traverse(n=>{n.geometry?.dispose(); if(n.material) {for(const m of Array.isArray(n.material)?n.material:[n.material]) {m.map?.dispose();m.dispose();}}});obj.parent=null;}}
function visible(e) {const filter=$('states').value; return filter==='all'||e.review_state===filter;}
function color(e) {return e.review_state==='rejected'?'#d66775':e.kind==='wall'?'#2784b2':e.kind==='opening'?'#2f9f82':'#d38535';}
function basis(e) {const r=e.geometry.basis_row_major;return new THREE.Matrix4().makeBasis(new THREE.Vector3(r[0],r[3],r[6]),new THREE.Vector3(r[1],r[4],r[7]),new THREE.Vector3(r[2],r[5],r[8]));}
function sprite(text,position,tint) {const canvas=document.createElement('canvas');canvas.width=512;canvas.height=80;const ctx=canvas.getContext('2d');ctx.fillStyle='#ffffffeb';ctx.fillRect(0,0,512,80);ctx.fillStyle=tint;ctx.font='500 34px system-ui';ctx.textAlign='center';ctx.fillText(text.slice(0,28),256,51);const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;const s=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false,transparent:true}));s.position.copy(position);s.scale.set(1.15,.18,1);return s;}
function rebuild() {
  dispose(semantics);
  if (!model) return;
  const elements=$('source').value==='original'?original.elements:model.elements;
  for(const e of elements.filter(visible)) {
    const s=e.geometry.size_m;
    const group=new THREE.Group();group.position.fromArray(e.geometry.center_m);group.quaternion.setFromRotationMatrix(basis(e));
    const geometry=new THREE.BoxGeometry(s[0],Math.max(s[1],.009),s[2]);
    const fill=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:color(e),transparent:true,opacity:e.element_id===selected ? .24 : e.review_state==='rejected' ? .035 : e.kind==='wall' ? .04 : .07,depthWrite:false,side:THREE.DoubleSide}));
    fill.userData.element_id=e.element_id;group.add(fill);
    const lines=new THREE.LineSegments(new THREE.EdgesGeometry(geometry),new THREE.LineBasicMaterial({color:e.element_id===selected?'#e5a928':color(e),transparent:true,opacity:e.review_state==='rejected'?.6:.95}));group.add(lines);semantics.add(group);
    if(e.kind==='object'&&(e.review_state==='accepted'||e.element_id===selected)) {const p=new THREE.Vector3(...e.geometry.center_m);p.y=mode==='plan'?2.6:p.y+s[2]/2+.15;semantics.add(sprite(e.label,p,color(e)));}
  }
  semantics.visible=$('layer').value!=='cloud';
  if(cloud) cloud.visible=$('layer').value!=='semantics';
  updatePerson();render();
  window.semanticReview.metrics.geometryCount=renderer.info.memory.geometries;
}
function view(next) {
  mode=next;$('perspective').classList.toggle('active',next==='perspective');$('plan').classList.toggle('active',next==='plan');
  controls.target.copy(center);camera.up.set(0,next==='plan'?0:1,next==='plan'?-1:0);
  camera.position.copy(center).add(next==='plan'?new THREE.Vector3(0,extent*1.8,0):new THREE.Vector3(extent*.95,extent*1.15,extent*1.25));
  controls.enableRotate=next!=='plan';controls.update();rebuild();
}
function pushUndo() {undo.push(clone(model.elements));if(undo.length>60) undo.shift();redo=[];}
function markDirty() {window.semanticReview.metrics.dirty=true;tell('Unsaved review changes. Add an evidence note, then save the draft.');}
function refreshQueue() {
  const counts={accepted:0,proposed:0,rejected:0};for(const e of model.elements)counts[e.review_state]++;
  $('counts').textContent=`${counts.accepted} accepted · ${counts.proposed} pending · ${counts.rejected} rejected`;
  const search=$('search').value.toLowerCase(), kind=$('kind').value;$('queue').replaceChildren();
  for(const e of model.elements.filter(e=>(kind==='all'||kind===e.kind)&&(e.label+' '+e.element_id).toLowerCase().includes(search))) {
    const row=document.createElement('div');row.className='queue-row'+(e.element_id===selected?' selected':'');row.dataset.id=e.element_id;
    const label=document.createElement('span');label.textContent=e.label;const tag=document.createElement('small');tag.className=e.review_state;tag.textContent=e.review_state==='proposed'?'pending':e.review_state;
    row.append(label,tag);row.addEventListener('click',()=>select(e.element_id));$('queue').append(row);
  }
  $('undo').disabled=!undo.length;$('redo').disabled=!redo.length;
}
function select(id) {
  const e=model.elements.find(e=>e.element_id===id);if(!e)return;selected=id;
  $('selected-title').textContent=e.element_id;$('label').value=e.label;$('category').value=e.category;
  for(const [j,axis] of ['x','y','z'].entries()) {$('c'+axis).value=String(e.geometry.center_m[j]);$('s'+axis).value=String(e.geometry.size_m[j]);}
  const r=e.geometry.basis_row_major;$('yaw').value=(Math.atan2(r[6],r[0])*180/Math.PI).toFixed(6);$('yaw').dataset.original=$('yaw').value;
  $('wall-endpoints').hidden=e.kind!=='wall';
  if(e.kind==='wall'){for(const [id,value] of [['ax',e.geometry.center_m[0]-r[0]*e.geometry.size_m[0]/2],['az',e.geometry.center_m[2]-r[6]*e.geometry.size_m[0]/2],['bx',e.geometry.center_m[0]+r[0]*e.geometry.size_m[0]/2],['bz',e.geometry.center_m[2]+r[6]*e.geometry.size_m[0]/2]]){$(id).value=String(value);$(id).dataset.original=String(value);}}
  $('decision').value=e.review_state;$('wall-reference').hidden=e.kind!=='opening';$('wall-id').replaceChildren();
  for(const wall of model.elements.filter(e=>e.kind==='wall')){const o=document.createElement('option');o.value=wall.element_id;o.textContent=wall.label;$('wall-id').append(o);}
  $('wall-id').value=e.wall_id||'';
  const row=checks.find(r=>r.element_id===id);$('issues').style.display=row?.issues?.length?'block':'none';
  $('issues').textContent=row?.issues?.map(i=>i.code.replaceAll('_',' ')+(i.segment_overrun_m?`: ${i.segment_overrun_m.toFixed(3)} m beyond wall`:'')).join('; ')||'';
  refreshQueue();rebuild();
}
function download(value,name) {const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function post(url,value) {const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});const answer=await r.json();if(!r.ok)throw new Error(answer.error||r.statusText);return answer;}
function contextText(f) {
  const result=f.context;const p=f.entity.position;
  $('context').replaceChildren();const position=document.createElement('div');position.textContent=`${f.name==='window'?'Window recording':'Hearth recording'} · measured (${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}) m`;$('context').append(position);
  if(result.status==='advisory') {const nearby=document.createElement('strong');nearby.textContent=result.nearby.length?result.nearby.map(n=>`${n.label}: ${n.ground_footprint_distance_m.toFixed(2)} m`).join(' · '):'No accepted object within 1 m';$('context').append(nearby);}else{const status=document.createElement('strong');status.textContent=`Unknown · ${result.reason}`;$('context').append(status);}
  const qualifier=document.createElement('div');qualifier.textContent='Distance to a provisional ground footprint; it does not establish sitting, activity, or room membership.';$('context').append(qualifier);
}
function updatePerson() {
  dispose(person);if(!fixtures.length)return;
  const fixture=fixtures[selectedFixture];contextText(fixture);if(!$('people').checked)return;
  const p=fixture.entity.position;const dot=new THREE.Mesh(new THREE.SphereGeometry(.09,18,12),new THREE.MeshBasicMaterial({color:'#c73f63'}));dot.position.set(p.x,.08,p.z);person.add(dot);
  const upright=new THREE.Mesh(new THREE.CylinderGeometry(.015,.015,1.45,8),new THREE.MeshBasicMaterial({color:'#c73f63'}));upright.position.set(p.x,.8,p.z);person.add(upright);
  person.add(sprite(fixture.name+' · recorded',new THREE.Vector3(p.x,mode==='plan'?2.9:1.65,p.z),'#b73559'));
}
async function loadCloud(full=false) {
  const prefix=full?'cloud-full':'cloud';const [xyz,rgb]=await Promise.all([fetch(`/${prefix}.xyz.f32`).then(r=>r.arrayBuffer()),fetch(`/${prefix}.rgb.u8`).then(r=>r.arrayBuffer())]);
  const positions=new Float32Array(xyz),bytes=new Uint8Array(rgb);if(positions.length!==bytes.length||positions.length%3)throw new Error('Cloud buffer length mismatch');
  const colors=new Float32Array(bytes.length),colorValue=new THREE.Color();
  for(let i=0;i<bytes.length;i+=3){colorValue.setRGB(bytes[i]/255,bytes[i+1]/255,bytes[i+2]/255,THREE.SRGBColorSpace);colors.set([colorValue.r,colorValue.g,colorValue.b],i);}
  if(cloud){scene.remove(cloud);cloud.geometry.dispose();cloud.material.dispose();}
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));geometry.computeBoundingBox();geometry.boundingBox.getCenter(center);center.y=.8;const dimensions=geometry.boundingBox.getSize(new THREE.Vector3());extent=Math.max(dimensions.x,dimensions.z);
  grid.position.x=center.x;grid.position.z=center.z;
  cloud=new THREE.Points(geometry,new THREE.PointsMaterial({size:.019,vertexColors:true,transparent:true,opacity:.8,depthWrite:true}));scene.add(cloud);
  $('metrics').textContent=`${(positions.length/3).toLocaleString()} display points / ${metadata.full_points.toLocaleString()} model-input points · metres`;
  window.semanticReview.metrics.displayPoints=positions.length/3;rebuild();
}
async function reload() {
  const state=await fetchJSON('/api/state');model=state.map;checks=state.diagnostics;fixtures=state.fixtures;savedRevision=model.revision_sha256;undo=[];redo=[];
  $('fixture').replaceChildren();fixtures.forEach((f,i)=>{const option=document.createElement('option');option.value=i;option.textContent=f.name==='window'?'Window · occupied Oct 3':'Hearth · occupied Oct 2';$('fixture').append(option);});
  window.semanticReview.metrics.dirty=false;window.semanticReview.metrics.revision=savedRevision;
  $('provenance').textContent=`${model.coordinate_frame} / metres / +Y up\nWorld revision: ${model.world_frame_revision}\nWorld transform SHA256: ${model.world_transform_sha256}\nModel: ${metadata.model} / seed ${metadata.seed}\n${model.conversion_note}\n\nSource fingerprints:\n`+model.source_artifacts.map(a=>`${a.role}: ${a.sha256}\n${a.path}`).join('\n\n');
  refreshQueue();select(selected||'sept.object.10');rebuild();tell('September proposals loaded. 17 provisional accepts, 2 rejections; remaining elements await review.');
}
window.semanticReview={metrics:{loaded:false,dirty:false},getMap:()=>clone(model),select:(id)=>select(id),view:(v)=>view(v),rendererInfo:renderer.info};
renderer.domElement.addEventListener('click',(event)=>{
  const rect=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);
  if(measuring){const p=new THREE.Vector3();raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,1,0),0),p);if(!Number.isFinite(p.x))return;measurementPoints.push(p);if(measurementPoints.length>2)measurementPoints=[p];dispose(measures);for(const v of measurementPoints){const dot=new THREE.Mesh(new THREE.SphereGeometry(.05),new THREE.MeshBasicMaterial({color:'#ae4264'}));dot.position.copy(v);measures.add(dot);}if(measurementPoints.length===2){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(measurementPoints),new THREE.LineBasicMaterial({color:'#ae4264'}));measures.add(line);$('measure-label').textContent=`${measurementPoints[0].distanceTo(measurementPoints[1]).toFixed(3)} m · display-picked endpoints, unqualified`;}else{$('measure-label').textContent='Choose the second floor endpoint';}render();return;}
  const hit=raycaster.intersectObjects(semantics.children,true).find(h=>h.object.userData.element_id);if(hit)select(hit.object.userData.element_id);
});
$('edit-form').addEventListener('submit',(event)=>{
  event.preventDefault();
  try{
    if(!selected)return;
    if($('source').value==='original')throw new Error('Switch Geometry to Review draft before editing.');
    if(!$('reason').value.trim())throw new Error('Add an evidence / reason note before editing.');
    const index=model.elements.findIndex(e=>e.element_id===selected),e=clone(model.elements[index]);
    e.label=$('label').value.trim();e.category=$('category').value.trim();
    e.geometry.center_m=['x','y','z'].map(v=>Number($('c'+v).value));
    e.geometry.size_m=['x','y','z'].map(v=>Number($('s'+v).value));
    if(![...e.geometry.center_m,...e.geometry.size_m,Number($('yaw').value)].every(Number.isFinite))throw new Error('Geometry must contain finite numbers.');
    if(e.geometry.size_m[0]<=0||e.geometry.size_m[2]<=0||e.geometry.size_m[1]<0||(e.kind==='object'&&e.geometry.size_m[1]===0))throw new Error('Dimensions must be positive; wall/opening width may remain zero.');
    if($('yaw').value!==$('yaw').dataset.original){const a=Number($('yaw').value)*Math.PI/180,c=Math.cos(a),s=Math.sin(a);e.geometry.basis_row_major=[c,s,0,0,0,1,s,-c,0];}
    if(e.kind==='wall'&&['ax','az','bx','bz'].some(id=>$(id).value!==$(id).dataset.original)){
      const ax=Number($('ax').value),az=Number($('az').value),bx=Number($('bx').value),bz=Number($('bz').value),dx=bx-ax,dz=bz-az,length=Math.hypot(dx,dz);
      if(!Number.isFinite(length)||length<.001)throw new Error('Wall endpoints must be finite and distinct.');
      e.geometry.center_m[0]=(ax+bx)/2;e.geometry.center_m[2]=(az+bz)/2;e.geometry.size_m[0]=length;
      const c=dx/length,s=dz/length;e.geometry.basis_row_major=[c,s,0,0,0,1,s,-c,0];
    }
    e.review_state=$('decision').value;e.review_actor=$('actor').value.trim();e.review_reason=$('reason').value.trim();
    if(e.kind==='opening')e.wall_id=$('wall-id').value;
    pushUndo();model.elements[index]=e;markDirty();select(selected);
  }catch(error){tell(error.message,true);}
});
$('save').onclick=async()=>{try{const answer=await post('/api/save',{base_revision:savedRevision,elements:model.elements,actor:$('actor').value,reason:$('reason').value});model=answer.map;savedRevision=model.revision_sha256;checks=answer.diagnostics||checks;const state=await fetchJSON('/api/state');fixtures=state.fixtures;undo=[];redo=[];window.semanticReview.metrics.dirty=false;window.semanticReview.metrics.revision=savedRevision;refreshQueue();select(selected);tell(answer.saved?'Draft saved with an immutable change record.':'No changes to save.');}catch(error){tell(error.message,true);}};
$('reload').onclick=()=>reload().catch(e=>tell(e.message,true));
$('export').onclick=async()=>{try{if(window.semanticReview.metrics.dirty)throw new Error('Save changes before exporting accepted elements.');const result=await post('/api/export',{});download(result.map,result.download_name);tell(`Exported ${result.map.elements.length} accepted elements in Noesis world metres; physical accuracy remains unqualified.`);}catch(error){tell(error.message,true);}};
$('download').onclick=()=>{if(window.semanticReview.metrics.dirty){tell('Save the draft first so the downloaded revision is valid.',true);return;}download(model,'living-room-review-draft.json');};
$('import').onclick=()=>$('import-file').click();
$('import-file').onchange=async(event)=>{try{const file=event.target.files[0];if(!file)return;if(file.size>2_000_000)throw new Error('Draft exceeds 2 MB');const incoming=JSON.parse(await file.text());for(const key of ['contract','contract_version','map_id','coordinate_frame','units','world_frame_revision','world_transform_sha256','proposal_revision_sha256'])if(incoming[key]!==model[key])throw new Error('Imported draft has an incompatible source/frame binding');if(!Array.isArray(incoming.elements))throw new Error('Missing elements');pushUndo();model.elements=incoming.elements;markDirty();refreshQueue();select(model.elements[0].element_id);tell('Draft imported for review. Save validates every element and records the changes.');}catch(error){tell(error.message,true);}finally{event.target.value='';}};
$('undo').onclick=()=>{if(!undo.length)return;redo.push(clone(model.elements));model.elements=undo.pop();markDirty();refreshQueue();select(selected);};
$('redo').onclick=()=>{if(!redo.length)return;undo.push(clone(model.elements));model.elements=redo.pop();markDirty();refreshQueue();select(selected);};
$('add').onclick=()=>{if(!$('reason').value.trim()){tell('Add evidence / reason before adding geometry.',true);return;}pushUndo();const e={element_id:`manual.object.${crypto.randomUUID()}`,kind:'object',category:'unknown',label:'New object',geometry:{center_m:[center.x,.5,center.z],size_m:[1,1,1],basis_row_major:[1,0,0,0,0,1,0,-1,0]},wall_id:null,review_state:'proposed',review_actor:$('actor').value,review_reason:$('reason').value,physical_qualification:'not_independently_qualified',source_kind:'manual_annotation'};model.elements.push(e);markDirty();select(e.element_id);};
$('search').oninput=refreshQueue;$('kind').onchange=refreshQueue;$('states').onchange=rebuild;$('layer').onchange=()=>{$('scene-label').textContent=$('layer').selectedOptions[0].text;rebuild();};$('source').onchange=()=>{$('scene-label').textContent=$('source').value==='original'?'Original model output · 39 unreviewed proposals':'Saved review draft · provisional decisions';rebuild();};
$('perspective').onclick=()=>view('perspective');$('plan').onclick=()=>view('plan');
$('measure').onclick=()=>{measuring=!measuring;$('measure').classList.toggle('active',measuring);$('measure-label').style.display=measuring?'block':'none';if(measuring){view('plan');$('measure-label').textContent='Choose two endpoints on the floor plane';}};
$('fixture').onchange=()=>{selectedFixture=Number($('fixture').value);updatePerson();render();};$('people').onchange=()=>{updatePerson();render();};
$('fullcloud').onclick=async()=>{try{$('fullcloud').disabled=true;await loadCloud(true);tell('All 536,074 inference-input points are displayed. Model geometry is unchanged.');}catch(error){tell(error.message,true);$('fullcloud').disabled=false;}};
try{[metadata,original]=await Promise.all([fetchJSON('/package.json'),fetchJSON('/proposals.json')]);if(metadata.seed_stability_url){const stability=await fetchJSON(metadata.seed_stability_url);$('stability-panel').hidden=false;$('stability-summary').textContent=Object.entries(stability.seed_comparison).map(([seed,c])=>`Seed ${seed}: ${c.walls} walls, ${c.doors} doors, ${c.windows} windows; ${c.categories.tv||0} TV(s), ${c.categories.curtain||0} curtains`).join(' · ');$('stability-image').src='/evidence/seed-stability.png';}await reload();await loadCloud();view('perspective');window.semanticReview.metrics.loaded=true;}catch(error){tell(error.message,true);console.error(error);}
window.addEventListener('beforeunload',(event)=>{if(window.semanticReview.metrics.dirty){event.preventDefault();event.returnValue='';}});
