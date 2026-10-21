import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const element=id=>document.getElementById(id);
const response=await fetch('/data.json');
if(!response.ok)throw new Error(`Artifact request failed (${response.status})`);
const data=await response.json();
const finiteVector=(value,size)=>Array.isArray(value)&&value.length===size&&value.every(Number.isFinite);
const validGeometry=geometry=>Array.isArray(geometry?.vertices)&&geometry.vertices.length>=3&&geometry.vertices.every(vertex=>finiteVector(vertex,3))&&Array.isArray(geometry.faces)&&geometry.faces.length>0&&geometry.faces.every(face=>finiteVector(face,3)&&face.every(value=>Number.isInteger(value)&&value>=0&&value<geometry.vertices.length));
function validateDisplayMetadata(geometry,label){
  if(geometry.vertex_components!==undefined&&(!Array.isArray(geometry.vertex_components)||geometry.vertex_components.length!==geometry.vertices.length||!geometry.vertex_components.every(value=>Number.isInteger(value)&&value>=0)))throw new Error(`${label} vertex_components must contain one nonnegative integer per vertex`);
  if(geometry.crack_faces!==undefined&&(!Array.isArray(geometry.crack_faces)||geometry.crack_faces.length!==geometry.faces.length||!geometry.crack_faces.every(value=>typeof value==='boolean')))throw new Error(`${label} crack_faces must contain one boolean per triangle`);
  if(geometry.fixed_vertices!==undefined&&(!Array.isArray(geometry.fixed_vertices)||!geometry.fixed_vertices.every(value=>Number.isInteger(value)&&value>=0&&value<geometry.vertices.length)||new Set(geometry.fixed_vertices).size!==geometry.fixed_vertices.length))throw new Error(`${label} fixed_vertices must contain unique valid vertex indices`);
}
if(!Array.isArray(data.bodies)||!data.bodies.length||!Array.isArray(data.frames)||!data.frames.length)throw new Error('The artifact needs bodies and at least one frame');
const frames=data.frames;
frames.forEach((frame,index)=>{
  if(!Number.isFinite(frame.time)||(index&&frame.time<frames[index-1].time))throw new Error('Frame times must be finite and increasing');
  if(!Array.isArray(frame.poses)||frame.poses.length!==data.bodies.length)throw new Error(`Frame ${index} pose count does not match the body count`);
  frame.poses.forEach(pose=>{if(!finiteVector(pose.position,3)||!finiteVector(pose.orientation,4)||Math.abs(Math.hypot(...pose.orientation)-1)>1e-4)throw new Error(`Frame ${index} contains an invalid position or quaternion`);});
  if(frame.geometries&&(!Array.isArray(frame.geometries)||frame.geometries.length!==data.bodies.length||!frame.geometries.every(validGeometry)))throw new Error(`Frame ${index} contains invalid deformable geometry`);
  frame.geometries?.forEach((geometry,bodyIndex)=>validateDisplayMetadata(geometry,`Frame ${index}, body ${bodyIndex}`));
});
element('title').textContent=data.title||'Recorded fragment motion';
const source=String(data.source||'unspecified');
const sourceKind=source.toLowerCase();
const staticPreview=data.metadata?.simulation_executed===false;
element('speed').value=String(data.metadata?.suggested_playback_speed??(sourceKind.includes('femfx')?.001:1));
element('source').textContent=staticPreview?'Static geometry preview':sourceKind.includes('femfx')?'FEMFX numerical fracture reference':sourceKind.includes('trained')?'Trained fracture prediction':sourceKind.includes('benchmark')?'Contact benchmark':sourceKind.includes('reference')?'Reference geometry':'Recorded artifact';
element('metadata').textContent=JSON.stringify({source,metadata:data.metadata||{}},null,2);
const partial=Boolean(data.metadata?.partial_label_surrogate);
element('status').textContent=staticPreview?`${data.bodies.length} ${data.metadata?.artifact_role==='reference_encoding'?'reference':'generated'} fragments. Static shape preview; no dynamics or physical validation.`:`${data.bodies.length} solid ${data.bodies.length===1?'body':'bodies'} · ${frames.length} recorded ${frames.length===1?'frame':'frames'}${partial?' · Partial-label training surrogate':''}. Poses are replayed from Python.`;
const canvas=element('scene');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.toneMapping=THREE.ACESFilmicToneMapping;
const scene=new THREE.Scene();
scene.background=new THREE.Color('#131c26');
const camera=new THREE.PerspectiveCamera(42,1,.001,1000);
camera.up.set(0,0,1);
const controls=new OrbitControls(camera,canvas);
controls.enableDamping=true;
scene.add(new THREE.HemisphereLight('#d9e9fa','#4e5663',2.4));
const light=new THREE.DirectionalLight('#fff1db',3.0);
light.position.set(4,-5,8);
scene.add(light);
const group=new THREE.Group();
scene.add(group);
const palette=['#b5cbd8','#cdab86','#8bb9ad','#ab9dc3','#d4bc8c','#8aaec6','#c39794'];
const componentPalette=['#86c9e8','#e6aa72','#9dd4a0','#c8a3e6','#e4d281','#80d3c1','#e698ad','#a4b8ee','#d0bb9a','#9ed3db'];
const componentColor=component=>component<componentPalette.length?new THREE.Color(componentPalette[component]):new THREE.Color().setHSL((component*.61803398875)%1,.48,.68);
const crackColor=new THREE.Color('#ff873d');
const displayGeometries=[...data.bodies,...frames.flatMap(frame=>frame.geometries||[])];
for(const [id,key] of [['components','vertex_components'],['cracks','crack_faces'],['supports','fixed_vertices']]){
  element(id).disabled=!displayGeometries.some(geometry=>geometry?.[key]!==undefined);
  if(element(id).disabled)element(id).parentElement.title='Unavailable in this artifact';
}
function updateGeometry(mesh,recorded){
  const positions=new Float32Array(recorded.faces.length*9),colors=new Float32Array(positions.length);
  recorded.faces.forEach((face,faceIndex)=>{
    const color=element('cracks').checked&&recorded.crack_faces?.[faceIndex]?crackColor:element('components').checked&&recorded.vertex_components?componentColor(recorded.vertex_components[face[0]]):mesh.userData.baseColor;
    face.forEach((vertex,corner)=>{
      const offset=faceIndex*9+corner*3;
      positions.set(recorded.vertices[vertex],offset);
      color.toArray(colors,offset);
    });
  });
  if(mesh.geometry.getAttribute('position')?.array.length!==positions.length){
    mesh.geometry.dispose();
    mesh.geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
    mesh.geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
    mesh.geometry.deleteAttribute('normal');
  }else{
    for(const [name,values] of [['position',positions],['color',colors]]){
      const attribute=mesh.geometry.getAttribute(name);attribute.array.set(values);attribute.needsUpdate=true;
    }
  }
  mesh.geometry.setIndex(null);
  mesh.geometry.computeVertexNormals();
  mesh.geometry.computeBoundingBox();
  mesh.geometry.computeBoundingSphere();
  const supportPositions=(recorded.fixed_vertices||[]).flatMap(vertex=>recorded.vertices[vertex]);
  const supportGeometry=mesh.userData.supports.geometry;
  if(supportGeometry.getAttribute('position')?.array.length!==supportPositions.length){
    supportGeometry.dispose();supportGeometry.setAttribute('position',new THREE.Float32BufferAttribute(supportPositions,3));
  }else{
    const attribute=supportGeometry.getAttribute('position');attribute.array.set(supportPositions);attribute.needsUpdate=true;
  }
  supportGeometry.computeBoundingSphere();
  mesh.userData.supports.visible=element('supports').checked&&supportPositions.length>0;
  mesh.userData.wireframe.visible=element('wireframe').checked;
  mesh.userData.recordedGeometry=recorded;
}
const meshes=data.bodies.map((body,index)=>{
  if(!Array.isArray(body.vertices)||body.vertices.length<4||!body.vertices.every(vertex=>finiteVector(vertex,3)))throw new Error(`Body ${index} has invalid vertices`);
  if(!Array.isArray(body.faces)||!body.faces.length||!body.faces.every(face=>finiteVector(face,3)&&face.every(value=>Number.isInteger(value)&&value>=0&&value<body.vertices.length)))throw new Error(`Body ${index} has invalid triangle indices`);
  validateDisplayMetadata(body,`Body ${index}`);
  const geometry=new THREE.BufferGeometry();
  const color=Array.isArray(body.color)?new THREE.Color().fromArray(body.color):new THREE.Color(body.color||palette[index%palette.length]);
  const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.72,metalness:.03,flatShading:true,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1}));
  const wireframe=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:'#10202b',wireframe:true,transparent:true,opacity:.55}));
  const supports=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:'#64fff0',size:.01,sizeAttenuation:true,depthTest:false}));
  supports.renderOrder=2;
  mesh.userData={baseColor:color,wireframe,supports};
  mesh.add(wireframe,supports);
  updateGeometry(mesh,body);
  group.add(mesh);
  return mesh;
});
const setPoses=frame=>frame.poses.forEach((pose,index)=>{
  const mesh=meshes[index];
  mesh.position.fromArray(pose.position);mesh.quaternion.fromArray(pose.orientation);
  if(frame.geometries){
    updateGeometry(mesh,frame.geometries[index]);
  }
});
setPoses(frames[0]);
const bounds=new THREE.Box3();
const transform=new THREE.Matrix4(),position=new THREE.Vector3(),orientation=new THREE.Quaternion(),scale=new THREE.Vector3(1,1,1);
meshes.forEach(mesh=>mesh.geometry.computeBoundingBox());
frames.forEach(frame=>frame.poses.forEach((pose,index)=>{transform.compose(position.fromArray(pose.position),orientation.fromArray(pose.orientation),scale);const box=frame.geometries?new THREE.Box3().setFromArray(frame.geometries[index].vertices.flat()):meshes[index].geometry.boundingBox.clone();bounds.union(box.applyMatrix4(transform));}));
const center=bounds.getCenter(new THREE.Vector3());
const radius=Math.max(bounds.getSize(new THREE.Vector3()).length(),1e-6);
meshes.forEach(mesh=>{mesh.userData.supports.material.size=radius*.009;});
camera.near=radius/1000;camera.far=radius*1000;
camera.position.copy(center).add(new THREE.Vector3(1.4,-2.1,1.5).multiplyScalar(radius));
controls.target.copy(center);controls.minDistance=radius*.05;controls.maxDistance=radius*30;
const grid=new THREE.GridHelper(radius*5,30,'#496174','#2a3949');
grid.rotation.x=Math.PI/2;grid.position.set(center.x,center.y,0);
grid.material.transparent=true;grid.material.opacity=.45;
scene.add(grid);
const topologyEdges=data.topology_edges;
if(Array.isArray(topologyEdges)&&topologyEdges.length){
  if(!topologyEdges.every(edge=>Array.isArray(edge)&&edge.length===2&&edge.every(point=>finiteVector(point,3))))throw new Error('Topology edges must contain pairs of world coordinates');
  const geometry=new THREE.BufferGeometry().setFromPoints(topologyEdges.flat().map(point=>new THREE.Vector3(...point)));
  const overlay=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:'#f3d6a1',transparent:true,opacity:.7,depthTest:false}));
  overlay.visible=false;scene.add(overlay);element('topology').style.display='block';
  element('topology').onclick=()=>{overlay.visible=!overlay.visible;element('topology').textContent=overlay.visible?'Hide topology':'Show topology';};
}
let index=0,playing=false,playhead=frames[0].time,lastTick=performance.now();
element('scrub').max=frames.length-1;
element('play').disabled=frames.length===1;
const metricKeys=[...new Set(frames.flatMap(frame=>Object.keys(frame.metrics||{})))].filter(key=>frames.some(frame=>Number.isFinite(frame.metrics?.[key])));
metricKeys.forEach(key=>{const option=document.createElement('option');option.value=key;option.textContent=key.replaceAll('_',' ');element('metric').append(option);});
if(metricKeys.length)element('metrics').style.display='flex';
function plot(){
  if(!metricKeys.length)return;
  const chart=element('plot'),scale=Math.min(devicePixelRatio,2),width=chart.clientWidth,height=chart.clientHeight;
  chart.width=width*scale;chart.height=height*scale;
  const context=chart.getContext('2d');context.scale(scale,scale);context.clearRect(0,0,width,height);
  const key=element('metric').value,values=frames.map(frame=>frame.metrics?.[key]);
  const finite=values.filter(Number.isFinite),minimum=Math.min(...finite),maximum=Math.max(...finite),span=maximum-minimum||1;
  const duration=frames.at(-1).time-frames[0].time||1;
  const x=i=>12+(width-24)*(frames[i].time-frames[0].time)/duration,y=value=>height-21-(height-35)*(value-minimum)/span;
  context.strokeStyle='#334454';context.beginPath();context.moveTo(12,height-20);context.lineTo(width-12,height-20);context.stroke();
  context.strokeStyle='#83b6c9';context.lineWidth=1.6;context.beginPath();let connected=false;
  values.forEach((value,i)=>{if(Number.isFinite(value)){if(connected)context.lineTo(x(i),y(value));else context.moveTo(x(i),y(value));connected=true;}else connected=false;});context.stroke();
  context.strokeStyle='#d2b688';context.beginPath();context.moveTo(x(index),8);context.lineTo(x(index),height-17);context.stroke();
  context.fillStyle='#9eafbd';context.font='10px Segoe UI';context.fillText(`${key.replaceAll('_',' ')}: ${Number.isFinite(values[index])?values[index].toExponential(3):'unavailable'}`,12,height-4);
}
function showFrame(next){
  index=next;setPoses(frames[index]);element('scrub').value=index;
  if(data.metadata?.format==='fracture-world.neural-rollout.v1')element('status').textContent=`Engine-free ${data.metadata.method} rollout · ${frames.length} frames · ${frames[index].metrics?.material_components??'?'} material components · ${frames[index].metrics?.broken_faces??'?'} broken faces`;
  const precision=frames.at(-1).time<.1?5:3;
  element('time').textContent=`${frames[index].time.toFixed(precision)} / ${frames.at(-1).time.toFixed(precision)} s`;plot();
}
element('play').onclick=()=>{playing=!playing;lastTick=performance.now();element('play').textContent=playing?'Pause':'Play';};
element('scrub').oninput=()=>{playing=false;element('play').textContent='Play';showFrame(Number(element('scrub').value));playhead=frames[index].time;};
element('metric').onchange=plot;
for(const id of ['components','cracks','wireframe','supports'])element(id).onchange=()=>meshes.forEach(mesh=>updateGeometry(mesh,mesh.userData.recordedGeometry));
element('screenshot').onclick=()=>{
  const capturedIndex=index;
  renderer.render(scene,camera);
  canvas.toBlob(blob=>{
    if(!blob){element('status').textContent='PNG export failed. The current frame could not be captured.';return;}
    const url=URL.createObjectURL(blob),link=document.createElement('a');
    const title=String(data.title||'fracture-world').replace(/[^a-z0-9_-]+/gi,'-').replace(/^-+|-+$/g,'')||'fracture-world';
    link.href=url;link.download=`${title}-frame-${String(capturedIndex).padStart(5,'0')}.png`;link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  },'image/png');
};
new ResizeObserver(()=>{const rectangle=canvas.parentElement.getBoundingClientRect();renderer.setSize(rectangle.width,rectangle.height,false);camera.aspect=rectangle.width/rectangle.height;camera.updateProjectionMatrix();plot();}).observe(canvas.parentElement);
showFrame(0);
function animate(now){
  const elapsed=Math.min((now-lastTick)/1000,.1);lastTick=now;
  if(playing){playhead+=elapsed*Number(element('speed').value);if(playhead>frames.at(-1).time){playhead=frames[0].time;index=0;}let next=index;while(next+1<frames.length&&frames[next+1].time<=playhead)next++;if(next!==Number(element('scrub').value))showFrame(next);}
  controls.update();renderer.render(scene,camera);requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
