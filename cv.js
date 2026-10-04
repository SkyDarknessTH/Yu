/* cv.js — offline registration engine (no dependencies, no network).
   Harris corners (grid-spread) → oriented BRIEF-256 → mutual ratio-test matching → RANSAC homography (DLT) → real confidence. */
(function(G){"use strict";
const T=0.019; // inlier threshold in normalized units ≈ 3 px @320
const gray=(d,n)=>{const g=new Float32Array(n);for(let i=0,j=0;i<n;i++,j+=4)g[i]=.299*d[j]+.587*d[j+1]+.114*d[j+2];return g};
function box(g,w,h,r){const t=new Float32Array(g.length),o=new Float32Array(g.length),n=2*r+1,cl=(v,m)=>v<0?0:v>m?m:v;
  for(let y=0;y<h;y++){let s=0;for(let x=-r;x<=r;x++)s+=g[y*w+cl(x,w-1)];for(let x=0;x<w;x++){t[y*w+x]=s/n;s+=g[y*w+cl(x+r+1,w-1)]-g[y*w+cl(x-r,w-1)]}}
  for(let x=0;x<w;x++){let s=0;for(let y=-r;y<=r;y++)s+=t[cl(y,h-1)*w+x];for(let y=0;y<h;y++){o[y*w+x]=s/n;s+=t[cl(y+r+1,h-1)*w+x]-t[cl(y-r,h-1)*w+x]}}return o}
const PAIRS=(()=>{let s=12345;const r=()=>(s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296,p=[],q=()=>(r()+r()+r()-1.5)*10;for(let i=0;i<256;i++)p.push([q(),q(),q(),q()]);return p})();
const pc=v=>{v=v-((v>>>1)&0x55555555);v=(v&0x33333333)+((v>>>2)&0x33333333);return Math.imul((v+(v>>>4))&0x0F0F0F0F,0x01010101)>>>24};
/* features(rgba,w,h) → {w,h,g,kp:[{x,y}],ds:Uint32Array(8n)}  (x,y in px) */
function features(rgba,w,h,per){per=per||14;const n=w*h,g=gray(rgba,n),Ix=new Float32Array(n),Iy=new Float32Array(n),a=new Float32Array(n),b=new Float32Array(n),c=new Float32Array(n);
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){const i=y*w+x,u=g[i+1]-g[i-1],v=g[i+w]-g[i-w];a[i]=u*u;b[i]=v*v;c[i]=u*v}
  const A=box(a,w,h,2),B=box(b,w,h,2),C=box(c,w,h,2),R=new Float32Array(n);let mx=0;
  for(let i=0;i<n;i++){R[i]=A[i]*B[i]-C[i]*C[i]-.04*(A[i]+B[i])*(A[i]+B[i]);if(R[i]>mx)mx=R[i]}
  const M=22,gx=8,gy=6,cell=[];for(let i=0;i<gx*gy;i++)cell.push([]);
  for(let y=M;y<h-M;y++)for(let x=M;x<w-M;x++){const i=y*w+x,r=R[i];if(r<mx*.002)continue;
    if(r>R[i-1]&&r>R[i+1]&&r>R[i-w]&&r>R[i+w]&&r>R[i-w-1]&&r>R[i-w+1]&&r>R[i+w-1]&&r>R[i+w+1])cell[Math.min(gy-1,(y*gy/h)|0)*gx+Math.min(gx-1,(x*gx/w)|0)].push([x,y,r])}
  const kp=[];cell.forEach(l=>{l.sort((p,q)=>q[2]-p[2]);l.slice(0,per).forEach(p=>kp.push({x:p[0],y:p[1]}))});
  const s=box(g,w,h,2),ds=new Uint32Array(kp.length*8);
  kp.forEach((k,j)=>{let m10=0,m01=0;for(let dy=-9;dy<=9;dy++)for(let dx=-9;dx<=9;dx++)if(dx*dx+dy*dy<=81){const v=s[(k.y+dy)*w+k.x+dx];m10+=dx*v;m01+=dy*v}
    const t=Math.atan2(m01,m10),co=Math.cos(t),si=Math.sin(t);
    for(let i=0;i<256;i++){const p=PAIRS[i],x1=Math.round(k.x+p[0]*co-p[1]*si),y1=Math.round(k.y+p[0]*si+p[1]*co),x2=Math.round(k.x+p[2]*co-p[3]*si),y2=Math.round(k.y+p[2]*si+p[3]*co);
      if(s[y1*w+x1]<s[y2*w+x2])ds[j*8+(i>>5)]|=1<<(i&31)}});
  return{w,h,g,kp,ds}}
function match(F,Rf){const na=F.kp.length,nb=Rf.kp.length,D=new Uint8Array(na*nb),bA=new Int32Array(na).fill(-1),bB=new Int32Array(nb).fill(-1),dA=new Uint8Array(nb).fill(255),out=[];
  for(let i=0;i<na;i++){let b1=999,b2=999,bj=-1;for(let j=0;j<nb;j++){let d=0;for(let k=0;k<8;k++)d+=pc(F.ds[i*8+k]^Rf.ds[j*8+k]);D[i*nb+j]=d;
      if(d<b1){b2=b1;b1=d;bj=j}else if(d<b2)b2=d;if(d<dA[j]){dA[j]=d;bB[j]=i}}
    if(bj>=0&&b1<=70&&b1<.8*b2)bA[i]=bj}
  for(let i=0;i<na;i++)if(bA[i]>=0&&bB[bA[i]]===i)out.push([i,bA[i]]);return out}
function solve(P){const A=Array.from({length:8},()=>new Float64Array(9));
  for(const[x,y,u,v]of P)for(const r of[[x,y,1,0,0,0,-u*x,-u*y,u],[0,0,0,x,y,1,-v*x,-v*y,v]])for(let i=0;i<8;i++){for(let j=0;j<8;j++)A[i][j]+=r[i]*r[j];A[i][8]+=r[i]*r[8]}
  for(let c=0;c<8;c++){let p=c;for(let r=c+1;r<8;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;if(Math.abs(A[p][c])<1e-9)return null;[A[c],A[p]]=[A[p],A[c]];
    for(let r=0;r<8;r++)if(r!==c){const f=A[r][c]/A[c][c];if(f)for(let k=c;k<9;k++)A[r][k]-=f*A[c][k]}}
  const H=[];for(let i=0;i<8;i++)H.push(A[i][8]/A[i][i]);H.push(1);return H}
const ap=(H,x,y)=>{const z=H[6]*x+H[7]*y+H[8];return[(H[0]*x+H[1]*y+H[2])/z,(H[3]*x+H[4]*y+H[5])/z,z]};
const mul=(A,B)=>{const o=[];for(let i=0;i<3;i++)for(let j=0;j<3;j++)o.push(A[i*3]*B[j]+A[i*3+1]*B[3+j]+A[i*3+2]*B[6+j]);return o};
function inv3(m){const[a,b,c,d,e,f,g,h,i]=m,A=e*i-f*h,B=f*g-d*i,C=d*h-e*g,det=a*A+b*B+c*C;return[A,c*h-b*i,b*f-c*e,B,a*i-c*g,c*d-a*f,C,b*g-a*h,a*e-b*d].map(v=>v/det)}
function ransac(P,seed){let s=seed||7;const rnd=n=>(s=(Math.imul(s,1103515245)+12345)>>>0)%n;let best=null,bi=[];
  for(let it=0;it<600;it++){const idx=new Set;while(idx.size<4)idx.add(rnd(P.length));const H=solve([...idx].map(i=>P[i]));if(!H)continue;
    const det=H[0]*H[4]-H[1]*H[3];if(!(det>.2&&det<5)||Math.abs(H[6])>2||Math.abs(H[7])>2)continue;
    const inl=[];for(let i=0;i<P.length;i++){const q=ap(H,P[i][0],P[i][1]);if(q[2]>0&&Math.hypot(q[0]-P[i][2],q[1]-P[i][3])<T)inl.push(i)}
    if(inl.length>bi.length){bi=inl;best=H;if(bi.length>.9*P.length)break}}
  if(!best||bi.length<6)return null;
  for(let r=0;r<3;r++){const H=solve(bi.map(i=>P[i]));if(!H)break;const inl=[];for(let i=0;i<P.length;i++){const q=ap(H,P[i][0],P[i][1]);if(q[2]>0&&Math.hypot(q[0]-P[i][2],q[1]-P[i][3])<T)inl.push(i)}
    if(inl.length<bi.length*.9)break;best=H;bi=inl}
  return{H:best,inl:bi}}
function bil(g,w,h,x,y){if(x<0||y<0||x>w-1||y>h-1)return NaN;const x0=x|0,y0=y|0,x1=Math.min(w-1,x0+1),y1=Math.min(h-1,y0+1),fx=x-x0,fy=y-y0;
  return(g[y0*w+x0]*(1-fx)+g[y0*w+x1]*fx)*(1-fy)+(g[y1*w+x0]*(1-fx)+g[y1*w+x1]*fx)*fy}
function similarity(gN,gR,w,h,Hi){let n=0,sa=0,sb=0,saa=0,sbb=0,sab=0;
  for(let y=2;y<h;y+=3)for(let x=2;x<w;x+=3){const q=ap(Hi,x,y),a=bil(gN,w,h,q[0],q[1]);if(a!==a)continue;const b=gR[y*w+x];n++;sa+=a;sb+=b;saa+=a*a;sbb+=b*b;sab+=a*b}
  if(n<w*h/9*.3)return 0;const va=saa-sa*sa/n,vb=sbb-sb*sb/n;return va>1e-6&&vb>1e-6?Math.max(0,(sab-sa*sb/n)/Math.sqrt(va*vb)):0}
const DEF={lim:10,acc:90,auto:75,rev:50};
/* register(F,Rf,opt) → alignment report. H maps NEW px → REF px (both at F.w×F.h). Always against the MASTER reference. */
function register(F,Rf,o){o=Object.assign({},DEF,o);const w=F.w,h=F.h,hw=w/2,hh=h/2,nrm=(x,y)=>[(x-hw)/hw,(y-hh)/hw];
  const r={st:"REJECT",conf:0,why:"",n:0,inl:0,ir:0,rms:0,tx:0,ty:0,rot:0,sc:1,persp:0,sim:0,H:null,crop:null,stab:"LOW",kpN:F.kp.length,kpR:Rf.kp.length};
  const M=match(F,Rf);r.n=M.length;if(M.length<10){r.why="not enough feature matches ("+M.length+")";return r}
  const P=M.map(([i,j])=>{const a=nrm(F.kp[i].x,F.kp[i].y),b=nrm(Rf.kp[j].x,Rf.kp[j].y);return[a[0],a[1],b[0],b[1]]});
  const R=ransac(P);if(!R){r.why="no consistent transform (RANSAC failed)";return r}
  const T=[1/hw,0,-1,0,1/hw,-hh/hw,0,0,1],Ti=[hw,0,hw,0,hw,hh,0,0,1],H=mul(mul(Ti,R.H),T).map(v=>v);const k=1/H[8];for(let i=0;i<9;i++)H[i]*=k;r.H=H;
  r.inl=R.inl.length;r.ir=r.inl/r.n;let e=0,x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;R.inl.forEach(i=>{const q=ap(R.H,P[i][0],P[i][1]);e+=(q[0]-P[i][2])**2+(q[1]-P[i][3])**2;const p=M[i],f=F.kp[p[0]];x0=Math.min(x0,f.x);x1=Math.max(x1,f.x);y0=Math.min(y0,f.y);y1=Math.max(y1,f.y)});
  r.rms=Math.sqrt(e/r.inl)*hw;const c=ap(H,hw,hh),a=ap(H,hw+1,hh),b=ap(H,hw,hh+1);r.tx=c[0]-hw;r.ty=c[1]-hh;
  const ax=a[0]-c[0],ay=a[1]-c[1],bx=b[0]-c[0],by=b[1]-c[1];r.sc=Math.sqrt(Math.abs(ax*by-ay*bx));r.rot=Math.atan2(ay,ax)*180/Math.PI;r.persp=(Math.abs(H[6])+Math.abs(H[7]))*w;
  const sp=(x1-x0)*(y1-y0)/(w*h),L=o.lim/100,bad=[];
  if(Math.hypot(r.tx,r.ty)/w>L)bad.push("translation");if(Math.abs(r.rot)>o.lim*.5)bad.push("rotation");if(Math.abs(r.sc-1)>L)bad.push("scale");if(r.persp>L)bad.push("perspective");
  r.sim=similarity(F.g,Rf.g,w,h,inv3(H));
  r.conf=Math.round(100*(.25*Math.min(1,r.ir/.7)+.2*Math.min(1,r.inl/40)+.2*Math.max(0,1-r.rms/3)+.25*r.sim+.1*Math.min(1,sp/.35)));
  r.stab=bad.length?"LOW":sp>.3&&r.inl>=25?"HIGH":"MEDIUM";
  const q=[[0,0],[w,0],[w,h],[0,h]].map(p=>ap(H,p[0],p[1])),cl=(v,m)=>Math.max(0,Math.min(m,v));
  r.crop=[cl(Math.max(q[0][0],q[3][0]),w)/w,cl(Math.max(q[0][1],q[1][1]),h)/h,cl(Math.min(q[1][0],q[2][0]),w)/w,cl(Math.min(q[3][1],q[2][1]),h)/h];
  if(r.crop[2]-r.crop[0]<.4||r.crop[3]-r.crop[1]<.4)bad.push("overlap");
  if(bad.length){r.st="REJECT";r.why="FRAME TOO DIFFERENT ("+bad.join(", ")+" over limit)";return r}
  r.st=r.conf>=o.acc?"ACCEPT":r.conf>=o.auto?"CORRECTED":r.conf>=o.rev?"REVIEW":"REJECT";if(r.st==="REJECT")r.why="alignment confidence too low ("+r.conf+"%)";return r}
/* warp(src RGBA, sw, sh, H(320-space, new→ref), w320, ow, oh) → RGBA at ref geometry ow×oh */
function warp(src,sw,sh,H,w,ow,oh){const Hi=inv3(H),k=ow/w,ks=sw/w,out=new Uint8ClampedArray(ow*oh*4);
  for(let y=0;y<oh;y++)for(let x=0;x<ow;x++){const X=(x+.5)/k,Y=(y+.5)/k,z=Hi[6]*X+Hi[7]*Y+Hi[8],u=(Hi[0]*X+Hi[1]*Y+Hi[2])/z*ks-.5,v=(Hi[3]*X+Hi[4]*Y+Hi[5])/z*ks-.5;
    if(u<0||v<0||u>sw-1||v>sh-1)continue;const x0=u|0,y0=v|0,x1=Math.min(sw-1,x0+1),y1=Math.min(sh-1,y0+1),fx=u-x0,fy=v-y0,o=(y*ow+x)*4;
    for(let c=0;c<3;c++)out[o+c]=(src[(y0*sw+x0)*4+c]*(1-fx)+src[(y0*sw+x1)*4+c]*fx)*(1-fy)+(src[(y1*sw+x0)*4+c]*(1-fx)+src[(y1*sw+x1)*4+c]*fx)*fy;out[o+3]=255}
  return out}
const intersect=l=>l.reduce((a,c)=>[Math.max(a[0],c[0]),Math.max(a[1],c[1]),Math.min(a[2],c[2]),Math.min(a[3],c[3])],[0,0,1,1]);
/* selfTest: synthetic textured scene + known transforms, report estimated-vs-true corner error */
function scene(w,h){let s=99;const r=()=>(s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296,d=new Uint8ClampedArray(w*h*4).fill(255);
  const rect=(x,y,a,b,v)=>{for(let j=Math.max(0,y|0);j<Math.min(h,y+b);j++)for(let i=Math.max(0,x|0);i<Math.min(w,x+a);i++){const o=(j*w+i)*4;d[o]=d[o+1]=d[o+2]=v}};
  rect(0,0,w,h,150);for(let i=0;i<90;i++)rect(r()*w,r()*h,6+r()*50,6+r()*40,r()*255);return d}
function selfTest(){const w=320,h=240,S=scene(w*2,h*2),F0=features(scene(w,h),w,h),out=[];
  const mk=(tx,ty,rd,sc,p)=>{const a=rd*Math.PI/180,c=Math.cos(a)*sc,s=Math.sin(a)*sc;return[c,-s,tx*w+(1-c)*w/2+s*h/2,s,c,ty*h+(1-c)*h/2-s*w/2,p,0,1]};
  [["translation 3%",.03,.02,0,1,0],["rotation 1°",0,0,1,1,0],["scale 2%",0,0,0,1.02,0],["trans+rot",.04,-.03,1.5,1,0],["perspective",0,0,0,1,.0004],["~10% displ.",.1,.06,3,1.05,0],["~20% displ.",.2,.12,6,1.1,0],["~30% displ.",.3,.18,9,1.15,0]].forEach(([n,tx,ty,rd,sc,p])=>{
    const Hg=mk(tx,ty,rd,sc,p),img=warp(scene(w,h),w,h,Hg,w,w,h),F=features(img,w,h),R=register(F,F0,{lim:n.startsWith("~30")?30:n.startsWith("~20")?20:10});
    let err=NaN;if(R.H){const Hi=inv3(Hg);err=0;[[0,0],[w,0],[w,h],[0,h]].forEach(p=>{const t=ap(Hg,p[0],p[1]),q=ap(R.H,t[0],t[1]);err=Math.max(err,Math.hypot(q[0]-p[0],q[1]-p[1]))})}
    out.push({case:n,st:R.st,conf:R.conf,inl:R.inl,n:R.n,errPx:err,why:R.why})});return out}
G.CV={features,match,register,warp,intersect,selfTest,inv3,ap,DEF};
})(typeof window!=="undefined"?window:globalThis);
