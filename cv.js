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
function features(rgba,w,h,per,mask){per=per||14;const n=w*h,g=gray(rgba,n),Ix=new Float32Array(n),Iy=new Float32Array(n),a=new Float32Array(n),b=new Float32Array(n),c=new Float32Array(n);
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){const i=y*w+x,u=g[i+1]-g[i-1],v=g[i+w]-g[i-w];a[i]=u*u;b[i]=v*v;c[i]=u*v}
  const A=box(a,w,h,2),B=box(b,w,h,2),C=box(c,w,h,2),R=new Float32Array(n);let mx=0;
  for(let i=0;i<n;i++){R[i]=A[i]*B[i]-C[i]*C[i]-.04*(A[i]+B[i])*(A[i]+B[i]);if(R[i]>mx)mx=R[i]}
  const M=22,gx=8,gy=6,cell=[];for(let i=0;i<gx*gy;i++)cell.push([]);
  for(let y=M;y<h-M;y++)for(let x=M;x<w-M;x++){const i=y*w+x,r=R[i];if(r<mx*.002)continue;
    if(r>R[i-1]&&r>R[i+1]&&r>R[i-w]&&r>R[i+w]&&r>R[i-w-1]&&r>R[i-w+1]&&r>R[i+w-1]&&r>R[i+w+1])cell[Math.min(gy-1,(y*gy/h)|0)*gx+Math.min(gx-1,(x*gx/w)|0)].push([x,y,r])}
  const kp=[];cell.forEach(l=>{l.sort((p,q)=>q[2]-p[2]);l.slice(0,per).forEach(p=>kp.push({x:p[0],y:p[1]}))});
  if(mask)for(let i=kp.length-1;i>=0;i--){const k=kp[i];if(mask.d[((k.y*mask.h/h)|0)*mask.w+((k.x*mask.w/w)|0)])kp.splice(i,1)} // หน้ากาก: ตัดจุดในพื้นที่ที่ผู้ใช้ระบายออก
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
function stats(g,w,h){let m=0,s=0,s2=0,n=0;for(let i=0;i<g.length;i++)m+=g[i];m/=g.length;for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){const i=y*w+x,v=4*g[i]-g[i-1]-g[i+1]-g[i-w]-g[i+w];s+=v;s2+=v*v;n++}return{mean:m,lap:s2/n-(s/n)**2}}
function spread(F){const s=new Set;F.kp.forEach(k=>s.add(((k.y*6/F.h)|0)*8+((k.x*8/F.w)|0)));return s.size/48}
function rescale(d,w,h,s){const g=gray(d,w*h),o=new Uint8ClampedArray(w*h*4);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const v=bil(g,w,h,w/2+(x-w/2)/s,h/2+(y-h/2)/s),i=(y*w+x)*4;o[i]=o[i+1]=o[i+2]=v!==v?128:v;o[i+3]=255}return o}
/* handle(R,m): ใช้ทั้งใน Web Worker และ main thread (fallback). ref = เก็บ features ของ Master/Extra, reg = จับคู่ทุก ref × 3 สเกล เลือกที่ conf สูงสุด */
function handle(R,m){try{const w=m.w,h=m.h;
  if(m.t==="ref"){const F=features(new Uint8ClampedArray(m.buf),w,h,0,m.mask);F.st=stats(F.g,w,h);R[m.key]=F;return{ok:1,nkp:F.kp.length,cover:spread(F)}}
  if(m.t==="refhi"){R[m.key]={g:gray(new Uint8ClampedArray(m.buf),w*h),w,h,mask:m.mask};return{ok:1}}
  if(m.t==="fine"){const Rh=R[m.key];if(!Rh)return{err:"no refhi"};const k=w/m.w0,K=[k,0,0,0,k,0,0,0,1],Ki=[1/k,0,0,0,1/k,0,0,0,1],gI=gray(new Uint8ClampedArray(m.buf),w*h),Tb=box(Rh.g,w,h,1),Ib=box(gI,w,h,1),pix=heldPix(Tb,w,h,Rh.mask),A0=inv3(nz0(mul(mul(K,m.H),Ki)));
    const a0=medRes(Tb,Ib,w,h,A0,null,pix);if(a0.length<50)return{err:"too few pixels"};const cap=3*Math.max(.02,a0[a0.length>>1]),e0=cm(a0,cap),cand={coarse:e0};let best={A:A0,F:null,e:e0,method:"COARSE"};
    const L=lk(Rh.g,gI,w,h,A0,Rh.mask,12);if(L){const e=cm(medRes(Tb,Ib,w,h,L.A,null,pix),cap);cand.lk=e;if(e<=e0*1.02)best={A:L.A,F:null,e,method:"LK"}}
    if(m.local){const F=flowTiles(Rh.g,gI,w,h,best.A,Rh.mask,8,6);if(F&&!F.bad){const e=cm(medRes(Tb,Ib,w,h,best.A,F,pix),cap);cand.flow=e;if(e<best.e*.97)best={A:best.A,F,e,method:best.method+"+FLOW"}}}
    const R2=flowTiles(Rh.g,gI,w,h,best.A,Rh.mask,8,6,best.F||{gx:8,gy:6,d:new Array(96).fill(0)}),mg=((R2&&R2.mags)||[]).slice().sort((p,q)=>p-q),
      res=mg.length?{med:mg[mg.length>>1],p95:mg[Math.min(mg.length-1,Math.floor(mg.length*.95))],max:mg[mg.length-1],n:mg.length,w}:null,Hn=nz0(mul(mul(Ki,nz0(inv3(best.A))),K));
    return{H:Hn,rms:L?L.rms:0,shift:shiftPx(m.H,Hn,m.w0,m.h0),method:best.method,cand,res,flow:best.F?{gx:8,gy:6,d:best.F.d}:null}}
  if(m.t==="warp")return{out:warp(new Uint8ClampedArray(m.buf),m.sw,m.sh,m.H,w,m.ow,m.oh,m.flow,m.cubic)};
  const d=new Uint8ClampedArray(m.buf);let best=null,lap=0,mean=0;
  for(const s of(m.scales||[1,.87,1.15,.75,1.3])){const F=features(s===1?d:rescale(d,w,h,s),w,h,0,m.mask);if(s===1){const st=stats(F.g,w,h);lap=st.lap;mean=st.mean}
    const S=[s,0,w/2*(1-s),0,s,h/2*(1-s),0,0,1];
    for(const k of m.keys){const Rf=R[k.key];if(!Rf)continue;const r=register(F,Rf,Object.assign({},m.opt,{pre:k.pre,post:s===1?null:S,mask:m.mask}));r.key=k.key;r.lapR=Rf.st.lap;r.scl=s;if(!best||(r.H&&(!best.H||r.conf>best.conf)))best=r}
    if(best&&best.H&&best.conf>=(m.opt.acc||90))break}
  if(best){best.lap=lap;best.mean=mean}return{r:best}}catch(e){return{err:String(e)}}}
const DEF={lim:10,acc:90,auto:75,rev:50};
/* register(F,Rf,opt) → alignment report. H maps NEW px → REF px (both at F.w×F.h). Always against the MASTER reference. */
const nz0=m=>{const k=1/m[8];return m.map(v=>v*k)};
function solve8(Hs,b){const A=[];for(let i=0;i<8;i++)A.push([...Hs.slice(i*8,i*8+8),b[i]]);
  for(let c=0;c<8;c++){let p=c;for(let r=c+1;r<8;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;if(Math.abs(A[p][c])<1e-12)return null;[A[c],A[p]]=[A[p],A[c]];
    for(let r=0;r<8;r++)if(r!==c){const f=A[r][c]/A[c][c];for(let k=c;k<9;k++)A[r][k]-=f*A[c][k]}}
  return A.map((r,i)=>r[8]/r[i])}
/* lk: Inverse-compositional Lucas-Kanade (homography 8 พารามิเตอร์) — ปรับละเอียดระดับ sub-pixel
   T=ref gray, I=ภาพใหม่ gray (ขนาดเท่ากัน), A = ref→new (3x3 px). ใช้เฉพาะพิกเซลขอบแรง, normalize เฉลี่ย/ส่วนเบี่ยงเบน (ทนแสงเปลี่ยน), น้ำหนัก Tukey ตัดวัตถุเคลื่อนที่, ข้ามพื้นที่ mask */
function lk(T,I,w,h,A,mask,it){const hw=w/2,cx=w/2,cy=h/2,Tb=box(T,w,h,1),Ib=box(I,w,h,1),n=w*h,mg=new Float32Array(n),hist=new Uint32Array(256);
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){const i=y*w+x,v=Math.hypot((Tb[i+1]-Tb[i-1])/2,(Tb[i+w]-Tb[i-w])/2);mg[i]=v;hist[Math.min(255,v|0)]++}
  let acc=0,th=255;for(let b=255;b>=0;b--){acc+=hist[b];if(acc>n*.25){th=b;break}}th=Math.max(th,3);
  const idx=[];for(let y=3;y<h-3;y++)for(let x=3;x<w-3;x++){const i=y*w+x;if(mg[i]>=th&&!(mask&&mask.d[((y*mask.h/h)|0)*mask.w+((x*mask.w/w)|0)]))idx.push(i)}
  const N=idx.length;if(N<200)return null;let tm=0,tv=0;for(const i of idx)tm+=Tb[i];tm/=N;for(const i of idx)tv+=(Tb[i]-tm)**2;
  const ts=Math.sqrt(tv/N)||1,Tn=new Float32Array(N),SD=new Float32Array(N*8);
  idx.forEach((i,j)=>{const x=i%w,y=(i/w)|0,xn=(x-cx)/hw,yn=(y-cy)/hw,gx=(Tb[i+1]-Tb[i-1])/2*hw/ts,gy=(Tb[i+w]-Tb[i-w])/2*hw/ts,o=j*8,q=gx*xn+gy*yn;Tn[j]=(Tb[i]-tm)/ts;
    SD[o]=gx*xn;SD[o+1]=gy*xn;SD[o+2]=gx*yn;SD[o+3]=gy*yn;SD[o+4]=gx;SD[o+5]=gy;SD[o+6]=-q*xn;SD[o+7]=-q*yn});
  const Tm=[1/hw,0,-cx/hw,0,1/hw,-cy/hw,0,0,1],Ti=[hw,0,cx,0,hw,cy,0,0,1];let rms=0;
  for(let k=0;k<(it||14);k++){const iv=new Float32Array(N),ok=new Uint8Array(N);let c=0,sI=0,sII=0;
    for(let j=0;j<N;j++){const i=idx[j],x=i%w,y=(i/w)|0,z=A[6]*x+A[7]*y+A[8],v=bil(Ib,w,h,(A[0]*x+A[1]*y+A[2])/z,(A[3]*x+A[4]*y+A[5])/z);if(v!==v)continue;iv[j]=v;ok[j]=1;c++;sI+=v;sII+=v*v}
    if(c<N*.5)return null;const mi=sI/c,si=Math.sqrt(Math.max(1e-9,sII/c-mi*mi)),r=new Float32Array(N),a=[];
    for(let j=0;j<N;j++)if(ok[j]){r[j]=(iv[j]-mi)/si-Tn[j];if(j%7===0)a.push(Math.abs(r[j]))}
    a.sort((p,q)=>p-q);const sg=Math.max(.03,a[Math.floor(a.length*.35)]/.454),Hs=new Float64Array(64),b=new Float64Array(8);let ws=0,e2=0;
    for(let j=0;j<N;j++)if(ok[j]){const u=Math.abs(r[j])/(4.685*sg);if(u>=1)continue;const wt=(1-u*u)**2,o=j*8;ws+=wt;e2+=wt*r[j]*r[j];
      for(let p=0;p<8;p++){const sp=wt*SD[o+p];b[p]+=sp*r[j];for(let q=p;q<8;q++)Hs[p*8+q]+=sp*SD[o+q]}}
    if(ws<N*.15)return null;for(let p=0;p<8;p++)for(let q=0;q<p;q++)Hs[p*8+q]=Hs[q*8+p];for(let p=0;p<8;p++)Hs[p*9]+=1e-6*ws;
    const dp=solve8(Hs,b);if(!dp||dp.some(v=>!isFinite(v)))return null;rms=Math.sqrt(e2/ws);
    A=nz0(mul(A,inv3(mul(mul(Ti,[1+dp[0],dp[2],dp[4],dp[1],1+dp[3],dp[5],dp[6],dp[7],1]),Tm))));
    if(Math.hypot(dp[0],dp[1],dp[2],dp[3],dp[4],dp[5])<2e-6)break}
  return{A,rms,N}}
/* ---------- LOCAL WARP: แก้ parallax/ความคลาดที่ homography ตัวเดียวแก้ไม่ได้ ด้วยการประมาณ flow ราย tile (8×6) แล้วทำให้เรียบ ---------- */
function fsm(F,nx,ny){const u=Math.min(F.gx-1,Math.max(0,nx*F.gx-.5)),v=Math.min(F.gy-1,Math.max(0,ny*F.gy-.5)),i=u|0,j=v|0,i1=Math.min(F.gx-1,i+1),j1=Math.min(F.gy-1,j+1),a=u-i,b=v-j,d=F.d,g=(x,y,k)=>d[(y*F.gx+x)*2+k];
  return[(g(i,j,0)*(1-a)+g(i1,j,0)*a)*(1-b)+(g(i,j1,0)*(1-a)+g(i1,j1,0)*a)*b,(g(i,j,1)*(1-a)+g(i1,j,1)*a)*(1-b)+(g(i,j1,1)*(1-a)+g(i1,j1,1)*a)*b]}
function medRes(Tb,Ib,w,h,A,F,pix){const v=[],t=[];let s=0,ss=0,st=0,stt=0;
  for(const i of pix){const x=i%w,y=(i/w)|0,z=A[6]*x+A[7]*y+A[8];let X=(A[0]*x+A[1]*y+A[2])/z,Y=(A[3]*x+A[4]*y+A[5])/z;if(F){const f=fsm(F,x/w,y/h);X+=f[0]*w;Y+=f[1]*w}
    const q=bil(Ib,w,h,X,Y);if(q!==q)continue;v.push(q);t.push(Tb[i]);s+=q;ss+=q*q;st+=Tb[i];stt+=Tb[i]**2}
  const n=v.length;if(n<50)return[];const mi=s/n,si=Math.sqrt(Math.max(1e-9,ss/n-mi*mi)),mt=st/n,sT=Math.sqrt(Math.max(1e-9,stt/n-mt*mt)),a=[];
  for(let k=0;k<n;k++)a.push(Math.abs((v[k]-mi)/si-(t[k]-mt)/sT));a.sort((p,q)=>p-q);return a}
function heldPix(Tb,w,h,mask){const n=w*h,mg=new Float32Array(n),hist=new Uint32Array(256);
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){const i=y*w+x,v=Math.hypot((Tb[i+1]-Tb[i-1])/2,(Tb[i+w]-Tb[i-w])/2);mg[i]=v;hist[Math.min(255,v|0)]++}
  let acc=0,th=255;for(let b=255;b>=0;b--){acc+=hist[b];if(acc>n*.35){th=b;break}}th=Math.max(th,3);const p=[];
  for(let y=3;y<h-3;y++)for(let x=3;x<w-3;x++){const i=y*w+x;if((x+y)%4===0&&mg[i]>=th&&!(mask&&mask.d[((y*mask.h/h)|0)*mask.w+((x*mask.w/w)|0)]))p.push(i)}
  return p.length>4000?p.filter((_,i)=>i%Math.ceil(p.length/4000)===0):p}
const cm=(a,cap)=>{if(!a.length)return 9;let s=0;for(const v of a)s+=Math.min(v,cap);return s/a.length};
function flowTiles(T,I,w,h,A,mask,gx,gy,F0){const Tb=box(T,w,h,1),Ib=box(I,w,h,1),Tb4=box(T,w,h,4),Ib4=box(I,w,h,4),n=w*h,mg=new Float32Array(n),hist=new Uint32Array(256);
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){const i=y*w+x,v=Math.hypot((Tb[i+1]-Tb[i-1])/2,(Tb[i+w]-Tb[i-w])/2);mg[i]=v;hist[Math.min(255,v|0)]++}
  let acc=0,th=255;for(let b=255;b>=0;b--){acc+=hist[b];if(acc>n*.35){th=b;break}}th=Math.max(th,3);
  const d=new Float32Array(gx*gy*2),ok=new Uint8Array(gx*gy),mags=[],mk=(x,y)=>mask&&mask.d[((y*mask.h/h)|0)*mask.w+((x*mask.w/w)|0)];
  for(let ty=0;ty<gy;ty++)for(let tx=0;tx<gx;tx++){const x0=Math.max(3,Math.floor(tx*w/gx)),x1=Math.min(w-3,Math.floor((tx+1)*w/gx)),y0=Math.max(3,Math.floor(ty*h/gy)),y1=Math.min(h-3,Math.floor((ty+1)*h/gy)),idx=[];
    for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){const i=y*w+x;if(mg[i]>=th&&!mk(x,y)&&(x+y)%4!==0)idx.push(i)}
    const N=idx.length;if(N<80)continue;const f0=F0?fsm(F0,(tx+.5)/gx,(ty+.5)/gy):[0,0];let dx=f0[0]*w,dy=f0[1]*w,good=false;const dx0=dx,dy0=dy;
    for(let it=0;it<8;it++){const Tq=it<4?Tb4:Tb,Iq=it<4?Ib4:Ib;const iv=new Float32Array(N),jx=new Float32Array(N),jy=new Float32Array(N),okp=new Uint8Array(N);let c=0,sI=0,sII=0,sT=0,sTT=0;
      for(let j=0;j<N;j++){const i=idx[j],x=i%w,y=(i/w)|0,z=A[6]*x+A[7]*y+A[8],X=(A[0]*x+A[1]*y+A[2])/z+dx,Y=(A[3]*x+A[4]*y+A[5])/z+dy,v=bil(Iq,w,h,X,Y),vx=bil(Iq,w,h,X+1,Y),vy=bil(Iq,w,h,X,Y+1);
        if(v!==v||vx!==vx||vy!==vy)continue;okp[j]=1;iv[j]=v;jx[j]=vx-v;jy[j]=vy-v;c++;sI+=v;sII+=v*v;sT+=Tq[i];sTT+=Tq[i]**2}
      if(c<N*.6)break;const mi=sI/c,si=Math.sqrt(Math.max(1e-9,sII/c-mi*mi)),mt=sT/c,st=Math.sqrt(Math.max(1e-9,sTT/c-mt*mt)),r=new Float32Array(N),a=[];
      for(let j=0;j<N;j++)if(okp[j]){r[j]=(iv[j]-mi)/si-(Tq[idx[j]]-mt)/st;if(j%3===0)a.push(Math.abs(r[j]))}
      a.sort((p,q)=>p-q);const sg=Math.max(.05,a[Math.floor(a.length*.35)]/.454);let a11=0,a12=0,a22=0,b1=0,b2=0,ws=0;
      for(let j=0;j<N;j++)if(okp[j]){const u=Math.abs(r[j])/(4.685*sg);if(u>=1)continue;const wt=(1-u*u)**2,gxn=jx[j]/si,gyn=jy[j]/si;ws+=wt;a11+=wt*gxn*gxn;a12+=wt*gxn*gyn;a22+=wt*gyn*gyn;b1+=wt*gxn*r[j];b2+=wt*gyn*r[j]}
      const det=a11*a22-a12*a12,tr=a11+a22,lm=(tr-Math.sqrt(Math.max(0,tr*tr-4*det)))/2;if(ws<N*.2||lm/ws<.004||det<1e-12)break;
      const sx=-(a22*b1-a12*b2)/det,sy=-(-a12*b1+a11*b2)/det;dx+=sx;dy+=sy;good=true;if(Math.abs(sx)+Math.abs(sy)<.01)break}
    if(good&&Math.abs(dx)<.02*w&&Math.abs(dy)<.02*w){d[(ty*gx+tx)*2]=dx/w;d[(ty*gx+tx)*2+1]=dy/w;ok[ty*gx+tx]=1;mags.push(Math.hypot(dx-dx0,dy-dy0))}}
  let nv=0;ok.forEach(v=>nv+=v);if(nv<gx*gy*.4)return{bad:true,mags};
  for(let p=0;p<4;p++){const o2=ok.slice(),d2=d.slice();for(let ty=0;ty<gy;ty++)for(let tx=0;tx<gx;tx++){if(ok[ty*gx+tx])continue;let sx=0,sy=0,c=0;
      for(let j=-1;j<=1;j++)for(let i=-1;i<=1;i++){const X=tx+i,Y=ty+j;if(X<0||Y<0||X>=gx||Y>=gy||!ok[Y*gx+X])continue;sx+=d[(Y*gx+X)*2];sy+=d[(Y*gx+X)*2+1];c++}
      if(c){d2[(ty*gx+tx)*2]=sx/c;d2[(ty*gx+tx)*2+1]=sy/c;o2[ty*gx+tx]=1}}d.set(d2);ok.set(o2)}
  const sm=d.slice();for(let ty=0;ty<gy;ty++)for(let tx=0;tx<gx;tx++){let sx=0,sy=0,c=0;for(let j=-1;j<=1;j++)for(let i=-1;i<=1;i++){const X=tx+i,Y=ty+j;if(X<0||Y<0||X>=gx||Y>=gy)continue;sx+=d[(Y*gx+X)*2];sy+=d[(Y*gx+X)*2+1];c++}sm[(ty*gx+tx)*2]=sx/c;sm[(ty*gx+tx)*2+1]=sy/c}
  return{gx,gy,d:Array.from(sm),mags}}
const shiftPx=(A,B,w,h)=>{let m=0;[[0,0],[w,0],[w,h],[0,h]].forEach(p=>{const a=ap(A,p[0],p[1]),b=ap(B,p[0],p[1]);m=Math.max(m,Math.hypot(a[0]-b[0],a[1]-b[1]))});return m};
/* finalize: แยก transform + crop + confidence + สถานะ จาก H (ใช้ซ้ำหลัง refine) */
function finalize(r,H,w,h,o){o=Object.assign({},DEF,o);const hw=w/2,hh=h/2,c=ap(H,hw,hh),a=ap(H,hw+1,hh),b=ap(H,hw,hh+1);r.H=H;r.tx=c[0]-hw;r.ty=c[1]-hh;
  const ax=a[0]-c[0],ay=a[1]-c[1],bx=b[0]-c[0],by=b[1]-c[1];r.sc=Math.sqrt(Math.abs(ax*by-ay*bx));r.rot=Math.atan2(ay,ax)*180/Math.PI;r.persp=(Math.abs(H[6])+Math.abs(H[7]))*w;
  const L=o.lim/100,bad=[];if(Math.hypot(r.tx,r.ty)/w>L)bad.push("translation");if(Math.abs(r.rot)>o.lim*.5)bad.push("rotation");if(Math.abs(r.sc-1)>L)bad.push("scale");if(r.persp>L)bad.push("มุมกล้องเอียง");
  r.conf=Math.round(100*(.25*Math.min(1,r.ir/.7)+.2*Math.min(1,r.inl/40)+.2*Math.max(0,1-r.rms/3)+.25*r.sim+.1*Math.min(1,r.sp/.35)));r.stab=bad.length?"LOW":r.sp>.3&&r.inl>=25?"HIGH":"MEDIUM";
  const q=[[0,0],[w,0],[w,h],[0,h]].map(p=>ap(H,p[0],p[1])),cl=(v,m)=>Math.max(0,Math.min(m,v));
  r.crop=[cl(Math.max(q[0][0],q[3][0]),w)/w,cl(Math.max(q[0][1],q[1][1]),h)/h,cl(Math.min(q[1][0],q[2][0]),w)/w,cl(Math.min(q[3][1],q[2][1]),h)/h];
  if(r.crop[2]-r.crop[0]<.4||r.crop[3]-r.crop[1]<.4)bad.push("overlap");r.why="";
  if(bad.length){r.st="REJECT";r.why="ภาพต่างจากต้นแบบมากเกินไป ("+bad.map(b=>({translation:"เลื่อน",rotation:"หมุน",scale:"ระยะ/ขนาด",perspective:"มุมกล้อง",overlap:"พื้นที่ซ้อนทับน้อย"}[b]||b)).join(", ")+" เกินที่ตั้งไว้)";return r}
  r.st=r.conf>=o.acc?"ACCEPT":r.conf>=o.auto?"CORRECTED":r.conf>=o.rev?"REVIEW":"REJECT";if(r.st==="REJECT")r.why="ความมั่นใจต่ำเกินไป ("+r.conf+"%)";return r}
/* register(F,Rf,o): H = NEW px → REF px — หยาบด้วย RANSAC แล้วปรับละเอียดด้วย lk (sub-pixel) เสมอเทียบ MASTER */
function register(F,Rf,o){o=Object.assign({},DEF,o);const w=F.w,h=F.h,hw=w/2,hh=h/2,nrm=(x,y)=>[(x-hw)/hw,(y-hh)/hw];
  const r={st:"REJECT",conf:0,why:"",n:0,inl:0,ir:0,rms:0,tx:0,ty:0,rot:0,sc:1,persp:0,sim:0,sp:0,H:null,crop:null,stab:"LOW",kpN:F.kp.length,kpR:Rf.kp.length,fine:0};
  const M=match(F,Rf);r.n=M.length;if(M.length<10){r.why="จับคู่จุดเด่นได้น้อยเกินไป ("+M.length+")";return r}
  const P=M.map(([i,j])=>{const a=nrm(F.kp[i].x,F.kp[i].y),b=nrm(Rf.kp[j].x,Rf.kp[j].y);return[a[0],a[1],b[0],b[1]]});
  const R=ransac(P);if(!R){r.why="หาการจัดภาพที่ลงตัวไม่ได้";return r}
  const Tm=[1/hw,0,-1,0,1/hw,-hh/hw,0,0,1],Ti=[hw,0,hw,0,hw,hh,0,0,1];let H=nz0(mul(mul(Ti,R.H),Tm));
  r.inl=R.inl.length;r.ir=r.inl/r.n;let e=0,x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
  R.inl.forEach(i=>{const q=ap(R.H,P[i][0],P[i][1]);e+=(q[0]-P[i][2])**2+(q[1]-P[i][3])**2;const f=F.kp[M[i][0]];x0=Math.min(x0,f.x);x1=Math.max(x1,f.x);y0=Math.min(y0,f.y);y1=Math.max(y1,f.y)});
  r.rms=Math.sqrt(e/r.inl)*hw;r.sp=(x1-x0)*(y1-y0)/(w*h);const IS=new Set(R.inl);r.pts=M.slice(0,150).map((p,i)=>[F.kp[p[0]].x,F.kp[p[0]].y,Rf.kp[p[1]].x,Rf.kp[p[1]].y,IS.has(i)?1:0]);
  if(o.refine!==false){const L=lk(Rf.g,F.g,w,h,inv3(H),o.mask,16);if(L){const Hn=nz0(inv3(L.A));if(shiftPx(H,Hn,w,h)<8){r.fineShift=shiftPx(H,Hn,w,h);H=Hn;r.fine=1;r.fineRms=L.rms}}}
  r.sim=similarity(F.g,Rf.g,w,h,inv3(H));if(o.post)H=nz0(mul(H,o.post));if(o.pre)H=nz0(mul(o.pre,H));
  return finalize(r,H,w,h,o)}
/* warp(src RGBA, sw, sh, H(320-space, new→ref), w320, ow, oh) → RGBA at ref geometry ow×oh */
function warp(src,sw,sh,H,w,ow,oh,flow,cubic){const Hi=inv3(H),k=ow/w,ks=sw/w,out=new Uint8ClampedArray(ow*oh*4),cr=(t)=>[(-.5*t*t*t+t*t-.5*t),(1.5*t*t*t-2.5*t*t+1),(-1.5*t*t*t+2*t*t+.5*t),(.5*t*t*t-.5*t*t)];
  for(let y=0;y<oh;y++)for(let x=0;x<ow;x++){const X=(x+.5)/k,Y=(y+.5)/k,z=Hi[6]*X+Hi[7]*Y+Hi[8];let u=(Hi[0]*X+Hi[1]*Y+Hi[2])/z*ks-.5,v=(Hi[3]*X+Hi[4]*Y+Hi[5])/z*ks-.5;
    if(flow){const f=fsm(flow,(x+.5)/ow,(y+.5)/oh);u+=f[0]*sw;v+=f[1]*sw}
    if(u<0||v<0||u>sw-1||v>sh-1)continue;const x0=u|0,y0=v|0,fx=u-x0,fy=v-y0,o=(y*ow+x)*4;
    if(cubic&&x0>0&&y0>0&&x0<sw-2&&y0<sh-2){const wx=cr(fx),wy=cr(fy);for(let c=0;c<3;c++){let a=0;for(let j=0;j<4;j++){let r=0;const row=((y0-1+j)*sw+x0-1)*4+c;for(let i=0;i<4;i++)r+=src[row+i*4]*wx[i];a+=r*wy[j]}out[o+c]=a}}
    else{const x1=Math.min(sw-1,x0+1),y1=Math.min(sh-1,y0+1);for(let c=0;c<3;c++)out[o+c]=(src[(y0*sw+x0)*4+c]*(1-fx)+src[(y0*sw+x1)*4+c]*fx)*(1-fy)+(src[(y1*sw+x0)*4+c]*(1-fx)+src[(y1*sw+x1)*4+c]*fx)*fy}out[o+3]=255}
  return out}
const intersect=l=>l.reduce((a,c)=>[Math.max(a[0],c[0]),Math.max(a[1],c[1]),Math.min(a[2],c[2]),Math.min(a[3],c[3])],[0,0,1,1]);
/* selfTest: synthetic textured scene + known transforms, report estimated-vs-true corner error */
function scene(w,h){let s=99;const r=()=>(s=(Math.imul(s,1664525)+1013904223)>>>0)/4294967296,d=new Uint8ClampedArray(w*h*4).fill(255);
  const rect=(x,y,a,b,v)=>{for(let j=Math.max(0,y|0);j<Math.min(h,y+b);j++)for(let i=Math.max(0,x|0);i<Math.min(w,x+a);i++){const o=(j*w+i)*4;d[o]=d[o+1]=d[o+2]=v}};
  rect(0,0,w,h,150);for(let i=0;i<90;i++)rect(r()*w,r()*h,6+r()*50,6+r()*40,r()*255);return d}
function selfTest(){const w=320,h=240,S=scene(w*2,h*2),B=scene(w,h),F0=features(B,w,h),out=[],R0={};handle(R0,{t:"ref",key:"m",buf:B.slice().buffer,w,h});
  const mk=(tx,ty,rd,sc,p)=>{const a=rd*Math.PI/180,c=Math.cos(a)*sc,s=Math.sin(a)*sc;return[c,-s,tx*w+(1-c)*w/2+s*h/2,s,c,ty*h+(1-c)*h/2-s*w/2,p,0,1]};
  let sd=5;const rn=()=>(sd=(Math.imul(sd,1664525)+1013904223)>>>0)/4294967296;
  const rp=(d,x,y,a,b,v)=>{for(let j=Math.max(0,y|0);j<Math.min(h,y+b);j++)for(let i=Math.max(0,x|0);i<Math.min(w,x+a);i++){const o=(j*w+i)*4;d[o]=d[o+1]=d[o+2]=v}};
  const bl=d=>{const g=box(gray(d,w*h),w,h,4),o=new Uint8ClampedArray(d.length);for(let i=0;i<g.length;i++){o[i*4]=o[i*4+1]=o[i*4+2]=g[i];o[i*4+3]=255}return o};
  const M={"ภาพเบลอ":bl,"มีของเคลื่อนที่เยอะ":d=>{for(let i=0;i<12;i++)rp(d,rn()*w,rn()*h,60,50,rn()*255);return d},"แสงเปลี่ยน":d=>{for(let i=0;i<d.length;i+=4)for(let k=0;k<3;k++)d[i+k]=d[i+k]*.55+40;return d},
    "ท้องฟ้าเปลี่ยน":d=>{rp(d,0,0,w,h*.3,210);for(let i=0;i<8;i++)rp(d,rn()*w,rn()*h*.3,40,20,rn()*255);return d},"คนเดินผ่าน":d=>{rp(d,130,60,45,150,30);return d},
    "ต้นไม้ไหว":d=>{const c=d.slice();for(let n=0;n<40;n++){const x=rn()*(w-30)|0,y=rn()*(h-30)|0,dx=(rn()*8-4)|0,dy=(rn()*8-4)|0;for(let j=0;j<18;j++)for(let i=0;i<18;i++){const a=((y+j)*w+x+i)*4,b=(Math.max(0,Math.min(h-1,y+j+dy))*w+Math.max(0,Math.min(w-1,x+i+dx)))*4;d[a]=d[a+1]=d[a+2]=c[b]}}return d}};
  const run=(n,Hg,lim,mut)=>{let img=warp(scene(w,h),w,h,Hg,w,w,h);if(mut)img=mut(img);const res=handle(R0,{t:"reg",keys:[{key:"m",pre:null}],w,h,buf:img.slice().buffer,opt:{lim}}).r;let st=res?res.st:"REJECT",why=res?res.why:"no result";
    if(res&&mut===M["ภาพเบลอ"]&&res.lapR&&res.lap/res.lapR<.3){st="REJECT";why="ภาพเบลอเกินไป (คมเพียง "+Math.round(100*res.lap/res.lapR)+"%)"}
    let err=NaN;if(res&&res.H){err=0;[[0,0],[w,0],[w,h],[0,h]].forEach(p=>{const t=ap(Hg,p[0],p[1]),q=ap(res.H,t[0],t[1]);err=Math.max(err,Math.hypot(q[0]-p[0],q[1]-p[1]))})}
    out.push({case:n,st,conf:res?res.conf:0,inl:res?res.inl:0,n:res?res.n:0,errPx:err,why})};
  [["เลื่อน 3%",.03,.02,0,1,0,10],["หมุน 1°",0,0,1,1,0,10],["ย่อ/ขยาย 2%",0,0,0,1.02,0,10],["เลื่อน+หมุน",.04,-.03,1.5,1,0,10],["มุมกล้องเอียง",0,0,0,1,.0004,15],["ภาพคลาดราว 10%",.1,.06,3,1.05,0,15],["ภาพคลาดราว 20%",.2,.12,6,1.1,0,20],["ภาพคลาดราว 30%",.3,.18,9,1.15,0,30],["ขยายภาพ 25%",0,0,0,1.25,0,30],["ย่อภาพ 22%",0,0,0,.78,0,30]].forEach(([n,tx,ty,rd,sc,p,l])=>run(n,mk(tx,ty,rd,sc,p),l));
  Object.keys(M).forEach(n=>run(n,mk(.02,.01,.5,1,0),10,M[n]));return out}
G.CV={flowTiles,fsm,features,match,register,finalize,warp,intersect,selfTest,inv3,ap,mul,handle,spread,DEF};
})(typeof window!=="undefined"?window:globalThis);
