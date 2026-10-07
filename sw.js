// Service Worker: ให้เปิดแอปได้โดยไม่ต้องมีอินเทอร์เน็ต (ภาพที่ถ่ายเก็บใน IndexedDB อยู่แล้ว)
const CACHE="tlcam-panel-v15",FILES=["./","./index.html","./manifest.webmanifest","./cv.js","./worker.js","./icon-180.png","./icon-192.png","./icon-512.png"];
self.addEventListener("install",e=>{e.waitUntil(caches.open(CACHE).then(c=>Promise.all(FILES.map(f=>c.add(f).catch(()=>{})))).then(()=>self.skipWaiting()))});
self.addEventListener("activate",e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==CACHE).map(x=>caches.delete(x)))).then(()=>self.clients.claim()))});
self.addEventListener("fetch",e=>{
  if(e.request.method!=="GET")return;
  e.respondWith(caches.match(e.request,{ignoreSearch:true}).then(hit=>{
    const net=fetch(e.request).then(r=>{if(r&&r.ok&&new URL(e.request.url).origin===location.origin){const cp=r.clone();caches.open(CACHE).then(c=>c.put(e.request,cp))}return r}).catch(()=>null);
    if(hit){e.waitUntil(net);return hit} // cache-first: ออฟไลน์ก็เปิดได้ และอัปเดตเบื้องหลังเมื่อมีเน็ต
    return net.then(r=>r||caches.match("./index.html"))}))});
