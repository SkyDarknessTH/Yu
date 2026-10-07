importScripts("cv.js");const R={};onmessage=e=>{const m=e.data,r=CV.handle(R,m);postMessage(Object.assign({id:m.id},r),r.out?[r.out.buffer]:[])};
