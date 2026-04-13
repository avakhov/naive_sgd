// ---- drawing helpers ----

function toCanvas(x, y, w, h) {
  const pad = 28;
  const uw = w - 2 * pad;
  const uh = h - 2 * pad;
  const range = VIEWPORT * 2;
  const cx = pad + (x + VIEWPORT) / range * uw;
  const cy = pad + (VIEWPORT - y) / range * uh;
  return [cx, cy];
}

function drawFigure(ctx, dataset, netX, netY, w, h) {
  ctx.clearRect(0, 0, w, h);

  // grid
  ctx.strokeStyle = '#f0f0f0';
  ctx.lineWidth = 1;
  for (let v = -1; v <= 1; v += 0.5) {
    const [x0] = toCanvas(v, -VIEWPORT, w, h);
    const [, y0] = toCanvas(-VIEWPORT, v, w, h);
    ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke();
  }

  // axes
  ctx.strokeStyle = '#ccc';
  ctx.lineWidth = 1;
  const [ax] = toCanvas(0, 0, w, h);
  const [, ay] = toCanvas(0, 0, w, h);
  ctx.beginPath(); ctx.moveTo(ax, 0); ctx.lineTo(ax, h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, ay); ctx.lineTo(w, ay); ctx.stroke();

  // target dots
  ctx.fillStyle = '#3b82f6';
  for (const [, dx, dy] of dataset) {
    const [cx, cy] = toCanvas(dx, dy, w, h);
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // network curve
  if (netX && netX.length > 0) {
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < netX.length; i++) {
      const [cx, cy] = toCanvas(netX[i], netY[i], w, h);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  }
}

function drawLoss(ctx, lossHistory, epochs, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, w, h);

  if (lossHistory.length < 2) return;

  const pad = { top: 8, right: 12, bottom: 18, left: 40 };
  const uw = w - pad.left - pad.right;
  const uh = h - pad.top - pad.bottom;

  const maxL = Math.max(...lossHistory) * 1.05;
  const minL = 0;

  function lx(i) { return pad.left + (i / (epochs - 1)) * uw; }
  function ly(v) { return pad.top + uh - ((v - minL) / (maxL - minL || 1)) * uh; }

  // grid
  ctx.strokeStyle = '#eee';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const yv = minL + (maxL - minL) * g / 4;
    const y = ly(yv);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    ctx.fillStyle = '#bbb';
    ctx.font = '9px Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(3), pad.left - 4, y + 3);
  }

  // loss curve
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < lossHistory.length; i++) {
    const x = lx(i);
    const y = ly(lossHistory[i]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // axis labels
  ctx.fillStyle = '#aaa';
  ctx.font = '9px Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('0', pad.left, h - 3);
  ctx.fillText(String(epochs), w - pad.right, h - 3);
  ctx.fillText('epoch', pad.left + uw / 2, h - 3);
}
