// ---- main app ----

const { useState, useRef, useEffect, useCallback } = React;

function App() {
  const [fig, setFig] = useState('circle');
  const [lr, setLr] = useState('0.1');
  const [epochs, setEpochs] = useState('500');
  const [batchSize, setBatchSize] = useState('8');
  const [seed, setSeed] = useState('123');
  const [speedIdx, setSpeedIdx] = useState(1);
  const [status, setStatus] = useState('idle');   // idle | training | done
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(null);

  const mainCanvas = useRef(null);
  const lossCanvas = useRef(null);
  const modelRef = useRef(null);
  const datasetRef = useRef(null);
  const genRef = useRef(null);
  const rafRef = useRef(null);
  const lossHistoryRef = useRef([]);
  const netCurveRef = useRef(null);

  // initial draw when figure changes (idle state)
  useEffect(() => {
    if (status !== 'idle') return;
    const canvas = mainCanvas.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const dataset = points(fig, 100);
    datasetRef.current = dataset;
    drawFigure(ctx, dataset, null, null, w, h);

    const lctx = lossCanvas.current.getContext('2d');
    drawLoss(lctx, [], parseInt(epochs) || 500, lossCanvas.current.width, lossCanvas.current.height);
  }, [fig, status]);

  function startTraining() {
    cancelAnimationFrame(rafRef.current);
    const ep = Math.max(1, parseInt(epochs) || 500);
    const bs = Math.max(1, parseInt(batchSize) || 8);
    const lrVal = parseFloat(lr) || 0.1;
    const sd = parseInt(seed) || 123;

    const dataset = points(fig, 100);
    datasetRef.current = dataset;
    const model = new SimpleNN(1, 20, 20, 2, sd);
    modelRef.current = model;
    lossHistoryRef.current = [];
    netCurveRef.current = null;
    setCurrentEpoch(0);
    setCurrentLoss(null);
    setStatus('training');

    const gen = sgd(model, dataset, lrVal, ep, bs, NUM_SNAPSHOTS, SNAP_POINTS);
    genRef.current = gen;

    const steps = SPEEDS[speedIdx].steps;

    function loop() {
      const mainCtx = mainCanvas.current.getContext('2d');
      const w = mainCanvas.current.width;
      const h = mainCanvas.current.height;
      const lctx = lossCanvas.current.getContext('2d');
      const lw = lossCanvas.current.width;
      const lh = lossCanvas.current.height;

      let done = false;
      let lastEpoch = 0;
      let lastLoss = 0;

      for (let s = 0; s < steps; s++) {
        const result = gen.next();
        if (result.done) { done = true; break; }
        const { epoch, loss } = result.value;
        lastEpoch = epoch;
        lastLoss = loss;
        lossHistoryRef.current.push(loss);
      }

      // draw current network output
      const tList = Array.from({ length: SNAP_POINTS + 1 }, (_, i) => i / SNAP_POINTS);
      const [netX, netY] = model.getGraph(tList);
      netCurveRef.current = { netX, netY };

      drawFigure(mainCtx, datasetRef.current, netX, netY, w, h);
      drawLoss(lctx, lossHistoryRef.current, ep, lw, lh);
      setCurrentEpoch(lastEpoch + 1);
      setCurrentLoss(lastLoss);

      if (done) {
        setStatus('done');
      } else {
        rafRef.current = requestAnimationFrame(loop);
      }
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  function reset() {
    cancelAnimationFrame(rafRef.current);
    genRef.current = null;
    modelRef.current = null;
    lossHistoryRef.current = [];
    netCurveRef.current = null;
    setStatus('idle');
    setCurrentEpoch(0);
    setCurrentLoss(null);
  }

  const isTraining = status === 'training';
  const ep = parseInt(epochs) || 500;
  const progress = ep > 0 ? Math.min(100, (currentEpoch / ep) * 100) : 0;

  return (
    <div className="app">
      <h1>Naive SGD Demo</h1>
      <div className="layout">
        {/* sidebar */}
        <div className="sidebar">
          <div className="field">
            <label>Figure</label>
            <select value={fig} onChange={e => setFig(e.target.value)} disabled={isTraining}>
              <option value="circle">circle</option>
              <option value="heart">heart</option>
              <option value="astroid">astroid</option>
              <option value="trefoil">trefoil</option>
              <option value="square">square</option>
            </select>
          </div>
          <div className="field">
            <label>Learning rate</label>
            <input type="number" value={lr} onChange={e => setLr(e.target.value)}
              step="0.01" min="0.001" max="10" disabled={isTraining} />
          </div>
          <div className="field">
            <label>Epochs</label>
            <input type="number" value={epochs} onChange={e => setEpochs(e.target.value)}
              step="100" min="10" max="5000" disabled={isTraining} />
          </div>
          <div className="field">
            <label>Batch size</label>
            <input type="number" value={batchSize} onChange={e => setBatchSize(e.target.value)}
              step="1" min="1" max="100" disabled={isTraining} />
          </div>
          <div className="field">
            <label>Seed</label>
            <input type="number" value={seed} onChange={e => setSeed(e.target.value)}
              step="1" min="0" disabled={isTraining} />
          </div>
          <div className="field">
            <label>Speed</label>
            <select value={speedIdx} onChange={e => setSpeedIdx(Number(e.target.value))} disabled={isTraining}>
              {SPEEDS.map((s, i) => (
                <option key={i} value={i}>{s.label}</option>
              ))}
            </select>
          </div>
          <button className="btn btn-train" onClick={startTraining} disabled={isTraining}>
            {isTraining ? 'training…' : status === 'done' ? 'train again' : 'train'}
          </button>
          <button className="btn btn-reset" onClick={reset} disabled={isTraining}>
            reset
          </button>

          {(isTraining || status === 'done') && (
            <div className="status-box">
              <div>epoch <span className="val">{currentEpoch}</span> / {ep}</div>
              {currentLoss !== null && (
                <div>loss <span className="val">{currentLoss.toFixed(5)}</span></div>
              )}
              {status === 'done' && <div style={{color:'#16a34a'}}>done ✓</div>}
              <div className="progress-bar-wrap">
                <div className="progress-bar-fill" style={{ width: progress + '%' }} />
              </div>
            </div>
          )}
        </div>

        {/* canvas area */}
        <div className="canvas-area">
          <div className="canvas-label">figure space</div>
          <canvas ref={mainCanvas} width={560} height={480}
            style={{ marginBottom: 14 }} />

          <div className="canvas-label">loss curve</div>
          <canvas ref={lossCanvas} width={560} height={100} />

          <div className="legend">
            <span><span className="legend-dot" style={{background:'#3b82f6'}} />target points</span>
            <span><span className="legend-line" style={{background:'#ef4444'}} />network output</span>
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
