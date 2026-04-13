// Seeded pseudo-random number generator (Mulberry32)
function makeRNG(seed) {
    let s = seed >>> 0;
    function next() {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    // Box-Muller transform for Gaussian samples
    function gauss(mean, std) {
        let u, v, sq;
        do {
            u = next() * 2 - 1;
            v = next() * 2 - 1;
            sq = u * u + v * v;
        } while (sq >= 1 || sq === 0);
        return mean + std * u * Math.sqrt(-2 * Math.log(sq) / sq);
    }
    function shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(next() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }
    return { next, gauss, shuffle };
}

// ---- figures.py port ----

function points(fig, n) {
    const ta = Array.from({ length: n + 1 }, (_, i) => i / n);
    const out = [];
    if (fig === 'heart') {
        for (const t of ta) {
            const angle = t * 2 * Math.PI;
            const x = 0.5 * Math.sin(angle) ** 3;
            const y = (13 * Math.cos(angle) - 5 * Math.cos(2 * angle) - 2 * Math.cos(3 * angle) - Math.cos(4 * angle)) / 30;
            out.push([t, x, y]);
        }
    } else if (fig === 'circle') {
        for (const t of ta) {
            const angle = t * 2 * Math.PI;
            const x = 0.5 * Math.cos(angle);
            const y = 0.5 * Math.sin(angle);
            out.push([t, x, y]);
        }
    } else if (fig === 'astroid') {
        for (const t of ta) {
            const angle = t * 2 * Math.PI;
            const x = 0.6 * Math.cos(angle) ** 3;
            const y = 0.6 * Math.sin(angle) ** 3;
            out.push([t, x, y]);
        }
    } else if (fig === 'trefoil') {
        for (const t of ta) {
            const angle = t * 2 * Math.PI;
            const r = 0.6 * Math.cos(3 * angle);
            const x = r * Math.cos(angle);
            const y = r * Math.sin(angle);
            out.push([t, x, y]);
        }
    } else if (fig === 'square') {
        for (const t of ta) {
            const s = t * 4;
            const side = Math.floor(s) % 4;
            const u = s - Math.floor(s);
            let x, y;
            if (side === 0) { x = -0.5 + u; y = -0.5; }
            else if (side === 1) { x = 0.5; y = -0.5 + u; }
            else if (side === 2) { x = 0.5 - u; y = 0.5; }
            else { x = -0.5; y = 0.5 - u; }
            out.push([t, x, y]);
        }
    } else {
        throw new Error('wrong fig name');
    }
    return out;
}

// ---- nn.py port ----

class SimpleNN {
    constructor(n0, n1, n2, n3, seed = 123) {
        this.n0 = n0;
        this.n1 = n1;
        this.n2 = n2;
        this.n3 = n3;
        this.rng = makeRNG(seed);
        this.w1 = this._randMatrix(n0, n1);
        this.b1 = this._randArray(n1);
        this.w2 = this._randMatrix(n1, n2);
        this.b2 = this._randArray(n2);
        this.w3 = this._randMatrix(n2, n3);
        this.b3 = this._randArray(n3);
        this.snapshots = [];
        this.sigma = Math.tanh;
        this.deriv = s => 1 - s * s;
    }

    forward(h0) {
        const h1 = [];
        for (let i = 0; i < this.n1; i++) {
            let h1i = 0.0;
            for (let j = 0; j < this.n0; j++)
                h1i += h0[j] * this.w1[j][i];
            h1i += this.b1[i];
            h1.push(this.sigma(h1i));
        }
        const h2 = [];
        for (let i = 0; i < this.n2; i++) {
            let h2i = 0.0;
            for (let j = 0; j < this.n1; j++)
                h2i += h1[j] * this.w2[j][i];
            h2i += this.b2[i];
            h2.push(this.sigma(h2i));
        }
        const h3 = [];
        for (let i = 0; i < this.n3; i++) {
            let h3i = 0.0;
            for (let j = 0; j < this.n2; j++)
                h3i += h2[j] * this.w3[j][i];
            h3i += this.b3[i];
            h3.push(this.sigma(h3i));
        }
        return [h1, h2, h3];
    }

    train(batch, lr) {
        const dL_w1 = this._zeroMatrix(this.n0, this.n1);
        const dL_b1 = this._zeroArray(this.n1);
        const dL_w2 = this._zeroMatrix(this.n1, this.n2);
        const dL_b2 = this._zeroArray(this.n2);
        const dL_w3 = this._zeroMatrix(this.n2, this.n3);
        const dL_b3 = this._zeroArray(this.n3);
        let L = 0.0;
        const N = batch.length;
        for (let b = 0; b < N; b++) {
            const x = Array.from({ length: this.n0 }, (_, i) => batch[b][i]);
            const y = Array.from({ length: this.n3 }, (_, m) => batch[b][this.n0 + m]);
            const [h1, h2, h3] = this.forward(x);
            // loss
            for (let m = 0; m < this.n3; m++)
                L += (h3[m] - y[m]) ** 2 / N;
            // backprop layer 3
            const dz3 = [];
            for (let m = 0; m < this.n3; m++)
                dz3.push(2.0 * (h3[m] - y[m]) * this.deriv(h3[m]));
            for (let m = 0; m < this.n3; m++) {
                dL_b3[m] += dz3[m] / N;
                for (let j = 0; j < this.n2; j++)
                    dL_w3[j][m] += dz3[m] * h2[j] / N;
            }
            // backprop layer 2
            const dh2 = [];
            for (let j = 0; j < this.n2; j++) {
                let dh2j = 0.0;
                for (let m = 0; m < this.n3; m++)
                    dh2j += dz3[m] * this.w3[j][m];
                dh2.push(dh2j);
            }
            const dz2 = [];
            for (let i = 0; i < this.n2; i++)
                dz2.push(dh2[i] * this.deriv(h2[i]));
            for (let i = 0; i < this.n2; i++) {
                dL_b2[i] += dz2[i] / N;
                for (let j = 0; j < this.n1; j++)
                    dL_w2[j][i] += dz2[i] * h1[j] / N;
            }
            // backprop layer 1
            const dh1 = [];
            for (let j = 0; j < this.n1; j++) {
                let dh1j = 0.0;
                for (let i = 0; i < this.n2; i++)
                    dh1j += dz2[i] * this.w2[j][i];
                dh1.push(dh1j);
            }
            const dz1 = [];
            for (let k = 0; k < this.n1; k++)
                dz1.push(dh1[k] * this.deriv(h1[k]));
            for (let k = 0; k < this.n1; k++) {
                dL_b1[k] += dz1[k] / N;
                for (let j = 0; j < this.n0; j++)
                    dL_w1[j][k] += dz1[k] * x[j] / N;
            }
        }
        // SGD step
        for (let k = 0; k < this.n1; k++) {
            this.b1[k] -= lr * dL_b1[k];
            for (let j = 0; j < this.n0; j++)
                this.w1[j][k] -= lr * dL_w1[j][k];
        }
        for (let i = 0; i < this.n2; i++) {
            this.b2[i] -= lr * dL_b2[i];
            for (let j = 0; j < this.n1; j++)
                this.w2[j][i] -= lr * dL_w2[j][i];
        }
        for (let m = 0; m < this.n3; m++) {
            this.b3[m] -= lr * dL_b3[m];
            for (let j = 0; j < this.n2; j++)
                this.w3[j][m] -= lr * dL_w3[j][m];
        }
        return L;
    }

    getGraph(tList) {
        const netX = [], netY = [];
        for (const t of tList) {
            const [, , h3] = this.forward([t]);
            netX.push(h3[0]);
            netY.push(h3[1]);
        }
        return [netX, netY];
    }

    _zeroArray(n) { return new Array(n).fill(0.0); }
    _zeroMatrix(n, m) { return Array.from({ length: n }, () => new Array(m).fill(0.0)); }
    _randArray(n) {
        return Array.from({ length: n }, () => this.rng.gauss(0, 1.0 / Math.sqrt(n)));
    }
    _randMatrix(n, m) {
        return Array.from({ length: n }, () =>
            Array.from({ length: m }, () => this.rng.gauss(0, 1.0 / Math.sqrt(n)))
        );
    }
}

// Generator — yields { epoch, loss } each epoch, like Python's sgd()
function* sgd(model, dataset, lr, epochs, batch_size = 32, num_snapshots = 20, snap_points = 100) {
    const t_list = Array.from({ length: snap_points + 1 }, (_, i) => i / snap_points);
    const snap_epochs = new Set();
    for (let i = 0; i < num_snapshots; i++)
        snap_epochs.add(Math.round(i * (epochs - 1) / (num_snapshots - 1)));

    for (let epoch = 0; epoch < epochs; epoch++) {
        const shuffled = [...dataset];
        model.rng.shuffle(shuffled);
        let total_loss = 0.0;
        let batches = 0;
        for (let i = 0; i < shuffled.length; i += batch_size) {
            const batch = shuffled.slice(i, i + batch_size);
            total_loss += model.train(batch, lr);
            batches++;
        }
        const loss = total_loss / batches;
        if (snap_epochs.has(epoch)) {
            const [netX, netY] = model.getGraph(t_list);
            model.snapshots.push({ epoch, netX, netY });
        }
        yield { epoch, loss };
    }
}
