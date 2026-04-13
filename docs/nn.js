function* sgd(model, dataset, lr, epochs, batch_size = 32, snap_points = 100) {
    const t_list = Array.from({ length: snap_points + 1 }, (_, i) => i / snap_points);
    for (let epoch = 0; epoch < epochs; epoch++) {
        const shuffled = [...dataset];
        model.random.shuffle(shuffled);
        let total_loss = 0.0;
        let batches = 0;
        for (let i = 0; i < shuffled.length; i += batch_size) {
            const batch = shuffled.slice(i, i + batch_size);
            total_loss += model.train(batch, lr);
            batches++;
        }
        const loss = total_loss / batches;
        yield { epoch, loss };
    }
}

class SimpleNN {
    constructor(n0, n1, n2, n3) {
        this.n0 = n0;
        this.n1 = n1;
        this.n2 = n2;
        this.n3 = n3;
        this.random = makeRandom(123);
        this.w1 = this._randMatrix(n0, n1);
        this.b1 = this._randArray(n1);
        this.w2 = this._randMatrix(n1, n2);
        this.b2 = this._randArray(n2);
        this.w3 = this._randMatrix(n2, n3);
        this.b3 = this._randArray(n3);
        this.snapshots = [];
        this.deriv = s => 1 - s * s;
    }

    forward(h0) {
        const h1 = [];
        for (let i = 0; i < this.n1; i++) {
            let h1i = 0.0;
            for (let j = 0; j < this.n0; j++) {
                h1i += h0[j] * this.w1[j][i];
            }
            h1i += this.b1[i];
            h1.push(Math.tanh(h1i));
        }
        const h2 = [];
        for (let i = 0; i < this.n2; i++) {
            let h2i = 0.0;
            for (let j = 0; j < this.n1; j++) {
                h2i += h1[j] * this.w2[j][i];
            }
            h2i += this.b2[i];
            h2.push(Math.tanh(h2i));
        }
        const h3 = [];
        for (let i = 0; i < this.n3; i++) {
            let h3i = 0.0;
            for (let j = 0; j < this.n2; j++) {
                h3i += h2[j] * this.w3[j][i];
            }
            h3i += this.b3[i];
            h3.push(Math.tanh(h3i));
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
            for (let m = 0; m < this.n3; m++) {
                L += (h3[m] - y[m]) ** 2 / N;
            }
            // layer 3
            const dz3 = [];
            for (let m = 0; m < this.n3; m++) {
                dz3.push(2.0 * (h3[m] - y[m]) * (1 - h3[m]**2));
            }
            for (let m = 0; m < this.n3; m++) {
                dL_b3[m] += dz3[m] / N;
                for (let j = 0; j < this.n2; j++) {
                    dL_w3[j][m] += dz3[m] * h2[j] / N;
                }
            }
            // layer 2
            const dh2 = [];
            for (let j = 0; j < this.n2; j++) {
                let dh2j = 0.0;
                for (let m = 0; m < this.n3; m++) {
                    dh2j += dz3[m] * this.w3[j][m];
                }
                dh2.push(dh2j);
            }
            const dz2 = [];
            for (let i = 0; i < this.n2; i++) {
                dz2.push(dh2[i] * (1 - h2[i]**2));
            }
            for (let i = 0; i < this.n2; i++) {
                dL_b2[i] += dz2[i] / N;
                for (let j = 0; j < this.n1; j++) {
                    dL_w2[j][i] += dz2[i] * h1[j] / N;
                }
            }
            // layer 1
            const dh1 = [];
            for (let j = 0; j < this.n1; j++) {
                let dh1j = 0.0;
                for (let i = 0; i < this.n2; i++) {
                    dh1j += dz2[i] * this.w2[j][i];
                }
                dh1.push(dh1j);
            }
            const dz1 = [];
            for (let k = 0; k < this.n1; k++) {
                dz1.push(dh1[k] * (1 - h1[k]**2));
            }
            for (let k = 0; k < this.n1; k++) {
                dL_b1[k] += dz1[k] / N;
                for (let j = 0; j < this.n0; j++) {
                    dL_w1[j][k] += dz1[k] * x[j] / N;
                }
            }
        }
        // SGD step
        for (let k = 0; k < this.n1; k++) {
            this.b1[k] -= lr * dL_b1[k];
            for (let j = 0; j < this.n0; j++) {
                this.w1[j][k] -= lr * dL_w1[j][k];
            }
        }
        for (let i = 0; i < this.n2; i++) {
            this.b2[i] -= lr * dL_b2[i];
            for (let j = 0; j < this.n1; j++) {
                this.w2[j][i] -= lr * dL_w2[j][i];
            }
        }
        for (let m = 0; m < this.n3; m++) {
            this.b3[m] -= lr * dL_b3[m];
            for (let j = 0; j < this.n2; j++) {
                this.w3[j][m] -= lr * dL_w3[j][m];
            }
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

    _zeroArray(n) {
      return new Array(n).fill(0.0);
    }

    _zeroMatrix(n, m) {
      return Array.from({ length: n }, () => {
        return new Array(m).fill(0.0);
      });
    }
    _randArray(n) {
        return Array.from({ length: n }, () => {
          return this.random.gauss(0, 1.0 / Math.sqrt(n))
        });
    }
    _randMatrix(n, m) {
        return Array.from({ length: n }, () => {
            return Array.from({ length: m }, () => {
              return this.random.gauss(0, 1.0 / Math.sqrt(n))
            })
        });
    }
}
