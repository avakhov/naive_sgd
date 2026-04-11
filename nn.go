package main

import (
	"math"
	"math/rand"
)

// Point represents one training sample: [t, x, y]
type Point [3]float64

type Snapshot struct {
	Epoch int
	X     []float64
	Y     []float64
}

type SimpleNN struct {
	n0, n1, n2, n3 int
	w1              [][]*Value // [n0][n1]
	b1              []*Value   // [n1]
	w2              [][]*Value // [n1][n2]
	b2              []*Value   // [n2]
	w3              [][]*Value // [n2][n3]
	b3              []*Value   // [n3]
	Snapshots       []Snapshot
}

func NewSimpleNN(rng *rand.Rand, n0, n1, n2, n3 int) *SimpleNN {
	nn := &SimpleNN{n0: n0, n1: n1, n2: n2, n3: n3}
	nn.w1 = randMatrix(rng, n0, n1)
	nn.b1 = randArray(rng, n1)
	nn.w2 = randMatrix(rng, n1, n2)
	nn.b2 = randArray(rng, n2)
	nn.w3 = randMatrix(rng, n2, n3)
	nn.b3 = randArray(rng, n3)
	return nn
}

func randArray(rng *rand.Rand, n int) []*Value {
	std := 1.0 / math.Sqrt(float64(n))
	out := make([]*Value, n)
	for i := range out {
		out[i] = newValue(rng.NormFloat64() * std)
	}
	return out
}

func randMatrix(rng *rand.Rand, n, m int) [][]*Value {
	std := 1.0 / math.Sqrt(float64(n))
	out := make([][]*Value, n)
	for i := range out {
		out[i] = make([]*Value, m)
		for j := range out[i] {
			out[i][j] = newValue(rng.NormFloat64() * std)
		}
	}
	return out
}

func (nn *SimpleNN) forward(h0 []*Value) []*Value {
	h1 := make([]*Value, nn.n1)
	for i := 0; i < nn.n1; i++ {
		acc := newValue(0.0)
		for j := 0; j < nn.n0; j++ {
			acc = addV(acc, mulV(h0[j], nn.w1[j][i]))
		}
		acc = addV(acc, nn.b1[i])
		h1[i] = tanhV(acc)
	}

	h2 := make([]*Value, nn.n2)
	for i := 0; i < nn.n2; i++ {
		acc := newValue(0.0)
		for j := 0; j < nn.n1; j++ {
			acc = addV(acc, mulV(h1[j], nn.w2[j][i]))
		}
		acc = addV(acc, nn.b2[i])
		h2[i] = tanhV(acc)
	}

	h3 := make([]*Value, nn.n3)
	for i := 0; i < nn.n3; i++ {
		acc := newValue(0.0)
		for j := 0; j < nn.n2; j++ {
			acc = addV(acc, mulV(h2[j], nn.w3[j][i]))
		}
		acc = addV(acc, nn.b3[i])
		h3[i] = tanhV(acc)
	}
	return h3
}

func (nn *SimpleNN) loss(batch []Point) *Value {
	out := newValue(0.0)
	for _, p := range batch {
		x := make([]*Value, nn.n0)
		for i := 0; i < nn.n0; i++ {
			x[i] = newValue(p[i])
		}
		y := make([]*Value, nn.n3)
		for m := 0; m < nn.n3; m++ {
			y[m] = newValue(p[nn.n0+m])
		}
		v := nn.forward(x)
		for m := 0; m < nn.n3; m++ {
			out = addV(out, powV(subV(v[m], y[m]), 2))
		}
	}
	out = mulV(out, newValue(1.0/float64(len(batch))))
	return out
}

func (nn *SimpleNN) predict(t float64) (float64, float64) {
	h0 := []*Value{newValue(t)}
	out := nn.forward(h0)
	return out[0].Data, out[1].Data
}

func (nn *SimpleNN) getGraph(tList []float64) ([]float64, []float64) {
	xs := make([]float64, len(tList))
	ys := make([]float64, len(tList))
	for i, t := range tList {
		xs[i], ys[i] = nn.predict(t)
	}
	return xs, ys
}

func (nn *SimpleNN) addSnapshot(epoch int, tList []float64) {
	xs, ys := nn.getGraph(tList)
	nn.Snapshots = append(nn.Snapshots, Snapshot{Epoch: epoch, X: xs, Y: ys})
}
