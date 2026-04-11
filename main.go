package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// ---------- autograd ----------

type Value struct {
	Data     float64
	Grad     float64
	m        float64
	v        float64
	prev     []*Value
	backward func()
}

func newValue(data float64) *Value {
	return &Value{Data: data, backward: func() {}}
}

func mulV(a, b *Value) *Value {
	out := &Value{Data: a.Data * b.Data, prev: []*Value{a, b}}
	out.backward = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}
	return out
}

func addV(a, b *Value) *Value {
	out := &Value{Data: a.Data + b.Data, prev: []*Value{a, b}}
	out.backward = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}
	return out
}

func subV(a, b *Value) *Value {
	out := &Value{Data: a.Data - b.Data, prev: []*Value{a, b}}
	out.backward = func() {
		a.Grad += out.Grad
		b.Grad -= out.Grad
	}
	return out
}

func powV(a *Value, n float64) *Value {
	out := &Value{Data: math.Pow(a.Data, n), prev: []*Value{a}}
	out.backward = func() {
		a.Grad += out.Grad * n * math.Pow(a.Data, n-1)
	}
	return out
}

func tanhV(a *Value) *Value {
	t := math.Tanh(a.Data)
	out := &Value{Data: t, prev: []*Value{a}}
	out.backward = func() {
		th := math.Tanh(a.Data)
		a.Grad += out.Grad * (1 - th*th)
	}
	return out
}

func buildTopo(v *Value, visited map[*Value]bool, topo *[]*Value) {
	if !visited[v] {
		visited[v] = true
		for _, child := range v.prev {
			buildTopo(child, visited, topo)
		}
		*topo = append(*topo, v)
	}
}

func (v *Value) doBackward() {
	topo := make([]*Value, 0, 64)
	visited := make(map[*Value]bool)
	buildTopo(v, visited, &topo)
	v.Grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backward()
	}
}

func (v *Value) step(lr float64) {
	topo := make([]*Value, 0, 64)
	visited := make(map[*Value]bool)
	buildTopo(v, visited, &topo)
	noop := func() {}
	for _, node := range topo {
		if len(node.prev) == 0 {
			node.Data -= lr * node.Grad
			node.Grad = 0.0
		} else {
			node.backward = noop
			node.prev = nil
		}
	}
}

// ---------- neural net ----------

type Point [3]float64

type Snapshot struct {
	Epoch int
	X     []float64
	Y     []float64
}

type SimpleNN struct {
	n0, n1, n2, n3 int
	w1              [][]*Value
	b1              []*Value
	w2              [][]*Value
	b2              []*Value
	w3              [][]*Value
	b3              []*Value
	Snapshots       []Snapshot
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

func NewSimpleNN(rng *rand.Rand, n0, n1, n2, n3 int) *SimpleNN {
	return &SimpleNN{
		n0: n0, n1: n1, n2: n2, n3: n3,
		w1: randMatrix(rng, n0, n1), b1: randArray(rng, n1),
		w2: randMatrix(rng, n1, n2), b2: randArray(rng, n2),
		w3: randMatrix(rng, n2, n3), b3: randArray(rng, n3),
	}
}

func (nn *SimpleNN) forward(h0 []*Value) []*Value {
	h1 := make([]*Value, nn.n1)
	for i := 0; i < nn.n1; i++ {
		acc := newValue(0.0)
		for j := 0; j < nn.n0; j++ {
			acc = addV(acc, mulV(h0[j], nn.w1[j][i]))
		}
		h1[i] = tanhV(addV(acc, nn.b1[i]))
	}
	h2 := make([]*Value, nn.n2)
	for i := 0; i < nn.n2; i++ {
		acc := newValue(0.0)
		for j := 0; j < nn.n1; j++ {
			acc = addV(acc, mulV(h1[j], nn.w2[j][i]))
		}
		h2[i] = tanhV(addV(acc, nn.b2[i]))
	}
	h3 := make([]*Value, nn.n3)
	for i := 0; i < nn.n3; i++ {
		acc := newValue(0.0)
		for j := 0; j < nn.n2; j++ {
			acc = addV(acc, mulV(h2[j], nn.w3[j][i]))
		}
		h3[i] = tanhV(addV(acc, nn.b3[i]))
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
	return mulV(out, newValue(1.0/float64(len(batch))))
}

func (nn *SimpleNN) predict(t float64) (float64, float64) {
	out := nn.forward([]*Value{newValue(t)})
	return out[0].Data, out[1].Data
}

func (nn *SimpleNN) addSnapshot(epoch int, tList []float64) {
	xs := make([]float64, len(tList))
	ys := make([]float64, len(tList))
	for i, t := range tList {
		xs[i], ys[i] = nn.predict(t)
	}
	nn.Snapshots = append(nn.Snapshots, Snapshot{Epoch: epoch, X: xs, Y: ys})
}

// ---------- data ----------

func heartPoints(n int) []Point {
	pts := make([]Point, n+1)
	for i := 0; i <= n; i++ {
		t := float64(i) / float64(n)
		angle := t * 2 * math.Pi
		x := 0.5 * math.Pow(math.Sin(angle), 3)
		y := (13*math.Cos(angle) - 5*math.Cos(2*angle) - 2*math.Cos(3*angle) - math.Cos(4*angle)) / 30
		pts[i] = Point{t, x, y}
	}
	return pts
}

// ---------- training ----------

func sgd(nn *SimpleNN, rng *rand.Rand, dataset []Point, lr float64, epochs, batchSize, snapshotEvery int) {
	tList := make([]float64, len(dataset))
	for i, p := range dataset {
		tList[i] = p[0]
	}
	shuffled := make([]Point, len(dataset))

	for epoch := 0; epoch < epochs; epoch++ {
		copy(shuffled, dataset)
		rng.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })

		totalLoss := 0.0
		batches := 0
		for i := 0; i < len(shuffled); i += batchSize {
			end := i + batchSize
			if end > len(shuffled) {
				end = len(shuffled)
			}
			L := nn.loss(shuffled[i:end])
			L.doBackward()
			L.step(lr)
			totalLoss += L.Data
			batches++
		}
		if epoch%10 == 0 {
			fmt.Printf("epoch=%d, loss=%.6f\n", epoch, totalLoss/float64(batches))
		}
		if epoch%snapshotEvery == 0 || epoch == epochs-1 {
			nn.addSnapshot(epoch, tList)
		}
	}
}

// ---------- main ----------

func main() {
	rng := rand.New(rand.NewSource(123))
	dataset := heartPoints(200)
	nn := NewSimpleNN(rng, 1, 20, 20, 2)

	start := time.Now()
	sgd(nn, rng, dataset, 0.3, 100, 8, 30)
	fmt.Printf("elapsed: %v\n", time.Since(start))

	targetX := make([]float64, len(dataset))
	targetY := make([]float64, len(dataset))
	for i, p := range dataset {
		targetX[i] = p[1]
		targetY[i] = p[2]
	}

	type snapshotJSON struct {
		Epoch int       `json:"epoch"`
		X     []float64 `json:"x"`
		Y     []float64 `json:"y"`
	}
	type outputJSON struct {
		Target struct {
			X []float64 `json:"x"`
			Y []float64 `json:"y"`
		} `json:"target"`
		Snapshots []snapshotJSON `json:"snapshots"`
	}

	var out outputJSON
	out.Target.X = targetX
	out.Target.Y = targetY
	out.Snapshots = make([]snapshotJSON, len(nn.Snapshots))
	for i, s := range nn.Snapshots {
		out.Snapshots[i] = snapshotJSON{s.Epoch, s.X, s.Y}
	}

	f, _ := os.Create("output_go.json")
	defer f.Close()
	json.NewEncoder(f).Encode(out)
	fmt.Println("output_go.json saved.")
}
