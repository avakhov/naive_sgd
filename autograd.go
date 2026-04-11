package main

import "math"

// Value is a node in the autograd computation graph.
type Value struct {
	Data     float64
	Grad     float64
	m        float64 // momentum / Adam first moment
	v        float64 // Adam second moment
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

func (v *Value) momentumStep(lr, beta float64) {
	topo := make([]*Value, 0, 64)
	visited := make(map[*Value]bool)
	buildTopo(v, visited, &topo)
	noop := func() {}
	for _, node := range topo {
		if len(node.prev) == 0 {
			node.m = beta*node.m + node.Grad
			node.Data -= lr * node.m
			node.Grad = 0.0
		} else {
			node.backward = noop
			node.prev = nil
		}
	}
}

func (v *Value) adamStep(lr float64, t int, beta1, beta2, eps float64) {
	topo := make([]*Value, 0, 64)
	visited := make(map[*Value]bool)
	buildTopo(v, visited, &topo)
	noop := func() {}
	for _, node := range topo {
		if len(node.prev) == 0 {
			node.m = beta1*node.m + (1-beta1)*node.Grad
			node.v = beta2*node.v + (1-beta2)*node.Grad*node.Grad
			mHat := node.m / (1 - math.Pow(beta1, float64(t)))
			vHat := node.v / (1 - math.Pow(beta2, float64(t)))
			node.Data -= lr * mHat / (math.Sqrt(vHat) + eps)
			node.Grad = 0.0
		} else {
			node.backward = noop
			node.prev = nil
		}
	}
}
