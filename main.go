package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

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

func sgd(nn *SimpleNN, rng *rand.Rand, dataset []Point, lr float64, epochs, batchSize int, snapshotEvery int) {
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
			batch := shuffled[i:end]
			L := nn.loss(batch)
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

type outputJSON struct {
	Target    targetJSON     `json:"target"`
	Snapshots []snapshotJSON `json:"snapshots"`
}

type targetJSON struct {
	X []float64 `json:"x"`
	Y []float64 `json:"y"`
}

type snapshotJSON struct {
	Epoch int       `json:"epoch"`
	X     []float64 `json:"x"`
	Y     []float64 `json:"y"`
}

func main() {
	rng := rand.New(rand.NewSource(123))

	dataset := heartPoints(200)

	nn := NewSimpleNN(rng, 1, 20, 20, 2)

	start := time.Now()
	sgd(nn, rng, dataset, 0.3, 100, 8, 30)
	elapsed := time.Since(start)
	fmt.Printf("elapsed: %v\n", elapsed)

	targetX := make([]float64, len(dataset))
	targetY := make([]float64, len(dataset))
	for i, p := range dataset {
		targetX[i] = p[1]
		targetY[i] = p[2]
	}

	snapshots := make([]snapshotJSON, len(nn.Snapshots))
	for i, s := range nn.Snapshots {
		snapshots[i] = snapshotJSON{Epoch: s.Epoch, X: s.X, Y: s.Y}
	}

	out := outputJSON{
		Target:    targetJSON{X: targetX, Y: targetY},
		Snapshots: snapshots,
	}

	f, err := os.Create("output_go.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	if err := enc.Encode(out); err != nil {
		panic(err)
	}
	fmt.Println("output_go.json saved.")
}
