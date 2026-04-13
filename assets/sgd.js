function* sgd(model, dataset, lr, epochs, batch_size = 32, num_snapshots = 20, snap_points = 100) {
    const t_list = Array.from({ length: snap_points + 1 }, (_, i) => i / snap_points);
    const snap_epochs = new Set();
    for (let i = 0; i < num_snapshots; i++)
        snap_epochs.add(Math.round(i * (epochs - 1) / (num_snapshots - 1)));

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
        if (snap_epochs.has(epoch)) {
            const [netX, netY] = model.getGraph(t_list);
            model.snapshots.push({ epoch, netX, netY });
        }
        yield { epoch, loss };
    }
}
