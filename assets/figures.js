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
            const x = 0.5 * Math.cos(angle) ** 3;
            const y = 0.5 * Math.sin(angle) ** 3;
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
