// Seeded pseudo-random number generator (Mulberry32)
function makeRandom(seed) {
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

