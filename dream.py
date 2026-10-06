"""Hopfield Dream Machine: a network of 144 neurons remembers pixel-art from a noisy mess.

Run:  python dream.py   ->  assets/dream.gif
"""
import os
import numpy as np
from PIL import Image

HEART = [
    "............",
    "..###..###..",
    ".#####.#####",
    "############",
    "############",
    "############",
    ".##########.",
    "..########..",
    "...######...",
    "....####....",
    ".....##.....",
    "............",
]
SMILE = [
    "...######...",
    ".##########.",
    "############",
    "###..##..###",
    "###..##..###",
    "############",
    "############",
    "##.######.##",
    "###.####.###",
    "####....####",
    ".##########.",
    "...######...",
]
STAR = [
    "............",
    ".....##.....",
    ".....##.....",
    "....####....",
    "############",
    ".##########.",
    "..########..",
    "..########..",
    ".####..####.",
    ".###....###.",
    "##........##",
    "............",
]

SIDE = 12
N = SIDE * SIDE
BG = np.array([22, 27, 34], dtype=np.uint8)
ON = np.array([255, 126, 182], dtype=np.uint8)
GRID = np.array([13, 17, 23], dtype=np.uint8)


def to_vec(art):
    return np.array([1 if ch == "#" else -1 for row in art for ch in row])


def train(patterns):
    """Hebbian learning: neurons that fire together wire together."""
    W = sum(np.outer(p, p) for p in patterns) / N
    np.fill_diagonal(W, 0)
    return W


def render(states, scale=14, gap=14):
    tiles = []
    for s in states:
        img = np.where((s > 0).reshape(SIDE, SIDE, 1), ON, BG).astype(np.uint8)
        img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
        img[::scale, :] = GRID
        img[:, ::scale] = GRID
        tiles.append(img)
    spacer = np.full((SIDE * scale, gap, 3), 13, dtype=np.uint8)
    row = []
    for i, t in enumerate(tiles):
        if i:
            row.append(spacer)
        row.append(t)
    body = np.concatenate(row, axis=1)
    pad = np.full((gap, body.shape[1], 3), 13, dtype=np.uint8)
    return Image.fromarray(np.concatenate([pad, body, pad], axis=0))


def main():
    os.makedirs("assets", exist_ok=True)
    rng = np.random.default_rng(7)
    patterns = [to_vec(HEART), to_vec(SMILE), to_vec(STAR)]
    W = train(patterns)

    # Corrupt each memory by flipping 30% of its pixels.
    states = []
    for p in patterns:
        s = p.copy()
        flip = rng.choice(N, size=int(0.30 * N), replace=False)
        s[flip] *= -1
        states.append(s)

    frames = [render(states)] * 8
    sweeps, record_every = 6, 12
    count = 0
    for _ in range(sweeps):
        orders = [rng.permutation(N) for _ in states]
        for k in range(N):
            for s, order in zip(states, orders):
                i = order[k]
                h = W[i] @ s
                if h != 0:
                    s[i] = 1 if h > 0 else -1
            count += 1
            if count % record_every == 0:
                frames.append(render(states))
    frames += [render(states)] * 20

    frames[0].save("assets/dream.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)
    for name, p, s in zip(("heart", "smiley", "star"), patterns, states):
        print(f"{name}: {int((p == s).sum())}/{N} pixels recalled")
    print("saved assets/dream.gif")


if __name__ == "__main__":
    main()
