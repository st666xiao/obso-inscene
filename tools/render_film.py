"""Render the three shots in world coordinates; the clasp shares its shoe's transform."""

import math
import subprocess
import tempfile
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
FILM = ROOT / "film"
W, H, FPS, SECONDS = 960, 540, 24, 10
RELEASE = 5.40
BICUBIC = Image.Resampling.BICUBIC
LANCZOS = Image.Resampling.LANCZOS
BG = Image.open(FILM / "palace.png").convert("RGB").resize((W, H), LANCZOS)
ACTOR = Image.open(FILM / "runner.png").convert("RGBA")
SHOE = Image.open(FILM / "slipper.png").convert("RGBA")
SHOE = SHOE.crop(SHOE.getchannel("A").getbbox())
CLASP = Image.open(ROOT / "design/oblique.png").convert("RGBA")
CLASP = CLASP.crop(CLASP.getchannel("A").getbbox())
GROUND = 384
STEP_W, STEP_H = 78, 8


def floor_y(x):
    return GROUND + STEP_H * math.floor(x / STEP_W)


def smooth(x):
    x = max(0, min(1, x))
    return x * x * (3 - 2 * x)


def pose(t):
    elapsed = max(0, t - 1.35)
    x = 90 + 142 * elapsed
    phase = elapsed * 2 * math.pi * 1.48
    ground = GROUND + STEP_H * x / STEP_W
    hip = ground - 69 - 4 * abs(math.sin(phase))
    return x, hip, phase


def leg(t, far):
    x, hip, p = pose(t)
    p += math.pi if far else 0
    ax = x + 32 * math.sin(p)
    ay = floor_y(ax) - 6 - 25 * max(0, math.cos(p))
    kx = x + 13 + 18 * math.sin(p + 0.71)
    ky = (hip + ay) / 2 - 5
    angle = 8 * math.cos(p) - 7 * math.sin(p)
    return ((x, hip), (kx, ky), (ax, ay)), angle


def shoe_pose(t):
    joints, angle = leg(min(t, RELEASE), False)
    x, y = joints[-1]
    if t <= RELEASE:
        return x, y, angle
    dt = t - RELEASE
    # The loose slipper moves with the foot for an instant before it hits a lower step.
    flight = 0.72
    k = min(dt / flight, 1)
    end_x = x + 74
    end_y = floor_y(end_x) - 8
    sx = x + 74 * k
    sy = y + (end_y - y) * k - 45 * 4 * k * (1 - k)
    if dt <= flight:
        return sx, sy, angle * (1 - k) - 34 * math.sin(k * math.pi)
    after = dt - flight
    return (
        end_x,
        end_y - 4 * abs(math.sin(16 * after)) * math.exp(-10 * after),
        3 * math.sin(21 * after) * math.exp(-9 * after),
    )


def make_shoe(marked):
    piece = SHOE.copy()
    if marked:
        # Oblique view from the same mesh as the design plate. The clasp sits on the vamp.
        height = round(piece.height * 0.37)
        width = max(1, round(height * CLASP.width / CLASP.height))
        buckle = CLASP.resize((width, height), LANCZOS).rotate(
            -23, BICUBIC, expand=True
        )
        piece.alpha_composite(
            buckle,
            (
                round(0.735 * piece.width - buckle.width / 2),
                round(0.60 * piece.height - buckle.height / 2),
            ),
        )
    return piece


def camera(t):
    x, _, _ = pose(t)
    if t < 1.35:
        return None
    if t < 4.65:
        z = 1.13
        return z, x + 74, floor_y(x) - 127
    sx, sy, _ = shoe_pose(t)
    if t < 7.15:
        return 3.2, sx + 13, sy - 38
    return 11.5 + 1.4 * smooth((t - 7.15) / 2.85), sx + 12, sy - 4


def paste_pivot(canvas, sprite, pivot, pos, width, angle):
    factor = width / sprite.width
    dim = (max(1, round(width)), max(1, round(sprite.height * factor)))
    sprite = sprite.resize(dim, LANCZOS)
    # Rotate about an explicit local attachment point and preserve its screen position.
    pad = max(dim) * 2 + 5
    mid = pad // 2
    tmp = Image.new("RGBA", (pad, pad))
    tmp.alpha_composite(
        sprite, (mid - round(pivot[0] * dim[0]), mid - round(pivot[1] * dim[1]))
    )
    if angle:
        tmp = tmp.rotate(angle, BICUBIC)
    canvas.paste(tmp, (round(pos[0] - mid), round(pos[1] - mid)), tmp)


@lru_cache(maxsize=40)
def stone_block(index):
    # A world-locked grain pattern; repeated projection never changes its seed.
    rng = np.random.default_rng(index % 1009)
    w, h = 390, 500
    paper = rng.normal(0, 3, (h, w))
    rough = Image.fromarray(rng.integers(0, 255, (16, 13), dtype=np.uint8))
    cloud = np.asarray(rough.resize((w, h), BICUBIC), dtype=float) / 28 - 4
    yy = np.arange(h)[:, None]
    brightness = (
        paper + cloud - 5 * np.exp(-(((yy - 22) / 9) ** 2)) + 3 * np.exp(-yy / 80)
    )
    pixels = brightness[:, :, None] + np.array(
        [60 + index % 3, 72 + index % 4, 75 + index % 5]
    )
    slab = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    pen = ImageDraw.Draw(slab)
    for j in range(15):
        x = int(rng.integers(1, w - 1))
        y = int(rng.integers(1, h // 2))
        pen.line(
            [(x, y), (x + 4, y), (x + int(rng.integers(6, 19)), y - 2)],
            fill=(76, 84, 83),
            width=1,
        )
    return slab


def frame(t, marked, shoes):
    if t < 1.35:
        k = smooth(t / 1.35)
        box = (int(84 - 14 * k), int(12 - 6 * k), int(348 + 18 * k), int(160 + 12 * k))
        return ImageOps.fit(BG.crop(box), (W, H), method=LANCZOS)
    z, cx, cy = camera(t)

    def screen(x, y):
        return ((x - cx) * z + W / 2, (y - cy) * z + H / 2)

    # Distant skyline moves less than the stairs when the camera tracks the runner.
    xoff = min(146, max(0, (cx - 160) * 0.16))
    shot = BG.resize((W, H), BICUBIC, box=(xoff, 22, xoff + 795, 469))
    if z > 2:
        shot = shot.filter(ImageFilter.GaussianBlur(2 if z < 5 else 5))
    d = ImageDraw.Draw(shot)
    first = math.floor((cx - W / 2 / z) / STEP_W) - 1
    last = math.ceil((cx + W / 2 / z) / STEP_W) + 1
    for n in range(first, last):
        x0, y0 = n * STEP_W, GROUND + n * STEP_H
        l, y = screen(x0, y0)
        r, _ = screen(x0 + STEP_W, y0)
        bl, bt = math.floor(l), math.floor(y)
        bw = max(1, math.ceil(r) - bl)
        bh = max(1, round(bw * 500 / 390))
        slab = stone_block(n).resize((bw, bh), BICUBIC)
        shot.paste(slab, (bl, bt))
        d.rectangle(
            (math.floor(l), math.floor(y), math.ceil(r), math.ceil(y + 3 * z)),
            fill="#8c9588",
        )
        d.line((l, y + 4 * z, r, y + 4 * z), fill="#55696d", width=max(1, round(z)))
        for j in range(3):
            u = (17 + n * 9 + j * 23) % 70
            a, b = screen(x0 + u, y0 + 6 + j * 4)
            d.line(
                (a, b, a + (3 + j) * z, b + z * 0.3),
                fill="#586b6d",
                width=max(1, round(z * 0.4)),
            )
    x, hip, p = pose(t)

    def paint_shoe(at, turn, side):
        paste_pivot(shot, shoes[side], (0.16, 0.32), screen(*at), 34 * z, turn)

    # Once off camera, skip allocating enlarged puppet layers for the insert.
    if t < 7.15 and abs(screen(x, hip)[0] - W / 2) < W * 0.85:
        for far in (True, False):
            joints, angle = leg(t, far)
            coords = [screen(*j) for j in joints]
            ink = "#805d55" if far else "#b5816f"
            for a, b, thick in zip(coords, coords[1:], (12, 8)):
                d.line((a, b), fill=ink, width=max(1, round(thick * z)))
            ax, ay = joints[-1]
            if far or t <= RELEASE:
                paint_shoe((ax, ay), angle, marked and not far)
            else:
                a, b = screen(ax, ay)
                d.ellipse((a - 3 * z, b - 2 * z, a + 9 * z, b + 4 * z), fill=ink)
        paste_pivot(
            shot, ACTOR, (0.67, 0.78), screen(x, hip + 5), 243 * z, 2.3 * math.sin(p)
        )
    if t > RELEASE:
        sx, sy, a = shoe_pose(t)
        paint_shoe((sx, sy), a, marked)
    return shot


def score(path):
    rate = 44100
    mix = np.zeros(rate * SECONDS)

    def tone(start, length, partials, decay, gain):
        i = int(start * rate)
        if i >= len(mix):
            return
        time = np.arange(min(int(length * rate), len(mix) - i)) / rate
        signal = sum(amp * np.sin(2 * np.pi * freq * time) for freq, amp in partials)
        mix[i : i + len(time)] += (
            gain * signal * np.exp(-decay * time) * np.minimum(1, time / 0.008)
        )

    for t in (0.06, 1.3, 2.54):
        tone(t, 3.5, [(220, 1), (526, 0.38), (736, 0.13), (1142, 0.07)], 1.23, 0.22)
    for i in range(20):
        t = 1.44 + i / (2 * 1.48)
        tone(t, 0.12, [(90, 1), (165, 0.45)], 34, 0.10 if t < 6.2 else 0.04)
    tone(RELEASE + 0.72, 2, [(1476, 0.7), (2250, 0.4), (3330, 0.22)], 3, 0.18)
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, rate, 0, "NONE", ""))
        f.writeframes((np.clip(mix, -1, 1) * 32000).astype("<i2").tobytes())


def main():
    shoes = {b: make_shoe(b) for b in (False, True)}
    with tempfile.TemporaryDirectory() as tmp:
        sound = Path(tmp) / "foley.wav"
        score(sound)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-i",
            str(sound),
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(FILM / "midnight.mp4"),
        ]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
            for n in range(SECONDS * FPS):
                pic = frame(n / FPS, True, shoes)
                enc.stdin.write(pic.tobytes())
            enc.stdin.close()
            if enc.wait():
                raise RuntimeError("ffmpeg failed")
    frame(3.26, True, shoes).save(FILM / "running.jpg", quality=93)
    for marked, name in ((True, "with-clasp"), (False, "without-clasp")):
        frame(8, marked, shoes).save(FILM / (name + ".png"))


if __name__ == "__main__":
    main()
