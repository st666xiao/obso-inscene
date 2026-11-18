"""Build the Twin Reed clasp in millimetres and export metre-based glTF and projections."""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon

HERE = Path(__file__).resolve().parent
BRONZE = (155, 109, 65)
GOLD = (187, 147, 84)


def curve(start, *segments):
    result = [np.array(start, dtype=float)]
    for a, b, c in segments:
        p = result[-1]
        for t in np.linspace(0, 1, 24)[1:]:
            result.append(
                (1 - t) ** 3 * p
                + 3 * (1 - t) ** 2 * t * np.array(a)
                + 3 * (1 - t) * t * t * np.array(b)
                + t**3 * np.array(c)
            )
    return np.array(result)


def silhouette():
    outline = curve(
        (0, 0),
        ((-4, 1), (-13, 10), (-15, 21)),
        ((-16, 26), (-15, 32), (-16, 36)),
        ((-10, 32), (-6, 27), (-4.2, 21.5)),
        ((-3.0, 18.2), (-2, 18.4), (-1.3, 22)),
        ((-0.7, 32), (5.5, 44), (15, 50)),
        ((15.8, 50.2), (14.4, 48.8), (14, 47.8)),
        ((11.3, 42), (14.8, 34), (12, 23)),
        ((10.1, 14), (5, 5), (0, 0)),
    )
    body = Polygon(outline).difference(Point(-10.25, 24).buffer(2.0, quad_segs=18))
    assert body.is_valid and len(body.interiors) == 1
    return body


def solid(poly, depth, base=0):
    mesh = trimesh.creation.extrude_polygon(poly, depth, engine="earcut")
    mesh.apply_translation((0, 0, base))
    return mesh


def rib(coords, radius):
    # A chased ridge narrows toward both ends. All rings follow the stroke in XY.
    count = 10
    verts = []
    for i, p in enumerate(coords):
        direction = coords[min(i + 1, len(coords) - 1)] - coords[max(i - 1, 0)]
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        width = radius * (0.12 + 0.88 * np.sin(i * np.pi / (len(coords) - 1)) ** 0.58)
        for a in np.arange(count) * 2 * np.pi / count:
            x, y = p + normal * width * np.cos(a)
            verts.append((x, y, 2.73 + 0.42 * width * np.sin(a)))
    faces = []
    for i in range(len(coords) - 1):
        for j in range(count):
            a = i * count + j
            b = i * count + (j + 1) % count
            faces.extend(((a, b, a + count), (b, b + count, a + count)))
    for j in range(1, count - 1):
        end = (len(coords) - 1) * count
        faces.extend(((0, j + 1, j), (end, end + j, end + j + 1)))
    mesh = trimesh.Trimesh(verts, faces)
    mesh.fix_normals()
    return mesh


def model():
    body = silhouette()
    parts = [
        ("cast edge", solid(body, 2.05), BRONZE),
        ("face bevel", solid(body.buffer(-0.5, quad_segs=4), 0.68, 2.05), BRONZE),
    ]
    strokes = [
        ((0.2, 3), ((3, 15), (8, 27), (12.2, 44.5))),
        ((-0.35, 4.2), ((-5, 11), (-8, 17), (-8.3, 20.4))),
        ((-11.6, 27), ((-12, 28.2), (-13.5, 31), (-14.2, 33))),
        ((5.7, 24), ((7.1, 25), (9.3, 28), (10.2, 31.2))),
    ]
    for i, (start, segment) in enumerate(strokes):
        parts.append(
            (
                f"raised vein {i + 1}",
                rib(curve(start, segment), 0.90 if i == 0 else 0.56),
                GOLD,
            )
        )
    # Cross section in X/Z. The bridge is an inverted U on the reverse.
    section = Polygon(
        [
            (-6.5, 0.2),
            (-6.5, -4.4),
            (-5.1, -5.8),
            (5.1, -5.8),
            (6.5, -4.4),
            (6.5, 0.2),
            (4.8, 0.2),
            (4.8, -3.7),
            (4.2, -4.1),
            (-4.2, -4.1),
            (-4.8, -3.7),
            (-4.8, 0.2),
        ]
    )
    strap = solid(section, 2.8)
    v = strap.vertices.copy()
    strap.vertices = np.column_stack((v[:, 0], 13.1 - v[:, 2], v[:, 1]))
    strap.fix_normals()
    parts.append(("sewing bridge", strap, (119, 84, 53)))
    return parts


def export(parts):
    s = trimesh.Scene()
    for name, mesh, rgb in parts:
        m = mesh.copy()
        m.apply_scale(0.001)
        m.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[*rgb, 255], metallicFactor=0.73, roughnessFactor=0.46
            )
        )
        s.add_geometry(m, node_name=name, geom_name=name)
    (HERE / "twin-reed.glb").write_bytes(s.export(file_type="glb"))


def view(
    parts, direction, up=(0, 1, 0), scale=10, light=(0.3, 0.65, 0.7), tint=(1, 1, 1)
):
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    vertical = np.cross(direction, right)
    light = np.array(light)
    light = light / np.linalg.norm(light)
    layers = []
    for _, mesh, color in parts:
        verts = mesh.vertices
        pts = np.column_stack((verts @ right, -verts @ vertical)) * scale
        tris = pts[mesh.faces]
        depth = (verts @ direction)[mesh.faces]
        dot = mesh.face_normals @ light
        levels = 0.49 + 0.61 * np.maximum(dot, 0)
        colors = np.clip(levels[:, None] * np.array(color) * tint, 0, 255).astype(
            "uint8"
        )
        layers.extend(zip(depth, tris, colors))
    bounds = np.concatenate([t for _, t, _ in layers])
    lo = np.floor(bounds.min(axis=0)) - 2
    size = np.ceil(bounds.max(axis=0) - lo + 3).astype(int)
    w, h = map(int, size)
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    zbuf = np.full((h, w), -np.inf)
    for depths, triangle, color in layers:
        a, b, c = triangle - lo
        left, top = np.maximum(0, np.floor(np.minimum(np.minimum(a, b), c))).astype(int)
        right_edge, bottom = np.minimum(
            (w, h), np.ceil(np.maximum(np.maximum(a, b), c)) + 1
        ).astype(int)
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-8:
            continue
        yy, xx = np.mgrid[top:bottom, left:right_edge] + 0.5
        u = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denom
        v = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denom
        depth = u * depths[0] + v * depths[1] + (1 - u - v) * depths[2]
        window = zbuf[top:bottom, left:right_edge]
        mask = (u >= 0) & (v >= 0) & (u + v <= 1) & (depth > window)
        window[mask] = depth[mask]
        pixels[top:bottom, left:right_edge][mask] = (*color, 255)
    im = Image.fromarray(pixels)
    return im.crop(im.getchannel("A").getbbox())


def font(n):
    for path in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(path, n)
        except OSError:
            pass
    return ImageFont.load_default(size=n)


def sheet(parts):
    paper = "#f1eadb"
    ink = "#322e26"
    gray = "#746954"
    im = Image.new("RGB", (1500, 820), paper)
    d = ImageDraw.Draw(im)
    d.text((46, 31), "Twin Reed", font=font(64), fill=ink)
    d.text(
        (48, 116),
        "Cast-bronze shoe clasp   /   Orthographic study",
        font=font(25),
        fill=gray,
    )
    d.text((1210, 66), "OBSO   /   01", font=font(31), fill=ink)
    d.line((48, 174, 1452, 174), fill=gray)
    views = [
        ("Front", (0, 0, 1), (0, 1, 0)),
        ("Right", (1, 0, 0), (0, 1, 0)),
        ("Top", (0, 1, 0), (0, 0, -1)),
        ("Reverse detail", (0.65, 0.25, -1), (0, 1, 0)),
    ]
    for i, (label, cam, up) in enumerate(views):
        x = 49 + 360 * i
        v = view(parts, cam, up, scale=8)
        im.paste(v, (int(x + 158 - v.width / 2), int(427 - v.height / 2)), v)
        d.line((x, 655, x + 316, 655), fill=gray)
        d.text((x, 673), label, font=font(30), fill=ink)
    d.text(
        (48, 761),
        "Nominal leaf height 50 mm     /     Perforation 4 mm     /     Rear bridge clearance 4.1 mm",
        font=font(22),
        fill=gray,
    )
    im.save(HERE / "orthographic.png")


def conditions(parts):
    im = Image.new("RGB", (1312, 610), "#eee8da")
    d = ImageDraw.Draw(im)
    d.text((44, 30), "Twin Reed  /  Surface and light", font=font(42), fill="#312e26")
    items = [
        ("Daylight", (1, 1, 1), (0.3, 0.65, 0.7), "#879188"),
        ("Firelight", (1.35, 0.83, 0.50), (-0.65, 0.5, 0.48), "#594039"),
        ("Under water", (0.68, 0.91, 1.08), (0.4, 0.75, 0.45), "#47676d"),
        ("Soot", (0.59, 0.61, 0.61), (0.45, 0.62, 0.4), "#78736c"),
    ]
    for i, (label, tint, light, bg) in enumerate(items):
        x = 44 + 309 * i
        d.rectangle((x, 114, x + 289, 482), fill=bg)
        v = view(parts, (0.23, 0.12, 1), scale=5.9, light=light, tint=tint)
        im.paste(v, (x + (289 - v.width) // 2, 114 + (368 - v.height) // 2), v)
        d.text((x, 511), label, font=font(26), fill="#322e26")
    im.save(HERE / "conditions.png")


if __name__ == "__main__":
    parts = model()
    export(parts)
    for name, direction in [
        ("front", (0, 0, 1)),
        ("oblique", (0.32, 0.17, 1)),
        ("rear", (0.6, 0.3, -1)),
    ]:
        view(parts, direction, scale=14).save(HERE / (name + ".png"))
    sheet(parts)
    conditions(parts)
