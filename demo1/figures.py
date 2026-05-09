import math

def points(fig, n):
    ta = [t / n for t in range(n + 1)]
    out = []
    if fig == "heart":
        for t in ta:
            angle = t * 2 * math.pi
            x = 0.5 * math.sin(angle) ** 3
            y = (13 * math.cos(angle) - 5 * math.cos(2 * angle) - 2 * math.cos(3 * angle) - math.cos(4 * angle)) / 30
            out.append([t, x, y])
    elif fig == "circle":
        for t in ta:
            angle = t * 2 * math.pi
            x = 0.5 * math.cos(angle)
            y = 0.5 * math.sin(angle)
            out.append([t, x, y])
    elif fig == "astroid":
        for t in ta:
            angle = t * 2 * math.pi
            x = math.cos(angle) ** 3
            y = math.sin(angle) ** 3
            out.append([t, x, y])
    elif fig == "trefoil":
        for t in ta:
            angle = t * 2 * math.pi
            r = math.cos(3 * angle)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            out.append([t, x, y])
    elif fig == "square":
        for t in ta:
            s = t * 4
            side = int(s) % 4
            u = s - int(s)
            if side == 0:
                x, y = -0.5 + u, -0.5
            elif side == 1:
                x, y = 0.5, -0.5 + u
            elif side == 2:
                x, y = 0.5 - u, 0.5
            else:
                x, y = -0.5, 0.5 - u
            out.append([t, x, y])
    else:
        raise ValueError("wrong fig name")
    return out
