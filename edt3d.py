import numpy as np
from line_profiler import LineProfiler


lp = LineProfiler()


@lp
def myedt3d(b):
    p, m, n = b.shape
    g = np.full((p, m, n), -1, dtype='int32')
    g_indices = np.full((p, m, n, 2), -1, dtype='int32')
    indices = np.full((p, m, n, 3), -1, dtype='int32')  # 用于存储最近边界点的坐标
    distances = np.full((p, m, n), np.inf, dtype='float64')  # 用于存储最近边界点的距离

    g[..., 0] = (b[:, :, 0] == 0) * n
    for y in range(1, n):
        g[..., y] = (b[:, :, y] == 0) * (1 + g[..., y - 1]) + (b[:, :, y] == 1) * 0

    for y in range(n - 2, -1, -1):
        mask = g[..., y + 1] < g[..., y]
        g[..., y] = mask * (1 + g[..., y + 1]) + (1 - mask) * g[..., y]
        g_indices[:, :, y, 0] = y + (mask * 2 - 1) * g[..., y]

    s, t = np.zeros((p, m), dtype='int32'), np.zeros((p, m), dtype='int32')

    for y in range(n):
        for z in range(p):
            q = 0
            s[z, 0] = 0
            t[z, 0] = 0

            def f(x, i): return (x - i) ** 2 + g[z, i, y] ** 2

            def Sep(i, u): return (u ** 2 - i ** 2 +
                                   g[z, u, y] ** 2 - g[z, i, y] ** 2) / (2 * (u - i))

            for u in range(1, m):
                while q >= 0 and f(t[z, q], s[z, q]) > f(t[z, q], u):
                    q -= 1
                if q < 0:
                    q = 0
                    s[z, 0] = u
                else:
                    w = 1 + Sep(s[z, q], u)
                    if w < m:
                        q = q + 1
                        s[z, q] = u
                        t[z, q] = w
            for u in range(m - 1, -1, -1):
                distances[z, u, y] = f(u, s[z, q])
                indices[z, u, y] = [z, s[z, q], g_indices[z, s[z, q], y, 0]]
                if u == t[z, q]:
                    q -= 1

    return distances ** 0.5, indices


# 测试函数
border_arr = np.array([
    [[0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0]],
    [[1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]
], dtype=np.int32)

distances, indices = myedt3d(border_arr)

lp.print_stats()
# print("Distance Transform:\n", distances)
# print("Nearest Coordinates:\n", indices)
