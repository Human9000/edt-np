import  numpy as np
from time import time

import numpy as np
import heapq
from line_profiler import LineProfiler

import cv2
from scipy.ndimage import distance_transform_edt

lp = LineProfiler()


def our_edtv1(border_arr):
    pxs, pys = np.where(border_arr == 1)
    w, h = border_arr.shape
    indices = np.full((w, h, 2), -1, dtype='int32')
    distances = np.full((w, h), np.inf)
    pq0 = []

    for px, py in zip(pxs, pys):
        indices[px, py] = [px, py]
        distances[px, py] = 0
        heapq.heappush(pq0, (0, px, py))  # push (distance, x, y)

    dxy = np.array([(0, 1), (0, -1), (1, 0), (-1, 0), ])
    pq = pq0.copy()

    while pq:
        dist, x, y = heapq.heappop(pq)

        if dist > distances[x, y]:
            continue

        for dx, dy in dxy:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                bx, by = indices[x, y]
                new_dist = (nx - bx) ** 2 + (ny - by) ** 2
                if new_dist < distances[nx, ny]:
                    distances[nx, ny] = new_dist
                    indices[nx, ny] = [bx, by]
                    heapq.heappush(pq, (new_dist, nx, ny))

    return np.sqrt(distances), indices


def our_edtv2(border_arr):
    pxs, pys = np.where(border_arr == 1)
    w, h = border_arr.shape
    indices = np.full((w, h, 2), -1, dtype='int32')
    distances = np.full((w, h), np.inf)
    pq = np.zeros((w * h, 3), dtype='int32')
    pql, pqr = 0, 0
    for px, py in zip(pxs, pys):
        indices[px, py] = [px, py]
        distances[px, py] = 0
        pq[pqr] = (0, px, py)
        pqr += 1
    # dxy = np.array([(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), ])
    dxy = np.array([(0, 1), (0, -1), (1, 0), (-1, 0), ])
    while pql != pqr:
        dist, x, y = pq[pql]
        pql += 1
        pql = pql % (w * h)

        if dist > distances[x, y]:
            continue

        for dx, dy in dxy:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                bx, by = indices[x, y]
                new_dist = (nx - bx) ** 2 + (ny - by) ** 2
                if new_dist < distances[nx, ny]:
                    distances[nx, ny] = new_dist
                    indices[nx, ny] = [bx, by]
                    pq[pqr] = (new_dist, nx, ny)
                    pqr += 1
                    pqr = pqr % (w * h)

    return np.sqrt(distances)


def our_edtv3(b):
    m, n = border_arr.shape
    g = np.full((m, n), -1, dtype='float32')
    for x in range(0, m):
        if b[x, 0]:
            g[x, 0] = 0
        else:
            g[x, 0] = float('inf')

        for y in range(1, n):
            if b[x, y]:  # 边界点
                g[x, y] = 0
            else:
                g[x, y] = 1 + g[x, y - 1]
        for y in range(n - 2, -1, -1):
            if g[x, y + 1] < g[x, y]:
                g[x, y] = 1 + g[x, y + 1]

    dt = np.full((m, n), np.inf, dtype='float32')
    # 对行进行扫描，对每个点(x,y)找到最近的背景点 (x1,y)，使得两点的欧式距离最小。
    s, t = np.zeros(m * n, dtype='int32'), np.zeros(m * n, dtype='int32')

    for y in range(n):
        q = 0
        s[0] = 0
        t[0] = 0
        # edt 欧式距离的f和Sep
        f = lambda x, i: (x - i) ** 2 + g[i, y] ** 2
        Sep = lambda i, u: (u ** 2 - i ** 2 + g[u, y] ** 2 - g[i, y] ** 2) / (2 * (u - i))
        for u in range(1, m):
            while q >= 0 and f(t[q], s[q]) > f(t[q], u):
                q -= 1
            if q < 0:
                q = 0
                s[0] = u
            else:
                w = 1 + Sep(s[q], u)
                if w < m:
                    q = q + 1
                    s[q] = u
                    t[q] = w
        for u in range(m - 1, -1, -1):
            dt[u, y] = f(u, s[q])
            if u == t[q]:
                q -= 1
    return dt ** 0.5


def our_edtv4(b):
    m, n = b.shape
    g = np.full((m, n), -1, dtype='int32')
    g_indices = np.full((m, n), -1, dtype='int32')
    indices = np.full((m, n, 2), -1, dtype='int32')  # 用于存储最近边界点的坐标
    distances = np.full((m, n), np.inf, dtype='float64')  # 用于存储最近边界点的距离
    s, t = np.zeros(m, dtype='int32'), np.zeros(m, dtype='int32')

    g[:, 0] = (b[:, 0] == 0) * n
    # :的部分可以并行
    for y in range(1, n):
        g[:, y] = (b[:, y] == 0) * (1 + g[:, y - 1])

    # :的部分可以并行
    for y in range(n - 2, -1, -1):
        mask = g[:, y + 1] < g[:, y]
        g[:, y] = mask * (1 + g[:, y + 1]) + (1 - mask) * g[:, y]
        g_indices[:, y] = y + (mask * 2 - 1) * g[:, y]

    # 这里的 y 可以并行
    for y in range(n):
        q = 0
        s[0] = 0
        t[0] = 0

        f = lambda x, i: (x - i) ** 2 + g[i, y] ** 2
        Sep = lambda i, u: (u ** 2 - i ** 2 + g[u, y] ** 2 - g[i, y] ** 2) / (2 * (u - i))

        for u in range(1, m):
            while q >= 0 and f(t[q], s[q]) > f(t[q], u):
                q -= 1
            if q < 0:
                q = 0
                s[0] = u
            else:
                w = 1 + Sep(s[q], u)
                if w < m:
                    q = q + 1
                    s[q] = u
                    t[q] = w
        for u in range(m - 1, -1, -1):
            distances[u, y] = f(u, s[q])
            indices[u, y] = [s[q], g_indices[s[q], y]]
            if u == t[q]:
                q -= 1
    return distances ** 0.5, indices


# Example usage
if __name__ == '__main__':
    # 创建一个空白的黑色图像
    image = np.zeros((500, 520, 3), dtype=np.uint8)
    d = (500 ** 2 + 520 ** 2) ** 0.5

    # 定义圆的厚度 (正数表示空心圆，-1 表示实心圆)
    thickness = 1

    # lp

    lp.add_function(our_edtv1)
    lp.add_function(our_edtv3)

    # 画空心圆
    cv2.circle(image, (250, 250), 100, (1, 1, 1), 1)
    cv2.circle(image, (150, 200), 50, (1, 1, 1), 1)
    cv2.circle(image, (450, 450), 50, (1, 1, 1), 1)

    border_arr = image[..., 0]
    border_arr[100] = 1
    border_arr[:, 100] = 1

    ts = [time()]
    distance3, indices3 = our_edtv4(border_arr)
    x, y = indices3[..., 0], indices3[..., 1]
    xyb3 = np.stack([x / 500, y / 520, border_arr], axis=-1)
    ts.append(time())
    distance4, indices4 = distance_transform_edt(1 - border_arr,
                                                 return_distances=True,
                                                 return_indices=True)
    x, y = indices4[0], indices4[1]
    xyb4 = np.stack([x / 500, y / 520, border_arr], axis=-1)
    ts.append(time())
    err = np.abs(distance3 - distance4)
    times =  np.diff(ts)
    print(err.sum(), times[0]/times[1])
    cv2.imshow('Image with distance', (distance3 / d) ** 0.5)
    cv2.imshow('Image with Circle4', (distance4 / d) ** 0.5)
    cv2.imshow('Image with indices+border', xyb3)
    cv2.imshow('Image with xyb4', xyb4)
    cv2.waitKey(0)

def our_edtv4_3d(b):
    p, m, n = b.shape
    g = np.full((p, m, n), -1, dtype='int32')
    g_indices = np.full((p, m, n, 2), -1, dtype='int32')
    indices = np.full((p, m, n, 3), -1, dtype='int32')  # 用于存储最近边界点的坐标
    distances = np.full((p, m, n), np.inf, dtype='float64')  # 用于存储最近边界点的距离

    g[:, :, 0] = (b[:, :, 0] == 0) * n
    for y in range(1, n):
        g[:, :, y] = (b[:, :, y] == 0) * (1 + g[:, :, y - 1]) + (b[:, :, y] == 1) * 0

    for y in range(n - 2, -1, -1):
        mask = g[:, :, y + 1] < g[:, :, y]
        g[:, :, y] = mask * (1 + g[:, :, y + 1]) + (1 - mask) * g[:, :, y]
        g_indices[:, :, y, 0] = y + (mask * 2 - 1) * g[:, :, y]

    s, t = np.zeros((p, m), dtype='int32'), np.zeros((p, m), dtype='int32')

    for y in range(n):
        for z in range(p):
            q = 0
            s[z, 0] = 0
            t[z, 0] = 0

            f = lambda x, i: (x - i) ** 2 + g[z, i, y] ** 2
            Sep = lambda i, u: (u ** 2 - i ** 2 + g[z, u, y] ** 2 - g[z, i, y] ** 2) / (2 * (u - i))

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

distances, indices = our_edtv4_3d(border_arr)
print("Distance Transform:\n", distances)
print("Nearest Coordinates:\n", indices)
