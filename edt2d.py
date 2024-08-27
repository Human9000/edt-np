 

import numpy as np

import cv2
from scipy.ndimage import distance_transform_edt as normedt

from line_profiler import LineProfiler
lp = LineProfiler() 
 
def Sep(i, u, g, y): 
    return (u ** 2 - i ** 2 + g[u, y] ** 2 - g[i, y] ** 2) / (2 * (u - i))

def f(x, i, g, y): 
    return (x - i) ** 2 + g[i, y] ** 2

def f_big(x, i, j, g, y): 
    return (x - i) ** 2 + g[i, y] ** 2 > (x - j) ** 2 + g[j, y] ** 2

@lp 
def myedt(b):
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
        for u in range(1, m):
            while q >= 0 and f_big(t[q], s[q], u, g, y):
                q -= 1
            if q < 0:
                q = 0
                s[0] = u
            else:
                w = 1 + Sep(s[q], u, g, y)
                if w < m:
                    q = q + 1
                    s[q] = u
                    t[q] = w
        for u in range(m - 1, -1, -1):
            distances[u, y] = f(u, s[q], g, y)
            indices[u, y] = [s[q], g_indices[s[q], y]]
            if u == t[q]:
                q -= 1
    return distances ** 0.5, indices


# Example usage
if __name__ == '__main__':
    # 创建一个空白的黑色图像
    h, w = 500, 520
    image = np.zeros((h, w, 3), dtype=np.uint8)
    d = (h ** 2 + w ** 2) ** 0.5

    # 定义圆的厚度 (正数表示空心圆，-1 表示实心圆)
    thickness = 1

    # 画空心圆
    cv2.circle(image, (250, 250), 100, (1, 1, 1), 1)
    cv2.circle(image, (150, 200), 50, (1, 1, 1), 1)
    cv2.circle(image, (450, 450), 50, (1, 1, 1), 1)

    # 画线
    border_arr = image[..., 0]
    border_arr[100] = 1
    border_arr[:, 100] = 1
    # ===============================
    #          my edt
    # ===============================
    # lp 性能统计工具 
    # lp.add_function(myedt)
    # lp.enable_by_count()
    my_distance, my_indices = myedt(border_arr)
    # lp.disable_by_count()

    # indices+border可视化
    my_indices_border = np.stack([my_indices[..., 0] / h, my_indices[..., 1] / w, border_arr], axis=-1)

    # ===============================
    #          norm edt
    # ===============================
    norm_distance, norm_indices = normedt(1 - border_arr, return_distances=True, return_indices=True)

    # indices+border可视化
    norm_indices_border = np.stack([norm_indices[0] / h, norm_indices[1] / w, border_arr], axis=-1)
    err = np.abs(my_distance - norm_distance)
    lp.print_stats()

    cv2.imshow('Image with my_distance', (my_distance / d) ** 0.5)
    cv2.imshow('Image with norm_distance', (norm_distance / d) ** 0.5)
    cv2.imshow('Image with my_indices_border', my_indices_border)
    cv2.imshow('Image with norm_indices_border', norm_indices_border)
    cv2.waitKey(0)
