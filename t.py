# 测试 edt3d 的性能
import numpy as np
from edt3d import myedt3d, lp

# 创建一个3D数组并设置一些边界点
test_array = np.zeros((10, 10, 10), dtype=int)

# 设置一些边界点
# 在x-y平面上画一个小正方形
test_array[2:8, 2, 5] = 1
test_array[2:8, 8, 5] = 1
test_array[2, 2:8, 5] = 1
test_array[8, 2:8, 5] = 1
 
# 画一个空心小球形状的边界
center = np.array([5, 5, 5])
outer_radius = 3
inner_radius = 2.5
x, y, z = np.ogrid[:10, :10, :10]
outer_mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= outer_radius**2
inner_mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= inner_radius**2
test_array[outer_mask & ~inner_mask] = 1



from scipy.ndimage import distance_transform_edt as normedt

# 运行myedt3d函数
# distances, indices = myedt3d(test_array)
norm_distance, norm_indices = normedt(1 - test_array, return_distances=True, return_indices=True)
distances = norm_distance 
indices = norm_indices.transpose(1,2,3,0)

# 打印性能统计信息
lp.print_stats()

# 输出结果的一些基本统计信息
print(f"距离数组形状: {distances.shape}")
print(f"最小距离: {np.min(distances)}")
print(f"最大距离: {np.max(distances)}")
print(f"平均距离: {np.mean(distances)}")
 
# 按照三个轴，分别做切片，然后保存正一张大图
import matplotlib.pyplot as plt
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 创建一个3x10的子图布局
fig, axes = plt.subplots(3, 10, figsize=(50, 15))

# 对每个轴进行切片
for i, axis in enumerate(['x', 'y', 'z']):
    for j in range(10):
        slice_index = j
        
        if axis == 'x':
            distance_slice = distances[slice_index, :, :]
            indices_slice = indices[slice_index, :, :, :]
            boundary_slice = test_array[slice_index, :, :]
        elif axis == 'y':
            distance_slice = distances[:, slice_index, :]
            indices_slice = indices[:, slice_index, :, :]
            boundary_slice = test_array[:, slice_index, :]
        else:  # z
            distance_slice = distances[:, :, slice_index]
            indices_slice = indices[:, :, slice_index, :]
            boundary_slice = test_array[:, :, slice_index]
        
        # 绘制距离图
        im = axes[i, j].imshow(distance_slice, cmap='viridis')
        axes[i, j].set_title(f'{axis}轴距离图 (切片 {j+1})')
        
        # 添加颜色条
        if j == 9:
            fig.colorbar(im, ax=axes[i, j], orientation='vertical', label='距离')
        
        # 移除坐标轴刻度
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig('edt3d_visualization_distances.png', dpi=300)
plt.close()

# 创建新的图形用于索引图
fig, axes = plt.subplots(3, 10, figsize=(50, 15))

for i, axis in enumerate(['x', 'y', 'z']):
    for j in range(10):
        slice_index = j
        
        if axis == 'x':
            indices_slice = indices[slice_index, :, :, :]
        elif axis == 'y':
            indices_slice = indices[:, slice_index, :, :]
        else:  # z
            indices_slice = indices[:, :, slice_index, :]
        
        # 绘制索引图
        indices_rgb = np.stack([indices_slice[..., k] / test_array.shape[k] for k in range(3)], axis=-1)
        axes[i, j].imshow(indices_rgb)
        axes[i, j].set_title(f'{axis}轴索引图 (切片 {j+1})')
        
        # 移除坐标轴刻度
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig('edt3d_visualization_indices.png', dpi=300)
plt.close()

# 创建新的图形用于边界图
fig, axes = plt.subplots(3, 10, figsize=(50, 15))

for i, axis in enumerate(['x', 'y', 'z']):
    for j in range(10):
        slice_index = j
        
        if axis == 'x':
            boundary_slice = test_array[slice_index, :, :]
        elif axis == 'y':
            boundary_slice = test_array[:, slice_index, :]
        else:  # z
            boundary_slice = test_array[:, :, slice_index]
        
        # 绘制边界图
        axes[i, j].imshow(boundary_slice, cmap='binary')
        axes[i, j].set_title(f'{axis}轴边界图 (切片 {j+1})')
        
        # 移除坐标轴刻度
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig('edt3d_visualization_boundaries.png', dpi=300)
plt.close()

print("图像已保存为 'edt3d_visualization_distances.png'、'edt3d_visualization_indices.png' 和 'edt3d_visualization_boundaries.png'")