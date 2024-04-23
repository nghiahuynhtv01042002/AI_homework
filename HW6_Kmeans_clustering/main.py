import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

# Tạo bộ dữ liệu mẫu về màu sắc
N = 300
data = np.random.randint(0, 256, size=(N, 3), dtype=np.uint8)

# Áp dụng thuật toán K-means clustering để phân đoạn màu
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Trích xuất các điểm trung tâm của các cụm
colors = kmeans.cluster_centers_.astype(int)

# Phân loại mỗi điểm dữ liệu vào một cụm
labels = kmeans.labels_

# Tạo một colormap tùy chỉnh
# colors_custom = [(colors[i, 0]/255 +50/255, colors[i, 1]/255 + 50/255, colors[i, 2]/255 + 50/255) for i in range(len(colors))]
# custom_cmap = ListedColormap(colors_custom)
custom_colors = np.array([[255, 165, 0],  # Màu cam
                          [50, 50, 50],  # Màu vàng
                          [128, 0, 128]]) / 255  # Màu tím
custom_cmap = ListedColormap(custom_colors)

# Hiển thị màu gốc
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data/255)
ax.set_title('Màu gốc')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

# Hiển thị màu được phân loại
ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap=custom_cmap)
ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], marker='o', s=150, c=range(len(colors)), cmap=custom_cmap, edgecolor='k')

ax.set_title('Phân đoạn màu')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

plt.show()
