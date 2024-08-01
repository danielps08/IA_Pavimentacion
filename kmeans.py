import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# Cargar la imagen
image = io.imread(r"D:\Mi Material\Maestria\Inteligencia Artificial\PROYECTO DETECCION DE GRIETAS\Deteccion de grietas\DATASET\grieta1.jpeg")
# Redimensionar la imagen para reducir el tiempo de cálculo (opcional)
# from skimage.transform import resize
# image = resize(image, (100, 100))

# Mostrar la imagen original
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title('Imagen Original')
plt.axis('off')
plt.show()

# Convertir la imagen en un arreglo de 2D donde cada fila es un píxel y cada columna es un canal de color
pixels = image.reshape(-1, 3)

# Aplicar K-Means para segmentar la imagen en k clústeres
kmeans = KMeans(n_clusters=2, random_state=42).fit(pixels)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]

# Convertir el resultado a una imagen con la forma original
segmented_img = segmented_img.reshape(image.shape)

# Mostrar la imagen segmentada
plt.figure(figsize=(8, 8))
plt.imshow(segmented_img.astype(np.uint8))
plt.title('Imagen Segmentada')
plt.axis('off')
plt.show()
