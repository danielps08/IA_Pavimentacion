import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Cargar una imagen
image = cv2.imread(r"D:\CESEL\CODIGOS\IA_PAVIMENTACION\Imagenes\Crack_prueba_yolo\data_3527.jpg")

# Convertir la imagen a RGB si está en BGR 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir la imagen a un array de NumPy
image_array = np.array(image)
print(image_array[0][0])

# Obtener las dimensiones de la imagen
height, width, channels = image_array.shape
print(height)
print(width)
# Convertir la imagen en un DataFrame de pandas
# Cada fila representa un píxel con sus valores R, G, B
data = []

for y in range(height):
    for x in range(width):
        r, g, b = image_array[y, x]
        data.append([r, g, b])

# Crear un DataFrame con las columnas 'R', 'G', 'B'
df = pd.DataFrame(data, columns=['R', 'G', 'B'])

print(df.head())

# Mostrar la imagen usando Matplotlib
plt.imshow(image_array)
plt.title("Imagen Procesada")
plt.axis('off')  # Para ocultar los ejes
plt.show()

################

from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns

# Aplicar el algoritmo K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

# Obtener las etiquetas de los clusters
labels = kmeans.labels_

# Agregar las etiquetas de los clusters al DataFrame
df['Cluster'] = labels

print(df.head())

###########

# Visualizar los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(x='R', y='G', hue='Cluster', data=df, palette='viridis', style=y, s=100)
plt.title('Clustering con K-means')
plt.xlabel('Canal RED')
plt.ylabel('Canal GREEN')
plt.legend(loc='upper right')
plt.show()

##########

# Visualizar los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(x='R', y='B', hue='Cluster', data=df, palette='viridis', style=y, s=100)
plt.title('Clustering con K-means')
plt.xlabel('Canal RED')
plt.ylabel('Canal BLUE')
plt.legend(loc='upper right')
plt.show()

##########

# Visualizar los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(x='G', y='B', hue='Cluster', data=df, palette='viridis', style=y, s=100)
plt.title('Clustering con K-means')
plt.xlabel('Canal GREEN')
plt.ylabel('Canal BLUE')
plt.legend(loc='upper right')
plt.show()

###########

# Convertir el DataFrame en un array de NumPy
imagen_kmeans = df['Cluster'].values
print(imagen_kmeans.shape)
# Reshape el array a sus dimensiones originales
imagen_kmeans = imagen_kmeans.reshape((height, width))

# Convertir la matriz a una imagen
#imagen_f = Image.fromarray(imagen_kmeans)
#print(imagen_f.size)

# Crear una matriz RGB de la misma forma que la matriz original
imagen_rgb = np.zeros((height, width, 3), dtype=np.uint8)

# Asignar colores específicos a cada valor
# Por ejemplo, 0 -> Negro (0, 0, 0), 1 -> Rojo (255, 0, 0), 2 -> Verde (0, 255, 0)
imagen_rgb[imagen_kmeans == 0] = [255, 255, 255]
imagen_rgb[imagen_kmeans == 1] = [0, 0, 0]
imagen_rgb[imagen_kmeans == 2] = [255, 0, 0]

# Mostrar la imagen usando Matplotlib
plt.imshow(imagen_rgb)
plt.title("Imagen Procesada")
plt.axis('off')  # Para ocultar los ejes
plt.show()

################

# Rango de valores de K
K = range(1, 10)

# Lista para almacenar la inercia para cada valor de K
inertias = []

# Calcular K-means para cada valor de K
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Graficar la inercia contra el número de clusters
plt.figure(figsize=(8, 6))
plt.plot(K, inertias, 'bo-')
plt.xlabel('Número de Clusters K')
plt.ylabel('Inercia')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.show()

#############

from sklearn.preprocessing import StandardScaler

# Normalizar el dataset
scaler = StandardScaler()
data_normalizada = scaler.fit_transform(data)

# Crear un DataFrame para facilitar la visualización
df_n = pd.DataFrame(data_normalizada, columns=['R', 'G', 'B'])

print(df_n.head())

##############

# Rango de valores de K
K = range(1, 10)

# Lista para almacenar la inercia para cada valor de K
inertias = []

# Calcular K-means para cada valor de K
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_normalizada)
    inertias.append(kmeans.inertia_)

# Graficar la inercia contra el número de clusters
plt.figure(figsize=(8, 6))
plt.plot(K, inertias, 'bo-')
plt.xlabel('Número de Clusters K')
plt.ylabel('Inercia')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.show()

###########

# Aplicar el algoritmo K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_normalizada)

# Obtener las etiquetas de los clusters
labels = kmeans.labels_

# Agregar las etiquetas de los clusters al DataFrame
df_n['Cluster'] = labels

print(df_n.head())

#############

# Visualizar los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(x='R', y='G', hue='Cluster', data=df_n, palette='viridis', style=y, s=100)
plt.title('Clustering con K-means')
plt.xlabel('Canal RED')
plt.ylabel('Canal GREEN')
plt.legend(loc='upper right')
plt.show()

############

# Crear una figura 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar los datos normalizados con los clusters asignados
scatter = ax.scatter(data_normalizada[:, 0], data_normalizada[:, 1], data_normalizada[:, 2], 
                     c=labels, cmap='viridis', s=50)

# Etiquetas y título
ax.set_title('Clustering K-means en 3D (Datos Normalizados)')
ax.set_xlabel('Característica R (normalizada)')
ax.set_ylabel('Característica G (normalizada)')
ax.set_zlabel('Característica B (normalizada)')

# Leyenda
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

# Mostrar el gráfico
plt.show()

################
df_n.shape

# Convertir el DataFrame en un array de NumPy
array_imagen = df_n['Cluster'].values
print(array_imagen.shape)
# Reshape el array a sus dimensiones originales
array_imagen = array_imagen.reshape((height, width))

# Crear una matriz RGB de la misma forma que la matriz original
imagen_rgb_n = np.zeros((height, width, 3), dtype=np.uint8)

# Asignar colores específicos a cada valor
# Por ejemplo, 0 -> Negro (0, 0, 0), 1 -> Rojo (255, 0, 0), 2 -> Verde (0, 255, 0)
imagen_rgb_n[array_imagen == 0] = [255, 255, 255]
imagen_rgb_n[array_imagen == 1] = [0, 0, 0]
imagen_rgb_n[array_imagen == 2] = [255, 0, 0]


# Mostrar la imagen usando Matplotlib
plt.imshow(imagen_rgb_n)
plt.title("Imagen Procesada")
plt.axis('off')  # Para ocultar los ejes
plt.show()

#########

# Configurar el modelo KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Ajustar el modelo a los datos
kmeans.fit(data_normalizada)

# Predecir los clusters
y_kmeans = kmeans.predict(data_normalizada)

print(y_kmeans.shape)

# Reshape el array a sus dimensiones originales
array_img_kpp = y_kmeans.reshape((height, width))

# Crear una matriz RGB de la misma forma que la matriz original
imagen_rgb_kpp = np.zeros((height, width, 3), dtype=np.uint8)

# Asignar colores específicos a cada valor
# Por ejemplo, 0 -> Negro (0, 0, 0), 1 -> Rojo (255, 0, 0), 2 -> Verde (0, 255, 0)
imagen_rgb_kpp[array_img_kpp == 0] = [0, 0, 0]
imagen_rgb_kpp[array_img_kpp == 1] = [255, 255, 255]
imagen_rgb_kpp[array_img_kpp == 2] = [255, 0, 0]


# Mostrar la imagen usando Matplotlib
plt.imshow(imagen_rgb_kpp)
plt.title("Imagen Procesada")
plt.axis('off')  # Para ocultar los ejes
plt.show()

##############

print(data_normalizada)

df_n = df_n.drop(['Cluster'], axis=1)
df_n

############

from sklearn.cluster import DBSCAN

# Establecer parámetros: epsilon (eps) y el número mínimo de muestras (min_samples)
eps = 0.1
min_samples = 5

# Ejecutar DBSCAN
db = DBSCAN(eps=eps, min_samples=min_samples).fit(df_n)

# Agregar los labels al dataframe
df_n['cluster2.0'] = db.labels_

# Mostrar los clusters
print(df_n.head())

val = df_n['cluster2.0'].unique()
print(val)

df_n['cluster2.0'].value_counts()

############

# Convertir el DataFrame en un array de NumPy
array_imagen = df_n['cluster2.0'].values
print(array_imagen.shape)
# Reshape el array a sus dimensiones originales
array_imagen = array_imagen.reshape((height, width))

# Crear una matriz RGB de la misma forma que la matriz original
imagen_rgb_n = np.zeros((height, width, 3), dtype=np.uint8)

# Asignar colores específicos a cada valor
# Por ejemplo, 0 -> Negro (0, 0, 0), 1 -> Rojo (255, 0, 0), 2 -> Verde (0, 255, 0)
imagen_rgb_n[array_imagen == 0] = [255, 255, 255]
imagen_rgb_n[array_imagen == 4] = [0, 0, 0]
imagen_rgb_n[array_imagen == -1] = [255, 0, 0]


# Mostrar la imagen usando Matplotlib
plt.imshow(imagen_rgb_n)
plt.title("Imagen Procesada")
plt.axis('off')  # Para ocultar los ejes
plt.show()
