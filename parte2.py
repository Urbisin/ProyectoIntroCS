import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

datos = pd.read_csv('smogonclusters.csv')
print(datos)
datos.drop(['grupo'], axis=1, inplace=True)  # eliminamos cluster para que no entre en datos
tabla = pd.DataFrame(data=datos)
print(tabla)
tabla.drop(tabla.columns[0], axis=1, inplace=True)  # eliminamos indice repetido
print(tabla)

pca = PCA(n_components=7)  # ncomponents comprime a n dimensiones
pca.fit(datos)
matrizpca = pca.transform(datos)
filas, columnas = tabla.shape
print('\nEl número de filas del dataframe original es ', filas, 'y de columnas', columnas)
filas1, columnas1 = matrizpca.shape
print('El número de filas de la matriz de componentes principales es ', filas1, 'y de columnas', columnas1, '\n')

cabeceras = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7']
tablapca = pd.DataFrame(data=matrizpca, columns=cabeceras)
print(tablapca)

# agrupamiento
km = KMeans(n_clusters=18, n_init=40)
listaclusters = km.fit_predict(tablapca)
print('\n', listaclusters)

tablapca['cluster'] = listaclusters
print('\n', tablapca)
tablapca.to_csv("smogonpca.csv")
