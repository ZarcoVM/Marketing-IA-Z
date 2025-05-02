from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar datos
sales_df = pd.read_csv("sales_data_sample.csv", encoding='latin1')
sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'], errors='coerce')
sales_df.drop("ORDERDATE", axis=1, inplace=True)

scaler = StandardScaler()
sales_df_scaled = scaler.fit_transform(sales_df.select_dtypes(include='number'))

graficas = []

def df_to_html(df):
    return df.to_html(classes="table table-striped table-bordered", border=0, justify="center")

def generar_graficas():
    if graficas:
        return

    graficas_data = [
        ("Pedidos por País", 'COUNTRY', 'Muestra el número total de pedidos realizados desde cada país.'),
        ("Pedidos por Estado", 'STATUS', 'Cantidad de pedidos agrupados por su estado.'),
        ("Pedidos por Línea de Producto", 'PRODUCTLINE', 'Visualiza qué líneas de productos tienen mayor cantidad de pedidos.'),
        ("Pedidos por Tamaño del Trato", 'DEALSIZE', 'Distribución de pedidos según el tamaño del trato comercial.'),
    ]

    for titulo, col, descripcion in graficas_data:
        vc = sales_df[col].value_counts()
        fig = px.bar(x=vc.index, y=vc.values, color=vc.index, title=titulo)
        tabla_df = vc.reset_index().rename(columns={'index': col, col: 'Cantidad'})
        graficas.append({
            'html': pio.to_html(fig, full_html=False),
            'titulo': titulo,
            'descripcion': descripcion,
            'tabla': df_to_html(tabla_df)
        })

    # Ventas por Mes
    sales_df_group = sales_df.groupby('MONTH_ID').sum(numeric_only=True)
    fig = px.line(x=sales_df_group.index, y=sales_df_group['SALES'], title='Ventas por Mes')
    tabla = sales_df_group[['SALES']].reset_index()
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'Ventas por Mes',
        'descripcion': 'Total de ventas agregadas por cada mes.',
        'tabla': df_to_html(tabla)
    })

    # Mapa de calor
    corr_matrix = sales_df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cbar=True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close('all')
    graficas.append({
        'html': f'<img src="data:image/png;base64,{img_base64}">',
        'titulo': 'Mapa de Calor de Correlaciones',
        'descripcion': 'Correlaciones entre variables numéricas del dataset.',
        'tabla': df_to_html(corr_matrix)
    })

    # Distribuciones
    for col in ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']:
        fig = ff.create_distplot([sales_df[col].astype(float)], [f"Distribución {col}"], show_hist=False)
        fig.update_layout(title_text=f'Distribución de {col}')
        estad = sales_df[[col]].describe().reset_index().rename(columns={'index': 'Estadístico'})
        graficas.append({
            'html': pio.to_html(fig, full_html=False),
            'titulo': f'Distribución de {col}',
            'descripcion': f'Distribución estadística de la variable {col}.',
            'tabla': df_to_html(estad)
        })

    # PCA 2D con clustering
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
    labels = kmeans.fit_predict(sales_df_scaled)
    pca = PCA(n_components=2)
    components = pca.fit_transform(sales_df_scaled)
    pca_df = pd.DataFrame(components, columns=['pca1', 'pca2'])
    pca_df['cluster'] = labels
    fig = px.scatter(pca_df, x='pca1', y='pca2', color=pca_df['cluster'].astype(str),
                     title='PCA 2D con Clustering')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'PCA 2D con Clustering',
        'descripcion': 'Visualización reducida a 2D con agrupación KMeans.',
        'tabla': df_to_html(pca_df.head(10))
    })

    # PCA 3D con clustering
    pca3d = PCA(n_components=3).fit_transform(sales_df_scaled)
    pca3d_df = pd.DataFrame(pca3d, columns=['PC1', 'PC2', 'PC3'])
    pca3d_df['cluster'] = labels
    fig = px.scatter_3d(pca3d_df, x='PC1', y='PC2', z='PC3', color=pca3d_df['cluster'].astype(str),
                        title='PCA 3D con Clustering')
    graficas.append({
        'html': pio.to_html(fig, full_html=False),
        'titulo': 'PCA 3D con Clustering',
        'descripcion': 'Agrupación visualizada en 3D con componentes principales.',
        'tabla': df_to_html(pca3d_df.head(10))
    })

@app.route('/')
@app.route('/grafica/<int:indice>')
def index(indice=0):
    generar_graficas()
    total = len(graficas)
    indice = max(0, min(indice, total - 1))
    grafica = graficas[indice]
    return render_template("index.html", grafica=grafica, indice=indice, total=total)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)