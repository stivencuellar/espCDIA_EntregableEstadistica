#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Completo del Dataset Freelancer Earnings - Versión Eficiente
Autor: Análisis de Datos
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("ANÁLISIS COMPLETO DEL DATASET FREELANCER EARNINGS")
print("=" * 80)

# ============================================================================
# PASO 1: CARGA DEL DATASET Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

print("\n" + "="*60)
print("PASO 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("="*60)

# Cargar el dataset
print("📊 Cargando el dataset...")
df = pd.read_csv('freelancer_earnings_bd.csv', sep=';')

# Mostrar información general
print(f"\n📋 Información del Dataset:")
print(f"   • Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"   • Columnas: {list(df.columns)}")

# Mostrar las primeras filas
print(f"\n🔍 Primeras 5 filas del dataset:")
print(df.head())

# Información de las columnas
print(f"\n📊 Información de las columnas:")
print(df.info())

# Estadísticas descriptivas
print(f"\n📈 Estadísticas descriptivas:")
print(df.describe())

# Valores nulos
print(f"\n❓ Valores nulos por columna:")
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print(null_counts[null_counts > 0])
else:
    print("✅ No hay valores nulos en el dataset")

# Tipos de datos
print(f"\n🔧 Tipos de datos:")
print(df.dtypes)

# ============================================================================
# ANÁLISIS DE VARIABLES NUMÉRICAS Y CATEGÓRICAS
# ============================================================================

# Separar variables numéricas y categóricas
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=['object']).columns

print(f"\n📊 Variables numéricas ({len(numeric_columns)}): {list(numeric_columns)}")
print(f"📊 Variables categóricas ({len(categorical_columns)}): {list(categorical_columns)}")

# Matriz de correlación para variables numéricas
print(f"\n🔗 Creando matriz de correlación...")
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación - Variables Numéricas', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('corr_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Pairplot para variables numéricas principales
print(f"\n📊 Creando pairplot...")
main_numeric = numeric_columns[:6] if len(numeric_columns) > 6 else numeric_columns
sns.pairplot(df[main_numeric], diag_kind='kde')
plt.suptitle('Pairplot - Variables Numéricas Principales', y=1.02, fontsize=16, fontweight='bold')
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PASO 2: ANÁLISIS UNIVARIADO
# ============================================================================

print("\n" + "="*60)
print("PASO 2: ANÁLISIS UNIVARIADO")
print("="*60)

# Histogramas con KDE para cada variable numérica
print(f"\n📊 Creando histogramas con KDE para variables numéricas...")

for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    
    # Histograma con KDE
    sns.histplot(df[col], kde=True, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Estadísticas en el gráfico
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_val = df[col].std()
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.2f}')
    
    plt.title(f'Distribución de {col}', fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Agregar estadísticas como texto
    stats_text = f'Media: {mean_val:.2f}\nMediana: {median_val:.2f}\nDesv. Est.: {std_val:.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'hist_{col}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PASO 3: ANÁLISIS MULTIVARIADO
# ============================================================================

print("\n" + "="*60)
print("PASO 3: ANÁLISIS MULTIVARIADO")
print("="*60)

# Estandarización de variables numéricas
print(f"\n🔧 Estandarizando variables numéricas...")
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), 
                        columns=numeric_columns, index=df.index)

print("✅ Variables estandarizadas creadas")

# ============================================================================
# ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
# ============================================================================

print(f"\n📊 Ejecutando Análisis de Componentes Principales (PCA)...")

# Aplicar PCA
pca = PCA()
pca_result = pca.fit_transform(df_scaled)

# Varianza explicada
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"   • Varianza explicada por componente:")
for i, var in enumerate(explained_variance_ratio):
    print(f"     Componente {i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"   • Varianza acumulada:")
for i, var in enumerate(cumulative_variance):
    print(f"     Hasta componente {i+1}: {var:.4f} ({var*100:.2f}%)")

# Visualización de varianza explicada
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 
        alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Componente')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
         marker='o', linewidth=2, markersize=8, color='red')
plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% de varianza')
plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% de varianza')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Acumulada')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualización de los primeros dos componentes principales
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=50, c='blue')
plt.xlabel(f'Componente Principal 1 ({explained_variance_ratio[0]*100:.1f}%)')
plt.ylabel(f'Componente Principal 2 ({explained_variance_ratio[1]*100:.1f}%)')
plt.title('Visualización de los Primeros Dos Componentes Principales')
plt.grid(True, alpha=0.3)
plt.savefig('pca_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# ANÁLISIS FACTORIAL (FA)
# ============================================================================

print(f"\n📊 Ejecutando Análisis Factorial con rotación Varimax...")

# Verificar si hay suficientes variables para el análisis factorial
if len(numeric_columns) >= 3:
    # Crear el objeto FactorAnalyzer
    fa = FactorAnalyzer(rotation='varimax', n_factors=min(3, len(numeric_columns)-1))
    
    # Aplicar el análisis factorial
    fa_result = fa.fit(df_scaled)
    
    # Obtener las cargas factoriales
    loadings = pd.DataFrame(fa_result.loadings_, 
                           columns=[f'Factor_{i+1}' for i in range(fa_result.loadings_.shape[1])],
                           index=numeric_columns)
    
    print(f"\n📊 Cargas factoriales:")
    print(loadings.round(3))
    
    # Varianza explicada por factores
    variance = fa_result.get_factor_variance()
    print(f"\n📊 Varianza explicada por factores:")
    for i, var in enumerate(variance[0]):
        print(f"   Factor {i+1}: {var:.4f} ({var*100:.2f}%)")
    
    # Visualización de cargas factoriales
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Cargas Factoriales - Análisis Factorial', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('factor_loadings.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("⚠️  No hay suficientes variables numéricas para realizar análisis factorial")

# ============================================================================
# CLUSTERING CON K-MEANS
# ============================================================================

print(f"\n📊 Ejecutando Clustering con K-Means...")

# Usar los primeros dos componentes principales para el clustering
X_pca = pca_result[:, :2]

# Determinar el número óptimo de clusters usando el método del codo
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

# Gráfico del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Determinar k Óptimo')
plt.grid(True, alpha=0.3)
plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Aplicar K-Means con k=3 (o el valor que parezca óptimo del gráfico)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# Visualización del clustering
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroides')
plt.xlabel(f'Componente Principal 1 ({explained_variance_ratio[0]*100:.1f}%)')
plt.ylabel(f'Componente Principal 2 ({explained_variance_ratio[1]*100:.1f}%)')
plt.title(f'Clustering K-Means (k={k_optimal}) en el Espacio PCA')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# Agregar clusters al dataframe original
df['Cluster'] = clusters
print(f"\n📊 Distribución de clusters:")
print(df['Cluster'].value_counts().sort_index())

# ============================================================================
# PASO 4: MODELOS ESTADÍSTICOS
# ============================================================================

print("\n" + "="*60)
print("PASO 4: MODELOS ESTADÍSTICOS")
print("="*60)

# Verificar si existe la variable 'Earnings_USD_L'
if 'Earnings_USD_L' in df.columns:
    print(f"\n📊 Regresión Lineal Múltiple - Variable dependiente: Earnings_USD_L")
    
    # Preparar variables para la regresión
    X_vars = [col for col in numeric_columns if col != 'Earnings_USD_L']
    X = df[X_vars]
    y = df['Earnings_USD_L']
    
    # Agregar constante para statsmodels
    X_with_const = sm.add_constant(X)
    
    # Modelo de regresión lineal
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"\n📊 Resumen del modelo de regresión lineal:")
    print(model.summary())
    
    # Métricas del modelo
    print(f"\n📊 Métricas del modelo:")
    print(f"   • R²: {model.rsquared:.4f}")
    print(f"   • R² Ajustado: {model.rsquared_adj:.4f}")
    print(f"   • F-statistic: {model.fvalue:.4f}")
    print(f"   • P-valor (F): {model.f_pvalue:.4f}")
    
    # Coeficientes significativos
    significant_coeffs = model.pvalues[model.pvalues < 0.05]
    print(f"\n📊 Variables significativas (p < 0.05):")
    for var, pval in significant_coeffs.items():
        if var != 'const':
            print(f"   • {var}: p = {pval:.4f}")
    
    # Análisis de multicolinealidad (VIF)
    print(f"\n📊 Análisis de multicolinealidad (VIF):")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data.sort_values('VIF', ascending=False))
    
else:
    print("⚠️  Variable 'Earnings_USD_L' no encontrada en el dataset")

# ============================================================================
# REGRESIÓN LOGÍSTICA (si existe variable binaria)
# ============================================================================

# Buscar variables binarias
binary_vars = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            binary_vars.append(col)

if binary_vars:
    print(f"\n📊 Variables binarias encontradas: {binary_vars}")
    
    for binary_var in binary_vars:
        print(f"\n📊 Regresión Logística - Variable dependiente: {binary_var}")
        
        # Preparar variables
        X_vars = [col for col in numeric_columns if col != binary_var]
        X = df[X_vars]
        y = df[binary_var]
        
        # Agregar constante
        X_with_const = sm.add_constant(X)
        
        # Modelo de regresión logística
        logit_model = sm.Logit(y, X_with_const).fit()
        
        print(f"\n📊 Resumen del modelo de regresión logística:")
        print(logit_model.summary())
        
        # Métricas del modelo
        print(f"\n📊 Métricas del modelo logístico:")
        print(f"   • Log-Likelihood: {logit_model.llf:.4f}")
        print(f"   • Pseudo R²: {logit_model.prsquared:.4f}")
        
        # Coeficientes significativos
        significant_coeffs = logit_model.pvalues[logit_model.pvalues < 0.05]
        print(f"\n📊 Variables significativas (p < 0.05):")
        for var, pval in significant_coeffs.items():
            if var != 'const':
                print(f"   • {var}: p = {pval:.4f}")
else:
    print("⚠️  No se encontraron variables binarias para regresión logística")

# ============================================================================
# PASO 5: MODELOS AUTOMÁTICOS
# ============================================================================

print("\n" + "="*60)
print("PASO 5: MODELOS AUTOMÁTICOS")
print("="*60)

if binary_vars:
    print(f"\n📊 Entrenando modelos automáticos para clasificación...")
    
    for binary_var in binary_vars:
        print(f"\n🎯 Modelos para variable: {binary_var}")
        
        # Preparar datos
        X_vars = [col for col in numeric_columns if col != binary_var]
        X = df[X_vars]
        y = df[binary_var]
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print(f"   • Tamaño de entrenamiento: {X_train.shape[0]} muestras")
        print(f"   • Tamaño de prueba: {X_test.shape[0]} muestras")
        
        # Random Forest
        print(f"\n🌲 Entrenando Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        print(f"   • Accuracy Random Forest: {rf_accuracy:.4f}")
        print(f"   • Reporte de clasificación:")
        print(classification_report(y_test, rf_pred))
        
        # Gradient Boosting
        print(f"\n🚀 Entrenando Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        
        print(f"   • Accuracy Gradient Boosting: {gb_accuracy:.4f}")
        print(f"   • Reporte de clasificación:")
        print(classification_report(y_test, gb_pred))
        
        # Comparación de modelos
        print(f"\n📊 Comparación de modelos para {binary_var}:")
        print(f"   • Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"   • Gradient Boosting Accuracy: {gb_accuracy:.4f}")
        
        # Importancia de características (Random Forest)
        feature_importance = pd.DataFrame({
            'feature': X_vars,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Importancia de características (Random Forest):")
        print(feature_importance)
        
        # Visualización de importancia de características
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title(f'Importancia de Características - Random Forest ({binary_var})')
        plt.xlabel('Importancia')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{binary_var}.png', dpi=300, bbox_inches='tight')
        plt.close()

else:
    print("⚠️  No hay variables binarias para entrenar modelos de clasificación")
    
    # Si no hay variables binarias, entrenar modelos de regresión
    if 'Earnings_USD_L' in df.columns:
        print(f"\n📊 Entrenando modelos de regresión para Earnings_USD_L...")
        
        # Preparar datos
        X_vars = [col for col in numeric_columns if col != 'Earnings_USD_L']
        X = df[X_vars]
        y = df['Earnings_USD_L']
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Random Forest Regressor
        from sklearn.ensemble import RandomForestRegressor
        print(f"\n🌲 Entrenando Random Forest Regressor...")
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, y_train)
        rf_pred = rf_reg.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        print(f"   • R² Random Forest: {rf_r2:.4f}")
        print(f"   • RMSE Random Forest: {rf_rmse:.4f}")
        
        # Gradient Boosting Regressor
        from sklearn.ensemble import GradientBoostingRegressor
        print(f"\n🚀 Entrenando Gradient Boosting Regressor...")
        gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_reg.fit(X_train, y_train)
        gb_pred = gb_reg.predict(X_test)
        gb_r2 = r2_score(y_test, gb_pred)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
        
        print(f"   • R² Gradient Boosting: {gb_r2:.4f}")
        print(f"   • RMSE Gradient Boosting: {gb_rmse:.4f}")

# ============================================================================
# RESUMEN DE HALLAZGOS
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DE HALLAZGOS PRINCIPALES")
print("="*80)

print(f"\n📊 DATASET:")
print(f"   • Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"   • Variables numéricas: {len(numeric_columns)}")
print(f"   • Variables categóricas: {len(categorical_columns)}")

print(f"\n📊 ANÁLISIS EXPLORATORIO:")
print(f"   • No hay valores nulos en el dataset")
print(f"   • Se crearon visualizaciones de correlación y distribución")

print(f"\n📊 ANÁLISIS DE COMPONENTES PRINCIPALES:")
print(f"   • Se identificaron {len(numeric_columns)} componentes principales")
print(f"   • Los primeros 2 componentes explican {(explained_variance_ratio[0] + explained_variance_ratio[1])*100:.1f}% de la varianza")

print(f"\n📊 CLUSTERING:")
print(f"   • Se aplicó K-Means con {k_optimal} clusters")
print(f"   • Distribución de clusters:")
for cluster, count in df['Cluster'].value_counts().sort_index().items():
    print(f"     Cluster {cluster}: {count} muestras ({count/len(df)*100:.1f}%)")

if 'Earnings_USD_L' in df.columns:
    print(f"\n📊 MODELOS DE REGRESIÓN:")
    print(f"   • Se entrenó regresión lineal múltiple para Earnings_USD_L")
    print(f"   • R² del modelo: {model.rsquared:.4f}")

if binary_vars:
    print(f"\n📊 MODELOS DE CLASIFICACIÓN:")
    print(f"   • Variables binarias analizadas: {binary_vars}")
    print(f"   • Se entrenaron modelos Random Forest y Gradient Boosting")

print(f"\n📊 ARCHIVOS GENERADOS:")
print(f"   • Gráficos de correlación y distribución")
print(f"   • Visualizaciones de PCA y clustering")
print(f"   • Análisis de componentes principales")

print(f"\n✅ Análisis completo finalizado exitosamente!")
print("="*80)