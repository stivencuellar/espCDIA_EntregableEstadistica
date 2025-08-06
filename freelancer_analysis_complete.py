#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Completo del Dataset Freelancer Earnings
=================================================

Este script realiza un análisis exhaustivo del dataset freelancer_earnings_bd.csv
incluyendo EDA, análisis univariado, multivariado, modelos estadísticos y automáticos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
print("Cargando el dataset...")
df = pd.read_csv('freelancer_earnings_bd.csv', sep=';')

# Mostrar información básica
print(f"\n📊 INFORMACIÓN GENERAL DEL DATASET:")
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")

# Mostrar las primeras filas
print(f"\n🔍 PRIMERAS 5 FILAS DEL DATASET:")
print(df.head())

# Información de las columnas
print(f"\n📋 INFORMACIÓN DE LAS COLUMNAS:")
print(df.info())

# Estadísticas descriptivas
print(f"\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe())

# Valores nulos
print(f"\n❌ VALORES NULOS POR COLUMNA:")
null_counts = df.isnull().sum()
null_percentages = (df.isnull().sum() / len(df)) * 100
null_info = pd.DataFrame({
    'Valores_Nulos': null_counts,
    'Porcentaje_Nulos': null_percentages
})
print(null_info[null_info['Valores_Nulos'] > 0])

# Tipos de datos
print(f"\n🔧 TIPOS DE DATOS:")
print(df.dtypes.value_counts())

# Identificar variables numéricas y categóricas
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 VARIABLES NUMÉRICAS ({len(numeric_columns)}):")
print(numeric_columns)
print(f"\n📊 VARIABLES CATEGÓRICAS ({len(categorical_columns)}):")
print(categorical_columns)

# Visualizaciones generales
print(f"\n📊 CREANDO VISUALIZACIONES GENERALES...")

# Matriz de correlación
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Matriz de Correlación - Variables Numéricas', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Pairplot para variables numéricas principales
print("Generando pairplot...")
numeric_subset = df[numeric_columns].sample(n=min(500, len(df)), random_state=42)
sns.pairplot(numeric_subset, diag_kind='kde')
plt.suptitle('Pairplot - Relaciones entre Variables Numéricas', y=1.02, fontsize=16)
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PASO 2: ANÁLISIS UNIVARIADO
# ============================================================================

print("\n" + "="*60)
print("PASO 2: ANÁLISIS UNIVARIADO")
print("="*60)

print("📊 CREANDO HISTOGRAMAS CON KDE PARA CADA VARIABLE NUMÉRICA...")

# Crear histogramas con KDE para cada variable numérica
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.ravel()

for i, col in enumerate(numeric_columns):
    if i < len(axes):
        # Histograma con KDE
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f'Distribución de {col}', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
        
        # Agregar estadísticas en el gráfico
        mean_val = df[col].mean()
        std_val = df[col].std()
        axes[i].axvline(mean_val, color='red', linestyle='--', 
                       label=f'Media: {mean_val:.2f}')
        axes[i].legend()

# Ocultar subplots vacíos
for i in range(len(numeric_columns), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.suptitle('Análisis Univariado - Distribuciones de Variables Numéricas', 
             y=1.02, fontsize=16)
plt.savefig('univariate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Estadísticas adicionales por variable
print(f"\n📈 ESTADÍSTICAS DETALLADAS POR VARIABLE:")
for col in numeric_columns:
    print(f"\n{col}:")
    print(f"  Media: {df[col].mean():.2f}")
    print(f"  Mediana: {df[col].median():.2f}")
    print(f"  Desv. Estándar: {df[col].std():.2f}")
    print(f"  Mínimo: {df[col].min():.2f}")
    print(f"  Máximo: {df[col].max():.2f}")
    print(f"  Asimetría: {df[col].skew():.2f}")
    print(f"  Curtosis: {df[col].kurtosis():.2f}")

# ============================================================================
# PASO 3: ANÁLISIS MULTIVARIADO
# ============================================================================

print("\n" + "="*60)
print("PASO 3: ANÁLISIS MULTIVARIADO")
print("="*60)

# Preparar datos para análisis multivariado
print("🔧 PREPARANDO DATOS PARA ANÁLISIS MULTIVARIADO...")

# Seleccionar variables numéricas relevantes (excluir IDs si existen)
analysis_columns = [col for col in numeric_columns if 'ID' not in col and 'id' not in col]
print(f"Variables seleccionadas para análisis: {analysis_columns}")

# Crear dataset para análisis
df_analysis = df[analysis_columns].copy()

# Verificar y manejar valores nulos
if df_analysis.isnull().sum().sum() > 0:
    print("⚠️  Valores nulos detectados. Aplicando imputación...")
    df_analysis = df_analysis.fillna(df_analysis.median())

# Estandarización
print("📊 APLICANDO ESTANDARIZACIÓN...")
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_analysis),
    columns=df_analysis.columns
)

print("✅ Estandarización completada")

# PCA - Análisis de Componentes Principales
print("\n🔍 EJECUTANDO ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)...")

# Determinar número óptimo de componentes
pca_full = PCA()
pca_full.fit(df_scaled)

# Calcular varianza explicada acumulada
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Visualizar varianza explicada
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Componente')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada')
plt.axhline(y=0.8, color='g', linestyle='--', label='80% de varianza')
plt.axhline(y=0.9, color='orange', linestyle='--', label='90% de varianza')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('pca_variance_explained.png', dpi=300, bbox_inches='tight')
plt.show()

# Determinar número de componentes para 80% de varianza
n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1

print(f"Componentes para 80% de varianza: {n_components_80}")
print(f"Componentes para 90% de varianza: {n_components_90}")

# Aplicar PCA con número óptimo de componentes
n_components_optimal = min(n_components_80, len(analysis_columns))
pca = PCA(n_components=n_components_optimal)
pca_result = pca.fit_transform(df_scaled)

# Crear DataFrame con componentes principales
pca_df = pd.DataFrame(
    pca_result,
    columns=[f'PC{i+1}' for i in range(n_components_optimal)]
)

print(f"\n📊 COMPONENTES PRINCIPALES CREADOS:")
print(f"Número de componentes: {n_components_optimal}")
print(f"Varianza explicada total: {sum(pca.explained_variance_ratio_):.3f}")

# Mostrar carga de variables en componentes principales
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components_optimal)],
    index=analysis_columns
)

print(f"\n📋 CARGA DE VARIABLES EN COMPONENTES PRINCIPALES:")
print(loadings)

# Visualizar primeros dos componentes principales
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} varianza)')
plt.title('Proyección en los Primeros Dos Componentes Principales')
plt.grid(True)
plt.savefig('pca_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Análisis Factorial (FA)
print("\n🔍 EJECUTANDO ANÁLISIS FACTORIAL...")

# Verificar si hay suficientes variables para FA
if len(analysis_columns) >= 3:
    try:
        # Crear objeto FactorAnalyzer
        fa = FactorAnalyzer(rotation='varimax', n_factors=min(3, len(analysis_columns)))
        fa.fit(df_scaled)
        
        # Obtener cargas factoriales
        loadings_fa = pd.DataFrame(
            fa.loadings_,
            columns=[f'Factor{i+1}' for i in range(fa.n_factors_)],
            index=analysis_columns
        )
        
        print(f"\n📊 CARGAS FACTORIALES (Rotación Varimax):")
        print(loadings_fa)
        
        # Visualizar cargas factoriales
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings_fa, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Cargas Factoriales - Análisis Factorial con Rotación Varimax')
        plt.tight_layout()
        plt.savefig('factor_analysis_loadings.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Error en Análisis Factorial: {e}")
else:
    print("⚠️  Insuficientes variables para Análisis Factorial")

# Clustering con KMeans
print("\n🎯 REALIZANDO CLUSTERING CON KMEANS...")

# Determinar número óptimo de clusters usando método del codo
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_result[:, :2])  # Usar solo los primeros 2 componentes
    inertias.append(kmeans.inertia_)

# Visualizar método del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Determinar Número Óptimo de Clusters')
plt.grid(True)
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# Aplicar KMeans con número óptimo de clusters
n_clusters_optimal = 3  # Basado en el método del codo
kmeans = KMeans(n_clusters=n_clusters_optimal, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_result[:, :2])

# Visualizar clusters en espacio PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                     c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} varianza)')
plt.title(f'Clustering KMeans (k={n_clusters_optimal}) en Espacio PCA')
plt.grid(True)
plt.savefig('kmeans_clustering_pca.png', dpi=300, bbox_inches='tight')
plt.show()

# Agregar etiquetas de cluster al dataset original
df['Cluster'] = cluster_labels

print(f"\n📊 DISTRIBUCIÓN DE CLUSTERS:")
print(df['Cluster'].value_counts().sort_index())

# ============================================================================
# PASO 4: MODELOS ESTADÍSTICOS
# ============================================================================

print("\n" + "="*60)
print("PASO 4: MODELOS ESTADÍSTICOS")
print("="*60)

# Verificar si existe la variable Earnings_USD_L
if 'Earnings_USD_L' in df.columns:
    print("📊 REALIZANDO REGRESIÓN LINEAL MÚLTIPLE...")
    
    # Preparar variables para regresión
    # Excluir la variable dependiente y variables no numéricas
    exclude_cols = ['Earnings_USD_L', 'Cluster']
    feature_cols = [col for col in numeric_columns if col not in exclude_cols]
    
    if len(feature_cols) > 0:
        X = df[feature_cols].copy()
        y = df['Earnings_USD_L']
        
        # Manejar valores nulos
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Agregar constante para statsmodels
        X_sm = sm.add_constant(X)
        
        # Regresión lineal múltiple con statsmodels
        model = sm.OLS(y, X_sm).fit()
        
        print(f"\n📈 RESULTADOS DE REGRESIÓN LINEAL MÚLTIPLE:")
        print(model.summary())
        
        # Verificar multicolinealidad
        print(f"\n🔍 ANÁLISIS DE MULTICOLINEALIDAD (VIF):")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_sm.columns
        vif_data["VIF"] = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
        print(vif_data)
        
        # Predicciones vs valores reales
        y_pred = model.predict(X_sm)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs Valores Reales')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        plt.title('Análisis de Residuos')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Coeficientes más importantes
        coef_df = pd.DataFrame({
            'Variable': model.params.index,
            'Coeficiente': model.params.values,
            'P-valor': model.pvalues.values
        })
        coef_df = coef_df.sort_values('P-valor')
        
        print(f"\n📊 COEFICIENTES ORDENADOS POR SIGNIFICANCIA:")
        print(coef_df)
        
    else:
        print("⚠️  No hay suficientes variables predictoras para regresión")
else:
    print("⚠️  Variable 'Earnings_USD_L' no encontrada")

# Regresión Logística (si existe variable binaria)
binary_vars = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            binary_vars.append(col)

if binary_vars:
    print(f"\n📊 VARIABLES BINARIAS ENCONTRADAS: {binary_vars}")
    
    for binary_var in binary_vars:
        print(f"\n🔍 REGRESIÓN LOGÍSTICA PARA {binary_var}...")
        
        # Preparar datos
        exclude_cols = [binary_var, 'Cluster']
        feature_cols = [col for col in numeric_columns if col not in exclude_cols]
        
        if len(feature_cols) > 0:
            X = df[feature_cols].copy()
            y = df[binary_var]
            
            # Manejar valores nulos
            X = X.fillna(X.median())
            y = y.fillna(y.mode()[0])
            
            # Agregar constante
            X_sm = sm.add_constant(X)
            
            # Regresión logística
            logit_model = sm.Logit(y, X_sm).fit(disp=0)
            
            print(f"\n📈 RESULTADOS DE REGRESIÓN LOGÍSTICA PARA {binary_var}:")
            print(logit_model.summary())
            
            # Odds ratios
            odds_ratios = np.exp(logit_model.params)
            conf_int = np.exp(logit_model.conf_int())
            
            odds_df = pd.DataFrame({
                'Variable': odds_ratios.index,
                'Odds_Ratio': odds_ratios.values,
                'CI_Lower': conf_int.iloc[:, 0],
                'CI_Upper': conf_int.iloc[:, 1]
            })
            
            print(f"\n📊 ODDS RATIOS:")
            print(odds_df)
else:
    print("⚠️  No se encontraron variables binarias para regresión logística")

# ============================================================================
# PASO 5: MODELOS AUTOMÁTICOS
# ============================================================================

print("\n" + "="*60)
print("PASO 5: MODELOS AUTOMÁTICOS")
print("="*60)

if binary_vars:
    print(f"🎯 ENTRENANDO MODELOS AUTOMÁTICOS PARA CLASIFICACIÓN...")
    
    for binary_var in binary_vars:
        print(f"\n📊 MODELOS PARA VARIABLE: {binary_var}")
        
        # Preparar datos
        exclude_cols = [binary_var, 'Cluster']
        feature_cols = [col for col in numeric_columns if col not in exclude_cols]
        
        if len(feature_cols) > 0:
            X = df[feature_cols].copy()
            y = df[binary_var]
            
            # Manejar valores nulos
            X = X.fillna(X.median())
            y = y.fillna(y.mode()[0])
            
            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Random Forest
            print(f"\n🌲 RANDOM FOREST:")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            
            print(f"Accuracy: {rf_accuracy:.4f}")
            print(f"Reporte de clasificación:")
            print(classification_report(y_test, rf_pred))
            
            # Gradient Boosting
            print(f"\n🚀 GRADIENT BOOSTING:")
            gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            gb_accuracy = accuracy_score(y_test, gb_pred)
            
            print(f"Accuracy: {gb_accuracy:.4f}")
            print(f"Reporte de clasificación:")
            print(classification_report(y_test, gb_pred))
            
            # Comparar modelos
            print(f"\n📊 COMPARACIÓN DE MODELOS:")
            print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
            print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
            
            # Visualizar importancia de características
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            rf_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(range(len(rf_importance)), rf_importance['importance'])
            plt.yticks(range(len(rf_importance)), rf_importance['feature'])
            plt.xlabel('Importancia')
            plt.title('Importancia de Características - Random Forest')
            
            plt.subplot(1, 2, 2)
            gb_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': gb_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(range(len(gb_importance)), gb_importance['importance'])
            plt.yticks(range(len(gb_importance)), gb_importance['feature'])
            plt.xlabel('Importancia')
            plt.title('Importancia de Características - Gradient Boosting')
            
            plt.tight_layout()
            plt.savefig(f'feature_importance_{binary_var}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Matrices de confusión
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            cm_rf = confusion_matrix(y_test, rf_pred)
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Matriz de Confusión - Random Forest ({binary_var})')
            plt.ylabel('Valor Real')
            plt.xlabel('Predicción')
            
            plt.subplot(1, 2, 2)
            cm_gb = confusion_matrix(y_test, gb_pred)
            sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Matriz de Confusión - Gradient Boosting ({binary_var})')
            plt.ylabel('Valor Real')
            plt.xlabel('Predicción')
            
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{binary_var}.png', dpi=300, bbox_inches='tight')
            plt.show()
else:
    print("⚠️  No se encontraron variables binarias para modelos de clasificación")

# ============================================================================
# RESUMEN DE HALLAZGOS
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DE HALLAZGOS PRINCIPALES")
print("="*80)

print("\n📊 HALLAZGOS DEL ANÁLISIS EXPLORATORIO:")
print("• Dimensiones del dataset:", df.shape)
print("• Variables numéricas:", len(numeric_columns))
print("• Variables categóricas:", len(categorical_columns))
print("• Valores nulos totales:", df.isnull().sum().sum())

if 'Earnings_USD_L' in df.columns:
    print(f"• Variable objetivo 'Earnings_USD_L' encontrada")
    print(f"  - Media: {df['Earnings_USD_L'].mean():.2f}")
    print(f"  - Mediana: {df['Earnings_USD_L'].median():.2f}")
    print(f"  - Rango: {df['Earnings_USD_L'].min():.2f} - {df['Earnings_USD_L'].max():.2f}")

print("\n📊 HALLAZGOS DEL ANÁLISIS MULTIVARIADO:")
print(f"• Componentes principales óptimos: {n_components_optimal}")
print(f"• Varianza explicada total: {sum(pca.explained_variance_ratio_):.3f}")
print(f"• Clusters identificados: {n_clusters_optimal}")

if binary_vars:
    print(f"\n📊 HALLAZGOS DE MODELOS DE CLASIFICACIÓN:")
    print(f"• Variables binarias encontradas: {binary_vars}")
    print("• Modelos entrenados: Random Forest y Gradient Boosting")

print("\n📊 RECOMENDACIONES:")
print("• Revisar correlaciones entre variables para identificar multicolinealidad")
print("• Considerar transformaciones para variables con distribuciones asimétricas")
print("• Evaluar la necesidad de más datos o variables adicionales")
print("• Validar modelos con datos externos si es posible")

print("\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE!")
print("="*80)