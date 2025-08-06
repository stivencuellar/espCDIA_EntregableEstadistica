#!/usr/bin/env python3
"""
Análisis Completo del Dataset de Freelancer Earnings
====================================================
Este script realiza un análisis completo del dataset freelancer_earnings_bd.csv
siguiendo los pasos solicitados: EDA, análisis univariado, multivariado, 
modelos estadísticos y modelos automáticos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS COMPLETO DEL DATASET DE FREELANCER EARNINGS")
print("="*80)

# ============================================================================
# PASO 1️⃣: CARGA DEL DATASET Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================
print("\n" + "="*50)
print("PASO 1️⃣: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("="*50)

# Cargar el dataset
df = pd.read_csv('/workspace/freelancer_earnings_bd.csv', sep=';')

print("\n🔍 PRIMERAS FILAS DEL DATASET:")
print(df.head())

print("\n📊 INFORMACIÓN GENERAL DEL DATASET:")
print(f"Dimensiones: {df.shape}")
print("\nTipos de datos:")
print(df.dtypes)
print("\nInformación detallada:")
df.info()

print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe())

print("\n❌ VALORES NULOS:")
print(df.isnull().sum())

# Identificar variables numéricas y categóricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 Variables numéricas ({len(numeric_cols)}): {numeric_cols}")
print(f"📊 Variables categóricas ({len(categorical_cols)}): {categorical_cols}")

# Crear variable binaria de éxito si no existe
if 'Success_Binary' not in df.columns:
    # Crear variable binaria basada en el percentil 75 de earnings
    earnings_threshold = df['Earnings_USD'].quantile(0.75)
    df['Success_Binary'] = (df['Earnings_USD'] >= earnings_threshold).astype(int)
    print(f"\n✅ Variable Success_Binary creada (threshold: ${earnings_threshold:.2f})")

# Crear variable logarítmica de Earnings si no existe
if 'Earnings_USD_L' not in df.columns:
    df['Earnings_USD_L'] = np.log1p(df['Earnings_USD'])
    print("✅ Variable Earnings_USD_L (log) creada")

# Actualizar lista de variables numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Matriz de correlación
plt.figure(figsize=(14, 10))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Matriz de Correlación - Variables Numéricas', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/correlation_matrix_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# Pairplot para visualizar relaciones
print("\n🎨 Creando visualizaciones de relaciones entre variables...")
# Seleccionar las variables más importantes para el pairplot
key_vars = ['Earnings_USD', 'Hourly_Rate', 'Job_Success_Rate', 'Client_Rating', 
           'Job_Completed', 'Success_Binary']
if len(key_vars) > 6:
    key_vars = key_vars[:6]

plt.figure(figsize=(15, 12))
sns.pairplot(df[key_vars], hue='Success_Binary', diag_kind='kde', height=2.5)
plt.suptitle('Relaciones entre Variables Principales', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/pairplot_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PASO 2️⃣: ANÁLISIS UNIVARIADO
# ============================================================================
print("\n" + "="*50)
print("PASO 2️⃣: ANÁLISIS UNIVARIADO")
print("="*50)

# Histogramas con KDE para cada variable numérica
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

plt.figure(figsize=(18, 6 * n_rows))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    
    # Histograma con KDE
    sns.histplot(data=df, x=col, kde=True, alpha=0.7, color='skyblue')
    plt.title(f'Distribución de {col}', fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    
    # Añadir estadísticas básicas
    mean_val = df[col].mean()
    median_val = df[col].median()
    plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Media: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Mediana: {median_val:.2f}')
    plt.legend()

plt.suptitle('Análisis Univariado - Distribuciones de Variables Numéricas', 
             fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('/workspace/univariate_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PASO 3️⃣: ANÁLISIS MULTIVARIADO
# ============================================================================
print("\n" + "="*50)
print("PASO 3️⃣: ANÁLISIS MULTIVARIADO")
print("="*50)

# Preparar datos para análisis multivariado
numeric_data = df[numeric_cols].copy()

# Estandarización
print("\n🔧 Estandarizando variables numéricas...")
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_data)
numeric_scaled_df = pd.DataFrame(numeric_scaled, columns=numeric_cols)

print("✅ Estandarización completada")
print("Estadísticas después de la estandarización:")
print(numeric_scaled_df.describe())

# ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
print("\n🔍 ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
print("-" * 40)

pca = PCA()
pca_result = pca.fit_transform(numeric_scaled)

# Varianza explicada
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Componentes necesarios para explicar 95% de varianza: {n_components_95}")
print("Varianza explicada por componente:")
for i, var in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

# Visualización de varianza explicada
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_, 'bo-', linewidth=2, markersize=8)
plt.title('Varianza Explicada por Componente', fontweight='bold')
plt.xlabel('Componente Principal')
plt.ylabel('Proporción de Varianza Explicada')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), 
         cumulative_variance, 'ro-', linewidth=2, markersize=8)
plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% Varianza')
plt.title('Varianza Explicada Acumulada', fontweight='bold')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/pca_variance_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                     c=df['Success_Binary'], cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='Success Binary')
plt.title('PCA - Primeros Dos Componentes Principales', fontweight='bold', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% varianza)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/pca_scatter_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# ANÁLISIS FACTORIAL
print("\n🔍 ANÁLISIS FACTORIAL")
print("-" * 40)

# Determinar número óptimo de factores
fa_test = FactorAnalyzer(rotation=None)
fa_test.fit(numeric_scaled)

# Usar criterio de Kaiser (eigenvalues > 1)
eigenvals = fa_test.get_eigenvalues()[0]
n_factors = sum(eigenvals > 1)
print(f"Número de factores sugerido (Kaiser): {n_factors}")
print("Eigenvalues:", eigenvals[:5])

# Análisis factorial con rotación varimax
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa.fit(numeric_scaled)

# Cargas factoriales
loadings = fa.loadings_
loadings_df = pd.DataFrame(loadings, 
                          index=numeric_cols, 
                          columns=[f'Factor {i+1}' for i in range(n_factors)])

print("\nCargas Factoriales:")
print(loadings_df.round(3))

# Visualización de cargas factoriales
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, 
            square=False, linewidths=0.5)
plt.title('Cargas Factoriales (Rotación Varimax)', fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/factor_loadings_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# CLUSTERING K-MEANS
print("\n🔍 ANÁLISIS DE CLUSTERING K-MEANS")
print("-" * 40)

# Determinar número óptimo de clusters usando el método del codo
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(numeric_scaled)
    inertias.append(kmeans.inertia_)

# Visualizar método del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.title('Método del Codo para K-Means', fontweight='bold')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/kmeans_elbow_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# Aplicar K-means con k=3 (asumiendo como óptimo)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(numeric_scaled)

# Añadir clusters al dataset
df['Cluster'] = clusters

print(f"Distribución de clusters:")
print(df['Cluster'].value_counts().sort_index())

# Visualizar clusters en espacio PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                     c=clusters, cmap='tab10', alpha=0.7, s=50)
plt.colorbar(scatter, label='Cluster')
plt.title('Clusters K-Means en Espacio PCA', fontweight='bold', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% varianza)')

# Añadir centroides
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroides')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/kmeans_clusters_pca_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PASO 4️⃣: MODELOS ESTADÍSTICOS
# ============================================================================
print("\n" + "="*50)
print("PASO 4️⃣: MODELOS ESTADÍSTICOS")
print("="*50)

# REGRESIÓN LINEAL MÚLTIPLE
print("\n📈 REGRESIÓN LINEAL MÚLTIPLE")
print("-" * 40)

# Preparar variables para regresión
target_var = 'Earnings_USD_L'
feature_cols = [col for col in numeric_cols if col not in [target_var, 'Earnings_USD', 'Success_Binary', 'Freelancer_ID']]

X = df[feature_cols]
y = df[target_var]

# Añadir constante para statsmodels
X_sm = sm.add_constant(X)

# Ajustar modelo
linear_model = sm.OLS(y, X_sm).fit()
print("RESUMEN DEL MODELO DE REGRESIÓN LINEAL MÚLTIPLE:")
print(linear_model.summary())

# Predicciones
y_pred_linear = linear_model.predict(X_sm)
r2_linear = r2_score(y, y_pred_linear)
print(f"\nR² del modelo: {r2_linear:.4f}")

# Gráfico de residuos
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_pred_linear, y - y_pred_linear, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Gráfico de Residuos')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')

plt.subplot(1, 3, 2)
plt.scatter(y, y_pred_linear, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'red', linestyle='--')
plt.title('Valores Reales vs Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')

plt.subplot(1, 3, 3)
from scipy import stats
stats.probplot(y - y_pred_linear, dist="norm", plot=plt)
plt.title('Q-Q Plot de Residuos')

plt.tight_layout()
plt.savefig('/workspace/linear_regression_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

# REGRESIÓN LOGÍSTICA
print("\n📊 REGRESIÓN LOGÍSTICA")
print("-" * 40)

# Variable dependiente binaria
y_binary = df['Success_Binary']

# Regresión logística simple (con una variable)
print("REGRESIÓN LOGÍSTICA SIMPLE:")
X_simple = df[['Job_Success_Rate']]
X_simple_sm = sm.add_constant(X_simple)

logit_simple = sm.Logit(y_binary, X_simple_sm).fit()
print(logit_simple.summary())

# Regresión logística múltiple
print("\nREGRESIÓN LOGÍSTICA MÚLTIPLE:")
X_logit = df[feature_cols]
X_logit_sm = sm.add_constant(X_logit)

logit_multiple = sm.Logit(y_binary, X_logit_sm).fit()
print(logit_multiple.summary())

# MODELO LINEAL GENERALIZADO (GLM)
print("\n🔧 MODELO LINEAL GENERALIZADO (GLM)")
print("-" * 40)

# GLM con familia binomial para Success_Binary
glm_model = sm.GLM(y_binary, X_logit_sm, family=Binomial()).fit()
print("RESUMEN DEL MODELO GLM (Familia Binomial):")
print(glm_model.summary())

# ============================================================================
# PASO 5️⃣: MODELOS AUTOMÁTICOS
# ============================================================================
print("\n" + "="*50)
print("PASO 5️⃣: MODELOS AUTOMÁTICOS")
print("="*50)

# Preparar datos para modelos de machine learning
X_ml = df[feature_cols]
y_ml = df['Success_Binary']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
)

print(f"Tamaño conjunto entrenamiento: {X_train.shape}")
print(f"Tamaño conjunto prueba: {X_test.shape}")

# RANDOM FOREST
print("\n🌲 RANDOM FOREST CLASSIFIER")
print("-" * 40)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Precisión Random Forest: {accuracy_rf:.4f}")
print("\nReporte de clasificación Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Importancia de características
feature_importance_rf = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportancia de características (Random Forest):")
print(feature_importance_rf)

# GRADIENT BOOSTING
print("\n🚀 GRADIENT BOOSTING CLASSIFIER")
print("-" * 40)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

print(f"Precisión Gradient Boosting: {accuracy_gb:.4f}")
print("\nReporte de clasificación Gradient Boosting:")
print(classification_report(y_test, y_pred_gb))

# Importancia de características
feature_importance_gb = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportancia de características (Gradient Boosting):")
print(feature_importance_gb)

# Visualización comparativa de importancia
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.barh(feature_importance_rf['feature'], feature_importance_rf['importance'])
plt.title('Importancia de Características - Random Forest')
plt.xlabel('Importancia')

plt.subplot(1, 2, 2)
plt.barh(feature_importance_gb['feature'], feature_importance_gb['importance'])
plt.title('Importancia de Características - Gradient Boosting')
plt.xlabel('Importancia')

plt.tight_layout()
plt.savefig('/workspace/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# RESUMEN DE HALLAZGOS
# ============================================================================
print("\n" + "="*80)
print("🎯 RESUMEN DE HALLAZGOS PRINCIPALES")
print("="*80)

print(f"""
📊 ESTADÍSTICAS GENERALES:
- Dataset con {df.shape[0]} observaciones y {df.shape[1]} variables
- {len(numeric_cols)} variables numéricas y {len(categorical_cols)} variables categóricas
- Variable Success_Binary creada: {df['Success_Binary'].sum()} freelancers exitosos ({df['Success_Binary'].mean()*100:.1f}%)

🔍 ANÁLISIS EXPLORATORIO:
- Correlación más fuerte: {correlation_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates().iloc[1]:.3f}
- Variables con mayor variabilidad: {df[numeric_cols].std().sort_values(ascending=False).head(3).index.tolist()}

📈 ANÁLISIS MULTIVARIADO:
- PCA: {n_components_95} componentes explican 95% de la varianza
- Análisis Factorial: {n_factors} factores principales identificados
- Clustering K-Means: {optimal_k} clusters óptimos identificados

📊 MODELOS ESTADÍSTICOS:
- Regresión Lineal R²: {r2_linear:.4f}
- Variables más significativas en regresión lineal: {linear_model.pvalues.sort_values().head(3).index.tolist()}

🤖 MODELOS AUTOMÁTICOS:
- Random Forest Accuracy: {accuracy_rf:.4f}
- Gradient Boosting Accuracy: {accuracy_gb:.4f}
- Mejor modelo: {'Random Forest' if accuracy_rf > accuracy_gb else 'Gradient Boosting'}

🎯 VARIABLES MÁS IMPORTANTES:
- Random Forest: {feature_importance_rf.iloc[0]['feature']} ({feature_importance_rf.iloc[0]['importance']:.3f})
- Gradient Boosting: {feature_importance_gb.iloc[0]['feature']} ({feature_importance_gb.iloc[0]['importance']:.3f})
""")

print("\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)