import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

# Configure plots
sns.set(style='whitegrid', palette='deep', font_scale=1.1, rc={'figure.figsize': (10, 6)})

# ------------------------------
# Paso 1️⃣: Carga del dataset y EDA
# ------------------------------
DATA_PATH = Path('/workspace/freelancer_earnings_bd.csv')

df = pd.read_csv(DATA_PATH, sep=';', decimal='.')

# Crear variable logarítmica de las ganancias
if 'Earnings_USD_L' not in df.columns:
    df['Earnings_USD_L'] = np.log1p(df['Earnings_USD'])

# Crear variable binaria de éxito (>80%)
if 'Success_Binary' not in df.columns and 'Job_Success_Rate' in df.columns:
    df['Success_Binary'] = (df['Job_Success_Rate'] >= 80).astype(int)

print("\n===== Primeras filas =====\n", df.head())
print("\n===== Información =====")
print(df.info())
print("\n===== Estadísticas descriptivas =====\n", df.describe())
print("\n===== Valores nulos por columna =====\n", df.isnull().sum())

# Visualización de la matriz de correlación para variables numéricas
num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de correlación de variables numéricas')
plt.tight_layout()
plt.savefig('/workspace/corr_matrix.png')
plt.close()

# Pairplot de variables numéricas principales (muestra)
sns.pairplot(df[num_cols].sample(n=min(300, len(df))), diag_kind='kde')
plt.suptitle('Relaciones entre variables numéricas (muestra)')
plt.savefig('/workspace/pairplot.png')
plt.close()

# ------------------------------
# Paso 2️⃣: Análisis Univariado
# ------------------------------
for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.savefig(f'/workspace/hist_{col}.png')
    plt.close()

# ------------------------------
# Paso 3️⃣: Análisis Multivariado
# ------------------------------
# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols])

# PCA
pca = PCA(n_components=min(len(num_cols), 10))
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
print("\n===== Varianza explicada por PCA =====")
for i, var in enumerate(explained_var, 1):
    print(f'PC{i}: {var:.4f}')

# Graficar varianza explicada acumulativa
plt.figure()
plt.plot(np.cumsum(explained_var), marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.title('PCA - Varianza explicada acumulada')
plt.grid()
plt.savefig('/workspace/pca_variance.png')
plt.close()

# Scatter de los dos primeros componentes
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot PCA (PC1 vs PC2)')
plt.savefig('/workspace/pca_scatter.png')
plt.close()

# Factor Analysis con rotación varimax
fa = FactorAnalyzer(rotation='varimax', n_factors=3)
fa.fit(X_scaled)
loadings = fa.loadings_
fa_df = pd.DataFrame(loadings, index=num_cols, columns=[f'Factor_{i+1}' for i in range(loadings.shape[1])])
print("\n===== Cargas factoriales (rotación varimax) =====\n", fa_df)

# Clustering KMeans (usando PCA para reducir dimensionalidad)
km = KMeans(n_clusters=3, random_state=42)
clusters = km.fit_predict(X_pca[:, :2])  # Usamos las dos primeras componentes

plt.figure()
palette = sns.color_palette('bright', 3)
for k in range(3):
    plt.scatter(X_pca[clusters == k, 0], X_pca[clusters == k, 1], label=f'Cluster {k}', alpha=0.6, color=palette[k])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('KMeans clustering en espacio PCA')
plt.savefig('/workspace/kmeans_pca.png')
plt.close()

# ------------------------------
# Paso 4️⃣: Modelos Estadísticos
# ------------------------------
# Regresión lineal múltiple sobre Earnings_USD_L
features = num_cols.drop('Earnings_USD_L')
X = df[features]
X = sm.add_constant(X)
y = df['Earnings_USD_L']
model_ols = sm.OLS(y, X).fit()
print("\n===== Resumen Regresión Lineal Múltiple =====")
print(model_ols.summary())

# Variables binarias disponibles
binary_cols = ['Success_Binary'] if 'Success_Binary' in df.columns else []

# Regresión logística simple
if binary_cols:
    target = binary_cols[0]
    # Asegurar que la variable objetivo sea numérica 0/1
    if df[target].dtype == 'object' or df[target].dtype.name == 'category':
        df[target] = df[target].astype('category').cat.codes
    
    # Regresión logística simple
    logit_simple = sm.Logit(df[target], sm.add_constant(df['Earnings_USD_L'])).fit(disp=False)
    print("\n===== Regresión Logística Simple =====")
    print(logit_simple.summary())

    # Regresión logística múltiple usando num_cols
    try:
        logit_mult = sm.Logit(df[target], sm.add_constant(df[num_cols])).fit(disp=False)
        print("\n===== Regresión Logística Múltiple =====")
        print(logit_mult.summary())
    except np.linalg.LinAlgError:
        print("\nRegresión Logística Múltiple: problema de singularidad, se omite el resumen.")

    # GLM logístico (alternativa)
    try:
        glm_logit = sm.GLM(df[target], sm.add_constant(df[num_cols]), family=sm.families.Binomial()).fit()
        print("\n===== GLM Binomial =====")
        print(glm_logit.summary())
    except np.linalg.LinAlgError:
        print("\nGLM Binomial: problema de singularidad, se omite el resumen.")

# GLM Poisson para Job_Completed (conteo)
if 'Job_Completed' in df.columns:
    glm_poisson = sm.GLM(df['Job_Completed'], sm.add_constant(df[num_cols]), family=sm.families.Poisson()).fit()
    print("\n===== GLM Poisson - Job_Completed =====")
    print(glm_poisson.summary())

# ------------------------------
# Paso 5️⃣: Modelos Automáticos (Random Forest & GB) si hay variable binaria
# ------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    if binary_cols:
        target = binary_cols[0]
        X_train, X_test, y_train, y_test = train_test_split(df[num_cols], df[target], test_size=0.3, random_state=42, stratify=df[target])

        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)

        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        acc_gb = accuracy_score(y_test, y_pred_gb)

        print(f"\n===== Modelos Automáticos - Accuracy =====\nRandom Forest: {acc_rf:.3f}\nGradient Boosting: {acc_gb:.3f}")

except ImportError:
    print("sklearn.ensemble not available.")

# ------------------------------
# Resumen final de hallazgos
# ------------------------------
print("\n===== Resumen de Hallazgos =====")
print("1. La varianza explicada por las 2 primeras componentes del PCA es: {:.2f}%".format(np.cumsum(explained_var)[1]*100))
print("2. Regresión lineal múltiple R^2: {:.3f}".format(model_ols.rsquared))
if binary_cols:
    print("3. Accuracy Random Forest: {:.3f}, Gradient Boosting: {:.3f}".format(acc_rf, acc_gb))
    print("4. La variable de éxito tiene una media de {:.2f}, indicando un {:.0f}% de freelancers exitosos.".format(df[target].mean(), df[target].mean()*100))
print("Los gráficos se han guardado en la carpeta /workspace para su revisión.")

print("\nAnálisis completado. Gráficos guardados en carpeta /workspace")