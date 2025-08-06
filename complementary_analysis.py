#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Complementario - Regresión Lineal y Análisis Adicionales
================================================================

Este script complementa el análisis principal con regresión lineal
usando Earnings_USD como variable dependiente.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
print("ANÁLISIS COMPLEMENTARIO - REGRESIÓN LINEAL Y MODELOS ADICIONALES")
print("=" * 80)

# Cargar el dataset
print("Cargando el dataset...")
df = pd.read_csv('freelancer_earnings_bd.csv', sep=';')

# ============================================================================
# REGRESIÓN LINEAL MÚLTIPLE CON EARNINGS_USD
# ============================================================================

print("\n" + "="*60)
print("REGRESIÓN LINEAL MÚLTIPLE - EARNINGS_USD")
print("="*60)

# Preparar variables para regresión
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Earnings_USD', 'Freelancer_ID']
feature_cols = [col for col in numeric_columns if col not in exclude_cols]

print(f"Variables predictoras: {feature_cols}")
print(f"Variable dependiente: Earnings_USD")

X = df[feature_cols].copy()
y = df['Earnings_USD']

# Manejar valores nulos si existen
X = X.fillna(X.median())
y = y.fillna(y.median())

# Agregar constante para statsmodels
X_sm = sm.add_constant(X)

# Regresión lineal múltiple con statsmodels
print("\n📊 EJECUTANDO REGRESIÓN LINEAL MÚLTIPLE...")
model = sm.OLS(y, X_sm).fit()

print(f"\n📈 RESULTADOS DE REGRESIÓN LINEAL MÚLTIPLE:")
print(model.summary())

# Verificar multicolinealidad
print(f"\n🔍 ANÁLISIS DE MULTICOLINEALIDAD (VIF):")
vif_data = pd.DataFrame()
vif_data["Variable"] = X_sm.columns
vif_data["VIF"] = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
print(vif_data.sort_values('VIF', ascending=False))

# Predicciones vs valores reales
y_pred = model.predict(X_sm)

# Métricas de rendimiento
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\n📊 MÉTRICAS DE RENDIMIENTO:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Visualizar predicciones vs valores reales
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(y, y_pred, alpha=0.6, s=50)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valores Reales (Earnings_USD)')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.grid(True)

# Análisis de residuos
residuals = y - y_pred

plt.subplot(2, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6, s=50)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')
plt.grid(True)

# Histograma de residuos
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')
plt.grid(True)

# Q-Q plot de residuos
plt.subplot(2, 2, 4)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot de Residuos')
plt.grid(True)

plt.tight_layout()
plt.savefig('regression_analysis_earnings.png', dpi=300, bbox_inches='tight')
plt.show()

# Coeficientes más importantes
coef_df = pd.DataFrame({
    'Variable': model.params.index,
    'Coeficiente': model.params.values,
    'P-valor': model.pvalues.values,
    'Significancia': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in model.pvalues]
})
coef_df = coef_df.sort_values('P-valor')

print(f"\n📊 COEFICIENTES ORDENADOS POR SIGNIFICANCIA:")
print(coef_df)

# ============================================================================
# MODELOS DE MACHINE LEARNING PARA REGRESIÓN
# ============================================================================

print("\n" + "="*60)
print("MODELOS DE MACHINE LEARNING PARA REGRESIÓN")
print("="*60)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Estandarizar variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos a evaluar
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🚀 ENTRENANDO {name}...")
    
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Métricas
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[name] = {
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae
    }
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")

# Comparar modelos
results_df = pd.DataFrame(results).T
print(f"\n📊 COMPARACIÓN DE MODELOS:")
print(results_df)

# Visualizar comparación de modelos
plt.figure(figsize=(15, 10))

# R² Score comparación
plt.subplot(2, 2, 1)
models_names = list(results.keys())
train_r2_scores = [results[model]['Train_R2'] for model in models_names]
test_r2_scores = [results[model]['Test_R2'] for model in models_names]

x = np.arange(len(models_names))
width = 0.35

plt.bar(x - width/2, train_r2_scores, width, label='Train R²', alpha=0.8)
plt.bar(x + width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
plt.xlabel('Modelos')
plt.ylabel('R² Score')
plt.title('Comparación de R² Score')
plt.xticks(x, models_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# RMSE comparación
plt.subplot(2, 2, 2)
rmse_scores = [results[model]['Test_RMSE'] for model in models_names]
plt.bar(models_names, rmse_scores, alpha=0.8)
plt.xlabel('Modelos')
plt.ylabel('RMSE')
plt.title('Comparación de RMSE')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# MAE comparación
plt.subplot(2, 2, 3)
mae_scores = [results[model]['Test_MAE'] for model in models_names]
plt.bar(models_names, mae_scores, alpha=0.8)
plt.xlabel('Modelos')
plt.ylabel('MAE')
plt.title('Comparación de MAE')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Predicciones del mejor modelo
best_model_name = max(results.keys(), key=lambda x: results[x]['Test_R2'])
if best_model_name == 'Linear Regression':
    best_model = LinearRegression()
    best_model.fit(X_train_scaled, y_train)
    best_pred = best_model.predict(X_test_scaled)
else:
    best_model = models[best_model_name]
    best_pred = best_model.predict(X_test)

plt.subplot(2, 2, 4)
plt.scatter(y_test, best_pred, alpha=0.6, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title(f'Mejor Modelo: {best_model_name}')
plt.grid(True)

plt.tight_layout()
plt.savefig('ml_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Importancia de características (para modelos de árboles)
if 'Random Forest' in results:
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importancia')
    plt.title('Importancia de Características - Random Forest')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# ANÁLISIS DE VARIABLES CATEGÓRICAS
# ============================================================================

print("\n" + "="*60)
print("ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("="*60)

# Análisis de Earnings por categorías
categorical_cols = ['Job_Category', 'Platform', 'Experience_Level', 'Client_Region', 'Payment_Method', 'Project_Type']

plt.figure(figsize=(20, 15))

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    
    # Calcular estadísticas por categoría
    earnings_by_cat = df.groupby(col)['Earnings_USD'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    
    # Mostrar solo las top 10 categorías para evitar gráficos muy congestionados
    top_categories = earnings_by_cat.head(10)
    
    plt.barh(range(len(top_categories)), top_categories['mean'], alpha=0.8)
    plt.yticks(range(len(top_categories)), top_categories.index)
    plt.xlabel('Earnings Promedio (USD)')
    plt.title(f'Earnings Promedio por {col}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('earnings_by_categories.png', dpi=300, bbox_inches='tight')
plt.show()

# Análisis de correlaciones con variables categóricas
print(f"\n📊 EARNINGS PROMEDIO POR CATEGORÍAS:")
for col in categorical_cols:
    print(f"\n{col}:")
    earnings_stats = df.groupby(col)['Earnings_USD'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
    print(earnings_stats.head())

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN FINAL DEL ANÁLISIS COMPLEMENTARIO")
print("="*80)

print(f"\n📊 HALLAZGOS PRINCIPALES:")
print(f"• Variable dependiente: Earnings_USD")
print(f"• Variables predictoras: {len(feature_cols)}")
print(f"• R² del modelo lineal: {r2:.4f}")
print(f"• RMSE del modelo lineal: {rmse:.2f}")

print(f"\n📊 MEJOR MODELO DE MACHINE LEARNING:")
print(f"• Modelo: {best_model_name}")
print(f"• R² Score: {results[best_model_name]['Test_R2']:.4f}")
print(f"• RMSE: {results[best_model_name]['Test_RMSE']:.2f}")

print(f"\n📊 VARIABLES MÁS IMPORTANTES (Regresión Lineal):")
top_coefs = coef_df[coef_df['Variable'] != 'const'].head(5)
for _, row in top_coefs.iterrows():
    print(f"• {row['Variable']}: {row['Coeficiente']:.4f} {row['Significancia']}")

print(f"\n📊 RECOMENDACIONES:")
print("• El modelo lineal explica aproximadamente {:.1%} de la varianza en Earnings_USD".format(r2))
print("• Considerar transformaciones logarítmicas para mejorar la linealidad")
print("• Evaluar la inclusión de interacciones entre variables")
print("• Validar el modelo con datos externos")

print("\n✅ ANÁLISIS COMPLEMENTARIO COMPLETADO!")
print("="*80)