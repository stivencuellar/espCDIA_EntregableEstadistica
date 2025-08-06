#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Completar Análisis - Agregar variable logarítmica y modelos de regresión
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPLETANDO ANÁLISIS - MODELOS DE REGRESIÓN")
print("=" * 80)

# Cargar el dataset
print("📊 Cargando el dataset...")
df = pd.read_csv('freelancer_earnings_bd.csv', sep=';')

# Crear variable logarítmica de Earnings_USD
print("📊 Creando variable logarítmica Earnings_USD_L...")
df['Earnings_USD_L'] = np.log1p(df['Earnings_USD'])

# Separar variables numéricas
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(f"📊 Variables numéricas disponibles: {list(numeric_columns)}")

# ============================================================================
# REGRESIÓN LINEAL MÚLTIPLE
# ============================================================================

print("\n" + "="*60)
print("REGRESIÓN LINEAL MÚLTIPLE - Earnings_USD_L")
print("="*60)

# Preparar variables para la regresión
X_vars = [col for col in numeric_columns if col not in ['Earnings_USD_L', 'Freelancer_ID']]
X = df[X_vars]
y = df['Earnings_USD_L']

print(f"📊 Variables predictoras: {list(X.columns)}")
print(f"📊 Variable objetivo: Earnings_USD_L")

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

# ============================================================================
# MODELOS AUTOMÁTICOS DE REGRESIÓN
# ============================================================================

print("\n" + "="*60)
print("MODELOS AUTOMÁTICOS DE REGRESIÓN")
print("="*60)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"📊 Tamaño de entrenamiento: {X_train.shape[0]} muestras")
print(f"📊 Tamaño de prueba: {X_test.shape[0]} muestras")

# Random Forest Regressor
print(f"\n🌲 Entrenando Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"   • R² Random Forest: {rf_r2:.4f}")
print(f"   • RMSE Random Forest: {rf_rmse:.4f}")

# Gradient Boosting Regressor
print(f"\n🚀 Entrenando Gradient Boosting Regressor...")
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train, y_train)
gb_pred = gb_reg.predict(X_test)
gb_r2 = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

print(f"   • R² Gradient Boosting: {gb_r2:.4f}")
print(f"   • RMSE Gradient Boosting: {gb_rmse:.4f}")

# Comparación de modelos
print(f"\n📊 Comparación de modelos de regresión:")
print(f"   • Regresión Lineal R²: {model.rsquared:.4f}")
print(f"   • Random Forest R²: {rf_r2:.4f}")
print(f"   • Gradient Boosting R²: {gb_r2:.4f}")

# ============================================================================
# IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================

print(f"\n📊 Importancia de características (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': X_vars,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualización de importancia de características
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Importancia de Características - Random Forest (Earnings_USD_L)')
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig('feature_importance_earnings.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# ANÁLISIS DE RESIDUOS
# ============================================================================

print(f"\n📊 Análisis de residuos del modelo lineal...")

# Calcular residuos
residuals = model.resid
fitted_values = model.fittedvalues

# Gráfico de residuos vs valores ajustados
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(fitted_values, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Valores Ajustados')
plt.ylabel('Residuos')
plt.title('Residuos vs Valores Ajustados')

# Histograma de residuos
plt.subplot(2, 2, 2)
plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')

# Q-Q plot
from scipy import stats
plt.subplot(2, 2, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot de Residuos')

# Gráfico de escala-locación
plt.subplot(2, 2, 4)
plt.scatter(fitted_values, np.sqrt(np.abs(residuals)), alpha=0.6)
plt.xlabel('Valores Ajustados')
plt.ylabel('√|Residuos|')
plt.title('Gráfico de Escala-Locación')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN FINAL DEL ANÁLISIS COMPLETO")
print("="*80)

print(f"\n📊 DATASET:")
print(f"   • Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"   • Variables numéricas: {len(numeric_columns)}")
print(f"   • Variable logarítmica creada: Earnings_USD_L")

print(f"\n📊 MODELOS DE REGRESIÓN:")
print(f"   • Regresión Lineal R²: {model.rsquared:.4f}")
print(f"   • Random Forest R²: {rf_r2:.4f}")
print(f"   • Gradient Boosting R²: {gb_r2:.4f}")

print(f"\n📊 VARIABLES MÁS IMPORTANTES (Random Forest):")
for i, row in feature_importance.head(5).iterrows():
    print(f"   • {row['feature']}: {row['importance']:.4f}")

print(f"\n📊 VARIABLES SIGNIFICATIVAS (Regresión Lineal):")
for var, pval in significant_coeffs.items():
    if var != 'const':
        print(f"   • {var}: p = {pval:.4f}")

print(f"\n📊 ARCHIVOS GENERADOS:")
print(f"   • feature_importance_earnings.png")
print(f"   • residual_analysis.png")

print(f"\n✅ Análisis completo finalizado exitosamente!")
print("="*80)