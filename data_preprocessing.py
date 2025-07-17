# --------------------------
# CARGA Y LIMPIEZA DE DATOS
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carga del dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Limpieza básica
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Codificación de la variable objetivo
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# --------------------------
# EXPLORACIÓN INICIAL
# --------------------------
print(df.info())
print(df.describe())
print(df['Contract'].value_counts())

# Visualización de churn
sns.countplot(x='Churn', data=df)
plt.title('Distribución de Churn')
plt.show()

# Matriz de correlación
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlación entre variables numéricas')
plt.show()

# --------------------------
# FEATURE ENGINEERING
# --------------------------
# Codificación LabelEncoder para binarios
df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])  # Male=1, Female=0

# One-hot encoding
df = pd.get_dummies(df, columns=['Contract', 'InternetService'], drop_first=False)

# Escalado de variables numéricas
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# --------------------------
# SEGMENTACIÓN
# --------------------------
# Ejemplo de grupo de alto riesgo
high_risk = df[(df['Contract_Month-to-month'] == 1) & (df['MonthlyCharges'] > 0.5)]
print(f"Clientes de alto riesgo: {len(high_risk)}")
print(f"Tasa de churn en este grupo: {high_risk['Churn'].mean():.2%}")

# Boxplot 
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Cargos mensuales por churn')
plt.show()


# Exportar
df.to_csv('churn_limpio.csv', index=False)
