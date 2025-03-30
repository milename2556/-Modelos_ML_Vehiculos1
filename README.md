# -Modelos_ML_Vehiculos1

# Graficamos la matriz de correlación
plt.figure(figsize=(10,6))
sns.heatmap(df_vehicles_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()
print(df_vehicles.columns)  # Esto te mostrará todas las columnas del DataFrame

df_vehicles.info()
df_vehicles.describe()
df_vehicles.isnull().sum()  # Revisar valores faltantes
# Convertimos las variables categóricas en variables numéricas con one-hot encoding
df_vehicles_encoded = pd.get_dummies(df_vehicles, drop_first=True)

# Ahora sí calculamos la matriz de correlación
plt.figure(figsize=(10,6))
sns.heatmap(df_vehicles_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()

# Ver las primeras filas
print(df_vehicles.head())

# Estadísticas descriptivas
print(df_vehicles.describe())

# Identificar valores atípicos con boxplots
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_vehicles[["selling_price", "km_driven", "year"]])
plt.title("Boxplot de las Variables Numéricas")
plt.show()

# Matriz de correlación
plt.figure(figsize=(8,6))
sns.heatmap(df_vehicles.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()

# Distribución del precio de los vehículos
plt.figure(figsize=(8,5))
sns.histplot(df_vehicles["selling_price"], bins=30, kde=True)
plt.title("Distribución de Precios de Vehículos")
plt.xlabel("Precio de Venta")
plt.ylabel("Frecuencia")
plt.show()

# Relación entre precio y año de fabricación
plt.figure(figsize=(8,5))
sns.scatterplot(x=df_vehicles["year"], y=df_vehicles["selling_price"], alpha=0.5)
plt.title("Precio vs Año de Fabricación")
plt.xlabel("Año")
plt.ylabel("Precio de Venta")
plt.show()
 Identificar y eliminar valores atípicos en 'selling_price' y 'km_driven'
plt.figure(figsize=(10,5))
sns.boxplot(data=df_vehicles[['selling_price', 'km_driven']])
plt.title("Valores Atípicos en selling_price y km_driven")
plt.show()

# Cálculo del Rango Intercuartil (IQR) para eliminar outliers
Q1 = df_vehicles[['selling_price', 'km_driven']].quantile(0.25)
Q3 = df_vehicles[['selling_price', 'km_driven']].quantile(0.75)
IQR = Q3 - Q1

# Filtramos los valores atípicos
df_vehicles = df_vehicles[~((df_vehicles[['selling_price', 'km_driven']] < (Q1 - 1.5 * IQR)) |
                            (df_vehicles[['selling_price', 'km_driven']] > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Tamaño del dataset después de eliminar outliers:", df_vehicles.shape)

# Convertir variables categóricas en variables numéricas con One-Hot Encoding
df_vehicles = pd.get_dummies(df_vehicles, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

# Eliminar la columna 'name' porque no aporta valor a la predicción del precio
df_vehicles.drop(columns=['name'], inplace=True, errors='ignore')

# Normalizar las variables numéricas (year y km_driven)
scaler = MinMaxScaler()
df_vehicles[['year', 'km_driven']] = scaler.fit_transform(df_vehicles[['year', 'km_driven']])

#  Verificar que los datos han sido preprocesados correctamente
print(df_vehicles.info())
print(df_vehicles.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Matriz de correlación
plt.figure(figsize=(12,6))
sns.heatmap(df_vehicles.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación entre Variables")
plt.show()
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Creamos el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Importancia de las características
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)

# Gráfico de importancia
plt.figure(figsize=(10,5))
feature_importances.plot(kind="bar", color="skyblue")
plt.title("Importancia de Características en la Predicción del Precio")
plt.show()

from sklearn.feature_selection import SelectKBest, f_regression

X = df_vehicles.drop(columns=['selling_price'])  # Variables independientes
y = df_vehicles['selling_price']  # Variable objetivo

# Aplicamos SelectKBest con la métrica F-regression
selector = SelectKBest(score_func=f_regression, k=5)  # Seleccionamos las 5 mejores
X_new = selector.fit_transform(X, y)

# Mostramos las características seleccionadas
selected_features = X.columns[selector.get_support()]
print("Características más relevantes:", selected_features.tolist())

# Punto 4: Dividir el dataset en Train y Test para evaluar correctamente el 
modelo
from sklearn.model_selection import train_test_split

# Definir las características (X) y la variable objetivo (y)
X = df_vehicles[['year', 'km_driven', 'fuel_Diesel', 'fuel_Petrol', 'seller_type_Individual']]
y = df_vehicles['selling_price']

# Dividir el dataset en 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar las dimensiones de los datos
print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)

# Punto 5: Entrenar el modelo configurando los diferentes hiperparámetros.
# 1️⃣ Importar bibliotecas
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2️⃣ Definir los modelos con hiperparámetros iniciales
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)  # Alpha controla la regularización
tree_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10)  # Evita sobreajuste

# 3️⃣ Entrenar los modelos
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# 4️⃣ Hacer predicciones en el conjunto de prueba
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

# 5️⃣ Evaluar los modelos
def evaluar_modelo(y_test, y_pred, nombre):
    print(f"📊 Evaluación del modelo: {nombre}")
    print(f"🔹 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    from sklearn.metrics import root_mean_squared_error 
    root_mean_squared_error(y_test, y_pred)

    print(f"🔹 R² Score: {r2_score(y_test, y_pred):.2f}\n")

evaluar_modelo(y_test, y_pred_linear, "Regresión Lineal")
evaluar_modelo(y_test, y_pred_ridge, "Regresión Ridge")
evaluar_modelo(y_test, y_pred_tree, "Árbol de Decisión")

# Punto 6: Evaluar el desempeño del modelo en el conjunto de Test con 
métricas como precisión, recall, F1-score, etc. 
from sklearn.tree import DecisionTreeRegressor

# 🔹 Definir y entrenar el modelo
modelo_arbol = DecisionTreeRegressor(random_state=42)
modelo_arbol.fit(X_train, y_train)  # Entrenar con los datos de entrenamiento

# 🔹 Hacer predicciones
y_pred = modelo_arbol.predict(X_test)

# 🔹 Evaluar el modelo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📊 Evaluación del Modelo en el conjunto de Test:")
print(f"🔹 MAE: {mae:.2f}")
print(f"🔹 MSE: {mse:.2f}")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R² Score: {r2:.2f}")

# Punto 6:Evaluar el desempeño del modelo en el conjunto de Test con 
métricas como precisión, recall, F1-score, etc. 
# Comparación de valores reales vs. predichos
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Valores Reales vs. Predichos")
plt.show()

# Distribución del error (residuos)
import seaborn as sns

errores = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(errores, bins=30, kde=True, color="purple")
plt.xlabel("Error de predicción")
plt.title("Distribución de errores")
plt.show()
#Importancia de características (para modelos basados en árboles)
