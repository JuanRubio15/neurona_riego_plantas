# Neurona artificial para el riego de plantas

Neurona artificial (2 entradas, 2 pesos, 1 sesgo, activación sigmoide) que decide si una planta necesita riego.
Los datos son didácticos y no representan una recomendación agronómica.

## Problema

- **x1**: humedad del suelo (%), escala 0–100.
- **x2**: temperatura ambiental (°C), escala aproximada 0–50.
- **y**: 1 = regar, 0 = no regar.

Se entrena con 10 ejemplos. Las entradas se normalizan dividiendo por `escala = [100, 50]`; la **misma** escala se usa para los datos nuevos. La salida (probabilidad) se convierte en 0/1 con un umbral de 0.5.

## Ejecución

```bash
uv sync
uv run main.py
```

Requiere Python ≥ 3.12 y NumPy (instalado con `uv add numpy`). La carpeta `.venv` no se versiona.

## Resultado de la configuración base (10000 épocas, tasa 0.5)

| Parámetro | Valor |
|---|---|
| Peso humedad | -14.9556 |
| Peso temperatura | 2.6007 |
| Sesgo | 5.2207 |
| Error final (ECM) | 0.030312 |

Probabilidades por caso de entrenamiento:

| Caso | Humedad | Temp. | Probabilidad | Respuesta | Esperado |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0.0030 | 0 | 0 |
| 2 | 70 % | 22 °C | 0.0162 | 0 | 0 |
| 3 | 65 % | 28 °C | 0.0455 | 0 | 0 |
| 4 | 55 % | 25 °C | 0.1539 | 0 | 0 |
| 5 | 50 % | 32 °C | 0.3560 | 0 | 0 |
| 6 | 40 % | 30 °C | 0.6897 | 1 | 1 |
| 7 | 35 % | 25 °C | 0.7836 | 1 | 1 |
| 8 | 30 % | 32 °C | 0.9167 | 1 | 1 |
| 9 | 20 % | 35 °C | 0.9829 | 1 | 1 |
| 10 | 10 % | 38 °C | 0.9967 | 1 | 1 |

**Signo de los pesos.** El peso de la humedad es negativo: a más humedad, menor `z` y menor probabilidad de regar. El peso de la temperatura es positivo: a más calor, mayor probabilidad de regar. El sesgo positivo desplaza la frontera de decisión hacia "regar" cuando ambas entradas son moderadas.

## Predicciones con condiciones nuevas (modelo base, umbral 0.5)

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0117 | 0 (no regar) |
| 45 % | 34 °C | 0.5644 | 1 (regar) |
| 25 % | 22 °C | 0.9325 | 1 (regar) |
| 50 % | 25 °C | 0.2775 | 0 (no regar) |
| 30 % | 40 °C | 0.9435 | 1 (regar) |

## Experimentos

Semilla fija (42) para los pesos iniciales, de modo que solo cambia el parámetro indicado.

| Experimento | Épocas | Tasa | Error final (ECM) | Aciertos | P(45 %, 34 °C) | Aprendizaje |
|-------------|--------|------|-------------------|----------|----------------|-------------|
| Prueba base | 10000 | 0.5 | 0.030312 | 10/10 | 0.5644 | Adecuado, sigue mejorando despacio |
| Pocas épocas| 100 | 0.5 | 0.212861 | 9/10 | 0.4903 | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.089640 | 10/10 | 0.5521 | Lento (probabilidades aún cerca de 0.5) |
| Más épocas | 20000 | 0.5 | 0.019446 | 10/10 | 0.5441 | Mejora moderada |
| Tasa pequeña | 10000 | 0.01 | 0.175605 | 10/10 | 0.5053 | Muy lento |
| Tasa moderada | 10000 | 0.1 | 0.066631 | 10/10 | 0.5686 | Lento |
| Tasa alta | 10000 | 1.0 | 0.019445 | 10/10 | 0.5441 | Rápido |
| Tasa muy alta | 10000 | 2.0 | 0.011636 | 10/10 | 0.5231 | Rápido y estable en estos datos |

## Prueba con umbrales (sin reentrenar, modelo base)

| Humedad | Temp. | Probabilidad | U = 0.4 | U = 0.5 | U = 0.6 |
|---|---|---|---|---|---|
| 75 % | 30 °C | 0.0117 | 0 | 0 | 0 |
| 45 % | 34 °C | 0.5644 | 1 | 1 | **0** |
| 25 % | 22 °C | 0.9325 | 1 | 1 | 1 |
| 50 % | 25 °C | 0.2775 | 0 | 0 | 0 |
| 30 % | 40 °C | 0.9435 | 1 | 1 | 1 |

En los 10 casos de entrenamiento ninguna respuesta cambia (la probabilidad más alta de los "0" es 0.356 y la más baja de los "1" es 0.690, ambas fuera de la franja 0.4–0.6). El único caso que cambia es (45 %, 34 °C): pasa de 1 a 0 con umbral 0.6, porque su probabilidad (0.5644) queda por debajo. El umbral se aplica **después** de calcular la probabilidad: solo interpreta la salida, no interviene en el gradiente ni en la actualización, por eso los pesos no cambian.

## Análisis

1. **Normalización.** La humedad llega hasta 100 y la temperatura hasta 50. Sin normalizar, la humedad dominaría la suma ponderada, z tomaría valores enormes, la sigmoide se saturaría (gradientes casi cero) y el entrenamiento sería lento o inestable. Con ambas en una escala cercana a 0–1, los pesos son comparables y el aprendizaje es más estable.
2. **Uso de X_normalizado y de X.** X_normalizado se usa en la suma ponderada (z = X_normalizado @ pesos + sesgo) y en el gradiente de los pesos (X_normalizado.T @ gradiente_z). X se conserva para mostrar los valores originales (%, °C) y facilitar la interpretación de los resultados.
3. **Solo 100 épocas.** El ECM se queda en 0.213 y todas las probabilidades están entre 0.41 y 0.56, es decir, la neurona apenas distingue los casos. Acierta 9/10, pero por poco margen: el caso 6 tiene 0.499 y se clasifica como 0 (debía ser 1). Es un aprendizaje insuficiente.
4. **¿Más épocas siempre mejoran?** No de forma importante. De 100 a 1000 épocas el ECM baja de 0.213 a 0.090 y de 1000 a 10000 a 0.030. De 10000 a 20000 solo baja de 0.030 a 0.019, y los aciertos siguen en 10/10 desde las 1000 épocas. Hay rendimientos decrecientes; además, con datos separables, los pesos siguen creciendo en magnitud para empujar las probabilidades hacia 0 y 1.
5. **Tasa demasiado pequeña.** Con 0.01 los pesos se mueven muy poco por época: tras 10000 épocas el ECM es 0.176 y P(45 %, 34 °C) = 0.505, casi indecisa. Clasifica 10/10 pero con poca confianza: el aprendizaje es lento.
6. **Tasa alta o muy alta.** En estos datos con 1.0 y 2.0 el entrenamiento fue más rápido y no se volvió inestable (ECM 0.019 y 0.012, sin oscilaciones). Esto se debe a que el gradiente del ECM con sigmoide es pequeño (se promedia sobre los casos y se multiplica por p(1-p)), así que incluso 2.0 da pasos moderados. Notar que 20000 épocas a 0.5 y 10000 épocas a 1.0 dan prácticamente los mismos pesos: lo que importa es aproximadamente tasa × épocas. Con tasas aún mayores o con otra función de pérdida sí podrían aparecer oscilaciones o saltos; aquí no se observaron.
7. **Signo del peso de la humedad.** Es negativo (-14.96): al aumentar la humedad baja la probabilidad de regar. Coincide con la intuición (suelo húmedo → no regar).
8. **Signo del peso de la temperatura.** Es positivo (+2.60): al aumentar la temperatura sube la probabilidad de regar (más calor → más evaporación). Su magnitud es mucho menor que la de la humedad, así que en esta neurona la humedad pesa más en la decisión.
9. **Umbral.** La sigmoide entrega un valor continuo entre 0 y 1 (una probabilidad), pero la decisión final es binaria (regar / no regar). El umbral (0.5) convierte esa probabilidad en una acción concreta; cambiarlo permite ser más o menos estricto (por ejemplo, un umbral bajo riega con más facilidad).
10. **Limitaciones.** Solo 10 datos didácticos y 2 variables; no considera especie, tipo de suelo, luz, lluvia, hora del día ni etapa de crecimiento; es un modelo lineal (frontera recta) y no captura relaciones más complejas; no hay conjunto de validación, por lo que la exactitud en entrenamiento no garantiza buen desempeño real; los datos no provienen de mediciones agronómicas.
