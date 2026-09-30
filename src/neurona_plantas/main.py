"""Neurona artificial para decidir si una planta necesita riego.

Entradas: humedad del suelo (%) y temperatura ambiental (°C).
Salida: 1 = regar, 0 = no regar. Activación: sigmoide.
"""

import numpy as np

# Datos de entrenamiento

X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# Normalización: cada columna se divide por su valor máximo esperado.
# La MISMA escala se usa para los datos nuevos.

escala = np.array([100, 50])
X_normalizado = X / escala

print("Datos normalizados:")
print(X_normalizado)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas, semilla=42, verbose=False):
    """Entrena la neurona (2 entradas, 2 pesos, 1 sesgo)."""
    rng = np.random.default_rng(semilla)
    pesos = rng.normal(0, 0.5, size=(2, 1))  # [peso_humedad], [peso_temperatura]
    sesgo = 0.0
    n = len(X_normalizado)
    historial = []

    for epoca in range(1, epocas + 1):
        
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)

        error = probabilidad - y
        ecm = np.mean(error ** 2)
        historial.append(ecm)

        gradiente_z = error * probabilidad * (1 - probabilidad)
        gradiente_pesos = X_normalizado.T @ gradiente_z / n
        gradiente_sesgo = np.mean(gradiente_z)

        pesos = pesos - tasa_aprendizaje * gradiente_pesos
        sesgo = sesgo - tasa_aprendizaje * gradiente_sesgo

        if verbose and (epoca == 1 or epoca % (epocas // 10) == 0):
            print(f"  Época {epoca:>6} | ECM = {ecm:.6f}")

    probabilidad = sigmoide(X_normalizado @ pesos + sesgo)
    ecm_final = np.mean((probabilidad - y) ** 2)
    return pesos, sesgo, ecm_final, probabilidad, historial


def predecir(humedad, temperatura, pesos, sesgo, umbral=0.5):
    """Normaliza con la misma escala y devuelve (probabilidad, decisión)."""
    entrada = np.array([humedad, temperatura], dtype=float) / escala
    probabilidad = float(sigmoide(entrada @ pesos + sesgo)[0])
    return probabilidad, int(probabilidad >= umbral)

# 1) Entrenamiento base (con seguimiento periódico del ECM)

print("\n=== Entrenamiento base: 10000 épocas, tasa 0.5 ===")
pesos, sesgo, ecm_final, prob_train, _ = entrenar(0.5, 10000, verbose=True)

print(f"\nPeso de la humedad     : {pesos[0, 0]: .4f}")
print(f"Peso de la temperatura : {pesos[1, 0]: .4f}")
print(f"Sesgo                  : {sesgo: .4f}")
print(f"Error final (ECM)      : {ecm_final:.6f}")

print("\nProbabilidad por caso de entrenamiento:")
print("Caso | Humedad | Temp | Prob.  | Respuesta | Esperado")
for i, (fila, p) in enumerate(zip(X, prob_train), start=1):
    resp = int(p[0] >= 0.5)
    print(f"{i:>4} | {fila[0]:>6.0f}% | {fila[1]:>3.0f}C | {p[0]:.4f} | {resp:>9} | {int(y[i-1, 0])}")

# 2) Pruebas con condiciones nuevas

nuevas = [(75, 30), (45, 34), (25, 22), (50, 25), (30, 40)]
print("\n=== Pruebas con condiciones nuevas (umbral 0.5) ===")
print("Humedad | Temp | Probabilidad | Decisión")
for h, t in nuevas:
    p, d = predecir(h, t, pesos, sesgo)
    print(f"{h:>6}% | {t:>3}C | {p:>12.4f} | {d} ({'regar' if d else 'no regar'})")

# 3) Experimentos con los parámetros

experimentos = [
    ("Prueba base",          10000, 0.5),
    ("Pocas épocas",           100, 0.5),
    ("Cantidad intermedia",   1000, 0.5),
    ("Más épocas",           20000, 0.5),
    ("Tasa pequeña",         10000, 0.01),
    ("Tasa moderada",        10000, 0.1),
    ("Tasa alta",            10000, 1.0),
    ("Tasa muy alta",        10000, 2.0),
]

print("\n=== Experimentos ===")
print("Experimento          | Épocas | Tasa | ECM final | Aciertos | P(45%,34C) | ECM@100 | ECM@1000 | w_hum   | w_temp  | sesgo")
for nombre, ep, lr in experimentos:
    w, b, ecm, prob, hist = entrenar(lr, ep)
    aciertos = int(np.sum((prob >= 0.5).astype(int) == y))
    p45, _ = predecir(45, 34, w, b)
    e100 = hist[99] if len(hist) >= 100 else float("nan")
    e1000 = hist[999] if len(hist) >= 1000 else float("nan")
    oscila = np.max(np.abs(np.diff(hist[-100:]))) if len(hist) >= 100 else 0.0
    print(f"{nombre:<20} | {ep:>6} | {lr:<4} | {ecm:.6f}  | {aciertos:>2}/10    | {p45:.4f}     | "
          f"{e100:.5f} | {e1000:.5f}  | {w[0,0]: .3f} | {w[1,0]: .3f} | {b: .3f} | osc={oscila:.2e}")

# 4) Prueba adicional con el umbral (sin volver a entrenar)

print("\n=== Umbrales 0.4 / 0.5 / 0.6 (mismos pesos de la prueba base) ===")
print("--- Datos de entrenamiento ---")
print("Caso | Prob.  | U=0.4 | U=0.5 | U=0.6")
for i, p in enumerate(prob_train[:, 0], start=1):
    print(f"{i:>4} | {p:.4f} | {int(p >= 0.4):>5} | {int(p >= 0.5):>5} | {int(p >= 0.6):>5}")

print("--- Condiciones nuevas ---")
print("Humedad | Temp | Prob.  | U=0.4 | U=0.5 | U=0.6")
for h, t in nuevas:
    p, _ = predecir(h, t, pesos, sesgo)
    print(f"{h:>6}% | {t:>3}C | {p:.4f} | {int(p >= 0.4):>5} | {int(p >= 0.5):>5} | {int(p >= 0.6):>5}")
