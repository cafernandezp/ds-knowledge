# ADR — Estrategia de validación cruzada (regresión de avance judicial)

> **Estado:** Propuesto
> **Fecha:** 2026-06-04
> **Ámbito:** Esquema de CV para diagnóstico del modelo base y evaluación del flujo de feature selection
> **Target:** `score` ∈ [0,1] — proporción de avance de procedimientos judiciales
> **Datos:** ~6.000 casos, cada fila = un procedimiento judicial **único** (sin filas repetidas)

---

## 1. Contexto

Se necesita un esquema de validación cruzada que sea **estable con muestra pequeña** y coherente con cómo se construyó el split, para:

- Diagnosticar el **overfitting del modelo base** (gap train–val).
- Comparar configuraciones a lo largo del flujo de selección.
- Producir, al final, un número de generalización **fiable**.

Condicionantes del problema:

- **Muestra reducida** (~6.000) → cada fold deja relativamente pocos casos; preocupa la varianza de la estimación.
- **El target distribuye distinto por agencia y por tipo de procedimiento** → folds no balanceados darían estimaciones sesgadas/inestables.
- El split train/test original se **estratificó por `agencia × tipo_procedimiento`**.
- **Sin componente temporal** → no aplica `TimeSeriesSplit`.
- Cada fila es un **procedimiento único** → no hay agrupamiento que fuerce `GroupKFold`.
- Decisión operativa tomada: **no reajustar los selectores dentro de cada fold** (coste de Boruta-SHAP + stability + backward × folds es inviable).

---

## 2. Alternativas consideradas

### 2.1 Número de folds (`k`)

| `k` | Train/fold | Val/fold | Efecto |
|---|---|---|---|
| 3 | ~67% | ~2000 | Entrena con menos datos de los que tendrá el modelo final → **pesimista** |
| **4** | ~75% | ~1500 | **Compromiso elegido**; respeta preferencia de pocos folds |
| 5 | ~80% | ~1200 | Menos sesgo aún; igualmente defendible |
| 10 | ~90% | ~600 | Val pequeño + folds muy solapados → más varianza, caro |

- **Mito corregido:** bajar `k` **no** da más estabilidad. Menos folds = entrenamiento más pequeño = **más sesgo** respecto al modelo final (que usa todo el train). La varianza no se arregla bajando `k`.

### 2.2 Cómo controlar la varianza de la estimación

| Alternativa | Veredicto |
|---|---|
| Bajar `k` | Descartado: ataca el síntoma equivocado, introduce sesgo |
| **Subir `n_repeats`** (repetir el K-Fold completo con particiones aleatorias distintas) | **Elegido**: estabiliza la media y da un `std` para medir dispersión |

### 2.3 Esquema de partición

| Alternativa | Pros | Contras | Veredicto |
|---|---|---|---|
| `KFold` simple | Sencillo | No balancea la distribución de `y` por fold | Descartado |
| `StratifiedKFold` sobre `y` crudo | — | `y` es **continuo** → no se puede estratificar directo | Inviable |
| `StratifiedKFold` sobre `agencia × tipo` crudo | Coherente con el split | Combos raros (< k casos) → estratos singleton / folds degenerados | Riesgoso solo |
| **`RepeatedStratifiedKFold` sobre `qcut(y)`** (+ categórica de baja cardinalidad si la cardinalidad aguanta) | Garantiza que cada fold cubra todo el rango de avance; evita singletons | Requiere chequear cardinalidad si se combina con categórica | **Elegido** |

### 2.4 ¿Reajustar selectores dentro de cada fold?

| Alternativa | Pros | Contras | Veredicto |
|---|---|---|---|
| Nested CV (selección re-ejecutada por fold) | Estimación **insesgada** del pipeline completo | Coste prohibitivo (selección × 32 folds) | Descartado por coste |
| **Selección una vez sobre todo el train; CV evalúa solo el modelo sobre features fijas** | Barato; el gap train–val sigue siendo válido para overfitting | La CV-MAE queda **optimista** como medida de generalización del flujo | **Elegido**, con compensación |

- Compensación obligatoria: el **número final insesgado** sale del **test externo intocado**, no de esta CV.

### 2.5 Métrica (`scoring`)

- `neg_mean_absolute_error` — MAE es la métrica de negocio; negativo por la convención de sklearn ("mayor = mejor"). Se recupera el MAE real con `-res["test_score"]`.

---

## 3. Decisión (conclusión)

- **`RepeatedStratifiedKFold`** con **`n_splits=4`, `n_repeats=8`** (`random_state=42`).
- **Estratificar por `qcut(y, 5)`** (bins por cuantiles del target continuo), no por `y` crudo ni por `agencia × tipo` sin verificar cardinalidad. Combinar con `tipo_procedimiento` **solo si** el diagnóstico de cardinalidad lo permite.
- **Controlar la varianza con `n_repeats`, no bajando `k`.**
- **Selección de features una sola vez sobre todo el train; `pipe = model`** dentro de la CV (la CV evalúa solo el modelo sobre el set fijo).
- **La selección nunca toca el test externo** (innegociable).
- **Número final insesgado = test externo intocado**, evaluado una única vez al final. La CV es para **diagnóstico de overfitting + tuning + comparación relativa**, no para reportar generalización del pipeline.
- Diagnóstico rápido de estabilidad: **CV% = std/media** y **z-score por fold** para detectar folds atípicos.

---

## 4. Implementación de referencia

```python
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import mean_absolute_error

# --- 0) Selección UNA vez, solo sobre el train (nunca el test externo) ---
selected_features = run_selection_pipeline(X_train, y_train)   # etapas 1→5
X_train_sel = X_train[selected_features]

# --- 1) Clave de estratificación para TARGET CONTINUO: bins por cuantiles de y ---
y_bins = pd.qcut(y_train, q=5, labels=False, duplicates="drop").astype(str)
strat  = y_bins   # opcional: + "||" + df["tipo_procedimiento"] si la cardinalidad aguanta

# --- 2) Diagnóstico de cardinalidad (StratifiedKFold exige >= k por estrato) ---
k = 4
print(pd.Series(strat).value_counts().lt(k).sum(), "estratos con < k muestras")

# --- 3) CV: pocos folds, MUCHOS repeats; evalúa SOLO el modelo (pipe = model) ---
pipe = model   # XGBRegressor del ADR de parámetros; sin paso "select" dentro
cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=8, random_state=42)

res = cross_validate(
    pipe, X_train_sel, y_train,
    cv=list(cv.split(X_train_sel, strat)),
    scoring="neg_mean_absolute_error",
    return_train_score=True, n_jobs=-1,
)

# --- 4) Lectura: MAE, gap train-val, estabilidad ---
mae   = -res["test_score"]
m, s  = mae.mean(), mae.std()
gap   = m + res["train_score"].mean()      # >0 grande = overfitting del base
cv_pct = 100 * s / m                        # dispersión relativa
z = (mae - m) / s
outliers = np.where(np.abs(z) > 2)[0]       # folds atípicos
print(f"MAE val = {m:.4f} ± {s:.4f}  (CV {cv_pct:.1f}%)  "
      f"train {-res['train_score'].mean():.4f}  gap {gap:.4f}")
print(f"folds atípicos (|z|>2): {outliers}")

# --- 5) Número final insesgado: test externo, una sola vez ---
final_mae = mean_absolute_error(
    y_test,
    model.fit(X_train_sel, y_train).predict(X_test[selected_features]),
)
```

### Guía rápida de lectura del CV%

| CV% | Interpretación |
|---|---|
| < 10% | Estable; fiable la media |
| 10–20% | Aceptable con muestra pequeña / estratos heterogéneos |
| > 20% | Inestable → subir `n_repeats`, revisar estratificación o estratos raros |

---

## 5. Consecuencias

- **Positivas:** estimación estable con muestra pequeña (varianza controlada por repeats); folds que cubren todo el rango de avance (estratificación por cuantiles); coste contenido al no reajustar selección por fold; gap train–val válido para diagnosticar overfitting.
- **A vigilar:**
  - La CV-MAE es **optimista** como medida del pipeline completo (selección fuera del bucle) → **no es el número de reporte**; el test externo lo es.
  - El `std` entre repeats **subestima** la varianza real (training sets solapados) → termómetro relativo, no IC formal.
  - Si se combina `qcut(y)` con categóricas, vigilar **estratos con < k muestras**; colapsar niveles raros en `"Other"` si aparecen.
  - El test externo debe permanecer **cerrado** hasta la evaluación final única.
- **Pendiente:** confirmar si `tipo_procedimiento` entra en la clave de estratificación tras el diagnóstico de cardinalidad; revisar `k=4` vs `k=5` comparando la media de MAE entre ambos.
