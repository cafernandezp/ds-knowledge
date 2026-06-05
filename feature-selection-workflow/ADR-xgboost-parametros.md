# ADR — Configuración del modelo base XGBoost (regresión de avance judicial)

> **Estado:** Propuesto
> **Fecha:** 2026-06-04
> **Ámbito:** Modelo base para flujo de feature selection
> **Target:** `score` ∈ [0,1] — proporción de avance de procedimientos judiciales
> **Datos:** ~10.000 casos, 365 features iniciales (numéricas, binarias 0/1, categóricas baja/alta cardinalidad)

---

## 1. Contexto

Se necesita un **modelo base** de XGBoost que sirva de referencia estable a lo largo de un flujo de feature selection por etapas. Requisitos:

- El target es una **proporción acotada en [0,1]** (no un valor libre ni una clase).
- La **métrica de negocio principal es MAE**, en escala original (puntos de proporción).
- Se usará **validación cruzada** (sin `eval_set` ni early stopping en el base) → presupuesto de árboles fijo.
- El modelo debe ser **suficientemente regularizado** para que el chequeo de overfitting (gap train–val) sea interpretable y no domine el ruido en las etapas de selección.
- Hay **categóricas de alta cardinalidad** → conviene soporte nativo en lugar de one-hot masivo.

Decisiones a documentar: `objective`, `eval_metric`, backend de árboles, presupuesto (`n_estimators` / `learning_rate`) y regularización.

---

## 2. Alternativas consideradas

### 2.1 Objetivo / pérdida

| Alternativa | Pros | Contras | Veredicto |
|---|---|---|---|
| `reg:squarederror` | Estándar, estable, rápido | Predice **fuera de [0,1]** (p. ej. 1.07); asume homocedasticidad, falsa para proporciones (varianza se encoge cerca de 0 y 1) | Descartado (requiere parche de clipping) |
| `reg:logistic` | Salida **acotada a (0,1)** por sigmoide interna; pérdida logística penaliza más los bordes → mejor para proporciones; acepta `y` continuo en [0,1] | Optimiza la media logística, no MAE directamente (sesgo media-vs-mediana leve) | **Elegido** |
| `reg:absoluteerror` | Optimiza **MAE (L1)** directamente → alineado con la métrica | **Pierde la cota [0,1]**; hessiano constante → árboles menos estables / convergencia más lenta | Reservado como comparación empírica futura |

### 2.2 Métrica de evaluación (`eval_metric`)

- Sin `eval_set` + `early_stopping_rounds`, `eval_metric` es **cosmético**: solo se reporta, **no afecta el entrenamiento** (no toca gradientes ni corta iteraciones).
- El control real de MAE vive en `scoring="neg_mean_absolute_error"` dentro de `cross_validate`, **no** en el modelo.
- Se deja `eval_metric="mae"` por claridad de intención; cobrará efecto solo si en el futuro se añade early stopping por fold (loop manual).

### 2.3 Presupuesto de árboles

- Sin early stopping → **no** se deja `n_estimators` alto sin control. Estrategia: **muchos árboles + learning rate bajo** (`600 @ 0.03`) para suavizar y dejar que la regularización limite el sobreajuste.

### 2.4 Regularización

- Profundidad baja (`max_depth=4`) + soporte mínimo por hoja (`min_child_weight=5`) + muestreo (`subsample`, `colsample_bytree=0.8`) + L2 (`reg_lambda=1.0`).
- Punto de partida **conservador**: prioriza un base interpretable para el diagnóstico train–val antes que el rendimiento máximo.

---

## 3. Decisión (conclusión)

- **`objective = reg:logistic`** — la cota física [0,1] del target pesa más que la alineación exacta con MAE; el sesgo media-vs-mediana es leve y tolerable.
- **MAE se gestiona en la CV** (`scoring="neg_mean_absolute_error"`), no en el objetivo ni en `eval_metric`.
- **`tree_method="hist"` + `enable_categorical=True`** — rapidez y soporte nativo de categóricas de alta cardinalidad sin one-hot.
- **Presupuesto fijo** `n_estimators=600`, `learning_rate=0.03` (sin early stopping en el base).
- **Regularización conservadora** como referencia estable del flujo de selección.
- **Pendiente / acción futura:** comparar empíricamente contra `objective="reg:absoluteerror"` con el mismo `scoring`; si el MAE mejora de forma estable y el clipping de rango no molesta, reconsiderar el objetivo. El esquema de CV (estratificación / agrupamiento) se refina por separado.

---

## 4. Implementación de referencia

```python
import xgboost as xgb

# Modelo base XGBoost para regresión de proporción [0,1]
model = xgb.XGBRegressor(
    # --- Objetivo: salida acotada a [0,1] por la sigmoide interna ---
    objective="reg:logistic",
    eval_metric="mae",          # cosmético sin early stopping; lo dejo por claridad de intención

    # --- Backend: rápido + soporte nativo de categóricas ---
    tree_method="hist",
    enable_categorical=True,

    # --- Presupuesto fijo (sin early stopping) → muchos árboles + LR bajo ---
    n_estimators=600,
    learning_rate=0.03,

    # --- Regularización de complejidad (controla overfitting del base) ---
    max_depth=4,                # árboles poco profundos
    min_child_weight=5,         # exige soporte mínimo por hoja
    gamma=0.0,                  # subir si quieres podar splits poco útiles

    # --- Regularización por muestreo (decorrelaciona árboles) ---
    subsample=0.8,
    colsample_bytree=0.8,

    # --- Regularización L1/L2 sobre pesos de hoja ---
    reg_lambda=1.0,             # L2
    reg_alpha=0.0,              # L1 (sube si quieres más sparsity)

    # --- Reproducibilidad / cómputo ---
    random_state=42,
    n_jobs=-1,
)
```

---

## 5. Consecuencias

- **Positivas:** predicciones físicamente válidas en [0,1]; base regularizado que permite leer el overfitting vía gap train–val; categóricas tratadas sin explosión dimensional.
- **A vigilar:** ligero desajuste objetivo (log-loss) ↔ métrica (MAE); presupuesto de árboles fijo elegido a mano (revisar si el gap train–val sugiere otro punto); valores de regularización son punto de partida, no óptimos ajustados.
