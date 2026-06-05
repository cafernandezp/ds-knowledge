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

#### 2.1.1 Justificación detallada de `reg:logistic`

La elección no es solo "acota a [0,1]". Hay cuatro razones técnicas que se refuerzan entre sí; conviene documentarlas porque la alternativa intuitiva (`reg:squarederror`) parece más simple pero es estadísticamente inadecuada para un target que es una **proporción**.

**(a) Naturaleza del target: es una proporción, no una cantidad libre.**
`score` mide la fracción de avance de un procedimiento. Vive en el intervalo cerrado [0,1] por definición del fenómeno, no por una restricción artificial. Un modelo cuyo espacio de salida es ℝ (como `reg:squarederror`) está **mal especificado**: asigna densidad/predicción a regiones imposibles (un avance de −5 % o de 112 % no existe). `reg:logistic` modela el **logit** del avance y devuelve la proporción vía sigmoide, de modo que el espacio de salida coincide con el espacio del fenómeno.

**(b) Mecánica interna: dónde entra la cota.**
XGBoost construye un *score crudo* aditivo `F(x) = Σ fₖ(x)` (suma de hojas), que es **no acotado**. La diferencia entre objetivos es la función de enlace aplicada a `F(x)`:

- `reg:squarederror`: predicción = `F(x)` directamente → puede ser cualquier real.
- `reg:logistic`: predicción = `σ(F(x)) = 1 / (1 + e^(−F(x)))` → **siempre en (0,1)** por construcción matemática, sin post-proceso ni clipping. La cota es estructural, no un parche.

**(c) Estructura del error: la varianza de una proporción no es constante.**
`reg:squarederror` minimiza MSE, que asume **homocedasticidad** (misma varianza del error en todo el rango). Para una proporción esto es falso: la varianza es máxima cerca de 0.5 y **se contrae hacia 0 en los extremos** (un avance del 99 % deja muy poco margen de error posible), análogo a la varianza binomial `p(1−p)`. La pérdida logística (entropía cruzada) que usa `reg:logistic` **pondera el error de forma coherente con esa estructura**: penaliza más los fallos cerca de los bordes y menos en el centro. Resultado: predicciones mejor calibradas a lo largo de todo el rango de avance, no solo en la media.

**(d) Qué pierde `reg:squarederror` en la práctica.**
- Sesgo sistemático en las colas: tiende a predecir < 0 para casos casi sin avance y > 1 para casos casi completos, justo los extremos que suelen importar (¿está parado? ¿está terminado?).
- Si se "arregla" con clipping a [0,1], se introduce un sesgo no modelado y se acumula masa artificial en 0 y 1.
- Trata por igual un error en la zona estable (centro) y en la zona de varianza contraída (bordes), degradando la calibración.

**Por qué `reg:logistic` y no la opción "estadísticamente pura" (regresión beta).**
El modelo formalmente correcto para una proporción continua en (0,1) es la **regresión beta**, que modela explícitamente media y dispersión. Se descarta porque: (i) XGBoost **no la trae como objetivo nativo**; (ii) la beta está definida en el intervalo **abierto** (0,1) y **rompe con 0.0 y 1.0 exactos**, que aquí existen (procedimientos sin iniciar o completados); (iii) requeriría transformaciones (p. ej. squeeze de Smithson-Verkuilen) que añaden complejidad y supuestos. `reg:logistic` captura el 90 % del beneficio (cota + ponderación tipo binomial del error) **tolerando 0.0 y 1.0 exactos** y sin salir del ecosistema XGBoost. Es el punto pragmático correcto.

**Coste asumido — desajuste objetivo ↔ métrica.**
`reg:logistic` optimiza la **media** bajo pérdida logística; MAE corresponde a optimizar la **mediana** (L1). Hay, por tanto, un pequeño desalineamiento entre lo que el objetivo minimiza y la métrica que reportamos. Se asume conscientemente porque (i) el sesgo media-vs-mediana es leve cuando la distribución condicional no es fuertemente asimétrica, y (ii) la **cota física [0,1]** aporta más valor que cerrar ese desajuste. La alternativa que lo cierra (`reg:absoluteerror`) sacrifica la cota; ver §2.1 y la acción futura en §3.

**Validación esperada (cómo se confirma la decisión).**
- Histograma de predicciones: con `reg:logistic` **ninguna** cae fuera de [0,1]; con `reg:squarederror` aparecen colas fuera de rango.
- Calibración por tramos de avance (predicho vs observado en bins): `reg:logistic` debe mantener el sesgo bajo también en los extremos.
- MAE medido en escala original vía `scoring`, no la log-loss interna.

#### 2.1.2 Por qué NO `reg:absoluteerror` (aun estando alineado con MAE)

`reg:absoluteerror` optimiza L1, es decir **exactamente** la métrica de reporte. Es la objeción más razonable a `reg:logistic`. Aun así se descarta, y el argumento **no es la velocidad** (hay cómputo de sobra). Son tres razones de fondo, más un matiz sobre el optimizador:

**(1) Rompe la cota [0,1] — estructural.**
`reg:absoluteerror` **no aplica función de enlace**: la predicción es el score crudo aditivo `F(x) = Σ fₖ(x)`, que vive en ℝ → puede devolver −0.08 o 1.06. Esto reintroduce justo el problema que `reg:logistic` resuelve. El parche (clipping a [0,1]) añade sesgo no modelado y acumula masa artificial en los bordes.

**(2) L1 optimiza la *mediana* condicional — mal estimador para una proporción con masa en los bordes.**
Minimizar L1 ⇒ el modelo predice la **mediana** de `y|x`; minimizar log-loss/L2 ⇒ la **media**. Con un target que probablemente tiene **masa en 0.0 (sin iniciar) y 1.0 (completado)**, la mediana se degenera:
- Si en una región del espacio de features > 50 % de los casos están en `1.0`, la **mediana predicha es exactamente 1.0**, *ignorando la dispersión del resto*.
- Efecto: predicciones que **se pegan a 0 y 1 en bloques**, vaciando la gradación intermedia — que suele ser la población de interés (los casos *a medio camino*).
- La **media** (`reg:logistic`) conserva información de toda la distribución condicional y produce estimaciones suaves en el interior → mejor discriminación entre casos intermedios.

**(3) "Alineado con la métrica" ≠ "menor MAE en test".**
La alineación es sobre la **pérdida de entrenamiento**, no sobre el MAE **out-of-sample**, que depende de sesgo-varianza y calibración. Un estimador de la media bien calibrado (`reg:logistic`), *medido con MAE*, puede lograr **menor MAE en test** que un estimador de la mediana que se degenera en los bordes o sobreajusta la estructura mediana del train. El supuesto "si optimizo MAE tendré el mejor MAE" es **falso en general** fuera de muestra; solo se cumpliría si la mediana condicional fuese el mejor predictor, que aquí no lo es por el punto (2).

**Matiz sobre el hessiano (calidad de ajuste, NO velocidad).**
XGBoost es un optimizador de **segundo orden**: usa gradiente y **hessiano** para elegir splits (`gain = G²/(H+λ)`) y pesos de hoja. Para L1 el gradiente es `sign(pred − y) ∈ {−1,+1}` y el **hessiano es 0**; XGBoost lo compensa con hessiano constante y recálculo de hojas por mediana de residuos. Consecuencias **independientes del cómputo**:
- La fórmula de ganancia de split **degenera** (sin curvatura) → cortes peor informados.
- **`reg_lambda` / `min_child_weight` se comportan distinto** (L2 sumada a `H≈0`) → los knobs de regularización del §2.4 dejan de ser comparables y predecibles. Más cómputo no lo arregla: ajusta un modelo distinto y peor informado, no uno que "converge lento".

**Conclusión.** El beneficio de `reg:absoluteerror` (alinear *training loss* con la métrica) **no domina** frente a perder la cota física, inducir un estimador-mediana degenerado en los bordes, y desestabilizar la regularización. Dado que la velocidad no limita, la postura correcta no es decidir por teoría sino **confirmarla con un bake-off** (ver acción en §3).

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

- **`objective = reg:logistic`** — el target es una **proporción** en [0,1]: el objetivo modela el logit y devuelve la salida vía sigmoide, garantizando la cota **por construcción** y ponderando el error de forma coherente con la varianza tipo binomial de una proporción (mayor en el centro, contraída en los bordes). Se prefiere sobre `reg:squarederror` (mal especificado: salida en ℝ, homocedástico) y sobre la regresión beta (no nativa en XGBoost y rota en 0.0/1.0 exactos). La cota física pesa más que la alineación exacta con MAE; el sesgo media-vs-mediana es leve y tolerable. Justificación completa en §2.1.1.
- **MAE se gestiona en la CV** (`scoring="neg_mean_absolute_error"`), no en el objetivo ni en `eval_metric`.
- **`tree_method="hist"` + `enable_categorical=True`** — rapidez y soporte nativo de categóricas de alta cardinalidad sin one-hot.
- **Presupuesto fijo** `n_estimators=600`, `learning_rate=0.03` (sin early stopping en el base).
- **Regularización conservadora** como referencia estable del flujo de selección.
- **Pendiente / acción futura — bake-off `reg:logistic` vs `reg:absoluteerror`:** como la velocidad no limita, confirmar empíricamente la decisión en lugar de cerrarla solo por teoría. Entrenar ambos con el **mismo `scoring="neg_mean_absolute_error"`** y la misma CV, y comparar en **tres ejes**: (1) **MAE en test externo**; (2) **% de predicciones fuera de [0,1]** en `reg:absoluteerror` y sesgo introducido por el clipping; (3) **calibración por tramos** y **masa pegada a 0/1** (histograma de predicciones). Regla: adoptar `reg:absoluteerror` **solo si** mejora el MAE de test de forma **estable entre repeats** *y* el comportamiento en bordes es aceptable; en empate o con bordes degenerados → mantener `reg:logistic`. El esquema de CV se refina por separado.

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
