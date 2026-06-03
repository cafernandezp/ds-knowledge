# Pipeline de Selección de Variables para Regresión con XGBoost

**Contexto del problema**

- Tarea: regresión, modelo final XGBoost.
- Datos: `df_train`, **≤ 10 000 filas**. Todas las decisiones de selección se toman **únicamente sobre `df_train`**; el conjunto de test permanece intacto hasta el final.
- Las fases se ejecutan **en cadena**, una tras otra: la salida (lista de features) de una fase es la entrada de la siguiente.
- Boruta se usa en su variante **Boruta-SHAP** (importancias SHAP en lugar de *gain*).

**Decisión transversal: un único estimador (XGBoost), no LightGBM**

Con ≤ 10 000 filas la velocidad no es un factor. XGBoost con `tree_method="hist"` ya usa el mismo algoritmo de histograma que motivaba a LightGBM. Mezclar dos estimadores introduce un *selection–model mismatch*: seleccionarías variables según el sesgo inductivo de LightGBM (binning propio, crecimiento *leaf-wise*, manejo distinto de interacciones y NaN) para luego entrenar XGBoost. Se selecciona **para** XGBoost, así que se selecciona **con** XGBoost.

---

## Mapa de las fases y qué controla cada una

| Fase | Método | Pregunta que responde | Tipo de selección | ¿Overfitting train/val es el riesgo clave? |
|------|--------|-----------------------|-------------------|--------------------------------------------|
| 1 | Spearman (umbral 0.85) | ¿Qué features son redundantes entre sí? | Filtro, *model-agnostic* | **No** — no ajusta modelo |
| 2 | Boruta-SHAP | ¿Qué features baten al ruido? | *All-relevant* (incluyente) | **No** — control por *shadows* |
| 3 | Stability selection | ¿Qué features se eligen de forma consistente? | Robustez por remuestreo | **No** — el remuestreo es el control |
| 4 | Backward selection (XGBoost) | ¿Cuál es el subconjunto mínimo óptimo? | *Minimal-optimal*, específico del modelo | **Sí — es el riesgo dominante** |
| 5 *(añadida)* | Baseline check | ¿La selección mejora/iguala usar todo? | Validación de sanidad | Sí, por CV |

La progresión es correcta: **barato → caro**, **agnóstico → específico del modelo**, **incluyente (all-relevant) → mínimo (minimal-optimal)**.

---

## Fase 1 — Correlación de Spearman (umbral 0.85)

### Qué hace el método

Spearman mide la **correlación de rangos**: en lugar de operar sobre los valores, opera sobre sus posiciones ordenadas. Captura cualquier relación **monótona** (no solo lineal), por eso es robusta a transformaciones monótonas y a *outliers*.

Definición (sin empates):

```
ρ = 1 − 6·Σ dᵢ² / (n·(n²−1))
```

donde `dᵢ` es la diferencia de rangos del par de observaciones `i`. Equivale a la correlación de Pearson aplicada sobre los rangos.

### Cómo se aplica aquí

- Se calcula la matriz `|ρ|` **feature–feature** (no feature–target).
- Para cada par con `|ρ| ≥ 0.85` se conserva **una** feature y se descarta la otra: son redundantes para un modelo.
- "Sin inferir null": no se hace test de significancia del coeficiente; se usa el umbral de magnitud directamente. Correcto, porque aquí no interesa si la correlación es "estadísticamente distinta de 0", sino si es **prácticamente alta**.

### Análisis crítico de overfitting

**No hay overfitting posible**: no se ajusta ningún modelo, no hay train ni validación. Spearman es una propiedad de la distribución conjunta de los datos. La métrica train/val no aplica.

### Crítica al diseño y ajustes recomendados

- **Los árboles toleran colinealidad.** XGBoost no necesita decorrelación como un modelo lineal: ante dos features correladas simplemente elige una en el split. Por tanto, el valor real de esta fase **no** es "ayudar a XGBoost a converger", sino **estabilizar las fases 2 y 3**: cuando dos features están muy correladas, se reparten la importancia (SHAP o frecuencia de selección), lo que diluye la señal y puede hacer que ambas parezcan débiles en Boruta y stability.
- **Tie-break informado, no aleatorio.** Al decidir cuál del par conservar, no elegir al azar. Criterios razonables: mayor `|ρ_Spearman|` con el target, menor % de NaN, o menor cardinalidad/coste. Documentar la regla.
- **Riesgo real:** descartar una feature que individualmente era más predictiva. Por eso 0.85 es un umbral prudente (alto): solo elimina redundancia casi-total.

**Veredicto:** se mantiene, pero entendiendo que su justificación es *estabilizar las fases de importancia downstream*, no una necesidad del modelo final.

---

## Fase 2 — Boruta-SHAP (guardar Confirmadas + Tentativas)

### Qué hace el método

Boruta es un método de selección **all-relevant**: busca **todas** las features con señal real, no el subconjunto mínimo. Mecánica:

1. Por cada feature real, crea una **shadow feature**: una copia con sus valores **permutados aleatoriamente**. Por construcción, la shadow no tiene relación con el target (es ruido con la misma distribución marginal).
2. Entrena el modelo (aquí XGBoost) sobre features reales + shadows.
3. Calcula importancias. En **Boruta-SHAP** se usan **valores SHAP** en vez de *gain*: SHAP atribuye a cada feature su contribución marginal promediada sobre coaliciones, lo que da importancias más consistentes y menos sesgadas hacia features de alta cardinalidad que el *gain*.
4. Una feature recibe un **"hit"** en esa iteración si su importancia supera el **máximo** de las importancias de **todas** las shadows.
5. Se repite muchas iteraciones. El número de hits de cada feature sigue, bajo la hipótesis nula de irrelevancia, una **binomial(n_iter, 0.5)**. Se aplica un test:
   - **Confirmada:** hits significativamente **por encima** de lo esperado por azar.
   - **Rechazada:** hits significativamente **por debajo**.
   - **Tentativa (indecisa):** ni una cosa ni la otra → señal ambigua.

### Por qué guardar Confirmadas + Tentativas

En esta fase conviene ser **incluyente**: es un colador ancho. Las features tentativas pueden tener señal débil o inestable que las fases 3 y 4 evaluarán mejor. Descartarlas aquí sería podar demasiado pronto.

### Análisis crítico de overfitting

**El gap train/val no es la lente relevante.** La lógica de Boruta es **autorreferencial**: compara cada feature contra ruido (shadows) entrenado en el **mismo** ajuste. Si el modelo sobreajusta e infla importancias, **también infla las de las shadows** (que son ruido con la misma distribución), de modo que la comparación relativa se mantiene válida. Un modelo moderadamente sobreajustado no rompe Boruta.

Matiz: las importancias SHAP calculadas **sobre las filas de train** reflejan cómo el modelo *ajustó* train, y un modelo muy sobreajustado puede sobre-acreditar features ruidosas a las que se "enganchó". El control de shadows compensa parcialmente esto, pero no del todo.

**Mitigaciones (preferibles a un split train/val explícito dentro de Boruta):**
- Mantener el estimador **regularizado** (el `xgb_fast` con `min_child_weight` moderado, `subsample`/`colsample` < 1).
- Calcular SHAP **out-of-fold** (sobre filas no usadas en ese ajuste) si se quiere robustez extra.

**Veredicto:** se mantiene tal cual. No añadiría un split train/val "para mirar overfitting" porque el mecanismo de shadows ya cumple esa función; sí mantendría regularización y consideraría SHAP OOF.

---

## Fase 3 — Stability Selection (umbral 0.7)

### Qué hace el método

Stability selection (Meinshausen & Bühlmann) ataca un problema distinto al de Boruta: **¿la selección es estable ante pequeñas perturbaciones de los datos?** Una feature puede batir al ruido en un ajuste concreto pero ser elegida de forma errática según qué filas entren.

Mecánica:

1. Se generan `B` submuestras de `df_train` (típicamente submuestreo del 50% sin reemplazo, o *bootstrap*).
2. En cada submuestra se ajusta el modelo y se determina un **conjunto seleccionado** (p. ej. top-*k* por importancia, o importancia SHAP > 0). Es necesario **definir explícitamente** esta regla de selección por submuestra.
3. Para cada feature `k` se calcula su **frecuencia de selección**:

```
Π_k = (1/B) · Σ_b  1[ feature k seleccionada en la submuestra b ]
```

4. Se conservan las features con `Π_k ≥ π_thr`, aquí **0.7**.

### Garantía teórica

Con `π_thr > 0.5`, el método acota el número esperado de falsos positivos:

```
E[V] ≤ (1 / (2·π_thr − 1)) · q² / p
```

donde `q` = nº medio de features seleccionadas por submuestra y `p` = total de features candidatas. Esto exige `π_thr > 0.5`; **0.7 es una elección sólida** (control de falsos positivos sin ser excesivamente conservador).

### Análisis crítico de overfitting

**El remuestreo *es* el control de varianza/overfitting.** Solo se usa la salida **binaria** "¿fue seleccionada en esta submuestra?". Las selecciones impulsadas por ruido aparecen de forma intermitente y no superan el umbral; las features con señal real reaparecen consistentemente. Vigilar el gap train/val dentro de cada ajuste es **redundante** con lo que la fase ya hace por diseño: la frecuencia entre submuestras *es* el proxy de generalización de la decisión de selección.

Los parámetros a vigilar no son train/val sino: `B` (suficientes submuestras, p. ej. 50–100), `q` (tamaño de selección por submuestra) y `π_thr`.

### Crítica al diseño

- **Redundancia parcial con Boruta.** Ambas son filtros basados en importancia de árboles. La diferencia clave: Boruta responde "¿supera al ruido?" y stability responde "¿se elige de forma consistente?". Una feature puede pasar Boruta y ser inestable → stability **sí aporta** información nueva, pero la solapación es real.
- **Mitigación de coste y solapamiento:** ejecutar stability **sobre el set ya reducido por Boruta** (como en esta cadena) la hace barata y la convierte en un filtro de robustez, no en un re-descubrimiento desde cero. Correcto.
- **Definir la regla de selección por submuestra** es imprescindible y a menudo se omite. Recomendado: top-*k* por SHAP, con *k* coherente con `q` de la fórmula anterior.

**Veredicto:** se mantiene, ejecutada barata sobre el set post-Boruta, con regla de selección por submuestra documentada.

---

## Fase 4 — Backward Selection con XGBoost

### Qué hace el método

Selección **minimal-optimal** y **específica del modelo**: parte del conjunto de features supervivientes y **elimina iterativamente** la menos útil, midiendo el rendimiento en cada paso, hasta encontrar el subconjunto más pequeño que no degrada (o mejora) la métrica.

Mecánica (eliminación hacia atrás, *greedy*):

1. Empezar con todas las features candidatas. Calcular el **score por validación cruzada**.
2. En cada paso, evaluar la eliminación de cada feature restante (o, más barato, eliminar la de menor importancia) y quedarse con la eliminación que **maximiza** el score CV.
3. Repetir, registrando el score CV frente al nº de features.
4. Elegir el subconjunto final según la curva.

Aquí, a diferencia de las fases 2–3, se usan los **hiperparámetros reales del modelo final** (`min_child_weight=100`, `eta=0.01`, `n_estimators=200`, `max_depth=5`, etc.), porque esta fase debe reflejar el modelo que se desplegará.

### Análisis crítico de overfitting — **el riesgo dominante de toda la cadena**

Esta es la **única** fase donde "mirar overfitting entre train y validación" es exactamente la lente correcta, por dos problemas distintos:

1. **Puntuar sobre train es inválido.** Quitar features casi nunca **empeora** el error de entrenamiento (la curva es monótona decreciente o plana en flexibilidad), así que un backward guiado por el score de train es ciego: elegiría siempre el conjunto completo o decisiones espurias. **Obligatorio usar score de validación cruzada.**

2. **Optimismo por selección (el más sutil).** Backward prueba **muchos** subconjuntos y se queda con el mejor sobre la validación. Ese "mejor de muchos" **sobreajusta la propia estimación de validación**: el score CV del subconjunto ganador es optimista respecto a su rendimiento real.

**Mitigaciones obligatorias:**
- **CV con folds fijos** a lo largo de todos los pasos (mismos folds en cada eliminación → comparaciones justas).
- **CV repetida** (varias semillas de partición) para estabilizar la decisión de qué eliminar.
- **Regla 1-SE:** elegir el subconjunto **más pequeño** cuyo score CV esté **dentro de 1 error estándar** del mejor score, no el del mejor score absoluto. Esto contrarresta directamente el optimismo por selección y favorece parsimonia.
- **Tocar el test una sola vez** al final, como estimación honesta. Nunca usarlo para guiar la eliminación.

### Crítica al diseño

- **Coste:** backward exhaustivo es `O(p²)` ajustes. Con el set ya reducido por las fases 1–3 es asumible. Si no, usar **eliminación por importancia** (quitar siempre la de menor SHAP) en vez de probar todas: mucho más barato, ligeramente menos óptimo.
- `min_child_weight=100` con ≤ 10 000 filas es **muy agresivo**: cada hoja exige ≥ 100 muestras efectivas. Con `max_depth=5` (hasta 32 hojas) esto poda con fuerza. Es regularización fuerte, defendible en datos pequeños para evitar sobreajuste, **pero** durante backward puede ocultar la utilidad de features de señal débil. Recomendación: o se trata como hiperparámetro a tunear **después** de fijar features, o se relaja durante esta fase y se restaura en el modelo final.

**Veredicto:** se mantiene, con CV repetida, folds fijos y **regla 1-SE** como salvaguardas no negociables.

---

## Fase 5 — *(Añadida)* Comprobación contra baseline

### Por qué añadirla

Tras 4 fases de decisiones tomadas sobre el **mismo** `df_train`, hay **optimismo acumulado**: cada fase eligió lo que parecía bueno en estos datos. Falta la evidencia más básica de que el esfuerzo sirvió.

### Qué hace

- Comparar, con la **misma CV** que la fase 4:
  - score del **subconjunto final seleccionado**, vs.
  - score usando **todas las features** (o el set post-Spearman).
- Criterio de aceptación: el subconjunto final debe **igualar o mejorar** la generalización con **menos** variables. Si no lo hace, alguna fase fue demasiado agresiva → revisar (típicamente backward con `min_child_weight=100`).
- *(Opcional)* Repetir todo el pipeline con varias semillas para medir la **estabilidad del conjunto seleccionado**. Si la lista de features cambia mucho entre semillas, la selección no es fiable.

---

## Configuración recomendada del estimador

```python
from xgboost import XGBRegressor

# Fases 2–3 (Boruta-SHAP, stability): barato y regularizado; aquí solo se RANKEA
xgb_fast = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    tree_method="hist",
    learning_rate=0.05,      # más alto que el final: no se afina, se ordena
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,     # más laxo que el 100 final: no podar señal débil aún
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=-1,
)

# Fase 4 (backward) y modelo final: hiperparámetros REALES de despliegue
init_params = {
    "n_estimators": 200,
    "max_depth": 5,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "eta": 0.01,
    "seed": SEED,
    "gamma": 0,
    "min_child_weight": 100,  # revisar/tunear tras fijar features (ver Fase 4)
}
```

---

## Diagnósticos y riesgos transversales

- **Fuga de datos (leakage):** todas las fases, **Spearman incluida**, deben ejecutarse solo sobre `df_train`. Si más adelante hay CV externa para evaluar el *pipeline completo*, la selección debe ir **dentro** de cada fold de esa CV, no antes. Como aquí se trabaja "solo sobre `df_train`", el test externo permanece como estimación honesta y se toca una sola vez.
- **Optimismo acumulado:** 4 decisiones data-dependientes sobre el mismo train → el test intacto es la única estimación no sesgada. No reutilizarlo.
- **Spearman elimina redundancia, no irrelevancia:** no confundir su rol con el de Boruta.
- **Boruta = all-relevant, Backward = minimal-optimal:** son objetivos distintos y complementarios; por eso la cadena tiene sentido.
- **Estabilidad de la semilla:** con ≤ 10 000 filas, las importancias y selecciones pueden ser sensibles a la partición. Fijar `random_state` en todo y, si es posible, verificar estabilidad entre semillas.

## Resumen de la crítica

| Fase | Decisión | Motivo |
|------|----------|--------|
| 1 Spearman | **Mantener** | Su valor real es estabilizar 2–3, no decorrelacionar para el árbol. Tie-break informado. |
| 2 Boruta-SHAP | **Mantener** | Control por shadows hace innecesario vigilar train/val. SHAP OOF opcional. |
| 3 Stability | **Mantener (barata, post-Boruta)** | El remuestreo es el control de overfitting. Redundancia parcial con Boruta, pero aporta robustez. Definir regla de selección por submuestra. |
| 4 Backward | **Mantener con salvaguardas** | **Único punto donde overfitting train/val es el riesgo central.** CV repetida + folds fijos + regla 1-SE. Revisar `min_child_weight=100`. |
| 5 Baseline | **Añadir** | Evidencia de que la selección iguala/mejora con menos features. |

**LightGBM:** innecesario en todo el flujo. Con ≤ 10 000 filas la velocidad no es factor y mezclar estimadores introduce *selection–model mismatch*. Todo con XGBoost (`tree_method="hist"` en las fases baratas).

---

## Implementación de referencia (Python, funcional)

Estilo procedural/funcional, sin OOP, funciones solo donde aportan. Dataset real de regresión (**California Housing**, integrado en scikit-learn, sin descarga externa), submuestreado a ≤ 10 000 filas para reflejar el contexto. `df` representa `df_train`: **todo se ejecuta solo sobre él**.

> Requiere `pip install xgboost shap scikit-learn scipy pandas numpy`. SHAP es obligatorio porque la fase 2 es Boruta-**SHAP** y la 3 usa importancia SHAP por coherencia.

### Setup, datos y estimadores

```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RepeatedKFold, cross_val_score
from xgboost import XGBRegressor
import shap

SEED = 42

# Regresión real; submuestra a <=10k. 'df' = df_train (única fuente para selección).
df = fetch_california_housing(as_frame=True).frame.sample(8000, random_state=SEED).reset_index(drop=True)
TARGET = "MedHouseVal"
X, y = df.drop(columns=[TARGET]), df[TARGET]

def make_fast_model():
    # Fases 2-3: barato y REGULARIZADO. Aquí solo se rankea, no se afina ->
    # learning_rate alto, subsample/colsample <1, min_child_weight laxo (no podar señal débil aún).
    return XGBRegressor(
        n_estimators=300, max_depth=5, tree_method="hist", learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        objective="reg:squarederror", random_state=SEED, n_jobs=-1,
    )

def make_final_model():
    # Fase 4 y modelo final: hiperparámetros REALES de despliegue.
    return XGBRegressor(
        n_estimators=200, max_depth=5, subsample=1.0, colsample_bytree=1.0,
        objective="reg:squarederror", learning_rate=0.01, gamma=0,
        min_child_weight=100, random_state=SEED, n_jobs=-1,
    )

def shap_importance(model, X):
    # Importancia = media de |SHAP| por feature (más consistente que 'gain').
    sv = shap.TreeExplainer(model).shap_values(X)
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
```

### Fase 1 — Spearman (umbral 0.85)

`threshold=0.85` alto: solo elimina redundancia casi-total. Tie-break **informado** (conserva la más correlada con el target), no aleatorio.

```python
def spearman_filter(X, y, threshold=0.85):
    corr = X.corr(method="spearman").abs()
    tcorr = X.apply(lambda c: abs(spearmanr(c, y).statistic))  # |corr| con el target
    cols, drop = list(X.columns), set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in drop or b in drop or corr.loc[a, b] < threshold:
                continue
            drop.add(a if tcorr[a] < tcorr[b] else b)  # descarta la menos útil del par
    return [c for c in cols if c not in drop]
```

### Fase 2 — Boruta-SHAP

`n_iter=30` itera shadows para que el test binomial tenga potencia. Decisión por test binomial vs. `p=0.5`. Se guardan **Confirmadas + Tentativas** (colador ancho).

```python
def boruta_shap(X, y, n_iter=30, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    hits = pd.Series(0, index=X.columns)
    for _ in range(n_iter):
        # Shadow = copia permutada (ruido con misma marginal).
        shadow = X.apply(lambda c: rng.permutation(c.values)).add_prefix("shadow_")
        Xa = pd.concat([X.reset_index(drop=True), shadow], axis=1)
        imp = shap_importance(make_fast_model().fit(Xa, y), Xa)
        smax = imp[shadow.columns].max()                  # techo de ruido
        hits[X.columns] += (imp[X.columns] > smax).astype(int)
    def decide(h):
        if binomtest(h, n_iter, 0.5, alternative="greater").pvalue < alpha: return "Confirmada"
        if binomtest(h, n_iter, 0.5, alternative="less").pvalue   < alpha: return "Rechazada"
        return "Tentativa"
    status = hits.map(decide)
    return status[status != "Rechazada"].index.tolist(), status
```

### Fase 3 — Stability selection (umbral 0.7)

`n_subsamples=50`, `frac=0.5` (submuestreo sin reemplazo). `top_k` define la regla de selección por submuestra (las `q` más importantes). `thr=0.7` cumple `>0.5` exigido por la cota de falsos positivos.

```python
def stability_selection(X, y, n_subsamples=50, frac=0.5, top_k=None, thr=0.7, seed=SEED):
    rng = np.random.default_rng(seed)
    top_k = top_k or max(1, X.shape[1] // 2)   # q: nº seleccionadas por submuestra
    counts, n = pd.Series(0, index=X.columns), len(X)
    for _ in range(n_subsamples):
        idx = rng.choice(n, int(frac * n), replace=False)
        Xs, ys = X.iloc[idx], y.iloc[idx]
        imp = shap_importance(make_fast_model().fit(Xs, ys), Xs)
        counts[imp.nlargest(top_k).index] += 1
    freq = counts / n_subsamples
    return freq[freq >= thr].index.tolist(), freq
```

### Fase 4 — Backward selection (XGBoost) con CV repetida y regla 1-SE

Eliminación **greedy por importancia** (`O(p)` ajustes, no `O(p²)`). **CV repetida con folds fijos** y **regla 1-SE** contra el optimismo por selección.

```python
def backward_selection(X, y, min_features=1, seed=SEED):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)  # folds fijos
    def cv_score(cols):
        s = cross_val_score(make_final_model(), X[cols], y, cv=cv,
                            scoring="neg_root_mean_squared_error")
        return s.mean(), s.std() / np.sqrt(len(s))   # media y error estándar
    feats, history = list(X.columns), []
    while True:
        history.append((list(feats), *cv_score(feats)))
        if len(feats) == min_features:
            break
        imp = shap_importance(make_final_model().fit(X[feats], y), X[feats])
        feats = [f for f in feats if f != imp.idxmin()]   # quita la menos importante
    best = max(history, key=lambda h: h[1])               # neg_rmse: mayor es mejor
    thr = best[1] - best[2]                                # dentro de 1 SE del mejor
    chosen = min((h for h in history if h[1] >= thr), key=lambda h: len(h[0]))
    return chosen[0], history                              # subconjunto más parsimonioso
```

### Fase 5 — Comprobación contra baseline

Evidencia de que la selección **iguala o mejora** con menos features (misma CV).

```python
def baseline_check(X, y, selected, seed=SEED):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
    def rmse(cols):
        s = cross_val_score(make_final_model(), X[cols], y, cv=cv,
                            scoring="neg_root_mean_squared_error")
        return -s.mean()
    return {"rmse_full": rmse(list(X.columns)), "n_full": X.shape[1],
            "rmse_selected": rmse(selected),    "n_selected": len(selected)}
```

### Orquestación (cadena: cada fase alimenta a la siguiente)

```python
f1 = spearman_filter(X, y, threshold=0.85)
f2, status = boruta_shap(X[f1], y)
f3, freq    = stability_selection(X[f2], y, thr=0.7)
f4, history = backward_selection(X[f3], y)
report      = baseline_check(X, y, f4)

print("Fase 1 — Spearman      :", f1)
print("Fase 2 — Boruta-SHAP   :", f2, "\n  status:", status.to_dict())
print("Fase 3 — Stability     :", f3, "\n  freq  :", freq.round(2).to_dict())
print("Fase 4 — Backward final:", f4)
print("Fase 5 — Baseline      :", report)
# Aceptar la selección solo si rmse_selected <= rmse_full (±SE) con menos features.
```

> **Nota:** California Housing tiene solo 8 features, por lo que el recorte será modesto — el código es **ilustrativo del flujo**. El valor del pipeline crece con la dimensionalidad. Para reproducibilidad, todo va con `random_state=SEED`; verifica estabilidad del conjunto seleccionado repitiendo con varias semillas si el problema lo justifica.
