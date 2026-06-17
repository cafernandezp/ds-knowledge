# Reporte técnico — Validación cruzada en scikit-learn: funciones de ejecución y estrategias K-Fold

> **Last update:** 2026-06-04
> **Ámbito:** Diferencias prácticas entre las tres funciones de ejecución de CV (`cross_val_score`, `cross_val_predict`, `cross_validate`) y entre los splitters K-Fold (`KFold`, `KFold(shuffle=True)`, `RepeatedKFold`, `StratifiedKFold`, `RepeatedStratifiedKFold`).
> **Audiencia:** uso en pipelines de modelado predictivo (regresión y clasificación).

---

## Modelo mental

La validación cruzada se compone de **dos piezas ortogonales**:

1. **El splitter** — *cómo* se reparten las filas en folds (`KFold`, `StratifiedKFold`, …).
2. **La función de ejecución** — *qué* se hace en cada fold y *qué* se devuelve (`cross_val_score`, `cross_validate`, `cross_val_predict`).

Se combinan libremente: cualquier función de ejecución acepta cualquier splitter vía el argumento `cv`. Este reporte cubre ambas piezas por separado.

```
                       FUNCIÓN DE EJECUCIÓN  (qué se devuelve)
                  ┌───────────────┬───────────────┬────────────────────┐
                  │ cross_val_    │ cross_        │ cross_val_         │
                  │ score         │ validate      │ predict            │
  S    ┌──────────┼───────────────┼───────────────┼────────────────────┤
  P    │ KFold    │               │               │                    │
  L    ├──────────┤               │               │                    │
  I    │ Shuffle  │   cualquier   │  cualquier    │    cualquier        │
  T    ├──────────┤   splitter    │  splitter     │    splitter         │
  T    │ Repeated │      ×        │     ×         │      ×              │
  E    ├──────────┤   cualquier   │  cualquier    │    cualquier        │
  R    │Stratified│   función     │  función      │    función          │
       ├──────────┤               │               │                    │
 (cómo │ Rep.Strat│               │               │                    │
 parte)└──────────┴───────────────┴───────────────┴────────────────────┘
         Eje 1: CÓMO se reparten        Eje 2: QUÉ se hace y se devuelve
         las filas en folds             en cada fold

  → Eliges UNA celda: un splitter (fila) + una función (columna).
    Son decisiones independientes.
```

---

# Sección 1 — Funciones de ejecución de CV

Las tres entrenan y evalúan el estimador fold a fold; difieren en **qué devuelven** y, por tanto, en **para qué sirven**.

## 1.1 `cross_val_score`

**Qué devuelve:** un array 1-D con **un score de validación por fold** (una sola métrica, solo test).

**Cuándo usar:**
- Chequeo rápido y de una línea del rendimiento de validación.
- Cuando solo necesitas la media/desviación de una única métrica.

**Riesgo / cuidado:**
- **Ciego al overfitting:** no devuelve score de entrenamiento → es imposible leer el gap train–val a partir de su salida.
- **Métrica por defecto dependiente del estimador** (R² para regresores, accuracy para clasificadores). Fija `scoring` explícito siempre.
- Una sola métrica por llamada → para varias métricas tendrías que llamar repetidamente (reentrenando cada vez).

**Veredicto:** el caso de uso más estrecho. En la práctica, `cross_validate` lo subsume sin coste extra.

## 1.2 `cross_validate`

**Qué devuelve:** un `dict` con `test_score`, y opcionalmente `train_score` (`return_train_score=True`), `fit_time`, `score_time` y `estimator` (`return_estimator=True`). Admite **varias métricas a la vez** (`scoring` como lista o dict).

**Cuándo usar:**
- **Diagnóstico de overfitting** → comparar `train_score` vs `test_score` (el gap).
- Reportar **varias métricas** en una sola pasada (p. ej. MAE + R²).
- Recuperar los estimadores ajustados o los tiempos de cómputo.

**Riesgo / cuidado:**
- `return_train_score=True` añade coste de scoring sobre el train; despréocupate si no vas a leer el gap, actívalo si sí.
- El `train_score` alto con `test_score` bajo es la **firma del sobreajuste**; no lo ignores aunque el test luzca aceptable.

**Por qué es más robusto que `cross_val_score`:** es un **superconjunto estricto**. Da la misma información de validación **más** la de entrenamiento, multimétrica y metadatos, al mismo coste de ajuste. Es la opción por defecto recomendada para diagnóstico y comparación de configuraciones.

## 1.3 `cross_val_predict`

**Qué devuelve:** **predicciones out-of-fold**, una por muestra, generada por el modelo del fold en que esa muestra cayó como validación. NO devuelve un score.

**Cuándo usar:**
- **Visualización**: predicho vs real, gráficos de residuos.
- **Predicciones out-of-fold** para *stacking* / *blending* (entrada de un meta-modelo).
- Inspección a nivel de muestra (qué casos se predicen mal).

**Riesgo / cuidado — el error clásico:**
- **No es un estimador válido de generalización.** Concatenar predicciones de modelos distintos y calcular una métrica global **no equivale** a la media de scores por fold. La documentación de sklearn lo advierte explícitamente: úsalo para visualización, **no** para reportar rendimiento.
- Las predicciones provienen de **modelos diferentes** (uno por fold), así que la métrica agregada mezcla modelos.
- Métricas que **no se descomponen por muestra** (p. ej. AUC, métricas que dependen del ranking global del conjunto) quedan especialmente distorsionadas si se calculan sobre el vector concatenado.

**Por qué no compite con las otras dos:** resuelve un problema distinto (obtener predicciones, no estimar performance). Si lo usas para medir generalización, obtendrás un número **sesgado**.

## 1.4 Esquema — qué devuelve cada función

Las tres recorren los mismos folds; lo que cambia es **qué guardan** y, por tanto, **si puedes leer el overfitting**.

```
══════════════════════════════════════════════════════════════════
  cross_val_score          (lo más simple)
══════════════════════════════════════════════════════════════════
  QUÉ HACE: guarda SOLO el score de validación de cada fold.

     Train completo
        │
        ├── Fold 1 → val MAE = 0.081
        ├── Fold 2 → val MAE = 0.079
        ├── Fold 3 → val MAE = 0.082
        ├── Fold 4 → val MAE = 0.080
        └── Fold 5 → val MAE = 0.078
                       ↓  promedio
              MAE val medio = 0.080

  OVERFITTING: ✗ no se puede saber.
               Solo tienes el lado de validación, nunca el de train.
══════════════════════════════════════════════════════════════════
  cross_validate           (recomendado para diagnóstico)
══════════════════════════════════════════════════════════════════
  QUÉ HACE: por cada fold guarda DOS scores → train y validación.
            (requiere return_train_score=True)

        ┌──────────────┬───────────────┐
        │    TRAIN     │   VALIDACIÓN  │
     ───┼──────────────┼───────────────┤
     F1 │  MAE = 0.055 │  MAE = 0.081  │
     F2 │  MAE = 0.058 │  MAE = 0.079  │
     F3 │  MAE = 0.054 │  MAE = 0.082  │
     F4 │  MAE = 0.057 │  MAE = 0.080  │
     F5 │  MAE = 0.056 │  MAE = 0.078  │
     ───┼──────────────┼───────────────┤
    prom│    0.056     │    0.080      │
        └──────┬───────┴───────┬───────┘
               └───── GAP ──────┘ = 0.080 − 0.056 = 0.024

     (también devuelve: fit_time, score_time, estimator, y multimétrica)

  OVERFITTING: ✓ inmediato.
               gap = MAE val − MAE train   (>0 grande = sobreajuste)
══════════════════════════════════════════════════════════════════
  cross_val_predict        (predicciones, NO scores)
══════════════════════════════════════════════════════════════════
  QUÉ HACE: una predicción OOF por FILA (cada fila la predice el
            modelo del fold en que quedó fuera de entrenamiento).

     PJ_id   y_real   y_pred_oof
       1      0.32      0.30
       2      0.78      0.71
       3      0.15      0.19
       4      0.65      0.69
       …       …         …
       N      0.54      0.60
                  ↓
       mean_absolute_error(y_real, y_pred_oof) = 0.080

  OVERFITTING: ✗ no directamente → no hay predicción/MAE de TRAIN,
               solo el lado OOF. Además cada fila viene de un modelo
               distinto (uno por fold), así que NO es la medida limpia
               de performance: úsalo para graficar (pred vs real,
               residuos) o como insumo de stacking.
══════════════════════════════════════════════════════════════════
```

**Equivalencia y matiz (corroborado):**
- `cross_val_score` y la rama de validación de `cross_validate` calculan **lo mismo** (score de val por fold); `cross_validate` solo añade el lado train + metadatos.
- El **MAE val medio** de `cross_validate` y el **MAE sobre predicciones OOF** de `cross_val_predict` **coinciden para MAE con folds del mismo tamaño** (el MAE se descompone por muestra). **No coinciden en general** para métricas no descomponibles (AUC, R²) o folds de tamaño desigual → para esas usa el promedio por fold de `cross_validate`, no el pooled de `cross_val_predict`.
- Las cifras son ilustrativas; lo que importa es **qué columnas existen**: sin la columna *train*, no hay diagnóstico de overfitting.

### Mecánica de `cross_val_predict` (cómo se calcula la predicción OOF)

```
  Datos divididos en k=5 folds:   [ F1 ][ F2 ][ F3 ][ F4 ][ F5 ]

  En cada ronda se ENTRENA con los otros folds y se PREDICE el que quedó fuera:

     Ronda 1   train[ ░░ F2 F3 F4 F5 ] → predice → F1   (el modelo NO vio F1)
     Ronda 2   train[ F1 ░░ F3 F4 F5 ] → predice → F2   (el modelo NO vio F2)
     Ronda 3   train[ F1 F2 ░░ F4 F5 ] → predice → F3   (el modelo NO vio F3)
     Ronda 4   train[ F1 F2 F3 ░░ F5 ] → predice → F4   (el modelo NO vio F4)
     Ronda 5   train[ F1 F2 F3 F4 ░░ ] → predice → F5   (el modelo NO vio F5)
                                                          │
  Se ENSAMBLAN los trozos en UN vector, en el orden original de las filas:
                                                          ▼
     ŷ_oof = [ pred(F1) │ pred(F2) │ pred(F3) │ pred(F4) │ pred(F5) ]
              └──── cada fila predicha por el modelo que NO la entrenó ────┘

  Resultado:  1 predicción por fila (ninguna fila se predice a sí misma)
              →  MAE = mean_absolute_error(y, ŷ_oof)

  Contraste:  cross_val_score/validate  →  guardan un SCORE por fold
              cross_val_predict          →  guarda la PREDICCIÓN de cada fila
```

## 1.5 Tabla comparativa

|                       | `cross_val_score`          | `cross_validate`                                        | `cross_val_predict`      |
| --------------------- | -------------------------- | ------------------------------------------------------- | ------------------------ |
| Devuelve              | scores de test (1 métrica) | dict: test (+train, tiempos, estimadores), multimétrica | predicciones out-of-fold |
| Ve overfitting (gap)  | ❌                          | ✅ (`return_train_score`)                                | ❌                        |
| Varias métricas       | ❌                          | ✅                                                       | n/a                      |
| Estima generalización | ✅ (limitado)               | ✅ (recomendado)                                         | ❌ **no usar para esto**  |
| Uso típico            | chequeo rápido             | diagnóstico + reporte                                   | visualización / stacking |

---

# Sección 2 — Estrategias de partición (splitters K-Fold)

Todos parten los datos en `k` folds y rotan cuál es validación. Difieren en **cómo asignan las filas** y en **qué garantizan**.

## 2.1 `KFold` (sin shuffle)

**Qué hace:** divide en `k` **bloques contiguos** según el orden de las filas.

```
  Datos en su orden original (k=4):  [ filas 1………………………N ]

  Fold 1:  ███ VAL ███  ░░░░░░░░░░░░░░░░░░░░░░░░ train
  Fold 2:  ░░░░░░░░░  ███ VAL ███  ░░░░░░░░░░░░░ train
  Fold 3:  ░░░░░░░░░░░░░░░░░░  ███ VAL ███  ░░░░ train
  Fold 4:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░  ███ VAL ███
           └── cada bloque = trozo CONTIGUO del dataset ──┘

  ⚠ Si las filas vienen ordenadas (fecha, id, categoría):
     VAL del Fold 1 = solo los "primeros" → no representativo → sesgo
```

**Cuándo usar:**
- Cuando el orden de las filas **ya es aleatorio** o irrelevante.

**Riesgo / cuidado:**
- Si el DataFrame viene **ordenado** (por fecha, por id, por categoría), los bloques quedan **no representativos** → estimación sesgada. Es el fallo silencioso más común.

**Veredicto:** rara vez la mejor opción salvo que controles el orden. Por defecto, prefiere shuffle.

## 2.2 `KFold(shuffle=True, random_state=…)`

**Qué hace:** baraja las filas antes de cortar en `k` folds → asignación aleatoria.

```
  1) BARAJAR las filas:   [1 2 3 4 5 6 7 8] → [5 2 8 1 7 3 6 4]
  2) Cortar en k bloques sobre lo ya barajado:

  Fold 1:  ▓VAL▓  ░░░░░░░░░░░░░░░  → val = {5,2}   (filas mezcladas)
  Fold 2:  ░░░░  ▓VAL▓  ░░░░░░░░░  → val = {8,1}
  Fold 3:  ░░░░░░░░  ▓VAL▓  ░░░░░  → val = {7,3}
  Fold 4:  ░░░░░░░░░░░░  ▓VAL▓     → val = {6,4}

  ✓ rompe el sesgo por orden    ⚠ pero si hay estructura latente:
     tiempo  → mezcla pasado/futuro    → usar TimeSeriesSplit
     grupos  → misma entidad en train Y val → usar GroupKFold (leakage)
```

**Cuándo usar:**
- Datos **i.i.d.** sin estructura temporal ni de grupo.

**Riesgo / cuidado:**
- **Introduce leakage si hay estructura latente:**
  - **Temporal** → barajar mezcla pasado y futuro; usa `TimeSeriesSplit`.
  - **Agrupamiento** (varias filas por entidad: cliente, expediente, paciente) → filas del mismo grupo en train y val a la vez; usa `GroupKFold`.
- Fija `random_state` para reproducibilidad.

**Por qué es más robusto que `KFold` simple:** elimina el sesgo por orden de filas. Pero traslada la responsabilidad a verificar que **no exista** estructura temporal o de grupo.

## 2.3 `RepeatedKFold`

**Qué hace:** ejecuta el K-Fold completo **varias veces** (`n_repeats`), cada repetición con una **partición aleatoria distinta**. Total = `n_splits × n_repeats` estimaciones.

```
  n_splits=4, n_repeats=3  →  3 barajadas distintas × 4 folds = 12 scores

  Repeat 1 (barajada A):  [F1 F2 F3 F4]  → 4 scores ┐
  Repeat 2 (barajada B):  [F1 F2 F3 F4]  → 4 scores ├─→ 12 estimaciones
  Repeat 3 (barajada C):  [F1 F2 F3 F4]  → 4 scores ┘   → media ± std

  Cada repeat RE-BARAJA quién cae en cada fold (NO repite el mismo fold).

  Efecto sobre la estimación:
     1 sola partición:   MAE = 0.082   (¿suerte del reparto?)
     12 particiones:     MAE = 0.080 ± 0.006   ← media estable + dispersión
```

```
  Sesgo vs Varianza — qué mueve cada palanca:

     ▲ subir n_repeats  → ↓ VARIANZA del estimador   (lo que quieres)
     ▼ bajar k          → ↓ train por fold → ↑ SESGO  (NO ayuda)
                          (entrenas con menos datos de los del modelo final)
```

**Cuándo usar:**
- **Muestra pequeña** o resultados que varían mucho según cómo caiga la partición.
- Cuando necesitas una **media estable** y un `std` para medir dispersión.

**Riesgo / cuidado:**
- Reduce la **varianza** de la estimación, **no el sesgo**.
- Los training sets de distintas repeticiones se **solapan** → los scores están correlacionados → el `std` **subestima** la varianza real. Úsalo como termómetro relativo, no como intervalo de confianza formal.
- Coste lineal en `n_repeats`; caro si dentro de la CV hay selección/búsqueda pesada.

**Punto clave:** la varianza del estimador se controla **subiendo `n_repeats`**, no bajando `k`. Bajar `k` reduce el tamaño de entrenamiento → **aumenta el sesgo** (entrenas con menos datos de los que tendrá el modelo final).

## 2.4 `StratifiedKFold`

**Qué hace:** preserva en cada fold la **proporción de clases** (clasificación) o de un **estrato** definido.

```
  Población:  ████████████░░░░  (75% clase A ███ , 25% clase B ░)

  KFold normal:           StratifiedKFold:
  Fold 1: █████░  (mezcla   Fold 1: ███░   ┐
  Fold 2: ███████  azarosa, Fold 2: ███░   ├ cada fold conserva
  Fold 3: ░░██░░  proporción Fold 3: ███░   │ ~75/25 → representativo
  Fold 4: ██████  variable)  Fold 4: ███░   ┘

  Target CONTINUO (no hay clases) → estratificar por BINS de cuantiles:
     y ∈ [0,1]  →  pd.qcut(y, 5)  →  [Q1|Q2|Q3|Q4|Q5]
     cada fold cubre TODO el rango de avance, no solo una franja.

  ⚠ cardinalidad: cada estrato necesita ≥ k muestras.
     estrato con < k  →  fold degenerado / error  →  colapsar en "Other"
```

**Cuándo usar:**
- Clasificación, sobre todo con **clases desbalanceadas** (por defecto en los clasificadores de sklearn).
- Regresión donde el **target distribuye distinto por estrato** → estratificar sobre **bins por cuantiles** del target continuo (`pd.qcut(y, …)`), ya que no se puede estratificar sobre `y` continuo directamente.

**Riesgo / cuidado:**
- **Cardinalidad:** cada estrato necesita ≥ `k` muestras; estratos raros (combos de categóricas, niveles poco frecuentes) producen folds degenerados o errores. Mitiga colapsando niveles raros en `"Other"` o estratificando solo por la variable de menor cardinalidad/mayor relevancia.
- Combinar varios criterios de estrato (p. ej. `qcut(y) × categoría`) **multiplica** el número de estratos → fragmentación; úsalo solo si la cardinalidad lo aguanta.

**Por qué es más robusto que `KFold` shuffle:** garantiza folds **representativos de la distribución relevante**, reduciendo la varianza entre folds y haciendo las estimaciones comparables — especialmente cuando hay desbalance o heterogeneidad por estrato.

## 2.5 `RepeatedStratifiedKFold`

**Qué hace:** combina 2.3 + 2.4 → estratificación **repetida** con particiones aleatorias distintas.

```
  = StratifiedKFold  ×  n_repeats

  Repeat 1:  [estratificado] → folds que respetan ~75/25 (o qcut(y))
  Repeat 2:  [estratificado] → otra repartición, MISMA proporción
  Repeat 3:  [estratificado] → otra repartición, MISMA proporción
              │                          │
       REPRESENTATIVIDAD          +   ESTABILIDAD
       (cada fold equilibrado)        (media sobre muchas particiones)
```

**Cuándo usar:**
- Muestra pequeña **y** distribución heterogénea por estrato (el caso más exigente).
- Es la opción por defecto razonable cuando quieres a la vez **representatividad** (estratificación) y **estabilidad** (repeticiones).

**Riesgo / cuidado:** hereda los de ambos — control de cardinalidad de estratos y subestimación del `std` por solapamiento. Coste = `n_splits × n_repeats`.

## 2.6 Tabla comparativa

| Splitter                  | Asignación        | Garantía                 | Riesgo principal            | Mejor para                     |
| ------------------------- | ----------------- | ------------------------ | --------------------------- | ------------------------------ |
| `KFold`                   | bloques contiguos | ninguna                  | datos ordenados → sesgo     | filas ya aleatorias            |
| `KFold(shuffle)`          | aleatoria         | quita sesgo por orden    | leakage si hay tiempo/grupo | datos i.i.d.                   |
| `RepeatedKFold`           | aleatoria ×N      | media estable            | `std` subestimado; coste    | muestra pequeña                |
| `StratifiedKFold`         | preserva estrato  | folds representativos    | estratos con < k muestras   | clases/estratos desbalanceados |
| `RepeatedStratifiedKFold` | estrato ×N        | representativo + estable | cardinalidad + coste        | muestra pequeña + heterogénea  |

---

## Cuándo elegir qué — guía rápida combinada

```
  ¿QUÉ SPLITTER?  (cómo partir)

  ¿Hay orden temporal? ──sí──► TimeSeriesSplit
        │ no
  ¿Varias filas por entidad? ──sí──► GroupKFold
        │ no
  ¿Clases/estratos desbalanceados
   o target heterogéneo? ──sí──► Stratified(+qcut(y) si y continuo)
        │ no                          │
  KFold(shuffle=True)                 ▼
        │                       ¿muestra pequeña / estimación inestable?
        └──────────┬──────sí──► añadir n_repeats  → Repeated[Stratified]KFold
                   │ no
                   └─► versión simple

  ¿QUÉ FUNCIÓN?  (qué obtener)

  ¿Necesitas ver overfitting o varias métricas? ──sí──► cross_validate ★
  ¿Predicciones por muestra (plot / stacking)?  ──sí──► cross_val_predict
  ¿Solo un número rápido de validación?         ──sí──► cross_val_score
```

| Situación                                    | Splitter                                  | Función                                   |
| -------------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| Chequeo de una métrica, datos limpios        | `KFold(shuffle)`                          | `cross_val_score`                         |
| Diagnóstico de overfitting / multimétrica    | cualquiera apropiado                      | `cross_validate(return_train_score=True)` |
| Clasificación desbalanceada                  | `StratifiedKFold`                         | `cross_validate`                          |
| Regresión con target heterogéneo por estrato | `RepeatedStratifiedKFold` sobre `qcut(y)` | `cross_validate`                          |
| Muestra pequeña, estimación inestable        | subir `n_repeats` (Repeated*)             | `cross_validate`                          |
| Predicho-vs-real, residuos, stacking         | cualquiera                                | `cross_val_predict`                       |
| Serie temporal                               | `TimeSeriesSplit` (no K-Fold)             | `cross_validate`                          |
| Filas agrupadas por entidad                  | `GroupKFold` (no K-Fold)                  | `cross_validate`                          |

---

## Advertencias transversales

- **El número de reporte debe ser insesgado.** Si seleccionas features o ajustas hiperparámetros usando `y` sobre los mismos folds que luego reportas, el score es **optimista**. La medida limpia sale de un **test externo intocado** o de **nested CV**.
- **Pasos que usan `y` van dentro de un `Pipeline`** para que se reajusten por fold; aplicarlos antes del split filtra información de validación.
- **Estratificar ≠ agrupar.** Estratificar conserva proporciones; agrupar impide que un mismo identificador se reparta entre train y val. Para entidades repetidas, lo que manda es `GroupKFold`.
- **`std` entre repeticiones** subestima la varianza real (training sets solapados) → úsalo como dispersión relativa, no como IC formal.

---

## Referencias

1. scikit-learn — *Cross-validation: evaluating estimator performance* (User Guide, `model_selection`). https://scikit-learn.org/stable/modules/cross_validation.html
2. scikit-learn — `cross_val_score`, `cross_validate`, `cross_val_predict` (API Reference). https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
3. scikit-learn — Nota sobre `cross_val_predict`: apropiado para visualización, no para estimar métricas de generalización. https://scikit-learn.org/stable/modules/cross_validation.html#obtaining-predictions-by-cross-validation
4. scikit-learn — `KFold`, `RepeatedKFold`, `StratifiedKFold`, `RepeatedStratifiedKFold`, `GroupKFold`, `TimeSeriesSplit` (API Reference). https://scikit-learn.org/stable/modules/classes.html#splitter-classes
5. Kuhn, M. & Johnson, K. — *Feature Engineering and Selection: A Practical Approach for Predictive Models*. https://feat.engineering/ (cap. 3, proceso de modelado y resampling; evitar leakage dentro de la CV).
