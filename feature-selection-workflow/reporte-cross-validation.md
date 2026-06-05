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

## 1.4 Tabla comparativa

| | `cross_val_score` | `cross_validate` | `cross_val_predict` |
|---|---|---|---|
| Devuelve | scores de test (1 métrica) | dict: test (+train, tiempos, estimadores), multimétrica | predicciones out-of-fold |
| Ve overfitting (gap) | ❌ | ✅ (`return_train_score`) | ❌ |
| Varias métricas | ❌ | ✅ | n/a |
| Estima generalización | ✅ (limitado) | ✅ (recomendado) | ❌ **no usar para esto** |
| Uso típico | chequeo rápido | diagnóstico + reporte | visualización / stacking |

---

# Sección 2 — Estrategias de partición (splitters K-Fold)

Todos parten los datos en `k` folds y rotan cuál es validación. Difieren en **cómo asignan las filas** y en **qué garantizan**.

## 2.1 `KFold` (sin shuffle)

**Qué hace:** divide en `k` **bloques contiguos** según el orden de las filas.

**Cuándo usar:**
- Cuando el orden de las filas **ya es aleatorio** o irrelevante.

**Riesgo / cuidado:**
- Si el DataFrame viene **ordenado** (por fecha, por id, por categoría), los bloques quedan **no representativos** → estimación sesgada. Es el fallo silencioso más común.

**Veredicto:** rara vez la mejor opción salvo que controles el orden. Por defecto, prefiere shuffle.

## 2.2 `KFold(shuffle=True, random_state=…)`

**Qué hace:** baraja las filas antes de cortar en `k` folds → asignación aleatoria.

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

**Cuándo usar:**
- Clasificación, sobre todo con **clases desbalanceadas** (por defecto en los clasificadores de sklearn).
- Regresión donde el **target distribuye distinto por estrato** → estratificar sobre **bins por cuantiles** del target continuo (`pd.qcut(y, …)`), ya que no se puede estratificar sobre `y` continuo directamente.

**Riesgo / cuidado:**
- **Cardinalidad:** cada estrato necesita ≥ `k` muestras; estratos raros (combos de categóricas, niveles poco frecuentes) producen folds degenerados o errores. Mitiga colapsando niveles raros en `"Other"` o estratificando solo por la variable de menor cardinalidad/mayor relevancia.
- Combinar varios criterios de estrato (p. ej. `qcut(y) × categoría`) **multiplica** el número de estratos → fragmentación; úsalo solo si la cardinalidad lo aguanta.

**Por qué es más robusto que `KFold` shuffle:** garantiza folds **representativos de la distribución relevante**, reduciendo la varianza entre folds y haciendo las estimaciones comparables — especialmente cuando hay desbalance o heterogeneidad por estrato.

## 2.5 `RepeatedStratifiedKFold`

**Qué hace:** combina 2.3 + 2.4 → estratificación **repetida** con particiones aleatorias distintas.

**Cuándo usar:**
- Muestra pequeña **y** distribución heterogénea por estrato (el caso más exigente).
- Es la opción por defecto razonable cuando quieres a la vez **representatividad** (estratificación) y **estabilidad** (repeticiones).

**Riesgo / cuidado:** hereda los de ambos — control de cardinalidad de estratos y subestimación del `std` por solapamiento. Coste = `n_splits × n_repeats`.

## 2.6 Tabla comparativa

| Splitter | Asignación | Garantía | Riesgo principal | Mejor para |
|---|---|---|---|---|
| `KFold` | bloques contiguos | ninguna | datos ordenados → sesgo | filas ya aleatorias |
| `KFold(shuffle)` | aleatoria | quita sesgo por orden | leakage si hay tiempo/grupo | datos i.i.d. |
| `RepeatedKFold` | aleatoria ×N | media estable | `std` subestimado; coste | muestra pequeña |
| `StratifiedKFold` | preserva estrato | folds representativos | estratos con < k muestras | clases/estratos desbalanceados |
| `RepeatedStratifiedKFold` | estrato ×N | representativo + estable | cardinalidad + coste | muestra pequeña + heterogénea |

---

## Cuándo elegir qué — guía rápida combinada

| Situación | Splitter | Función |
|---|---|---|
| Chequeo de una métrica, datos limpios | `KFold(shuffle)` | `cross_val_score` |
| Diagnóstico de overfitting / multimétrica | cualquiera apropiado | `cross_validate(return_train_score=True)` |
| Clasificación desbalanceada | `StratifiedKFold` | `cross_validate` |
| Regresión con target heterogéneo por estrato | `RepeatedStratifiedKFold` sobre `qcut(y)` | `cross_validate` |
| Muestra pequeña, estimación inestable | subir `n_repeats` (Repeated*) | `cross_validate` |
| Predicho-vs-real, residuos, stacking | cualquiera | `cross_val_predict` |
| Serie temporal | `TimeSeriesSplit` (no K-Fold) | `cross_validate` |
| Filas agrupadas por entidad | `GroupKFold` (no K-Fold) | `cross_validate` |

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
