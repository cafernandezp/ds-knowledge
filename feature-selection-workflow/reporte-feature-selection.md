# Reporte técnico — Metodología de Feature Selection (flujo corregido end-to-end)

> **Last update:** 2026-06-04
> **Proyecto:** Regresión de avance de procedimientos judiciales
> **Target:** `score` ∈ [0,1] — proporción de avance (continuo, acotado)
> **Datos:** ~6.000 casos; cada fila = un procedimiento judicial **único**; 365 features iniciales (numéricas, binarias 0/1, categóricas de baja y alta cardinalidad)
> **Modelo:** XGBoost (`reg:logistic`, `tree_method="hist"`, `enable_categorical=True`)
> **Métrica de negocio:** MAE en escala original

---

## 0. Resumen ejecutivo

Flujo por etapas consecutivas (las features que sobreviven pasan a la siguiente). El flujo original colapsaba a **1 sola variable**, síntoma de **leakage / proxy del target** combinado con una **cascada de selectores supervisados** que sobre-poda. Este reporte documenta el flujo corregido, las razones de cada cambio y los controles anti-leakage.

**Cambios principales frente al flujo original:**

1. **Test externo intocado** como único juez insesgado (antes: todo sobre la muestra completa).
2. **Pre-filtro no supervisado** ampliado: cubre también pares **numérica ↔ categórica** (antes: hueco sin filtrar).
3. **Umbral de Cramér's V a 0.7** (antes 0.9, inalcanzable por su cota) + corrección de sesgo por alta cardinalidad.
4. **Desempate homogéneo por MI con `y`** (antes: "mayor Spearman con target", indefinido para categóricas nominales).
5. **Un selector supervisado principal**, no tres en cascada (antes: Boruta + stability + backward apilados → colapso).
6. **`objective="reg:logistic"`** para respetar la cota [0,1]; MAE gestionado vía `scoring`.
7. **CV de diagnóstico** con `RepeatedStratifiedKFold` y `cross_validate(return_train_score=True)` para leer el overfitting (antes: `cross_val_score`, ciego al gap).

---

## 1. Datos, supuestos y restricciones

- **Sin componente temporal** relevante → no aplica `TimeSeriesSplit`.
- **Filas únicas** (un procedimiento por fila) → no hay agrupamiento que fuerce `GroupKFold`.
- **El target distribuye distinto por agencia y por tipo de procedimiento** → la partición debe ser estratificada.
- El split train/test original se **estratificó por `agencia × tipo_procedimiento`**.
- Decisión operativa: **no reajustar los selectores dentro de cada fold de CV** (coste prohibitivo). Consecuencia asumida en §4.3.

---

## 2. Diagnóstico del flujo original (problemas detectados)

| # | Problema | Severidad | Corrección |
|---|---|---|---|
| 1 | **Colapso a 1 variable** | Crítico | Investigar leakage/proxy antes de afinar nada (§3.1) |
| 2 | Cascada de **3 selectores supervisados** apilados → sobre-poda y composición de varianza | Alto | Un selector principal; resto opcional y ligero (§3.5–3.6) |
| 3 | **Sin test externo**: selección y evaluación sobre los mismos datos con `y` | Alto | Test intocado / nested CV (§3.0, §4.3) |
| 4 | Etapa 1 **no cruza numérica ↔ categórica** | Medio | Añadir η/η² o MI para esos pares (§3.2) |
| 5 | **Umbral Cramér's V = 0.9**: casi nunca se alcanza (V acotada por dimensiones de la tabla) → no elimina categóricas | Medio | Bajar a **V > 0.7** + corrección de sesgo (§3.2) |
| 6 | **Desempate "mayor Spearman con target"** indefinido para categórica nominal vs `y` | Medio | Métrica de relevancia **homogénea: MI con `y`** (§3.2, §5) |
| 7 | **`cross_val_score`** no permite ver overfitting | Medio | `cross_validate(return_train_score=True)` (§4.2) |
| 8 | **`scoring` por defecto = R²** en regresor | Bajo | Fijar `neg_mean_absolute_error` (§4.1) |
| 9 | **`objective` no acotado** puede predecir fuera de [0,1] | Bajo | `reg:logistic` (§4.1) |

---

## 3. Flujo corregido end-to-end

```
[0] Split test externo intocado (estratificado)
        │
[1] Pre-filtro NO supervisado  (sobre train)
     NZV · % missing · redundancia (num/bin · cat · num↔cat)
        │   desempate por MI con y
[2] Modelo base + diagnóstico overfitting (CV: train vs val gap)
        │
[3] UN selector supervisado principal  (Boruta-SHAP  O  stability)
        │
[4] (opcional) RFECV / backward ligero, puntuado por CV
        │
[5] Validación final en TEST EXTERNO  → número insesgado de reporte
```

### 3.0 Split del test externo

- Separar un test intocado **antes de todo**, estratificado por la misma lógica del split (`agencia × tipo_procedimiento`, o `qcut(y)` si la cardinalidad lo exige).
- **No se toca hasta §3.5.** Es el único juez insesgado del flujo completo, porque la selección usa `y` y eso contamina cualquier evaluación hecha sobre los mismos folds.

### 3.1 Cierre obligatorio del colapso a 1 variable (antes de continuar)

El colapso no es "selección agresiva": es la firma de una variable que **predice el target de forma casi determinista**.

- **Verificar sobre el sobreviviente:**
  - `|ρ_spearman(feature, y)|` ≈ 1, y/o
  - su `gain` / SHAP domina > 80 % de la importancia total.
- **Decidir disponibilidad temporal:** ¿esa variable existe en el **momento de predecir**? Si se construye a partir del avance (p. ej. `pasos_completados`, `etapa_actual`, una fecha que sigue monotónicamente el progreso) → **es leakage**, sacarla.
- Mientras no se descarte esto, el resto del tuning es secundario.

### 3.2 Pre-filtro NO supervisado (feature vs feature)

Elimina redundancia sin mirar `y` para reducir el espacio antes de los métodos caros. **Cubre todos los tipos de par:**

| Par | Medida | Umbral redundancia |
|---|---|---|
| Numérica ↔ numérica / binaria | **Spearman ρ** (robusto, monotónico) | \|ρ\| > 0.85–0.90 |
| Categórica ↔ categórica | **Cramér's V** (χ²) **con corrección de sesgo** (Bergsma) | **V > 0.7** |
| **Numérica ↔ categórica** *(hueco del flujo original)* | **η/η²** (ANOVA) o **MI** | η > 0.7 / η² > 0.5 |

Pasos previos: **NZV / varianza casi cero**, **% missing alto** (drop si > 50 %).

**Por qué los umbrales difieren:** Spearman/Pearson escalan con la varianza compartida (ρ=0.8 → 64 % compartido), mientras que **Cramér's V está acotada por las dimensiones de la tabla** y rara vez se acerca a 1 → V=0.7 ya implica dependencia fuerte. Por eso 0.9 era inalcanzable y no eliminaba nada.

**Cuidado — alta cardinalidad infla Cramér's V** → usar la versión con corrección de sesgo y construir la tabla con `pd.crosstab(..., dropna=False)` (NaN como categoría suele ser señal).

### 3.3 Regla de desempate (cuál se queda del par redundante)

Aplicada en orden (detalle y matices en §5):

1. Mayor **relevancia a `y`** → medida **homogénea: MI con `y`** (`mutual_info_regression`), válida para todos los tipos. (No "mayor Spearman", que no existe para nominales.)
2. Empate → **menor % de missing**.
3. Empate → **más interpretable / más barata** de obtener.
4. Si una es **derivada** de la otra → drop la derivada (salvo señal no-lineal clara).

> **Matiz importante:** el desempate es **univariado**. Puede descartar una feature que solo aporta **en combinación** con otras. Aceptable como pre-filtro barato; el selector supervisado de §3.5 recupera parte de esto.

### 3.4 Modelo base + diagnóstico de overfitting

- Modelo base regularizado (ver ADR de parámetros): `reg:logistic`, `hist`, `enable_categorical`, presupuesto fijo (`n_estimators=600`, `lr=0.03`), regularización conservadora.
- CV con **`cross_validate(return_train_score=True)`** → leer el **gap train–val** (única forma de ver el sobreajuste; `cross_val_score` no lo permite).
- Esquema de CV: §4.2.

### 3.5 Selector supervisado principal (UNO, no tres)

Apilar Boruta + stability + backward compone varianza y sobre-poda → **contribuye directamente al colapso a 1 variable**. Elegir **uno**:

- **Boruta-SHAP** — crea *shadow features* (permutaciones) y conserva las features consistentemente más importantes que la mejor shadow. Potente, pero **caro** con XGBoost.
- **Stability selection** — subsamplea muchas veces, ajusta el selector y retiene las features elegidas por encima de un umbral de frecuencia (p. ej. 60 % de las corridas). Reduce la varianza de selección.

**Alternativa más robusta a un embudo estricto:** combinar selectores por **votación** (unión/intersección con umbral) en vez de pasarlos en cascada.

### 3.6 Ajuste fino opcional

- **RFECV** o backward ligero, **puntuado por CV** sobre el train. RFECV suele ser **más barato** que un backward manual si quedan muchos survivors.
- Mantener ligero: ya hubo un selector principal en §3.5.

### 3.7 Validación final

- Ajustar el modelo con las features finales sobre **todo el train** y evaluar **una sola vez** en el **test externo** (§3.0).
- Comparar contra el modelo base: el set reducido debe mantener (o mejorar) el MAE con menos features.

---

## 4. Decisiones transversales

### 4.1 Objetivo y métrica

- **`objective="reg:logistic"`** — salida acotada a (0,1) por sigmoide interna; pérdida logística adecuada a una proporción (penaliza más los bordes, donde la varianza se encoge). `reg:squarederror` puede predecir fuera de [0,1].
- **MAE vive en `scoring="neg_mean_absolute_error"`** (selección + reporte), **no** en el objetivo ni en `eval_metric`.
- **`eval_metric` es cosmético sin early stopping**: no toca gradientes ni corta iteraciones. Solo contaría con `eval_set` + early stopping (loop manual por fold).
- Negativo en `neg_mean_absolute_error` = convención de sklearn ("mayor = mejor"); se recupera con `-res["test_score"]`.
- **Comparación futura:** `objective="reg:absoluteerror"` optimiza MAE directamente (L1) pero **pierde la cota [0,1]** y es menos estable (hessiano constante). Trade-off: cota física vs alineación con la métrica; por defecto gana la cota.

### 4.2 Esquema de validación cruzada (diagnóstico)

- **`RepeatedStratifiedKFold`**, **`n_splits=4`, `n_repeats=8`**, `random_state=42`.
- **Estratificar por `qcut(y, 5)`** (bins por cuantiles del target continuo) — no se puede estratificar sobre `y` continuo directamente; combinar con `tipo_procedimiento` **solo si** la cardinalidad lo aguanta.
- **Varianza controlada con `n_repeats`, no bajando `k`.** Bajar `k` reduce el train → **aumenta el sesgo** (entrenas con menos datos de los que tendrá el modelo final).
- **Diagnóstico rápido:** `CV% = std/media` (estabilidad global) y `z-score` por fold (folds atípicos).
- `cross_validate(return_train_score=True)` para el gap train–val.

### 4.3 Selección fuera del bucle de CV (decisión operativa)

- La selección se ejecuta **una vez sobre todo el train**; la CV evalúa **solo el modelo** sobre el set fijo → `pipe = model`.
- **Consecuencia asumida:** la CV-MAE queda **optimista** como medida de generalización del pipeline completo (los val-folds fueron vistos por los selectores). Por eso el **número de reporte sale del test externo**, no de la CV.
- La CV sigue siendo **válida** para: diagnóstico de overfitting (gap del mismo set fijo), tuning y comparación **relativa** entre configuraciones.
- Lo **innegociable**: la selección **nunca** toca el test externo.
- La alternativa insesgada del pipeline completo sin gastar test es **nested CV** (re-ejecutar selección en el bucle externo) — descartada por coste. Con 6.000 casos, el test externo es la opción sensata.

---

## 5. Regla de drop de pares redundantes (detalle)

Cuando una variable está correlacionada con varias por encima del umbral:

1. **Relevancia a `y`** (quedarse con la más relevante) — medida con **MI con `y`** para que el criterio sea **homogéneo entre tipos**. Spearman-con-target solo aplica a numéricas/ordinales; MI cubre también nominales.
2. **% missing** (quedarse con la de menos faltantes).
3. **Interpretabilidad / coste** (quedarse con la más clara o barata).
4. **Derivación** (drop la derivada de la otra, salvo señal no-lineal añadida).

**Cuidado adicional:**
- El criterio es **univariado** → riesgo de tirar features útiles solo en interacción. Mitigado en parte por el selector multivariado de §3.5.
- **VIF** complementa el análisis por pares: dos features con `|r|` baja pueden generar **multicolinealidad** con una tercera. Revisar VIF **después** del filtrado por pares; iterar (drop mayor VIF, refit) hasta VIF < 5–10.
- **Siempre visualizar** antes de decisiones finales (Anscombe: misma correlación, formas radicalmente distintas).

---

## 6. Implementación de referencia

```python
import numpy as np, pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb

RANDOM_STATE = 42

# =========================================================
# [0] SPLIT TEST EXTERNO INTOCADO (estratificado)  → fuera de este script
#     X_train, X_test, y_train, y_test ya separados
# =========================================================

# =========================================================
# [1] PRE-FILTRO NO SUPERVISADO (feature vs feature, sobre train)
# =========================================================
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# 1a) NZV y % missing
nzv  = [c for c in num_cols if X_train[c].nunique() <= 1]
miss = X_train.columns[X_train.isna().mean() > 0.50].tolist()
drop0 = set(nzv) | set(miss)

# Imputación para los cálculos de redundancia (no contamina el modelo XGBoost)
num_imp = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X_train[num_cols]),
    columns=num_cols, index=X_train.index)
cat_imp = X_train[cat_cols].fillna("Missing")

# 1b) num/bin ↔ num/bin: Spearman
rho = num_imp.corr(method="spearman", min_periods=30).abs()
pairs_num = (rho.where(np.triu(np.ones(rho.shape), 1).astype(bool))
                .stack().reset_index())
pairs_num.columns = ["a", "b", "rho"]
red_num = pairs_num[pairs_num["rho"] > 0.85]

# 1c) cat ↔ cat: Cramér's V con corrección de sesgo (Bergsma)
def cramers_v_corrected(x, y):
    cm = pd.crosstab(x, y, dropna=False)
    chi2 = stats.chi2_contingency(cm)[0]
    n = cm.sum().sum()
    phi2 = chi2 / n
    r, k = cm.shape
    phi2c = max(0, phi2 - (k-1)*(r-1)/(n-1))
    rc = r - (r-1)**2/(n-1)
    kc = k - (k-1)**2/(n-1)
    return np.sqrt(phi2c / max(min(kc-1, rc-1), 1e-12))

red_cat = [(a, b, cramers_v_corrected(cat_imp[a], cat_imp[b]))
           for i, a in enumerate(cat_cols) for b in cat_cols[i+1:]]
red_cat = [(a, b, v) for a, b, v in red_cat if v > 0.7]

# 1d) num ↔ cat: eta² (ANOVA)  [hueco del flujo original]
def eta_squared(cat, cont):
    groups = [cont[cat == c] for c in cat.unique()]
    gm = cont.mean()
    ss_b = sum(len(g)*(g.mean()-gm)**2 for g in groups)
    ss_t = ((cont-gm)**2).sum()
    return ss_b/ss_t if ss_t > 0 else 0.0

red_mix = [(c, n, eta_squared(cat_imp[c], num_imp[n]))
           for c in cat_cols for n in num_cols]
red_mix = [(c, n, e) for c, n, e in red_mix if e > 0.5]

# 1e) DESEMPATE homogéneo: MI con y  (válido para todos los tipos)
mi = pd.Series(
    mutual_info_regression(num_imp, y_train, random_state=RANDOM_STATE),
    index=num_cols)
# (para categóricas: encode + mutual_info_regression, o mutual_info_classif sobre bins de y)
# Regla: del par redundante, quedarse con la de mayor MI; empate → menor %missing → más barata

# ... aplicar reglas §3.3/§5 para construir `selected_features` ...

# =========================================================
# [2] MODELO BASE  (ADR de parámetros)
# =========================================================
model = xgb.XGBRegressor(
    objective="reg:logistic", eval_metric="mae",
    tree_method="hist", enable_categorical=True,
    n_estimators=600, learning_rate=0.03,
    max_depth=4, min_child_weight=5, gamma=0.0,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.0,
    random_state=RANDOM_STATE, n_jobs=-1,
)

# =========================================================
# [3-4] SELECCIÓN SUPERVISADA (una vez, sobre train)
#       UN selector principal (Boruta-SHAP O stability) + RFECV ligero opcional
# =========================================================
# selected_features = run_supervised_selection(X_train[surv], y_train)   # placeholder

# =========================================================
# CV DE DIAGNÓSTICO (evalúa SOLO el modelo sobre features fijas)
# =========================================================
X_tr_sel = X_train[selected_features]
strat = pd.qcut(y_train, 5, labels=False, duplicates="drop").astype(str)
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=8, random_state=RANDOM_STATE)

res = cross_validate(
    model, X_tr_sel, y_train,
    cv=list(cv.split(X_tr_sel, strat)),
    scoring="neg_mean_absolute_error",
    return_train_score=True, n_jobs=-1,
)
mae = -res["test_score"]; m, s = mae.mean(), mae.std()
gap = m + res["train_score"].mean()           # >0 grande = overfitting
z = (mae - m) / s
print(f"MAE val={m:.4f} ± {s:.4f} (CV {100*s/m:.1f}%)  "
      f"train={-res['train_score'].mean():.4f}  gap={gap:.4f}")
print(f"folds atípicos (|z|>2): {np.where(np.abs(z) > 2)[0]}")

# =========================================================
# [5] NÚMERO FINAL INSESGADO: TEST EXTERNO, UNA SOLA VEZ
# =========================================================
final_mae = mean_absolute_error(
    y_test, model.fit(X_tr_sel, y_train).predict(X_test[selected_features]))
print(f"MAE test externo = {final_mae:.4f}")
```

> El bloque de Boruta-SHAP / stability se deja como `placeholder` porque su API depende de la librería elegida; el contrato es: **recibe `X_train[survivors], y_train` y devuelve la lista final de columnas**, sin tocar el test externo.

---

## 7. Checklist anti-leakage y riesgos residuales

**Checklist:**
- [ ] Test externo separado **antes** de cualquier paso con `y`, y **cerrado** hasta §3.5.
- [ ] Colapso a 1 variable investigado (ρ≈1 / SHAP dominante / disponibilidad temporal) **antes** de afinar.
- [ ] Pre-filtro cubre los **tres tipos de par** (num, cat, num↔cat).
- [ ] Cramér's V con **corrección de sesgo** y umbral **0.7**.
- [ ] Desempate por **MI con `y`** (homogéneo), no por Spearman-con-target.
- [ ] **Un** selector supervisado principal, no tres en cascada.
- [ ] `objective="reg:logistic"`; MAE en `scoring`.
- [ ] CV con `return_train_score=True` para leer el gap.
- [ ] Número de reporte = **test externo**, no la CV.

**Riesgos residuales:**
- **CV optimista** del pipeline completo (selección fuera del bucle) → mitigado por el test externo, no eliminado.
- **Desempate univariado** puede descartar features útiles solo en combinación.
- **`std` entre repeats** subestima la varianza real (training sets solapados) → termómetro, no IC.
- **Estratos raros** al combinar `qcut(y)` con categóricas → colapsar niveles en `"Other"`.
- **Desajuste objetivo↔métrica** (log-loss vs MAE) → leve; revisar con `reg:absoluteerror` si MAE manda.

---

## 8. Referencias

1. Kuhn, M. & Johnson, K. — *Feature Engineering and Selection: A Practical Approach for Predictive Models*. https://feat.engineering/ (cap. 3 proceso de modelado y leakage; cap. 10–12 selección de features).
2. scikit-learn — *Cross-validation* y *Feature selection* (User Guide). https://scikit-learn.org/stable/modules/cross_validation.html · https://scikit-learn.org/stable/modules/feature_selection.html
3. scikit-learn — `RepeatedStratifiedKFold`, `cross_validate`, `mutual_info_regression` (API Reference). https://scikit-learn.org/stable/modules/classes.html
4. XGBoost — *Learning Task Parameters* (`reg:logistic`, `reg:absoluteerror`, `reg:squarederror`). https://xgboost.readthedocs.io/en/stable/parameter.html
5. Bergsma, W. (2013) — *A bias correction for Cramér's V and Tschuprow's T*. (Corrección de sesgo de Cramér's V para alta cardinalidad.)
6. Documentos internos del proyecto: *ADR — Configuración del modelo base XGBoost*; *ADR — Estrategia de validación cruzada*; *Bivariate Association Cheatsheet*.
