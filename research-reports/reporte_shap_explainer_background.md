# SHAP Explainer, Background Data y reutilización en datos nuevos

## 1. Idea principal

En SHAP conviene separar dos conceptos:

-   **`explainer`**: objeto configurado para calcular explicaciones SHAP
    de un modelo.
-   **`shap_values`**: explicaciones ya calculadas para un conjunto
    concreto de observaciones.

Ejemplo:

``` python
import shap

explainer = shap.Explainer(model, X_background)
shap_values = explainer(X_nuevo)
```

La interpretación es:

> Usa `X_background` como población de referencia y explica las
> predicciones que el modelo genera para `X_nuevo`.

Por tanto:

-   `X_background` = **contra qué población explico**.
-   `X_nuevo` = **qué observaciones quiero explicar**.

------------------------------------------------------------------------

## 2. ¿Qué es `shap.Explainer`?

Cuando hacemos:

``` python
explainer = shap.Explainer(model, X_background)
```

no estamos entrenando un segundo modelo.

El modelo predictivo ya está entrenado:

``` python
model.fit(X_train, y_train)
```

`shap.Explainer` construye/configura un mecanismo para explicar las
predicciones de ese modelo. Dependiendo del modelo y del algoritmo SHAP
utilizado, el background puede ser necesario para establecer una
distribución de referencia.

Conceptualmente, SHAP descompone una predicción como:

\[ f(x) = E\[f(X)\] + `\sum`{=tex}\_{j=1}\^{p} `\phi`{=tex}\_j \]

donde:

-   (f(x)): predicción del modelo para una observación.
-   (E\[f(X)\]): **base value**, asociado a la población de referencia.
-   (`\phi`{=tex}\_j): contribución SHAP de la variable (j).

Ejemplo conceptual:

``` text
Base value                  0.50
antiguedad                 +0.18
tipo_procedimiento         +0.09
cuantia                    +0.07
otras variables            -0.03
                           -----
Predicción                  0.81
```

SHAP explica cómo se pasa de una predicción de referencia a la
predicción concreta del caso.

------------------------------------------------------------------------

## 3. ¿Para qué sirve reutilizar el `explainer`?

Supongamos que hemos definido:

``` python
explainer = shap.Explainer(model, X_background)
```

Podemos utilizar la misma referencia para explicar diferentes datasets:

``` python
shap_test = explainer(X_test)
shap_agosto = explainer(X_agosto)
shap_septiembre = explainer(X_septiembre)
shap_nuevo = explainer(X_nuevo)
```

Esto permite explicar observaciones que **no existían cuando se
construyó el modelo o cuando se realizó el análisis SHAP inicial**.

### Usos principales

**Explicabilidad individual**

``` python
shap_caso = explainer(X_caso)
shap.plots.waterfall(shap_caso[0])
```

Permite responder:

> ¿Qué variables hicieron que este expediente obtuviera este score?

**Explicabilidad de nuevos batches**

``` python
shap_mes = explainer(X_mes)
```

Permite analizar qué variables están impulsando las predicciones en una
nueva cartera o período.

**Monitoring**

Manteniendo una referencia fija pueden compararse las contribuciones del
modelo entre períodos:

``` text
SHAP test
   vs.
SHAP agosto
   vs.
SHAP septiembre
```

Esto complementa el análisis tradicional de drift en las variables.

**Auditoría y trazabilidad**

Si posteriormente se necesita explicar una predicción histórica, pueden
recuperarse las features del caso y calcular sus SHAP values utilizando
la misma configuración de referencia.

------------------------------------------------------------------------

## 4. ¿Por qué no usar siempre `X_nuevo` como background?

Podríamos hacer:

``` python
explainer = shap.Explainer(model, X_nuevo)
shap_values = explainer(X_nuevo)
```

Esto **no es necesariamente incorrecto**, pero responde a una pregunta
diferente.

### Background fijo

``` python
explainer = shap.Explainer(model, X_background_train)
shap_values = explainer(X_nuevo)
```

Pregunta:

> ¿Por qué el modelo produce estas predicciones para `X_nuevo` respecto
> a una población de referencia estable?

### Background = datos nuevos

``` python
explainer = shap.Explainer(model, X_nuevo)
shap_values = explainer(X_nuevo)
```

La explicación pasa a estar referenciada al propio batch nuevo:

> ¿Por qué cada observación se diferencia de la referencia definida por
> este nuevo conjunto?

Puede ser útil si esa es explícitamente la pregunta de negocio, pero
tiene una desventaja importante: **el baseline puede cambiar cada vez
que cambia el batch**.

------------------------------------------------------------------------

## 5. Ejemplo: por qué importa mantener fijo el background

Supongamos que el modelo tiene estas predicciones medias:

``` text
Train / referencia     0.50
Agosto                 0.70
Septiembre             0.35
```

Si hacemos:

``` python
explainer_ago = shap.Explainer(model, X_agosto)
explainer_sep = shap.Explainer(model, X_septiembre)
```

la referencia utilizada para explicar agosto puede ser diferente de la
utilizada para septiembre.

Conceptualmente:

``` text
Agosto
baseline ≈ 0.70
    ↓
SHAP relativos a agosto

Septiembre
baseline ≈ 0.35
    ↓
SHAP relativos a septiembre
```

Estamos moviendo el punto de referencia.

En cambio:

``` python
explainer = shap.Explainer(model, X_background_train)

shap_ago = explainer(X_agosto)
shap_sep = explainer(X_septiembre)
```

obtenemos:

``` text
                 referencia fija
                     ≈ 0.50
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Agosto             Septiembre
          SHAP                  SHAP
```

Ambos períodos se explican respecto a la misma referencia.

Esto hace mucho más consistente la interpretación y comparación
temporal.

------------------------------------------------------------------------

## 6. ¿Debe utilizarse todo `X_train` como background?

No necesariamente.

Una alternativa práctica es seleccionar una muestra representativa del
train:

``` python
X_background = shap.sample(
    X_train,
    1000,
    random_state=42
)

explainer = shap.Explainer(
    model,
    X_background
)
```

El tamaño adecuado depende del tipo de explainer, modelo, volumen de
datos y coste computacional.

### Ventajas

-   Mantiene una **referencia estable**.
-   Representa la población utilizada para desarrollar el modelo.
-   Reduce el coste computacional frente a utilizar todo el train cuando
    el algoritmo depende del background.
-   Evita definir la referencia utilizando datos futuros.
-   Facilita reproducibilidad y comparaciones entre períodos.

La muestra debe ser **representativa**. Si existen segmentos de negocio
relevantes, conviene verificar que no queden infrarrepresentados.

------------------------------------------------------------------------

## 7. Arquitectura recomendada

Una estructura conceptual robusta sería:

``` text
Modelo final
│
├── model
├── feature_list
│
└── configuración SHAP
      │
      └── X_background fijo
              │
              ├── X_test       → SHAP_test
              ├── X_agosto     → SHAP_agosto
              ├── X_septiembre → SHAP_septiembre
              └── X_nuevo      → SHAP_nuevo
```

La regla es:

\[ `\boxed{\text{mismo modelo} + \text{misma referencia} \Rightarrow
\text{explicaciones más consistentes y comparables}}`{=tex} \]

------------------------------------------------------------------------

## 8. ¿Qué conviene almacenar?

Para un pipeline reproducible, priorizaría guardar:

``` text
artifacts/
├── model.pkl
├── features.json
├── shap_background.parquet
└── shap_values_test.pkl       # opcional
```

### Modelo

Es el artefacto fundamental para reproducir las predicciones.

### Lista y orden de features

Debe preservarse exactamente la estructura de entrada esperada por el
modelo.

### Background SHAP

Permite reconstruir posteriormente el mismo esquema de explicación:

``` python
model = ...
X_background = ...

explainer = shap.Explainer(model, X_background)
```

### SHAP values ya calculados

Si calcular SHAP es costoso, puede ser útil almacenar:

``` python
shap_values_test = explainer(X_test)
```

y reutilizarlos posteriormente para análisis y visualizaciones sin
recalcularlos.

------------------------------------------------------------------------

## 9. ¿Guardar el `explainer` o reconstruirlo?

También es posible serializar el objeto:

``` python
import joblib

joblib.dump(explainer, "shap_explainer.pkl")
explainer = joblib.load("shap_explainer.pkl")
```

Sin embargo, para reproducibilidad a largo plazo suele ser más robusto
conservar los componentes necesarios para reconstruirlo:

-   modelo;
-   features y su orden;
-   background;
-   configuración relevante;
-   versiones de las librerías.

Los objetos Python serializados pueden ser sensibles a cambios de
versiones de `shap`, `numpy`, `scikit-learn`, `xgboost`, etc.

Por tanto, guardar el `explainer` puede ser una **optimización
operativa**, pero no debería ser el único mecanismo de reproducibilidad.

------------------------------------------------------------------------

## 10. Regla práctica

### Quiero explicar `X_test`

``` python
explainer = shap.Explainer(model, X_background)
shap_test = explainer(X_test)
```

### Mañana quiero explicar nuevos datos

``` python
shap_new = explainer(X_new)
```

o reconstruyo el explainer con el mismo background:

``` python
explainer = shap.Explainer(model, X_background)
shap_new = explainer(X_new)
```

### Quiero comparar períodos

Mantengo fijo:

``` python
model
X_background
feature schema
```

y cambio únicamente el dataset que quiero explicar:

``` python
shap_ago = explainer(X_ago)
shap_sep = explainer(X_sep)
```

### Quiero explicar cada batch respecto a sí mismo

Entonces sí puede tener sentido:

``` python
explainer = shap.Explainer(model, X_nuevo)
shap_values = explainer(X_nuevo)
```

pero debe hacerse conscientemente, porque **la referencia cambia entre
batches** y las explicaciones dejan de tener un baseline común.

------------------------------------------------------------------------

## Conclusión

La distinción fundamental es:

> **El background define la referencia; el dataset pasado al explainer
> define qué queremos explicar.**

Para un modelo productivo donde interesa analizar test, nuevos
expedientes, períodos y evolución temporal, una estrategia sólida es
mantener un **background fijo y representativo del universo de
desarrollo**, y reutilizar esa referencia para calcular SHAP sobre los
diferentes datasets.

Usar `X_nuevo` simultáneamente como background y como dataset explicado
no es automáticamente más robusto: simplemente cambia la pregunta que
SHAP está respondiendo.
