# Arquitectura de MLflow: Modo Local Embebido (SQLite) vs. Tracking Server

Este documento consolida la arquitectura técnica, los componentes de almacenamiento, las diferencias de concurrencia y la matriz de pros y contras entre ejecutar MLflow con almacenamiento SQLite embebido frente a un MLflow Tracking Server dedicado.

---

## 1. Arquitectura de Almacenamiento en MLflow

MLflow separa estrictamente su capa de persistencia en dos componentes independientes:

```
+-------------------------------------------------------------+
|                      MLflow Run / Client                    |
+-------------------------------------------------------------+
          |                                        |
          v                                        v
+-----------------------+              +-----------------------+
|     Backend Store     |              |     Artifact Store    |
| (Metadatos relacionales) |              |  (Archivos binarios)  |
+-----------------------+              +-----------------------+
| - Run ID / Experiment |              | - Modelos (.pkl, etc) |
| - Parámetros          |              | - Curvas / Gráficos   |
| - Métricas y pasos    |              | - Tablas / Reportes   |
| - Tags y Alias        |              | - Pesos y Checkpoints |
+-----------------------+              +-----------------------+
```

1. **Backend Store:** Base de datos relacional (manejada internamente vía SQLAlchemy) responsable de indexar experimentos, ejecuciones, parámetros, métricas numéricas por paso y metadatos del Model Registry.
2. **Artifact Store:** Sistema de archivos o almacenamiento de objetos donde se escriben los binarios pesados (modelos serializados, artefactos de texto, imágenes de diagnóstico y datasets).

---

## 2. Los Dos Modos de Ejecución

### 2.1 Modo Local Embebido (`sqlite:///mlflow.db`)
* **Mecanismo:** El script de Python actúa como ejecutor y gestor al mismo tiempo. Mediante el driver interno de SQLite / SQLAlchemy, lee y escribe directamente en el archivo `.db` en el disco local.
* **Procesos:** No existe ningún servidor web, demonio o API REST activo en segundo plano.
* **Consumo de interfaz:** Si se desea ver la interfaz gráfica, se lanza de forma independiente mediante:
  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

### 2.2 Modo Tracking Server (`http://<host>:<port>`)
* **Mecanismo:** El script actúa como cliente HTTP. Cada llamada (`mlflow.log_param`, `mlflow.log_metric`, `mlflow.register_model`) realiza peticiones REST hacia un servidor central (típicamente Gunicorn/Uvicorn orquestando la aplicación MLflow).
* **Persistencia:** El servidor central se conecta a una base de datos relacional empresarial (PostgreSQL, MySQL) y a un repositorio de artefactos remoto (S3, GCS, Azure Blob Storage).
* **Comando de despliegue típico:**
  ```bash
  mlflow server \
      --backend-store-uri postgresql://mlflow_user:password@db-host:5432/mlflow_db \
      --default-artifact-root s3://mi-bucket-mlflow/artifacts \
      --host 0.0.0.0 \
      --port 5000
  ```

---

## 3. Comparativa Técnica Directa

| Dimensión | SQLite Embebido (`sqlite:///mlflow.db`) | Tracking Server Dedicado (`http://<host>:<port>`) |
| :--- | :--- | :--- |
| **Arquitectura** | In-process (driver directo al archivo de disco) | Cliente-Servidor (REST API + demonio MLflow) |
| **Concurrencia** | Bloqueo a nivel de archivo (*Database-level write lock*) | Concurrencia real con pooling de conexiones en RDBMS |
| **Almacenamiento de Artefactos** | Sistema de archivos local (`./mlruns`) | Proxied Artifact Storage (S3, GCS, Azure Blob, HDFS) |
| **Model Registry** | Aislado en la máquina del usuario | Centralizado, compartido entre equipos y CI/CD |
| **Infraestructura** | Cero procesos externos requeridos | Requiere máquina virtual, contenedor Docker o Kubernetes |
| **Autenticación / RBAC** | No soportado | Integración con HTTP Basic Auth, OIDC, OAuth |

---

## 4. Pros y Contras Detallados

### 4.1 SQLite Embebido (`sqlite:///mlflow.db`)

#### Pros
* **Cero sobrecarga operativa:** No requiere aprovisionar servidores, configurar redes, firewalls ni administrar servicios externos.
* **Autocontenido y transportable:** Todo el estado de metadatos se conserva en un único archivo físico `.db`, fácil de respaldar o copiar entre carpetas.
* **Latencia de I/O mínima:** Sin sobrecarga de red ni transporte HTTP; las operaciones de registro son inmediatas a nivel de disco.
* **Óptimo para prototipado individual:** Ideal para exploración local, pruebas de concepto (PoCs) y trabajo offline sin conexión a internet.

#### Contras
* **Bloqueos por concurrencia (`sqlite3.OperationalError: database is locked`):** Si múltiples workers de entrenamiento (ej. Ray, multiprocessing, o varios notebooks) intentan registrar datos al mismo tiempo, SQLite bloquea el archivo y falla.
* **Aislamiento de artefactos:** Los modelos y binarios quedan guardados localmente; no pueden ser consumidos por otros miembros del equipo sin transferencias manuales.
* **Inaccesible para pipelines de CI/CD / Producción:** Un runner remoto o cluster de inferencia no puede acceder al Model Registry sin montar el sistema de archivos físico.
* **Sin control de acceso ni auditoría:** Cualquier usuario o script con permisos sobre el archivo puede modificar o eliminar registros sin trazabilidad de autoría.

### 4.2 MLflow Tracking Server (`http://<server>:<port>`)

#### Pros
* **Concurrencia y escalabilidad de grado empresarial:** Soporta decenas de pipelines y workers concurrentes apoyándose en PostgreSQL o MySQL.
* **Model Registry corporativo unificado:** Punto único de verdad organizacional para auditar versiones, comparar modelos y gestionar alias de ciclo de vida (`@champion`, `@challenger`).
* **Proxy seguro de artefactos:** Permite delegar la carga y descarga de modelos a buckets en la nube sin exponer credenciales cloud directas en cada entorno de entrenamiento de los científicos de datos.
* **Integración nativa con CI/CD y Serving:** Pipelines de inferencia en producción y microservicios pueden resolver modelos mediante URIs estándar como `models:/mi_modelo@champion`.

#### Contras
* **Sobrecarga de infraestructura:** Requiere desplegar, monitorizar y costear cómputo para el servidor MLflow y una base de datos relacional administrada.
* **Latencia de red en telemetría:** Cada llamada (`log_metric`, `log_params`) ejecuta peticiones HTTP/HTTPS, lo que puede ralentizar bucles de entrenamiento si no se usa logging por lote.
* **Punto único de fallo por conectividad:** Si el servidor se apaga o la red se interrumpe, el pipeline de entrenamiento fallará a menos que se implemente tolerancia a desconexión manual.
* **Costos operativos:** Implica costos fijos de instancias cloud, base de datos externa y transferencia de red.

---

## 5. Matriz de Decisión

| Escenario de Proyecto | Arquitectura Recomendada | Razón Principal |
| :--- | :--- | :--- |
| Desarrollo individual, prototipos rápidos, modo offline | **SQLite Embebido** | Simplicidad, portabilidad y nula necesidad de infraestructura. |
| Hiperoptimización local masiva (e.g., Optuna en paralelo) | **PostgreSQL Local / Servidor** | SQLite colapsa por bloqueos concurrentes de escritura. |
| Equipo de Data Science con múltiples colaboradores | **Tracking Server Dedicado** | Visibilidad compartida, comparación cruzada y un solo Model Registry. |
| Pipelines automatizados de CI/CD y despliegue a producción | **Tracking Server Dedicado** | Consumo centralizado de modelos mediante alias (`@champion`). |