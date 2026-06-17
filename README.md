# ds-knowledge



# Greenfield project
 Step 1: Configure development environment

* Configure Github Codespaces or the equivalent
* Create scaffold for structure of project: 'Makefile' 'requirements'
* Optional (setup virtualenv) ipython outside of requirements.txt)


# Brownfield project



# Preferred tech stack for data science
- Python 3
- FastAPI
- uv: package manager
- ruff: code formatter
- docker
- pytest: testing code
- click: command line tool
- mlflow: model and experiment versioning
- Pydantic: To force data types
- Makefile: File with instructions for install, lint, format, refactor (format + lint)
- logging: For creating logs instead of prints
- optuna
- shap
- xgboost
- other models depending on the problem (random forest, gaussian processes)

# Preferred behavior in coding
- Paths:  Use always relative paths instead of absolute paths for importing modules, reading  data, configs or parameter files.
- OOP vs functional: Avoid overengineering with object oriented programming methodology, try to keep it simple with the minimum amount of functions.
- Writing data: When we need to write datasets, parquet format is prefered vs csv or other format files
- Writing functions: Always add explicit arguments when calling a function (keyword naming, not postional only)
- Do not use sklearn.Pipeline. Explicit transformations and preprocessing is prefered

# Tests
Leave this part for the phase "Production". Not mandatory for other phases of the project (check section "Phases in machine learning projects")
1. Unit tests
2. Integration tests
3. Complete tests
4. Data tests


# Preferred project structure



# Phases in machine learning projects

**Developing Phase**
Developing format: Mainly in jupyter notebooks in the folder "notebooks/"
1. Exploration and data analysis
2. ETL Perimetro: Creating the joins and modification of data with the goal of create the sample for model training with typical id columns (reference date, id customer or id contract, segments, others)
3. ETL Features: Creation of new features o ready-to-go features from different sources, typically divided by families
4. ETL Target:
5. Modelling: feature analysis (univariate, bivariate), feature selection (drift, multivariate, hyperoptimization), 
6. Model performance evaluation

**Production Phase**
Developing format: code in folder of the same name of the project. Example: <repo_name>/<repo_name>/
1. ETL Train: Refactor of the code created in the developing phase to have a reproducible pipeline for the creation of the training/test dataset. Necessary to have an updated sample if needed based on a reference date and for future retraining processses.
2. ETL Inference: Creation of the inference population that is needed for model inference periodically.

**Specific observations for both phases**
- ETL's stages are typically developed in PySpark
- Rest of the phases are typically developed in python


# Any Decision Record

folder for writing in .md: docs/adr/

skill: use skill create-adr


# Memory

folder for writing in .md: docs/memory/


# Plans

Folder docs/

