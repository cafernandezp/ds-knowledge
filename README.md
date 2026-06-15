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


# Preferred behavior in coding
-Paths:  Use always relative paths instead of absolute paths for importing modules, reading  data, configs or parameter files.
-OOP vs functional: Avoid overengineering with object oriented programming methodology, try to keep it simple with the minimum amount of functions.
- Writing data: When we need to write datasets, parquet format is prefered vs csv or other format files
- Writing functions: Always add explicit arguments when calling a function (keyword naming, not postional only)

# Tests
1. Unit tests
2. Integration tests
3. Complete tests
4. Data tests


# Preferred project structure



# Workflow for machine learning projects

1. Exploration
2. ETL: Creating the joins and modification of data with the goal of create the training dstaset
3. Modelling
4. Evaluation of inference population
5. Deployment

