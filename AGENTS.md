# AGENTS.md

## Commands
- **Install dependencies**: `uv sync`
- **Run main script**: `python main.py`
- **Run Jupyter notebooks**: `jupyter notebook` or `jupyter lab`

## Code Style Guidelines

### Python Standards
- Use Python 3.10+ features
- Follow PEP 8 naming conventions (snake_case for variables/functions, PascalCase for classes)
- Use type hints for function parameters and return values
- Import standard library first, then third-party packages, then local modules
- Group imports: standard library, blank line, third-party, blank line, local

### ML/Data Science Best Practices
- Use descriptive variable names for datasets (X_train, y_test, features_df)
- Document data preprocessing steps in comments or docstrings
- Handle missing values explicitly with imputation or removal
- Split data before preprocessing to avoid data leakage
- Use sklearn's train_test_split with random_state for reproducibility

### Error Handling
- Use try-except blocks for file I/O operations
- Validate data shapes and types before model training
- Log warnings for data quality issues rather than failing silently

### File Organization
- Keep data files in `data/` directory
- Use `.ipynb` for exploratory analysis, `.py` for production code
- Separate data ingestion, preprocessing, modeling, and evaluation into modules