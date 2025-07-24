# Bangladesh Macroeconomic Models - Development Guide

This guide provides comprehensive information for developers working on the Bangladesh Macroeconomic Models simulation project.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [Development Workflow](#development-workflow)
5. [Testing Strategy](#testing-strategy)
6. [Code Quality Standards](#code-quality-standards)
7. [Configuration Management](#configuration-management)
8. [Performance Guidelines](#performance-guidelines)
9. [Documentation Standards](#documentation-standards)
10. [Troubleshooting](#troubleshooting)

## Project Overview

This project implements a comprehensive suite of macroeconomic models specifically designed for Bangladesh's economy, including:

- **DSGE Models**: Dynamic Stochastic General Equilibrium models
- **CGE Models**: Computable General Equilibrium models
- **ABM Models**: Agent-Based Models
- **Time Series Models**: SVAR, VAR, and other econometric models
- **Game Theory Models**: Strategic interaction models
- **Financial Models**: Banking and financial sector models

### Key Features

- Modular architecture for easy extension
- Comprehensive testing framework
- Performance monitoring and optimization
- Configuration management system
- Automated CI/CD pipeline
- Extensive documentation

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd BD_macro_models_sim
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .[dev]  # Install with development dependencies
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify installation**:
   ```bash
   pytest tests/ -v
   ```

### Development Dependencies

The project includes several categories of dependencies:

- **Core**: NumPy, Pandas, SciPy for numerical computation
- **Statistical**: Statsmodels, Scikit-learn for econometric analysis
- **Optimization**: CVXPY, Pyomo for mathematical programming
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Development**: Pytest, Black, Flake8, MyPy
- **Documentation**: Sphinx, MkDocs

## Project Structure

```
BD_macro_models_sim/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── dsge_model.py
│   │   ├── cge_model.py
│   │   ├── abm_model.py
│   │   └── ...
│   ├── analysis/                 # Analysis modules
│   │   ├── forecasting.py
│   │   ├── policy_analysis.py
│   │   └── validation.py
│   ├── utils/                    # Utility modules
│   │   ├── data_processing.py
│   │   ├── logging_config.py
│   │   ├── error_handling.py
│   │   └── performance_monitor.py
│   ├── config/                   # Configuration management
│   │   ├── config_manager.py
│   │   └── __init__.py
│   └── scripts/                  # Automation scripts
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── validation/               # Validation tests
├── config/                       # Configuration files
│   └── default.yaml
├── data/                         # Data directory
├── outputs/                      # Output directory
├── docs/                         # Documentation
├── .github/workflows/            # CI/CD workflows
├── requirements.txt              # Dependencies
├── pyproject.toml               # Project configuration
├── pytest.ini                  # Test configuration
└── .pre-commit-config.yaml     # Code quality hooks
```

### Key Directories

- **`src/models/`**: Contains all model implementations
- **`src/analysis/`**: Analysis and post-processing modules
- **`src/utils/`**: Shared utilities and helper functions
- **`tests/`**: Comprehensive test suite
- **`config/`**: Configuration files for different environments
- **`docs/`**: Project documentation

## Development Workflow

### Git Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Push and create pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Convention

Use conventional commits format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

### Code Review Process

1. All changes must go through pull requests
2. At least one reviewer approval required
3. All CI checks must pass
4. Code coverage should not decrease

## Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`):
   - Test individual functions and classes
   - Fast execution (< 1 second per test)
   - High coverage (>90%)

2. **Integration Tests** (`tests/integration/`):
   - Test interaction between components
   - End-to-end workflow testing
   - Model interoperability

3. **Validation Tests** (`tests/validation/`):
   - Data quality validation
   - Economic theory compliance
   - Statistical property verification

### Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/validation/

# Run with coverage
pytest --cov=src --cov-report=html

# Run performance tests
pytest -m performance

# Run tests requiring external data
pytest -m requires_data
```

### Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.validation`: Validation tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.requires_data`: Tests requiring external data
- `@pytest.mark.requires_network`: Tests requiring network access
- `@pytest.mark.performance`: Performance benchmarks

### Writing Tests

```python
import pytest
import numpy as np
from src.models.dsge_model import DSGEModel, DSGEParameters

class TestDSGEModel:
    @pytest.fixture
    def model_params(self):
        return DSGEParameters(
            beta=0.99,
            alpha=0.33,
            delta=0.025
        )
    
    @pytest.fixture
    def dsge_model(self, model_params):
        return DSGEModel(model_params)
    
    def test_model_initialization(self, dsge_model):
        assert dsge_model is not None
        assert hasattr(dsge_model, 'parameters')
    
    @pytest.mark.slow
    def test_model_simulation(self, dsge_model):
        result = dsge_model.simulate(periods=100)
        assert 'status' in result
        assert result['status'] == 'success'
```

## Code Quality Standards

### Code Formatting

- **Black**: Automatic code formatting
- **isort**: Import sorting
- **Line length**: 88 characters (Black default)

### Linting

- **Flake8**: Style guide enforcement
- **MyPy**: Static type checking
- **Bandit**: Security linting

### Type Hints

All new code should include type hints:

```python
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

def process_data(
    data: pd.DataFrame,
    columns: List[str],
    method: str = 'linear'
) -> Dict[str, np.ndarray]:
    """Process data with specified method.
    
    Args:
        data: Input dataframe
        columns: Columns to process
        method: Processing method
        
    Returns:
        Dictionary of processed arrays
    """
    result = {}
    for col in columns:
        result[col] = data[col].values
    return result
```

### Documentation Standards

- **Docstrings**: Google style for all public functions/classes
- **Type hints**: Required for all function signatures
- **Comments**: Explain complex logic and algorithms
- **README**: Keep updated with project changes

### Pre-commit Hooks

The following hooks run automatically:

- Black (code formatting)
- isort (import sorting)
- Flake8 (linting)
- MyPy (type checking)
- Bandit (security)
- Trailing whitespace removal
- YAML formatting

## Configuration Management

### Configuration Files

The project uses YAML configuration files:

```yaml
# config/default.yaml
project:
  name: "Bangladesh Macroeconomic Models"
  version: "1.0.0"

logging:
  level: "INFO"
  format: "json"
  
models:
  dsge:
    default_periods: 100
    convergence_tolerance: 1e-6
  cge:
    max_iterations: 1000
    damping_factor: 0.5
```

### Environment-Specific Configs

```python
from src.config import ConfigManager

# Load configuration
config = ConfigManager()
config.load_config('config/default.yaml')

# Override for development
config.load_config('config/development.yaml')

# Access configuration
log_level = config.get('logging.level')
dsge_periods = config.get('models.dsge.default_periods')
```

### Environment Variables

Sensitive configuration through environment variables:

```bash
export WORLD_BANK_API_KEY="your-api-key"
export BANGLADESH_BANK_API_KEY="your-api-key"
export LOG_LEVEL="DEBUG"
```

## Performance Guidelines

### Performance Monitoring

```python
from src.utils.performance_monitor import monitor_performance

@monitor_performance
def expensive_computation(data):
    # Your computation here
    return result

# Or use context manager
with performance_context("model_simulation"):
    result = model.simulate(periods=1000)
```

### Optimization Tips

1. **Use NumPy vectorization** instead of Python loops
2. **Profile before optimizing** using cProfile or line_profiler
3. **Cache expensive computations** using functools.lru_cache
4. **Use appropriate data types** (float32 vs float64)
5. **Parallelize when possible** using multiprocessing or joblib

### Memory Management

```python
import gc
from src.utils.performance_monitor import MemoryProfiler

# Monitor memory usage
with MemoryProfiler() as profiler:
    large_computation()
    
print(f"Peak memory usage: {profiler.peak_memory_mb:.1f} MB")

# Explicit garbage collection for large objects
del large_object
gc.collect()
```

## Error Handling

### Custom Exceptions

```python
from src.utils.error_handling import (
    ModelConfigurationError,
    ModelConvergenceError,
    DataError
)

def calibrate_model(data):
    if not validate_data(data):
        raise DataError("Invalid input data format")
        
    try:
        result = optimization_routine(data)
    except ConvergenceError:
        raise ModelConvergenceError(
            "Model failed to converge",
            context={"iterations": 1000, "tolerance": 1e-6}
        )
    
    return result
```

### Error Handling Decorator

```python
from src.utils.error_handling import handle_errors

@handle_errors(reraise=True, log_errors=True)
def risky_function():
    # Function that might raise exceptions
    pass
```

## Logging

### Logging Configuration

```python
from src.utils.logging_config import get_logger, get_model_logger

# General logger
logger = get_logger(__name__)
logger.info("Starting analysis")

# Model-specific logger
model_logger = get_model_logger("dsge", model_id="bd_dsge_v1")
model_logger.info("Model calibration started")
```

### Performance Logging

```python
from src.utils.logging_config import log_performance

@log_performance
def expensive_function():
    # Function implementation
    pass
```

## Continuous Integration

### GitHub Actions Workflow

The CI pipeline includes:

1. **Code Quality Checks**:
   - Black formatting
   - isort import sorting
   - Flake8 linting
   - MyPy type checking
   - Bandit security scanning

2. **Testing**:
   - Unit tests across Python versions
   - Integration tests
   - Coverage reporting

3. **Security**:
   - Dependency vulnerability scanning
   - Secret detection

4. **Documentation**:
   - Documentation building
   - Link checking

### Local CI Simulation

```bash
# Run all quality checks locally
pre-commit run --all-files

# Run full test suite
pytest tests/ --cov=src --cov-report=html

# Security scan
bandit -r src/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure package is installed in development mode
   pip install -e .
   ```

2. **Test Failures**:
   ```bash
   # Run tests with verbose output
   pytest -v -s
   
   # Run specific test
   pytest tests/unit/test_dsge_model.py::TestDSGEModel::test_initialization -v
   ```

3. **Performance Issues**:
   ```python
   # Profile your code
   python -m cProfile -o profile.stats your_script.py
   
   # Analyze with snakeviz
   snakeviz profile.stats
   ```

4. **Memory Issues**:
   ```python
   # Monitor memory usage
   from memory_profiler import profile
   
   @profile
   def your_function():
       pass
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# Or use ipdb for better interface
import ipdb; ipdb.set_trace()
```

### Getting Help

1. Check existing issues in the repository
2. Review documentation and examples
3. Run tests to ensure environment is set up correctly
4. Create detailed issue reports with:
   - Python version
   - Operating system
   - Error messages
   - Minimal reproduction example

## Contributing

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Performance impact considered
- [ ] Security implications reviewed

### Review Criteria

1. **Functionality**: Does the code work as intended?
2. **Testing**: Are there adequate tests?
3. **Performance**: Is the code efficient?
4. **Maintainability**: Is the code readable and well-structured?
5. **Documentation**: Is the code properly documented?
6. **Security**: Are there any security concerns?

---

For more information, see the [Project Improvement Plan](PROJECT_IMPROVEMENT_PLAN.md) and individual module documentation.