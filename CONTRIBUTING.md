# Contributing

Thank you for your interest in contributing to MONAI Physio! This guide will help you get started.

## Ways to Contribute

- Report bugs and issues
- Suggest new features
- Improve documentation
- Submit code contributions
- Share example workflows

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/monai-physio.git
   cd monai-physio
   git lfs install
   git lfs pull
   ```

   The test baseline images are stored with [Git LFS](https://git-lfs.com/);
   without it they are text pointers and every baseline comparison fails.

3. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install in development mode**:

   ```bash
   pip install uv
   uv pip install -e ".[dev_cuda12]" --torch-backend=cu128 -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
   ```

   `dev_cuda12` is the full developer environment: CUDA 12.8 PyTorch and
   CuPy plus the test, lint, and documentation tooling. For CUDA 13, see the
   [installation guide](https://project-monai.github.io/monai-physio/installation.html).

5. **Install pre-commit hooks**:

   ```bash
   pre-commit install --hook-type pre-commit --hook-type pre-push
   ```

   The second hook type installs the fast-test hook that runs on `git push`.

If you work with an AI coding assistant, use the graphify knowledge graph to
navigate the codebase - `graphify query "<question>"` returns a scoped
subgraph rather than raw search output. See the
[AI Assistants guide](https://project-monai.github.io/monai-physio/developer/ai_assistants.html).

## IDE Setup (VS Code / Cursor)

### Recommended Extensions

For the best development experience with VS Code or Cursor, install these extensions:

**Required:**

- [charliermarsh.ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) - Ruff linting and formatting
- [ms-python.python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) - Python language support
- [ms-python.vscode-pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) - IntelliSense and type checking

**Recommended:**

- [ms-python.debugpy](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy) - Python debugger
- [njpwerner.autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) - Generate docstrings automatically

**Not Needed (Replaced by Ruff):**

- ms-python.black-formatter - No longer needed
- ms-python.isort - No longer needed
- ms-python.flake8 - No longer needed
- ms-python.pylint - No longer needed

### VS Code Settings

The repository includes `.vscode/settings.json` with optimal configuration. Key settings:

```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    },
    "editor.rulers": [88]
  },
  "ruff.enable": true,
  "python.analysis.typeCheckingMode": "basic"
}
```

This configuration:

- Uses Ruff for all formatting and linting
- Automatically formats code on save
- Organizes imports automatically
- Shows a ruler at 88 characters (line length limit)
- Enables basic type checking with Pylance

### Experiment Scripts

Experiments in the `experiments/` directory are `# %%` percent-format
Python scripts. They run end-to-end as plain Python (`python <script>.py`)
or cell-by-cell via the VS Code / Cursor Python extension's "Run Cell"
feature.

- Ruff formats these cell-separated scripts automatically
- Type checking is less strict in experiment scripts (expected for exploratory work)

### First-Time Setup Checklist

After cloning the repository:

1. Install Python 3.11+ and create virtual environment
2. Install development dependencies with the command in step 4 of
   *Getting Started*
3. Install pre-commit hooks:
   `pre-commit install --hook-type pre-commit --hook-type pre-push`
4. Install Ruff extension in VS Code/Cursor
5. Remove old formatter extensions (black, isort, flake8, pylint)
6. Verify settings: Open a Python file and save to test auto-formatting
7. Run tests: `pytest tests/ -m "not slow"` to verify setup

## Code Style

MONAI Physio follows strict code quality standards using modern, fast tooling.

### Formatting and Linting with Ruff

We use **Ruff** for all formatting and linting (line length: 88, double quotes):

```bash
# Check and fix linting issues
ruff check . --fix

# Format code
ruff format .

# Check without making changes
ruff check . --diff
ruff format --check .
```

### Type Checking with mypy

We use **mypy** for static type checking:

```bash
# Run type checking
mypy src/
```

### Pre-commit Hooks

Run all checks automatically before committing:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

The pre-commit hooks will automatically:

- Run Ruff linter with auto-fixes
- Run Ruff formatter
- Run mypy type checking (on commit)
- Strip outputs and execution counts from notebooks (on commit), so a
  notebook's diff shows only code changes. Run it by hand on every tracked
  notebook with `python utils/strip_notebook_outputs.py`.
- Run fast unit tests (on push)

## Development Workflow

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following code style guidelines

3. **Add tests** for new functionality

4. **Run tests**:

   ```bash
   pytest tests/
   ```

5. **Commit your changes**:

   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork**:

   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request** on GitHub

## Pull Request Guidelines

- **Clear description**: Explain what and why
- **Reference issues**: Link related [issues](https://github.com/Project-MONAI/monai-physio/issues) with #123
- **Pass all tests**: CI must pass
- **Update documentation**: Document new features
- **Add release note**: Document user-facing changes in the pull request
- **Log breaking changes**: If the change breaks a public API, add an entry to
  `docs/developer/migration_next.md` in the same commit, covering what
  changed, why it benefits future users, before/after code, and the script that
  automates the conversion. Do not add deprecation shims instead.

## Testing

### Write Tests

Add tests in the `tests/` directory:

```python
# tests/test_my_feature.py
import pytest
from monai_physio import MyNewFeature


def test_my_feature():
    feature = MyNewFeature()
    result = feature.do_something()
    assert result == expected_value
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_my_feature.py -v

# Run with coverage
pytest tests/ --cov=src/monai_physio --cov-report=html

# Default invocation auto-skips slow/GPU/Simpleware/tutorial
pytest tests/

# Opt into specific buckets
pytest tests/ --run-slow
pytest tests/ --run-gpu --run-slow   # typical local GPU profile
pytest tests/ --run-physicsnemo      # needs PhysicsNeMo; requires Python >= 3.11
# --run-all turns on every --run-* bucket at once (used by self-hosted CI):
pytest tests/ --run-all
```

If your machine has no GPU, run the default suite and say so in your pull
request; a maintainer can run the GPU tests on it by adding the
`run-gpu-tests` label.

## Documentation

Documentation is built with Sphinx and published to GitHub Pages at
<https://project-monai.github.io/monai-physio/>.

### Build Docs Locally

```bash
# Install documentation dependencies (the dev environment from Getting Started)
uv pip install -e ".[dev_cuda12]" --torch-backend=cu128 -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

# Build HTML documentation
cd docs
make html

# Open in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Documentation Style

- Use **reStructuredText** (.rst) for documentation
- Follow existing structure and formatting
- Include code examples with proper syntax highlighting
- Add docstrings to all public classes and methods

### Contributing Scripts vs Experiments

When contributing new workflows or examples:

**Production Code (src/monai_physio/cli/):**

- **DO contribute here** for production-ready CLI implementations
- Must include proper error handling and validation
- Should follow all code style and testing requirements
- Serves as definitive usage examples for users
- Will be referenced in documentation

**Research Code (experiments/ directory):**

- **May contribute here** for exploratory research and design experiments
- Can have hardcoded paths and minimal error handling
- Should document what was learned and how it informed production code
- Helps others understand adaptation possibilities for new domains
- Should reference corresponding production implementation in CLI commands or `src/monai_physio/cli/`

### Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description with more details about what the function does,
    any important notes, and usage examples.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
        RuntimeError: When something else fails

    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        True
    """
    return True
```

## Code Review Process

All contributions go through code review:

1. **Automated checks** run via GitHub Actions
2. **Maintainer review** for code quality and design
3. **Feedback** may request changes
4. **Approval** and merge when ready

### Review Criteria

- **Correctness**: Does it work as intended?
- **Code quality**: Is it clean and well-structured?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it properly documented?
- **Performance**: Are there any performance concerns?
- **Compatibility**: Does it avoid needless breaking changes, and is every
  unavoidable break recorded in `docs/developer/migration_next.md` with a
  conversion path rather than a deprecation shim?

## Reporting Issues

Report bugs and request features via
[GitHub Issues](https://github.com/Project-MONAI/monai-physio/issues).

### Bug Reports

When reporting bugs, include:

- **Python version**
- **MONAI Physio version**
- **Operating system**
- **GPU/CUDA version** (if applicable)
- **Minimal code** to reproduce
- **Error messages** and stack traces
- **Expected vs actual behavior**

### Feature Requests

When suggesting features:

- **Clear description** of the feature
- **Use cases** and motivation
- **Proposed API** or interface
- **Potential challenges** or limitations

## Release Process

### Versioning

MONAI Physio uses calendar versioning: `YYYY.0M.PATCH`

- **YYYY**: Year
- **0M**: Zero-padded month
- **PATCH**: Patch number within month

Example: `2026.09.0`

### Making a Release

Maintainers only:

```bash
# Bump version
bumpver update --patch

# Archive the migration guide under the new version, then start a fresh one
VERSION=$(bumpver show --no-fetch | sed -n "s/^Current Version: //p")
git mv docs/developer/migration_next.md "docs/developer/migration_$VERSION.md"
# Retitle the archived file to "Migration Guide - $VERSION"
# Recreate docs/developer/migration_next.md from its entry template

# Commit, tag, and push
git commit -am "Bump version to $VERSION"
git tag "$VERSION"
git push origin main "$VERSION"
```

Then publish a GitHub release for the tag. Publishing it runs
`.github/workflows/release.yml`, which checks that the tag equals the
`pyproject.toml` version, builds the sdist and wheel, and uploads them to PyPI
through trusted publishing; nothing is uploaded by hand. To try a build first,
run that workflow manually with `testpypi` as the target.

A broken release cannot be overwritten, because PyPI never reuses a version
number. Yank it on PyPI, fix the problem, and release the next patch version.

The `Developer Guides` toctree in `docs/index.rst` globs
`developer/migration_*`, so archived guides appear in the sidebar without
further edits.

## Community Guidelines

- **Be respectful** and professional
- **Be constructive** in feedback
- **Be patient** with reviews
- **Help others** in discussions
- **Share knowledge** and examples

## Getting Help

- **[GitHub Issues](https://github.com/Project-MONAI/monai-physio/issues)**: Report bugs and request features
- **[GitHub Discussions](https://github.com/Project-MONAI/monai-physio/discussions)**: Ask questions and share ideas
- **Documentation**: Check the [docs](https://project-monai.github.io/monai-physio/) first
- **Conduct**: Follow the Community Guidelines above

## License

By contributing, you agree that your contributions will be licensed under the
Apache 2.0 License.

## Acknowledgments

Thank you to all contributors who help make MONAI Physio better!

## See Also

- [Architecture](https://project-monai.github.io/monai-physio/architecture.html) - System architecture
- [Testing](https://project-monai.github.io/monai-physio/testing.html) - Testing guide
- [GitHub Repository](https://github.com/Project-MONAI/monai-physio)
