# Publishing Relay to PyPI

This guide explains how to manually publish the Relay package to PyPI.

## Prerequisites

1. Create an account on [PyPI](https://pypi.org/account/register/)

2. Create an API token:
   - Go to your PyPI account settings
   - Create an API token with appropriate permissions
   - Save the token securely

3. Install build tools:
   ```bash
   pip install build twine
   ```
   
   Or use the Makefile:
   ```bash
   make build-deps
   ```

## Building the Package

1. Update version numbers in:
   - `setup.py` (version field)
   - `pyproject.toml` (version field)
   - `relay/__init__.py` (if you add a __version__)

2. Clean previous builds:
   ```bash
   make clean
   # or manually:
   rm -rf build/ dist/ *.egg-info
   ```

3. Build the package:
   ```bash
   make build
   # or:
   python -m build
   ```

   This creates:
   - `dist/relay-llm-0.1.0.tar.gz` (source distribution)
   - `dist/relay_llm-0.1.0-py3-none-any.whl` (wheel)

## Publishing to PyPI

```bash
make publish
# or:
twine upload dist/*
```

If you have `~/.pypirc` configured, credentials will be read automatically. Otherwise, you'll be prompted for your PyPI credentials.

## Using API Tokens

Instead of username/password, you can use API tokens. Twine will automatically read from `~/.pypirc`:

1. **Create `~/.pypirc`** (recommended):
   ```ini
   [distutils]
   index-servers =
       pypi

   [pypi]
   username = __token__
   password = pypi-your-token-here
   ```
   
   **Note:** Twine reads from `~/.pypirc` (in your home directory), not from `.pypirc` in the project directory. If you have `.pypirc` in the project, copy it to `~/.pypirc`.

2. **Or use environment variables**:
   ```bash
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=pypi-your-token-here
   ```

3. **Or use the Makefile** - it will automatically use `~/.pypirc` if it exists:
   ```bash
   make publish
   ```

## Version Management

- Follow [Semantic Versioning](https://semver.org/):
  - MAJOR.MINOR.PATCH (e.g., 1.2.3)
  - MAJOR: Breaking changes
  - MINOR: New features (backward compatible)
  - PATCH: Bug fixes

- Update version in:
  - `relay/__init__.py` (source of truth)
  - `pyproject.toml` (for packaging)
  - Note: `setup.py` reads version from `relay/__init__.py` automatically

## Post-Publication

After publishing:

1. Create a git tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. Create a GitHub release with release notes

3. Update CHANGELOG.md (if you maintain one)

## Troubleshooting

- **"File already exists"**: The version already exists on PyPI. Bump the version.
- **"Invalid credentials"**: Check your token/username/password
- **"Package name conflict"**: The name `relay-llm` might be taken. Choose a different name in `setup.py` and `pyproject.toml`
