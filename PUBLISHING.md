# Publishing Relay to PyPI

This guide explains how to publish the Relay package to PyPI, both manually and automatically via GitHub Actions.

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

## Automatic Publishing with GitHub Actions

You can set up automatic publishing to PyPI using GitHub Actions. Two workflows are included:

### Option 1: Publish on Git Tags (Recommended)

The workflow `.github/workflows/publish-on-tag.yml` automatically publishes when you push a version tag:

1. **Set up PyPI Trusted Publishing** (one-time setup):
   - Go to your PyPI project settings: https://pypi.org/manage/project/relay-llm/settings/
   - Scroll to "Publishing" → "Add a new pending publisher"
   - Select "GitHub Actions"
   - Enter your GitHub username and repository name (`neelguha/relay`)
   - Enter the workflow filename: `.github/workflows/publish-on-tag.yml`
   - Enter the environment name: `__pypi__` (or leave blank)
   - Click "Add"

2. **Update version and create a tag**:
   ```bash
   # Update version in relay/__init__.py and pyproject.toml
   git add relay/__init__.py pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git tag v0.1.1
   git push origin main --tags
   ```

3. The workflow will automatically:
   - Extract the version from the tag
   - Update version files
   - Build the package
   - Publish to PyPI

### Option 2: Publish on GitHub Releases

The workflow `.github/workflows/publish.yml` publishes when you create a GitHub release:

1. Set up PyPI Trusted Publishing (same as above, but use workflow: `.github/workflows/publish.yml`)

2. Create a GitHub release:
   - Go to your repository → Releases → Create a new release
   - Choose a tag (or create a new one)
   - Fill in release notes
   - Click "Publish release"

3. The workflow will automatically build and publish to PyPI

### Option 3: Manual Trigger

You can also manually trigger the workflow:
- Go to Actions → "Publish to PyPI" → "Run workflow"
- Enter the version number
- Click "Run workflow"

**Note:** The workflows use PyPI's Trusted Publishing feature, which is more secure than storing API tokens as secrets. No secrets need to be configured in GitHub.

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

After publishing (manually or automatically):

1. Create a git tag (if not already created):
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. Create a GitHub release with release notes (if using manual publishing)

3. Update CHANGELOG.md (if you maintain one)

## Troubleshooting

- **"File already exists"**: The version already exists on PyPI. Bump the version.
- **"Invalid credentials"**: Check your token/username/password
- **"Package name conflict"**: The name `relay-llm` might be taken. Choose a different name in `setup.py` and `pyproject.toml`
