"""Setup script for Relay package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from package
version = {}
with open(this_directory / "relay" / "__init__.py", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

setup(
    name="relay-llm",
    version=version.get("__version__", "0.1.0"),
    description="A Python package for batch API calls to commercial LLM APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Neel Guha",
    author_email="neelguha@gmail.com",
    url="https://github.com/neelguha/relay",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm batch api openai anthropic together ai",
    project_urls={
        "Bug Reports": "https://github.com/neelguha/relay/issues",
        "Source": "https://github.com/neelguha/relay",
        "Documentation": "https://github.com/neelguha/relay#readme",
    },
)
