import re
from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).parent / "README.md"

version_file = Path("tracker") / "__init__.py"
version = re.search(
    r'__version__ = ["\']([^"\']+)["\']',
    version_file.read_text(), re.M
).group(1)

setup(
    name="llm-cost-tracker",                            
    version=version,
    description="LLM API call token usage-based expense tracker",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Heondeuk Lee",
    author_email="leehd1995@gmail.com",
    url="https://github.com/hundredeuk2/TrackMyLLM",
    packages=find_packages(exclude=["tutorial", "tests*"]),
    install_requires=[
        "PyYAML>=5.1",
        "tabulate>=0.9.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        "tracker": ["pricing.yaml"],
    },
    license="MIT",
)
