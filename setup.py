from setuptools import setup, find_packages

setup(
    name="crypto-bot",
    version="0.1.0",
    description="A cryptocurrency trading bot",
    author="John P",
    author_email="hjpdev@gmail.com",
    packages=find_packages(),
    python_requires=">=3.11.5",
    install_requires=[
        "sqlalchemy>=2.0.38",
        "alembic>=1.14.1",
        "ccxt>=4.1.87",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.1",
        "pandas>=2.2.1",
        "pandas-ta>=0.3.14b0",
        "numpy>=1.24.1",
        "psycopg2-binary>=2.9.9",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.1",
            "flake8>=6.1.0"
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-bot=app.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)