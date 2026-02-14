from setuptools import setup, find_packages

setup(
    name="simple-job-source-agent",
    version="0.1.0",
    description="A linkedin job crawling agent",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "requests",
        "bs4",
        "playwright",
        "google-search-results",
        "langchain",
        "langchain-core",
        "langchain-openai",
        "pandas",
    ],
    python_requires=">=3.9",
)