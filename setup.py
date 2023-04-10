from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## editing variables to fit requirements
REPO_NAME = "Book-Recommender-System-Using-Machine-Learning-Apriori"
AUTHOR_USER_NAME = "ianmlolwa"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ['streamlit', 'numpy', 'seaborn']


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    desscription="A Small Book Recommender System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="ianmlolwa2@gmail.com",
    packages=[SRC_REPO],
    python_requires=">=3.9",
    install_requires=LIST_OF_REQUIREMENTS
)