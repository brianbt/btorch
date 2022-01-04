from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='btorch',
    version='0.0.1',
    author="Chan Cheong",
    author_email='brianchan.xd@gmail.com',
    description='an advanced pytorch library',
    long_description=long_description,
    url='https://github.com/brianbt/btorch',
    license='MIT',
    install_requires=['torch', 'torchvision', 'numpy', 'pandas'],
)
