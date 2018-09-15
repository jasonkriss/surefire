from setuptools import setup, find_packages

VERSION = '0.2.0'

setup(
    name='surefire',
    version=VERSION,
    author='Jason Kriss',
    author_email='jasonkriss@gmail.com',
    url='https://github.com/jasonkriss/surefire',
    description='PyTorch models for heterogeneous inputs',
    license='MIT',
    packages=find_packages(),
    zip_safe=True,
    test_suite='tests',
    install_requires=[
        'torch'
    ]
)
