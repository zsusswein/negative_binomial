from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='negative_binomial',
    version='0.1.0',
    description='Package for fitting negative binomial distributions',
    long_description=readme,
    author='Zachary Susswein',
    author_email='zsusswein@gmail.com',
    url='https://github.com/zsusswein/negative_binomial',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
