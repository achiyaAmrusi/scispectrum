from setuptools import setup, find_packages

setup(
    name='scispectrum',
    version='0.3',
    python_requires='>=3.11',
    packages=find_packages(),
    url='https://github.com/Achiya/scispectrum',
    license='MIT license',
    author='Achiya Yosef Amrusi',
    author_email='ahia.amrosi@mail.huji.ac.il',
    description='spectrum and peak analysis tools',
    install_requires=[
        'numpy>=2.0.0,<3.0',
        'pandas>=2.3,<4.0',
        'scipy>=1.14.0',
        'xarray>=2024.6.0',
        'uncertainties>=3.1'],
)
