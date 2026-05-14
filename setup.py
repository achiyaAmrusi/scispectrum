from setuptools import setup, find_packages

setup(
    name='PySpectrum',
    version='0.2',
    python_requires='>=3.9',
    packages=find_packages(),
    url='https://github.com/achiyaAmrusi/pySpectrum',
    license='MIT license',
    author='Achiya Yosef Amrusi',
    author_email='ahia.amrosi@mail.huji.ac.il',
    description='spectrum and peak analysis tools',
    install_requires=[
        'xarray>=2022.6',
        'numpy>=1.22.0,<3.0',
        'uncertainties>=3.1',
        'pandas>=1.5.0,<4.0',
        'scipy>=1.8',
    ],
)
