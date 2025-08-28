from setuptools import setup

setup(
    name='PySpectrum',
    version='0.2',
    packages=['pyspectrum', 'pyspectrum.peak', 'pyspectrum.spectrum', 'pyspectrum.calibration',
              'pyspectrum.peak_fitting', 'pyspectrum.detector_parser', 'pyspectrum.peak_identification'],
    url='https://github.com/achiyaAmrusi/pySpectrum',
    license='MIT license',
    author='Achiya Yosef Amrusi',
    author_email='ahia.amrosi@mail.huji.ac.il',
    description='spectrum and peak analysis tools'
)
