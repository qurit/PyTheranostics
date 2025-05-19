from setuptools import setup

setup(name='pytheranostics',
      version='0.1.0',
      description='A library of tools to process nuclear medicine scans and take them through the dosimetry workflow to calculate the absorbed dose in target organs.',
      url='https://github.com/qurit/PyTheranostics',
      author='Carlos Uribe, PhD, MCCPM',
      author_email='curibe@bccrc.ca',
      license='MIT',
      packages=[
        'pytheranostics',
        'pytheranostics.calibrations',
        'pytheranostics.dicomtools',
        'pytheranostics.dosimetry',
        'pytheranostics.fits',
        'pytheranostics.plots',
        'pytheranostics.qc',
        'pytheranostics.segmentation',
        'pytheranostics.registration',
        'pytheranostics.shared'
      ],
      include_package_data=True,
      install_requires=[
          'numpy',
          'matplotlib',
          'pandas',
          'pydicom',
          'openpyxl',
          'rt-utils',
          'scikit-image',
          'simpleitk',
          'lmfit'
      ],
      extras_require={
          'dev': [
              'pytest>=7.0',
              'pytest-cov>=4.0',
              'flake8>=6.0',
              'black>=23.0',
              'mypy>=1.0',
              'sphinx>=7.0',
              'sphinx-rtd-theme>=1.0'
          ]
      },
      python_requires='>=3.8',
      zip_safe=False)