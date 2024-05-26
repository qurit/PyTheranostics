from setuptools import setup

setup(name='PyTheranostics',
      version='0.1',
      description='A library of tools to process nuclear medicine scans and take them through the dosimetry workflow to calculate the absorbed dose in target organs. ',
      url='https://github.com/qurit/PyTheranostics',
      author='Carlos Uribe, PhD, MCCPM',
      author_email='curibe@bccrc.ca',
      license='LICENSE',
      packages=[
        'doodle',
        'doodle.calibrations',
        'doodle.dicomtools',
        'doodle.dosimetry',
        'doodle.fits',
        'doodle.plots',
        'doodle.qc',
        'doodle.segmentation',
        'doodle.registration',
        'doodle.shared'],
      include_package_data=True,
      install_requires = [
          'numpy','matplotlib','pandas','pydicom','openpyxl','rt-utils','scikit-image', 'simpleitk'
      ],
      zip_safe=False)