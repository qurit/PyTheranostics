from setuptools import setup

setup(name='DOODLE',
      version='0.1',
      description='3D dOsimetry fOr raDiopharmaceuticaL thErapies',
      url='https://github.com/carluri/doodle.git',
      author='Carlos Uribe',
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
        'doodle.shared'],
      include_package_data=True,
      install_requires = [
          'numpy','matplotlib','pandas==1.5.3','pydicom','openpyxl','rt_utils','scikit-image'
      ],
      zip_safe=False)