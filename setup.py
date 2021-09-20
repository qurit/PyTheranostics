from setuptools import setup

setup(name='DOODLE',
      version='0.1',
      description='3D dOsimetry fOr raDiopharmaceuticaL thErapies',
      url='https://github.com/carluri/doodle.git',
      author='Carlos Uribe',
      author_email='curibe@bccrc.ca',
      license='LICENSE',
      packages=['doodle','doodle.tests'],
      include_package_data=True,
      install_requires = [
          'numpy','matplotlib','pandas','pydicom'
      ],
      zip_safe=False)