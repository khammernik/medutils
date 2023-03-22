import os.path
import shutil

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    readme = f.read()

setup(name='medutils-mri',
  version='0.1.1',
  description='Utils for medical image processing and reconstruction',
  long_description=readme,
  author='Kerstin Hammernik',
  author_email='k.hammernik@tum.de',
  url='https://github.com/khammernik/medutils/',
  packages=['medutils', 'medutils.optimization', 'medutils.optimization_th'],
  install_requires=['numpy', 'scipy', 'h5py', 'matplotlib', 'scikit-image'],
  license="Apache 2.0",
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: Unix',
      'Programming Language :: Python :: 3.6',
      ],
 )
