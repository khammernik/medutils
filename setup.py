import os.path
import shutil

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    readme = f.read()

setup(name='Medutils',
  version='0.1',
  description='Utils for medical image processing and reconstruction',
  long_description=readme,
  author='Kerstin Hammernik',
  author_email='hammernik@icg.tugraz.com',
  url='https://github.com/khammernik/medutils/',
  packages=['medutils'],
  install_requires=['numpy', 'scipy', 'h5py', 'matplotlib', 'scikit-image'],
  license="Apache 2.0",
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: Linux',
      'Programming Language :: Python :: 3.6',
      ],
 )
