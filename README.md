medutils
========

This repository contains useful functions for medical image processing and reconstruction.

- **io**: loadmat function to handle mat files saved with different versions
- **measures**: common evaluation measures
- **mri**: (centered) fft2 / ifft2, MRI Cartesian 2D operators, post-processing functions, radial and spiral trajectories
- **optimization**: Conjugate gradient for arbitrary operators 
- **visualization**: slice viewer, contrast enhancement, image brightening, functions to show and save images and k-space data, image flipping

Installation
------------
To install the package, simply execute following commands. All dependencies will
be installed automatically.
~~~
$ python setup.py bdist_wheel
$ pip install dist/<your-medutils-wheel-package>.whl
~~~
