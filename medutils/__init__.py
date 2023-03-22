"""
This file is part of medutils.

Copyright (C) 2019 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
Institute of Computer Graphics and Vision, Graz University of Technology
https://www.tugraz.at/institute/icg/research/team-pock/
"""

from . import measures
from . import mri
from . import io
from . import optimization
try:
    from . import optimization_th
except:
    print(Warning("medutils.optimization_th is only available if pytorch installed."))
from . import visualization
from . import complex

import sys
import os

# setup bart toolbox
try:
    sys.path.append(os.environ['TOOLBOX_PATH'] + '/python/')
    from bart import bart
except Exception:
    print(Warning("BART toolbox not setup properly or not available"))