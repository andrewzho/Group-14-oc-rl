"""
NumPy compatibility patch for mlagents_envs
This fixes the np.bool deprecation issue by monkey patching numpy
"""

import numpy as np
import warnings

# Check if numpy already has 'bool' attribute
if not hasattr(np, 'bool'):
    # Add bool alias to numpy to fix mlagents_envs
    setattr(np, 'bool', np.bool_)
    warnings.warn(
        "Applied monkey patch for np.bool deprecation. This is needed for mlagents_envs compatibility.",
        DeprecationWarning, stacklevel=2
    ) 