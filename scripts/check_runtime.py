import json
import platform

import numpy as np
import tensorflow as tf

print(json.dumps({
    "python": platform.python_version(), "tensorflow": tf.__version__,
    "numpy": np.__version__, "physical_devices": [d.name for d in tf.config.list_physical_devices()],
    "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
}, indent=2))
