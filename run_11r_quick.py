import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tf_keras
sys.modules['keras'] = tf_keras
sys.modules['keras.models'] = tf_keras.models
sys.modules['keras.callbacks'] = tf_keras.callbacks
sys.modules['keras.optimizers'] = tf_keras.optimizers
sys.modules['keras.layers'] = tf_keras.layers
sys.modules['keras.backend'] = tf_keras.backend
sys.modules['keras.regularizers'] = tf_keras.regularizers

import numpy as np
from test_key_recovery import test

n = 20
print(f"Running {n} trials of 11-round Speck32/64 key recovery attack (Huber matching)...")
print("=" * 60)

arr1, arr2, good = test(n)

success = np.sum((arr1 == 0) & (arr2 == 0))
print("=" * 60)
print(f"Success: {success}/{n} = {success/n*100:.1f}%")
print(f"Baseline (L2 norm): 52%")
print(f"Delta: {success/n*100 - 52:+.1f}%")
