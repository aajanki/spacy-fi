# Plot the pretrain loss.
#
# Usage:
# python3 tools/plot_pretrain_loss.py < models/pretrain/log.jsonl

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt

epochs = []
for line in sys.stdin:
    epochs.append(json.loads(line))

df = pd.DataFrame(epochs).iloc[1:]
df.plot(x='epoch', y='epoch_loss')
plt.show()
