# Plot the pretrain loss.
#
# Usage:
# python3 tools/plot_pretrain_loss.py < models/pretrain/log.jsonl

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

data = []
for line in sys.stdin:
    data.append(json.loads(line))

df = pd.DataFrame(data)
df = df.groupby('epoch').last().reset_index()
df.plot(x='epoch', y='epoch_loss')
loc = mticker.MaxNLocator(nbins='auto', integer=True)
plt.gca().xaxis.set_major_locator(loc)
plt.show()
