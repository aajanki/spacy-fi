# Plot pretraining loss
#
# Usage:
#
# python3 tools/plot_pretrain_loss.py model1/log.jsonl
#
# Compare two models:
#
# python3 tools/plot_pretrain_loss.py --xwords model1/log.jsonl model2/log.jsonl

import json
import sys
import typer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional
from pathlib import Path
from typing_extensions import Annotated

def main(
  paths: Annotated[Optional[List[Path]], typer.Argument()] = None,
  xwords: Annotated[bool, typer.Option(help="number of words on x-axis (instead of epoch)")] = False
):
  xvariable = 'nr_word' if xwords else 'epoch'
  data = []
  if paths is not None and len(paths) > 0:
    for i, path in enumerate(paths):
      for line in path.open():
        linedata = json.loads(line)
        linedata['model'] = path
        data.append(linedata)
  else:
    for line in sys.stdin:
        linedata = json.loads(line)
        linedata['model'] = 'pretraining'
        data.append(linedata)

  df = pd.DataFrame(data)
  legend = 'auto' if paths and len(paths) > 1 else False
  sns.lineplot(data=df, x=xvariable, y='epoch_loss', hue='model', legend=legend)
  loc = mticker.MaxNLocator(nbins='auto', integer=True)
  plt.gca().xaxis.set_major_locator(loc)
  plt.show()

if __name__ == '__main__':
  typer.run(main)
