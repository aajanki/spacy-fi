# Plot pretraining loss
#
# Usage:
#
# python3 tools/plot_pretrain_loss.py model1/log.jsonl
#
# Compare two models:
#
# python3 tools/plot_pretrain_loss.py --xwords --yperword model1/log.jsonl model2/log.jsonl

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
  xwords: Annotated[bool, typer.Option(help="number of words on x-axis (instead of epoch)")] = False,
  yperword: Annotated[bool, typer.Option(help="loss per word on y-axis (instead of epoch_loss)")] = False
):
  xvariable = 'nr_word' if xwords else 'epoch'
  yvariable = 'loss_per_word' if yperword else 'epoch_loss'
  dfs = []
  if paths is not None and len(paths) > 0:
    for i, path in enumerate(paths):
      data = [json.loads(line) for line in path.open()]
      df = pd.DataFrame(data)
      df['model'] = path
      words_per_epoch = df.iloc[0]['nr_word']
      df['loss_per_word'] = df['epoch_loss']/words_per_epoch
      dfs.append(df)
  else:
    data = [json.loads(line) for line in sys.stdin]
    df = pd.DataFrame(data)
    df['model'] = 'pretraining'
    words_per_epoch = df.iloc[0]['nr_word']
    df['loss_per_word'] = df['epoch_loss']/words_per_epoch
    dfs.append(df)

  df = pd.concat(dfs)
  legend = 'auto' if paths and len(paths) > 1 else False
  sns.lineplot(data=df, x=xvariable, y=yvariable, hue='model', legend=legend)
  loc = mticker.MaxNLocator(nbins='auto', integer=True)
  plt.gca().xaxis.set_major_locator(loc)
  plt.show()

if __name__ == '__main__':
  typer.run(main)
