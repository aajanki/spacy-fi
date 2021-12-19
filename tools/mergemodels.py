import copy
import json
import re
import shutil
import typer
from pathlib import Path


def main(
        tagger_path: Path = typer.Argument(..., help='Tagger+parser model directory'),
        ner_path: Path = typer.Argument(..., help='NER model directory'),
        merged_path: Path = typer.Argument(..., help='Output directory')
):
    if merged_path.exists():
        shutil.rmtree(merged_path)

    shutil.copytree(tagger_path, merged_path)
    shutil.copytree(ner_path / 'ner', merged_path / 'ner')

    merged_meta = merge_meta(json.load(open(tagger_path / 'meta.json')),
                             json.load(open(ner_path / 'meta.json')))
    json.dump(merged_meta, open(merged_path / 'meta.json', 'w'), indent=4)

    open(merged_path / 'config.cfg', 'w').write(
        merge_configs(open(tagger_path / 'config.cfg').read()))


def merge_meta(tagger_meta, ner_meta):
    merged_meta = copy.deepcopy(tagger_meta)
    merged_meta['pipeline'] = merged_meta['pipeline'] + ['ner']
    merged_meta['components'] = merged_meta['components'] + ['ner']
    merged_meta['performance']['ents_p'] = ner_meta['performance']['ents_p']
    merged_meta['performance']['ents_r'] = ner_meta['performance']['ents_r']
    merged_meta['performance']['ents_f'] = ner_meta['performance']['ents_f']
    merged_meta['performance']['ents_per_type'] = ner_meta['performance']['ents_per_type']
    merged_meta['labels']['ner'] = ner_meta['labels']['ner']
    return merged_meta


def merge_configs(tagger_config):
    config = tagger_config + """

[components.ner]
factory = "ner"
incorrect_spans_key = null
moves = null
scorer = {"@scorers":"spacy.ner_scorer.v1"}
update_with_oracle_cut_size = 100

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}
upstream = "*"
"""

    config = re.sub(r'(pipeline = \[.+)]', r'\1,"ner"]', config, re.MULTILINE)
    return config


if __name__ == '__main__':
    typer.run(main)
