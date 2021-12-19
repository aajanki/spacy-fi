import copy
import filecmp
import json
import re
import shutil
import typer
from pathlib import Path
from spacy import util


def main(
        tagger_path: Path = typer.Argument(..., help='Tagger+parser model directory'),
        ner_path: Path = typer.Argument(..., help='NER model directory'),
        merged_path: Path = typer.Argument(..., help='Output directory')
):
    if merged_path.exists():
        shutil.rmtree(merged_path)

    assert filecmp.cmp(tagger_path / 'tokenizer', ner_path / 'tokenizer', shallow=False), \
        'Tokenizers differ!'

    _, mismatch, errors = \
        filecmp.cmpfiles(tagger_path / 'vocab',
                         ner_path / 'vocab',
                         ['key2row', 'lookups.bin', 'vectors', 'vectors.cfg'],
                         shallow=False)
    assert not mismatch, 'Vocab files differ!'
    assert not errors, 'Some vocab files could not be compared!'

    shutil.copytree(tagger_path, merged_path)
    shutil.copytree(ner_path / 'ner', merged_path / 'ner')

    with open(merged_path / 'meta.json', 'w') as meta_out:
        json.dump(merge_meta(tagger_path, ner_path), meta_out, indent=4)

    with open(merged_path / 'config.cfg', 'w') as config_out:
        config_out.write(merge_configs(tagger_path, ner_path).to_str())


def merge_meta(tagger_path, ner_path):
    ner_meta = json.load(open(ner_path / 'meta.json'))
    merged_meta = json.load(open(tagger_path / 'meta.json'))
    merged_meta['pipeline'] = merged_meta['pipeline'] + ['ner']
    merged_meta['components'] = merged_meta['components'] + ['ner']
    merged_meta['performance']['ents_p'] = ner_meta['performance']['ents_p']
    merged_meta['performance']['ents_r'] = ner_meta['performance']['ents_r']
    merged_meta['performance']['ents_f'] = ner_meta['performance']['ents_f']
    merged_meta['performance']['ents_per_type'] = ner_meta['performance']['ents_per_type']
    merged_meta['labels']['ner'] = ner_meta['labels']['ner']
    return merged_meta


def merge_configs(tagger_path, ner_path):
    cfg = util.load_config(tagger_path / 'config.cfg')
    cfg_ner = util.load_config(ner_path / 'config.cfg')

    cfg['nlp']['pipeline'] = cfg['nlp']['pipeline'] + ['ner']
    cfg['components']['ner'] = cfg_ner['components']['ner']
    return cfg


if __name__ == '__main__':
    typer.run(main)
