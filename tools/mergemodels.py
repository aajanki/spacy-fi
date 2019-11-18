import copy
import json
import os.path
import shutil
import sys


def main():
    tagger_path, ner_path, merged_path = sys.argv[1:4]

    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)
    os.makedirs(merged_path)

    tagger_meta = json.load(open(os.path.join(tagger_path, 'meta.json')))
    ner_meta = json.load(open(os.path.join(ner_path, 'meta.json')))

    merged_meta = copy.deepcopy(tagger_meta)
    merged_meta['pipeline'] = ['tagger', 'parser', 'ner']
    merged_meta['accuracy']['ents_p'] = ner_meta['accuracy']['ents_p']
    merged_meta['accuracy']['ents_r'] = ner_meta['accuracy']['ents_r']
    merged_meta['accuracy']['ents_f'] = ner_meta['accuracy']['ents_f']
    merged_meta['accuracy']['ents_per_type'] = ner_meta['accuracy']['ents_per_type']
    merged_meta['labels']['ner'] = ner_meta['labels']['ner']

    with open(os.path.join(merged_path, 'meta.json'), 'w') as f:
        json.dump(merged_meta, f, indent=2)

    shutil.copy(
        os.path.join(tagger_path, 'tokenizer'),
        os.path.join(merged_path, 'tokenizer')
    )
    shutil.copytree(
        os.path.join(tagger_path, 'vocab'),
        os.path.join(merged_path, 'vocab'))
    shutil.copytree(
        os.path.join(tagger_path, 'parser'),
        os.path.join(merged_path, 'parser'))
    shutil.copytree(
        os.path.join(tagger_path, 'tagger'),
        os.path.join(merged_path, 'tagger'))
    shutil.copytree(
        os.path.join(ner_path, 'ner'),
        os.path.join(merged_path, 'ner'))


if __name__ == '__main__':
    main()
