import numpy as np
from spacy.tokens import Doc
from spacy.attrs import POS, HEAD, DEP


def get_doc_from_text(text, tokenizer, pos, heads, deps):
    tokens = tokenizer(text)
    return get_doc(vocab=tokens.vocab, words=[t.text for t in tokens],
                   pos=pos, heads=heads, deps=deps)


def get_doc(vocab, words, pos, heads, deps):
    assert len(pos) == len(words)
    assert len(heads) == len(words)
    assert len(deps) == len(words)

    headings = []
    values = []
    annotations = [pos, heads, deps]
    possible_headings = [POS, HEAD, DEP]
    for a, annot in enumerate(annotations):
        headings.append(possible_headings[a])
        if annot is not heads:
            values.extend(annot)
    for value in values:
        vocab.strings.add(value)

    doc = Doc(vocab, words=words)

    attrs = np.zeros((len(doc), len(headings)), dtype=np.uint64)

    for j, annot in enumerate(annotations):
        if annot is heads:
            # Overflow negative values on purpose because that's how spacy
            # expacts them.
            attrs[:, j] = np.array(heads).astype(np.uint64)
        else:
            for i in range(len(words)):
                attrs[i, j] = doc.vocab.strings[annot[i]]

    doc.from_array(headings, attrs)

    return doc
