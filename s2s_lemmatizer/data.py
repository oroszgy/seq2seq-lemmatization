import glob

import pandas as pd

import torchtext
from seq2seq.dataset import SourceField, TargetField

MAX_LEN = 50


def _parse_ud(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                yield {
                    "word": parts[0],
                    "lemma": parts[1],
                    "pos": parts[2],
                    "tags": parts[3]
                }


def read_ud(path):
    raw_data = [tok for fpath in glob.glob(path) for tok in list(_parse_ud(fpath))]
    df = pd.DataFrame(raw_data)
    df = df[df.word.str.len() < MAX_LEN].sample(frac=1)
    return df


def _prepare_dataset(df, fields):
    fields = [('src', fields[0]), ('tgt', fields[1])]
    examples = []
    for example in df.itertuples():
        examples.append(
            torchtext.data.Example.fromlist(
                [example.word, example.lemma],
                fields
            )
        )
    return torchtext.data.Dataset(
        examples,
        fields
    )


def make_datasets(train_df, dev_df):
    src = SourceField(tokenize=list)
    tgt = TargetField(tokenize=list)
    train = _prepare_dataset(
        train_df,
        (src, tgt)
    )
    dev = _prepare_dataset(
        dev_df,
        (src, tgt)
    )
    src.build_vocab(train)
    tgt.build_vocab(train)
    return train, dev, src, tgt