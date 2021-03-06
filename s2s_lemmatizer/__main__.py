import plac

from s2s_lemmatizer.data import read_ud, make_datasets
from s2s_lemmatizer.model import build_model, evaluate_model
from s2s_lemmatizer.utils import EXPERIMENT


@plac.annotations(
    hidden_size=("", "option"),
    epochs=("", "option"),
    mini_batch_size=("", "option"),
    bidirectional=("", "option"),
    dropout=("", "option"),
    attention=("", "option"),
    init_value=("", "option"),
)
def main(train_path, dev_path, hidden_size=256, epochs=10, mini_batch_size=128,
         bidirectional=False, dropout=.2, attention=True,
         init_value=0.08):
    train_df, dev_df = read_ud(train_path), read_ud(dev_path)
    train, dev, src, tgt = make_datasets(train_df, dev_df)

    hidden_size = int(hidden_size)
    epochs = int(epochs)
    mini_batch_size = int(mini_batch_size)
    bidirectional = bool(bidirectional)
    dropout = float(dropout)
    attention = bool(attention)

    seq2seq, trainer = build_model(
        src, tgt,
        hidden_size, mini_batch_size,
        bidirectional, dropout, attention,
        init_value
    )
    EXPERIMENT.param("epochs", epochs)
    seq2seq = trainer.train(
        seq2seq, train,
        num_epochs=epochs, dev_data=dev,
        resume=False
    )
    evaluate_model(seq2seq, dev_df, src, tgt)
    EXPERIMENT.end()


if __name__ == '__main__':
    plac.call(main)
