import torch

from s2s_lemmatizer.data import MAX_LEN
from s2s_lemmatizer.utils import EXPERIMENT
from seq2seq.evaluator import Predictor
from seq2seq.loss import Perplexity
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.trainer import SupervisedTrainer


def build_model(src, tgt, hidden_size, bidirectional, dropout, attention, init_value):
    EXPERIMENT.param("Hidden", hidden_size)
    EXPERIMENT.param("Bidirectional", bidirectional)
    EXPERIMENT.param("Dropout", dropout)
    EXPERIMENT.param("Attention", attention)
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    encoder = EncoderRNN(len(src.vocab), MAX_LEN, hidden_size,
                         rnn_cell="lstm",
                         bidirectional=bidirectional,
                         dropout_p=dropout,
                         variable_lengths=False)
    decoder = DecoderRNN(len(tgt.vocab), MAX_LEN, hidden_size,  # * 2 if bidirectional else hidden_size,
                         rnn_cell="lstm",
                         use_attention=attention,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        seq2seq.cuda()
        loss.cuda()
    for param in seq2seq.parameters():
        param.data.uniform_(-init_value, init_value)
    trainer = SupervisedTrainer(loss=loss, batch_size=10,
                                checkpoint_every=5000, random_seed=42,
                                print_every=1000)
    return seq2seq, trainer


def evaluate_model(model, data, src_field, tgt_field, file_props={}):
    predictor = Predictor(model, src_field.vocab, tgt_field.vocab)
    data["pred_lemma"] = ["".join(predictor.predict(
        list(e.word))[:-1]) for e in data.itertuples()]
    acc = 0
    for word in data.itertuples():
        acc += int(word.pred_lemma == word.lemma)
    acc /= len(data.lemma)
    EXPERIMENT.metric("Dev accuracy", acc)
    data.to_csv("./dev_{}.csv".format("-".join("{}={}".format(k, v) for k, v in file_props)))
    EXPERIMENT.log("Incorrect predictions")
    EXPERIMENT.log(str(data[data["lemma"] != data["pred_lemma"]]
                       [["word", "lemma", "pred_lemma"]]))