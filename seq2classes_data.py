import torchtext
from seq2seq_data import seq2seqData
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext import data, datasets
from torch.nn import init
import dill
import os


class seq2classesData(seq2seqData):
    def __init__(self, in_tokenize, cols, batch_size, device, data_path='.', vectors=None):
        super(seq2classesData, self).__init__(in_tokenize, in_tokenize, cols, batch_size, device, data_path, vectors)
        self.INPUT_TEXT = data.Field(batch_first=True, sequential=True, tokenize=in_tokenize, lower=True)
        self.OUTPUT_TEXT = data.Field(batch_first=True, sequential=False, use_vocab=False)

    def buildVocabulary(self, train, val):
        # build vocab
        if os.path.exists(self.input_vocab_path):
            print('using exist vocabulary')
            with open(self.input_vocab_path, 'rb')as f:
                self.INPUT_TEXT.vocab = dill.load(f)

        else:
            # test dataset may exist word out of vocabulary if build_vocab operation no include test
            # but more true
            print('create vocabulary')
            self.INPUT_TEXT.build_vocab(train, val, vectors=self.vectors)
            if not (self.vectors is None):
                self.INPUT_TEXT.vocab.vectors.unk_init = init.xavier_uniform

            with open(self.input_vocab_path, 'wb')as f:
                dill.dump(self.INPUT_TEXT.vocab, f)

        # return self.INPUT_TEXT.vocab, self.OUTPUT_TEXT.vocab
        return

    def getEmneddingMatrix(self):
        return self.INPUT_TEXT.vocab.vectors