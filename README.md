# nlp_pre
nlp module

## Capsule(in_units,in_channels, num_capsule, dim_capsule, routings=3) \[ [源码](capsule.py) \]
> A Capsule Implement with pytorch

### input: (batch_size,in_channel,in_units)
- **in_units**   : input unit's num
- **in_channels**: input vec's channel
- **num_capsule**: num of output capsule
- **dim_capsule**: each capsule's dim
### return: (batch_size,dim_capsule,num_capsule)
***

## mnist_data \[ [源码](mnist_data.py) \]
> get mnist data in pytorch
### get mnist train data and test date in data_loader mode(pytorch)
- train_loader
- test_loader
### usage:
```
test_iter = iter(test_loader)
imgs,labels = next(test_iter)
```
or 
```
for (data, target) in train_loader:
    ...
```
***

# seq2seqData \[ [源码](seq2seq_data.py) \]
> deal with seq to seq nlp data. (e.g. translate English to French)

**param:**
  - **input_fied:**    input text's field
  - **output_field:**   output text's field
  - **cols:**           ["input_col_name","output_col_name"]
  - **batch_size:**     iterator's batch size
  - **device:**         iterator's device torch.device('cpu') or torch.device('cuda')
  - **data_path:**      tran.csv, val.csv and test.csv（format:  input_text, output_text)'s dir path, default current dir
  - **in_vectors:**     pre-trained word embeddings
  - **out_vectors:**    pre-trained word embeddings

    vectors: one of or a list containing instantiations of the GloVe, CharNGram, or Vectors classes. Alternatively, one
    of or a list of available pretrained vectors:
    - charngram.100d
    - fasttext.en.300d
    - fasttext.simple.300d
    - glove.42B.300d
    - glove.840B.300d
    - glove.twitter.27B.25d
    - glove.twitter.27B.50d
    - glove.twitter.27B.100d
    - glove.twitter.27B.200d
    - glove.6B.50d
    - glove.6B.100d
    - glove.6B.200d
    - glove.6B.300d
    
    Remaining keyword arguments: Passed to the constructor of Vectors classes.
## define Field
class Field(RawField):
    """Defines a datatype together with instructions for converting to Tensor.

    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.

    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: torch.long.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab.
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy tokenizer is
            used. If a non-serializable function is passed as an argument,
            the field will not be able to be serialized. Default: string.split.
        tokenizer_language: The language of the tokenizer to be constructed.
            Various languages currently supported only in SpaCy.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
        truncate_first: Do the truncating of the sequence at the beginning. Default: False
        stop_words: Tokens to discard during the preprocessing step. Default: None
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False
    """
### e.g.
```bash
from torchtext import data
input_field = data.Field(batch_first=True, tokenize=in_tokenize, lower=True)
output_field = Field(batch_first=True, tokenize=out_tokenize, init_token="<sos>", eos_token="<eos>", lower=True)
```

# seq2classesData \[ [源码](seq2classes_data.py) \]
> deal with seq to classes data . (e.g. sentiment analysis, emotion analysis ...)

**param:**
  - **input_fied:**    input text's field
  - **output_field:**   output text's field
  - **cols:**           ["input_col_name","output_col_name"]
  - **batch_size:**     iterator's batch size
  - **device:**         iterator's device torch.device('cpu') or torch.device('cuda')
  - **data_path:**      tran.csv, val.csv and test.csv（format:  input_text, output_text)'s dir path, default current dir
  - **vectors:**        pre-trained word embeddings
    
    available vectors input:
      - "glove.6B.100d"
      - ...