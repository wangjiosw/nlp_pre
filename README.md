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
  - **in_tokenize:**    INPUT_TEXT's tokenize
  - **out_tokenize:**   OUT_TEXT's tokenize
  - **cols:**           ["input_col_name","output_col_name"]
  - **batch_size:**     iterator's batch size
  - **device:**         iterator's device torch.device('cpu') or torch.device('cuda')
  - **data_path:**      tran.csv, val.csv and test.csv（format:  input_text, output_text)'s dir path, default current dir
  - **vectors:**        pre-trained word embeddings
    
    available vectors input:
      - "glove.6B.100d"
      - ...

# seq2classesData \[ [源码](seq2classes_data.py) \]
> deal with seq to classes data . (e.g. sentiment analysis, emotion analysis ...)

**param:**
  - **in_tokenize:**    INPUT_TEXT's tokenize
  - **cols:**           ["input_col_name","output_col_name"]
  - **batch_size:**     iterator's batch size
  - **device:**         iterator's device torch.device('cpu') or torch.device('cuda')
  - **data_path:**      tran.csv, val.csv and test.csv（format:  input_text, output_text)'s dir path, default current dir
  - **vectors:**        pre-trained word embeddings
    
    available vectors input:
      - "glove.6B.100d"
      - ...