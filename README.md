# nlp_pre
nlp module

## Capsule(in_units,in_channels, num_capsule, dim_capsule, routings=3)
> A Capsule Implement with pytorch

### input: (batch_size,in_channel,in_units)
- **in_units**   : input unit's num
- **in_channels**: input vec's channel
- **num_capsule**: num of output capsule
- **dim_capsule**: each capsule's dim
### return: (batch_size,dim_capsule,num_capsule)
***

## mnist_data.py
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