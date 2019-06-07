# nlp_pre
nlp preprocess

## Capsule(in_units,in_channels, num_capsule, dim_capsule, routings=3)

### input: (batch_size,in_channel,in_units)
- **in_units**   : input unit's num
- **in_channels**: input vec's channel
- **num_capsule**: num of output capsule
- **dim_capsule**: each capsule's dim
### return: (batch_size,dim_capsule,num_capsule)
