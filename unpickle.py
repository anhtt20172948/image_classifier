def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dict = unpickle('data/cifar-10-batches-py/data_batch_1')
print(dict.keys())
print(dict[b'data'])