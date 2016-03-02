import numpy as np

def shuffle(x,y):
    d = np.append(x,y,axis=1)
    np.random.shuffle(d)
    _x = d[:,:x.shape[1]]
    _y = d[:,x.shape[1]:]
    return _x,_y

def split_train_test(x,y,ratio=0.75):
    split_size = int(len(x)*ratio)
    if len(x) != len(y):
        raise Exception('dimension unmatched')
    return x[:split_size],y[:split_size],x[split_size:],y[split_size:]

def cross_validation(model,x,y,k):
    if len(x) != len(y):
        raise Exception('dimension unmatched')
    step_size = int(len(x)/k)
    _x,_y = shuffle(x,y)
    scores = []
    for i in range(0,len(x),step_size):
        train_x = _x[i:i+step_size]
        train_y = _y[i:i+step_size]
        test_x = np.append(x[:i],x[i+step_size:],axis=0)
        test_y = np.append(y[:i],y[i+step_size:],axis=0)
        model.train(train_x,train_y)
        scores.append(model.score(test_x,test_y))
    return scores

def resampling(x,y,ratio=1.):
    sampling_size = int(len(x)*ratio)
    _x,_y = shuffle(x,y)

    idx = np.random.randint(0,sampling_size,sampling_size)
    resampled_x = _x[idx]
    resampled_y = _y[idx]

    return resampled_x,resampled_y