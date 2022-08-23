import numpy as np


def gd(gradient, start, lr, n_iter=1000, tolerance= 1e-06):
    print('99')
    vec= start
    for _ in range(n_iter):
        diff = -lr * gradient(vec)
        if np.all(np.abs(diff) <= tolerance):
            break
        vec += diff
    print(vec)
    return vec


def ssr_gd(
            gradient, x, y,
            start, lr=0.1,
            n_iter = 100,
            tolerance = 1e-6
            ):
    vec = start
    for _ in range(n_iter):
        diff = -lr*np.array(gradient(x, y, vec))
        if np.all(np.abs(diff) <= tolerance):
            break
        vec += diff
    return vec


def ssr_gradient(x, y, b):
    res = b[0] + b[1]*x - y
    return res.mean(), (res*x).mean()



def gradient_descent(
    gradient, x, y, start, learn_rate=0.1, n_iter=50, tolerance=1e-06,
    dtype="float64"
    ):
    # Checking if the gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")

    # Setting up the data type for NumPy arrays
    dtype_ = np.dtype(dtype)

    # Converting x and y to NumPy arrays
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    if x.shape[0] != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")

    # Initializing the values of the variables
    vector = np.array(start, dtype=dtype_)

    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")

    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

    # Setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    # Performing the gradient descent loop
    for _ in range(n_iter):
        # Recalculating the difference
        diff = -learn_rate * np.array(gradient(x, y, vector), dtype_)

        # Checking if the absolute difference is small enough
        if np.all(np.abs(diff) <= tolerance):
            break

        # Updating the values of the variables
        vector += diff
    print(vector)
    return vector if vector.shape else vector.item()



def sgd(
        gradient, x, y, start, lr=0.05, batch_size = 8, n_iter = 1000,
        tolerance = 1e-6, dtype = 'float64', random_state = None
):
    assert (callable(gradient), f'Gradient should be callable, got None')
    dtype_ = np.dtype(dtype)

    x, y = np.array(x, dtype = dtype_), np.array(y, dtype = dtype_)
    x_d1 = x.shape[0]
    assert (x_d1 == y.shape[0], f'x and y should have same length, current x, y is {x_d1, y.shape[0]}')

    xy = np.c_[x.reshape(x_d1, -1), y.reshape(x_d1, 1)]
    seed = None if random_state is None else int(random_state)
    rdg = np.random.default_rng(seed=seed)

    vec = np.array(start, dtype = dtype_)
    lr = np.array(lr, dtype= dtype_)
    assert np.any(lr > 0)

    batch_size = int(batch_size)
    assert 0 < batch_size < x_d1

    tolerance = np.array(tolerance, dtype = dtype_)
    assert tolerance > 0

    for _ in range(n_iter):

        rdg.shuffle(xy)
        for rstart in range(0, x_d1, batch_size):
            stop = rstart + batch_size
            x_batch, y_batch = xy[rstart:stop, :-1], xy[rstart:stop, -1:]
            grad = np.array(gradient(x_batch, y_batch, vec), dtype_)
            diff = -lr* grad

            if np.all(np.abs(diff) <= tolerance):
                break
            vec += diff
    print(vec)
    return vec if vec.shape else vec.item()


def sgd_decay(
        gradient, x, y, n_vars = None, start = None, lr=0.05, decay_rate = 0.0, batch_size = 8, n_iter = 1000,
        tolerance = 1e-6, dtype = 'float64', random_state = None
    ):
    assert (callable(gradient), f'Gradient should be callable, got None')
    dtype_ = np.dtype(dtype)

    x, y = np.array(x, dtype = dtype_), np.array(y, dtype = dtype_)
    x_d1 = x.shape[0]
    assert (x_d1 == y.shape[0], f'x and y should have same length, current x, y is {x_d1, y.shape[0]}')

    xy = np.c_[x.reshape(x_d1, -1), y.reshape(x_d1, 1)]
    seed = None if random_state is None else int(random_state)
    rdg = np.random.default_rng(seed=seed)

    vec = np.array(
    rdg.normal(size = int(n_vars)). astype(dtype_)
    if start is None else
    np.array(start, dtype = dtype_)
    )
    lr = np.array(lr, dtype= dtype_)
    assert np.any(lr > 0)

    decay_rate = np.array(decay_rate, dtype = dtype_)
    assert (0 < decay_rate < 1, f'the decay rate should in the range 0, 1, got {decay_rate}')


    batch_size = int(batch_size)
    assert 0 < batch_size < x_d1

    tolerance = np.array(tolerance, dtype = dtype_)
    assert tolerance > 0


    diff = 0

    for _ in range(n_iter):

        rdg.shuffle(xy)
        for rstart in range(0, x_d1, batch_size):
            stop = rstart + batch_size
            x_batch, y_batch = xy[rstart:stop, :-1], xy[rstart:stop, -1:]
            grad = np.array(gradient(x_batch, y_batch, vec), dtype_)
            diff = decay_rate*diff - lr* grad

            if np.all(np.abs(diff) <= tolerance):
                break
            vec += diff
    print(vec)
    return vec if vec.shape else vec.item()









if __name__ == '__main__':

    x = np.array([5, 15, 25, 35, 45, 55])
    y = np.array([5, 20, 14, 32, 22, 38])

    gradient_descent(ssr_gradient, x, y, start = [0.5, 0.5], learn_rate = 0.0006, n_iter = 100000)

    sgd_decay(
    ssr_gradient, x, y, n_vars = 2, lr = 0.0006, decay_rate = 0.88,
    batch_size= 2, n_iter = 100000, random_state = 5
    )


    sgd(ssr_gradient, x, y, start = [0.5, 0.5], lr = 0.0006,
    batch_size= 2, n_iter = 100000, random_state = 5)




    print('dd')
    gd(
    gradient = lambda x: 4*x**3 - 10*x - 3, start = 0.0, lr = 0.1, n_iter = 50
    )
