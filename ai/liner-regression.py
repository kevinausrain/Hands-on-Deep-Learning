from d2l import torch as d2l
import torch
from torch.utils import data


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2.0, -3.0, 4.0])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def show_result():
    d2l.set_figsize()
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    d2l.plt.show()


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load():
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    print(next(iter(data_iter)))


load()



