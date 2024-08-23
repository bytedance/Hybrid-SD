import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('agg')

def plot_dist_histogram(path, cpu_tensor, title, bins=2048, figsize=(16, 8)):
    print(f"Plotting {title}")
    shape_str = "x".join(map(str,list(cpu_tensor.size())))+"({})".format(np.prod(cpu_tensor.numpy().shape))
    data = cpu_tensor.view(-1).numpy()
    min_value, max_value = float(data.min()), float(data.max()) 
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, log=True)
    plt.title(f'{title} {shape_str}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(min_value, max_value)
    plt.grid(linestyle='-.')     
    plt.savefig(path, bbox_inches='tight', dpi=100)
    print(f"Save 2d dist plot to {path}")

def plot_3d_dist_histogram(path, cpu_tensor, name):
    ax = plt.figure().add_subplot(projection='3d')
    shape = list(cpu_tensor.shape)
    if cpu_tensor.ndim > 2:
        # conv weights
        cpu_tensor = cpu_tensor.view(cpu_tensor.size(0), -1)

    x_label = "Cin"
    y_label = "Cout"
    cout, cin = cpu_tensor.size()
    data = cpu_tensor.numpy()
    x = list(range(cin))
    for i in range(cout):
        y = i * np.ones(cin)
        z = data[i, :]
        ax.plot(x, y, z)
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.view_init(22, 28)
    ax.set_zlabel("Value")
    plt.subplots_adjust(0, 0, 1, 1)
    plt.suptitle(f'{name} ({shape})')
    plt.savefig(path, dpi=200)
    print(f"Save 3d hist plot to {path}")

def layer_boxplot(path, tensor_dict, title, figsize=(32, 16)):
    print(f"Plotting {title}")
    name_list = []
    data_list = []
    for name, item in tensor_dict.items():
        data = item["data"].detach().cpu().view(-1).numpy()
        class_name = item['class_name']
        # print(name, class_name, data.shape)
        name_list.append(name)
        data_list.append(data)
    plt.figure(figsize=figsize)
    plt.boxplot(data_list, showbox=False)
    plt.xticks(list(range(1, len(name_list)+1)), name_list, rotation='vertical', fontsize=20)
    plt.title(f'{title}', fontsize=40)
    plt.xlabel('Layer', fontsize=30)
    plt.ylabel('Values', fontsize=30)
    plt.grid(axis='x')
    plt.savefig(path, bbox_inches='tight', dpi=100)
    print(f"Save boxplot to {path}")
 