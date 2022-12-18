from itertools import product, chain


# Training parameters:
# model, mini_batch_size, criterion, optimizer, nb_epochs,
# weight_loss_classes, weight_loss_pairs,
# freeze_epochs,
# convert_classes_to_one_hot, convert_ioe_to_one_hot

# Model parameters:
# Siamese: conv_block_parameters, fc_parameters, predict
# CNN: conv_block_parameters, fc_parameters, predict
# FC: fc_parameters, predict
# ConvBlock:       ch1=64, ch2=64,
#                  conv1_kernel=3, conv2_kernel=2,
#                  use_max_pool1=True, max_pool1_kernel=3, max_pool_stride1=1,
#                  use_max_pool2=True, max_pool2_kernel=2, max_pool_stride2=1,
#                  dropout1=0.0, dropout2=0.0,
#                  activation1=nn.ReLU(), activation2=nn.ReLU(),
#                  use_batch_norm=True, use_skip_connections
# FCBlock:         fc, out=10,
#                  dropout=0.0,
#                  activation1=nn.ReLU(), activation2=nn.ReLU(),
#                  use_batch_norm=True

def parameters_combinations(*params, labels: list) -> list:
    """
    Generate all combinations of parameters
    :param params: list[list] parameters to combine
    :param labels: list[str] labels of parameters
    :return: list[dict] list of parameters combinations
    """
    return [dict(zip(labels, p)) for p in product(*params)]


def parameters_combinations_from_dict(params_dict: dict) -> list:
    """
    Generate all combinations of parameters
    :param params_dict: dict of parameters to combine
    :return: list of parameters combinations
    """
    return parameters_combinations(*params_dict.values(), labels=list(params_dict.keys()))


def mix_parameters_combinations(*params_combinations) -> list:
    """
    Mix all combinations of parameters
    :param params_combinations: list[list[dict]] of parameters to combine
    :return: list of parameters combinations
    """
    return [dict(zip(chain(*[combination.keys() for combination in multi_combination]),
                     chain(*[combination.values() for combination in multi_combination])))
            for multi_combination in product(*params_combinations)]


def print_parameters_combinations(params_combinations: list, highlight: bool = True):
    """
    Print all combinations of parameters
    :param params_combinations: list[dict] of parameters to print
    :param highlight: bool if True, highlight the differences between two consecutive combinations
    """
    previous = None
    for p in params_combinations:
        s = str(p)
        if previous is not None and highlight:
            add_end = False
            for i in range(len(s)):
                if s[i] != previous[i]:
                    print("\033[1;31m" + s[i], end="")
                    add_end = True
                elif add_end:
                    print("\033[0;0m" + s[i], end="")
                    add_end = False
                else:
                    print(s[i], end="")
            print("\033[0;0m")
        else:
            print(s)
            previous = s


def _test():
    params = parameters_combinations_from_dict(
        {'a': [1, 2], 'b': [3, 4], 'c': [5]})
    print(params)
    params = parameters_combinations([1, 2], [3, 4], [5], labels=['a', 'b', 'c'])
    print(params)
    params = mix_parameters_combinations(
        parameters_combinations_from_dict(
            {'a': [1], 'b': [2, 3]}),
        parameters_combinations_from_dict(
            {'c': [5, 6], 'd': [7]}))
    print(params)


if __name__ == '__main__':
    _test()
