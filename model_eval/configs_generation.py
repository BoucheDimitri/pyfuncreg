from collections.abc import Iterable
import itertools


def configs_combinations(params_dict, exclude_list=()):
    """
    For a dictionary, generate dictionaries by combining all the possible values for the keys pointing to iterables.

    Parameters
    ----------
    params_dict: dict
        Input dictionary
    exclude_list: list or tuple
        keys corresponding to Iterables to exclude from the combination stripping process

    Returns
    -------
    list
        List of dictionaries
    """
    expe_queue = []
    multi_params = []
    uni_params = []
    multi_vals = []
    for key in params_dict.keys():
        if isinstance(params_dict[key], Iterable) and not isinstance(params_dict[key], str) and key not in exclude_list:
            multi_params.append(key)
            multi_vals.append(params_dict[key])
        else:
            uni_params.append(key)
    for params_set in itertools.product(*multi_vals):
        new_dict = {key: params_dict[key] for key in uni_params}
        count = 0
        for param in params_set:
            new_dict[multi_params[count]] = param
            count += 1
        expe_queue.append(new_dict)
    return expe_queue


def subconfigs_combinations(subconfig_name, params_dict, exclude_list=()):
    """
    For a dictionary, generate dictionaries by combining all the possible values for the keys pointing to iterables.

    Parameters
    ----------
    subconfig_name: str
        The name of the subconfig
    params_dict: dict
        Input dictionary
    exclude_list: list or tuple
        keys corresponding to Iterables to exclude from the combination stripping process

    Returns
    -------
    list
        List of tuples of the form (subconfig_name, config_dict)
    """
    expe_queue = configs_combinations(params_dict, exclude_list)
    return [(subconfig_name, config) for config in expe_queue]


def combine_config_lists(configs1, configs2):
    """
    Combine two lists of configs using cartesian product

    Parameters
    ----------
    configs1: list
        List of configuration dictionaries
    configs2: list
        List of configuration dictionaries

    Returns
    -------
    list
        List of tuples of possible configuration dictionaries
    """
    return list(itertools.product(configs1, configs2))





