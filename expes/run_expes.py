import os
import sys
import pickle

from data import loading, processing
from model_eval import parallel_tuning


def create_output_folder(root_path, output_folder, parent="/ouptuts"):
    try:
        os.mkdir(root_path + parent)
    except FileExistsError:
        pass
    try:
        os.mkdir(root_path + parent + output_folder)
    except FileExistsError:
        pass
    rec_path = root_path + parent + output_folder
    return rec_path


def extract_key_speech(argv):
    # return "LA"
    try:
        key = argv[1]
        return key
    except IndexError:
        raise IndexError(
            'You need to define a vocal tract subproblem '
            'in the set {"LA", "LP", "TBCL", "VEL", "GLO", "TTCL", "TTCD"}')


def run_subexpe_speech(X ,Y, configs, regs, key, seed, input_indexing, output_indexing, n_folds, n_procs, min_nprocs):
    Xtrain, Ytrain_full_ext, Ytrain_full, Xtest, Ytest_full_ext, Ytest_full = processing.process_speech(
        X, Y, shuffle_seed=seed, n_train=300, normalize_domain=True, normalize_values=True)
    Ytrain_ext, Ytrain, Ytest_ext, Ytest \
        = Ytrain_full_ext[key], Ytrain_full[key], Ytest_full_ext[key], Ytest_full[key]
    # Cross validation of the regressors
    best_config, best_result, score_test = parallel_tuning.parallel_tuning(
        regs, Xtrain, Ytrain_ext, Xtest, Ytest, Xpred_train=None, Ypred_train=Ytrain,
        input_indexing=input_indexing, output_indexing=output_indexing,
        configs=configs, n_folds=n_folds, n_procs=n_procs, min_nprocs=min_nprocs)
    return best_config, best_result, score_test


def run_expe_speech(configs, regs, seeds, data_path, rec_path, input_indexing,
                    output_indexing, n_folds, n_procs, min_nprocs):
    scores_test, best_results, best_configs = list(), list(), list()
    X, Y = loading.load_raw_speech_dataset(data_path)
    key = extract_key_speech(sys.argv)
    for i in range(len(seeds)):
        best_config, best_result, score_test = run_subexpe_speech(
            X ,Y, configs, regs, key, seeds[i], input_indexing, output_indexing, n_folds, n_procs, min_nprocs)
        best_configs.append(best_config)
        best_results.append(best_results)
        scores_test.append(score_test)
        with open(rec_path + "/" + str(i) + "_" + key + ".pkl", "wb") as out:
            pickle.dump((best_configs, best_results, scores_test), out, pickle.HIGHEST_PROTOCOL)
    return best_configs, best_results, scores_test