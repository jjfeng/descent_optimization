import time
from Bio import Geo
import numpy as np
from common import testerror_logistic_grouped
import hillclimb_realdata_grouped_lasso_fullcv as hc
import gridsearch_grouped_lasso as gs_grouped
import gridsearch_lasso as gs
from method_results import MethodResults
from method_results import MethodResult
from realdata_common import *

NUM_ITERS = 1

TRAIN_SIZE = 40
VALIDATE_SIZE = 10
INIT_LAMBDAS = [0.5, 0.2]
GENE_EXPR_FILENAME = "realdata/GDS1615_full.soft"
PICKLE_DATA_FILENAME = "colitis_data.pkl"
PICKLE_BETAS_FILENAME = "colitis_betas.pkl"
CONTROL_LABEL = 0
DISEASE_LABEL = 1
KFOLDS = 5


def read_gene_expr_data(geneset_dict):
    """
    Read gene expression data from file. Returns
    X - array of unnormalized gene expression data, grouped by genesets
    y - control and disease labels
    geneset - filtered genesets that match the given geneset dictionary, same order as returned X array
    """
    handle = open(GENE_EXPR_FILENAME)
    records = Geo.parse(handle)

    # gsm ids of the normal subjects
    normal_subjects = []

    # geneset row ids
    X_groups = {}
    for k in geneset_dict.keys():
        X_groups[k] = set()

    X = []
    y = []

    i = 0
    for record in records:
        i += 1
        if i == 3:
            # Read patient labels so we can make the y vector
            attr = record.entity_attributes
            assert(attr["subset_description"] == "normal")
            normal_subjects = attr["subset_sample_id"].split(",")

        if i == 7:
            # Read actual gene expression data
            col_names = record.table_rows[0]
            gsm_idxs = []
            for idx, col_name in enumerate(col_names):
                if "GSM" == col_name[0:3]:
                    gsm_idxs.append(idx)

                    # populate the y matrix
                    # 1 means diseased. 0 means control.
                    y.append(CONTROL_LABEL if col_name in normal_subjects else DISEASE_LABEL)

            geneid_idx = col_names.index("Gene ID")

            feature_idx = 0
            for row in record.table_rows[1:]:
                geneid = row[geneid_idx]
                geneset = get_geneset_from_dict(geneset_dict, geneid)
                if geneset is not None:
                    # add feature idx to correct geneset
                    X_groups[geneset].add(feature_idx)

                    # append the gene expression data
                    X.append([float(row[i]) for i in gsm_idxs])

                    feature_idx += 1

    # Make feature groups
    X = np.matrix(X).T
    X_genesets = []
    genesets_included = []
    for geneset_key, geneset_col_idxs in X_groups.iteritems():
        if len(geneset_col_idxs) == 0:
            continue
        X_genesets.append(X[:, list(geneset_col_idxs)])
        genesets_included.append(geneset_key)

    y = np.matrix(y).T
    return X_genesets, y, genesets_included


def get_grouped_betas(beta, feature_group_sizes):
    final_betas =[]
    start_feature_idx = 0
    for feature_group_size in feature_group_sizes:
        end_feature_idx = start_feature_idx + feature_group_size
        final_betas.append(
            beta[start_feature_idx:end_feature_idx]
        )
        start_feature_idx = end_feature_idx
    return final_betas


def main():
    seed = int(np.random.rand() * 1e15)
    print "seed", seed
    np.random.seed(seed)

    geneset_dict = read_geneset_file()
    X_genesets, y, genesets = read_gene_expr_data(geneset_dict)
    print "num features", sum([X_genesets[i].shape[1] for i in range(0, len(X_genesets))])
    print "total genesets ever", len(X_genesets)
    X_genesets = normalize_data(X_genesets)

    hc_results = MethodResults("HC")
    gs_grouped_results = MethodResults("GS_Grouped")
    gs_results = MethodResults("GS_Lasso")
    for i in range(0, NUM_ITERS):
        X_groups_train_validate, y_train_validate, X_groups_test, y_test = shuffle_and_split_data_full_cv(
            X_genesets, y, TRAIN_SIZE + VALIDATE_SIZE)
        feature_group_sizes = [Xg.shape[1] for Xg in X_groups_train_validate]
        X_test = np.hstack(X_groups_test)

        for j in range(0, 3):
            start = time.time()
            hc_complete_beta, hc_validate_cost, hc_cost_path = hc.run_for_lambdas(X_groups_train_validate, y_train_validate, feature_group_sizes, KFOLDS, init_lambdas=INIT_LAMBDAS)
            hc_betas = get_grouped_betas(hc_complete_beta, feature_group_sizes)
            hc_runtime = time.time() - start
            hc_nonzeros = get_num_nonzero_betas(hc_betas, genesets, threshold=1e-6)
            print "hc 1e-6", hc_nonzeros

            hc_test, hc_rate = testerror_logistic_grouped(X_test, y_test, hc_betas)
            print "hc_test", hc_test, "hc_rate", hc_rate
            hc_results.append(MethodResult(test_err=hc_test, validation_err=hc_validate_cost, sensitivity=hc_rate, runtime=hc_runtime))
            hc_results.print_results()

        # start = time.time()
        # gs_grouped_complete_beta, gs_grouped_validate_cost = gs_grouped.run_classify_fullcv(X_groups_train_validate, y_train_validate, feature_group_sizes, KFOLDS)
        # gs_grouped_betas = get_grouped_betas(gs_grouped_complete_beta, feature_group_sizes)
        # gs_grouped_runtime = time.time() - start
        # gs_grouped_nonzeros = get_num_nonzero_betas(gs_grouped_betas, genesets, threshold=1e-6)
        # print "gs_grouped 1e-6", gs_grouped_nonzeros
        #
        # start = time.time()
        # gs_complete_beta, gs_validate_cost = gs.run_classify_fullcv(X_groups_train_validate, y_train_validate, feature_group_sizes, KFOLDS)
        # gs_betas = get_grouped_betas(gs_complete_beta, feature_group_sizes)
        # gs_runtime = time.time() - start
        # gs_nonzeros = get_num_nonzero_betas(gs_betas, genesets, threshold=1e-6)
        # print "gs 1e-6", gs_nonzeros
        #
        # print "================= hc ======================"
        # hc_test, hc_rate = testerror_logistic_grouped(X_test, y_test, hc_betas)
        # print "hc_test", hc_test, "hc_rate", hc_rate
        # hc_results.append(MethodResult(test_err=hc_test, validation_err=hc_validate_cost, sensitivity=hc_rate, runtime=hc_runtime))
        #
        # print "================= gs grouped ======================"
        # gs_grouped_test, gs_grouped_rate = testerror_logistic_grouped(X_test, y_test, gs_grouped_betas)
        # print "gs_grouped_test", gs_grouped_test, "gs_grouped_rate", gs_grouped_rate
        # gs_grouped_results.append(MethodResult(test_err=gs_grouped_test, validation_err=gs_grouped_validate_cost, sensitivity=gs_grouped_rate, runtime=gs_grouped_runtime))
        #
        # print "================= gs ======================"
        # gs_test, gs_rate = testerror_logistic_grouped(X_test, y_test, gs_betas)
        # print "gs_test", gs_test, "gs_rate", gs_rate
        # gs_results.append(MethodResult(test_err=gs_test, validation_err=gs_validate_cost, sensitivity=gs_rate, runtime=gs_runtime))
        #
        # print "ITERATION", i
        # hc_results.print_results()
        # gs_grouped_results.print_results()
        # gs_results.print_results()


if __name__ == "__main__":
    main()
