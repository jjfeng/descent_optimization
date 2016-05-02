import time
from Bio import Geo
import numpy as np
from common import testerror_logistic_grouped
import hillclimb_realdata_grouped_lasso as hc
import gridsearch_grouped_lasso as gs_grouped
import gridsearch_lasso as gs
from method_results import MethodResults
from method_results import MethodResult
from realdata_common import *

TRAIN_SIZE = 52
VALIDATE_SIZE = 13
INIT_LAMBDAS = [7 * 1e-2]
GENE_EXPR_FILENAME = "realdata/GSE27562_family.soft"
GENE_SERIES_FILENAME = "realdata/GSE27562_series_matrix.txt"
PICKLE_DATA_FILE = "breastcancer_data.pkl"
CONTROL_LABEL = 0
DISEASE_LABEL = 1
NUM_ITERS = 1


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
    X_grouped = {}
    for k in geneset_dict.keys():
        X_grouped[k] = set()

    X = []
    y = []

    i = 0
    probe_dict = {}
    num_probes = 0
    for record in records:
        if record.entity_type == "PLATFORM":
            table_header = record.table_rows[0]
            geneid_idx = table_header.index("ENTREZ_GENE_ID")

            feature_idx = 0
            for r in record.table_rows[1:]:
                entrez_gene_id = r[geneid_idx]
                geneset = get_geneset_from_dict(geneset_dict, entrez_gene_id)
                if geneset is not None:
                    probe_dict[r[0]] = feature_idx
                    X_grouped[geneset].add(feature_idx)
                    feature_idx += 1
            num_probes = len(probe_dict.keys())

        if record.entity_type == "SAMPLE":
            print record.entity_id
            phenotype = record.entity_attributes["Sample_characteristics_ch1"][0].replace("phenotype: ", "")
            if phenotype == "Malignant" or phenotype == "Pre-Surgery (aka Malignant)":
                y.append(DISEASE_LABEL)
            elif phenotype == "Benign":
                y.append(CONTROL_LABEL)
            else:
                continue

            sample_data = []
            sample_dict = {}
            for row in record.table_rows[1:]:
                probe_id = row[0]
                sample_dict[probe_id] = float(row[1])

            X_sample = [0] * num_probes
            for probe_id, feature_idx in probe_dict.iteritems():
                X_sample[feature_idx] = sample_dict[probe_id]
            X.append(X_sample)

    X = np.matrix(X)
    X_genesets = []
    genesets_included = []
    for geneset_key, geneset_col_idxs in X_grouped.iteritems():
        if len(geneset_col_idxs) == 0:
            continue
        X_genesets.append(X[:, list(geneset_col_idxs)])
        genesets_included.append(geneset_key)

    y = np.matrix(y).T

    return X_genesets, y, genesets_included


def main():
    seed = int(np.random.rand() * 1e15)
    print "seed", seed
    np.random.seed(seed)

    # geneset_dict = read_geneset_file()
    # X_genesets, y, genesets = read_gene_expr_data(geneset_dict)
    # print "num features", sum([X_genesets[i].shape[1] for i in range(0, len(X_genesets))])
    # print "total genesets ever", len(X_genesets)
    # X_genesets = normalize_data(X_genesets)

    hc_results = MethodResults("HC")
    gs_grouped_results = MethodResults("GS_Grouped")
    gs_results = MethodResults("GS_Lasso")
    for i in range(0, NUM_ITERS):
        # X_groups_train, y_train, X_groups_validate, y_validate, X_groups_test, y_test = shuffle_and_split_data(
        #     X_genesets, y, TRAIN_SIZE, VALIDATE_SIZE)
        # pickle_data(PICKLE_DATA_FILE, X_groups_train, y_train, X_groups_validate, y_validate, X_groups_test, y_test, genesets)

        X_groups_train, y_train, X_groups_validate, y_validate, X_groups_test, y_test, genesets = load_pickled_data(PICKLE_DATA_FILE)
        print "np.sum(y_train)", np.sum(y_train)
        print "y_train.size", y_train.size

        X_validate = np.hstack(X_groups_validate)
        X_test = np.hstack(X_groups_test)

        start = time.time()
        hc_betas = []
        hc_validate_cost = 1e10
        for init_lambda in INIT_LAMBDAS:
            betas, cost_path = hc.run(X_groups_train, y_train, X_groups_validate, y_validate, initial_lambda=init_lambda)
            validate_cost, _ = testerror_logistic_grouped(X_validate, y_validate, betas)
            if validate_cost < hc_validate_cost:
                hc_validate_cost = validate_cost
                hc_betas = betas

            if len(cost_path) > 2:
                break
        hc_runtime = time.time() - start
        print "hc 1e-6", get_num_nonzero_betas(hc_betas, genesets, threshold=1e-6)
        print "hc 1e-8", get_num_nonzero_betas(hc_betas, genesets, threshold=1e-8)
        print "hc 1e-10", get_num_nonzero_betas(hc_betas, genesets, threshold=1e-10)

        print "hc_validate_cost", hc_validate_cost

        # start = time.time()
        # gs_grouped_betas, gs_grouped_cost = gs_grouped.run_classify(X_groups_train, y_train, X_groups_validate, y_validate)
        # gs_grouped_runtime = time.time() - start
        # print "gs_grouped 1e-6", get_num_nonzero_betas(gs_grouped_betas, genesets, threshold=1e-6)
        # print "gs_grouped 1e-8", get_num_nonzero_betas(gs_grouped_betas, genesets, threshold=1e-8)
        # print "gs_grouped 1e-10", get_num_nonzero_betas(gs_grouped_betas, genesets, threshold=1e-10)
        # gs_grouped_validate_cost, gs_grouped_validate_rate = testerror_logistic_grouped(X_validate, y_validate, gs_grouped_betas)
        # print "gs_grouped_validate_cost", gs_grouped_validate_cost

        start = time.time()
        gs_betas, gs_cost = gs.run_classify(X_groups_train, y_train, X_groups_validate, y_validate)
        gs_runtime = time.time() - start
        print "gs 1e-6", get_num_nonzero_betas(gs_betas, genesets, threshold=1e-6)
        print "gs 1e-8", get_num_nonzero_betas(gs_betas, genesets, threshold=1e-8)
        print "gs 1e-10", get_num_nonzero_betas(gs_betas, genesets, threshold=1e-10)
        gs_validate_cost, gs_validate_rate = testerror_logistic_grouped(X_validate, y_validate, gs_betas)
        print "gs_validate_cost", gs_validate_cost

        # print "================= hc ======================"
        # hc_test, hc_rate = testerror_logistic_grouped(X_test, y_test, hc_betas)
        # print "hc_test", hc_test
        # hc_results.append(MethodResult(test_err=hc_test, validation_err=hc_validate_cost, sensitivity=hc_rate, runtime=hc_runtime))

        # print "================= gs grouped ======================"
        # gs_grouped_test, gs_grouped_rate = testerror_logistic_grouped(X_test, y_test, gs_grouped_betas)
        # print "gs_grouped_test", gs_grouped_test
        # gs_grouped_results.append(MethodResult(test_err=gs_grouped_test, validation_err=gs_grouped_validate_cost, sensitivity=gs_grouped_rate, runtime=gs_grouped_runtime))

        # print "================= gs ======================"
        # gs_test, gs_rate = testerror_logistic_grouped(X_test, y_test, gs_betas)
        # print "gs_test", gs_test
        # gs_results.append(MethodResult(test_err=gs_test, validation_err=gs_validate_cost, sensitivity=gs_rate, runtime=gs_runtime))

        print "ITERATION", i
        hc_results.print_results()
        gs_grouped_results.print_results()
        gs_results.print_results()

if __name__ == "__main__":
    main()
