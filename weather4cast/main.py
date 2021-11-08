import argparse
from functools import partial

import datasets
import ensemble
import models
from models import rnn3_model, rnn4_model, rnn5_model, crr_combo_model


def get_ensemble_weights(weights="ridge"):
    assert (weights in ["equal", "ridge", "ridge_lagrange"])
    if weights == "equal":
        w = {
            "temperature": [1/5]*5,
            "crr_intensity": [1/2]*2,
            "asii_turb_trop_prob": [1/3]*3,
            "cma": [1/2]*2,
        }
    elif weights == "ridge":
        w = {
            "temperature": [0.1455, 0.2666, 0.0904, 0.2487, 0.2457],
            "crr_intensity": [0.5206, 0.5320],
            "asii_turb_trop_prob": [0.2722, 0.2941, 0.4344],
            "cma": [0.5165, 0.4864],
        }
    elif weights == "ridge_lagrange":
        w = {
            "temperature": [0.1455, 0.2666, 0.0904, 0.2487, 0.2457],
            "crr_intensity": [0.5122, 0.4878],
            "asii_turb_trop_prob": [0.2722, 0.2941, 0.4344],
            "cma": [0.5165, 0.4864],        
        }
    return w


def get_model(model_type):
    model_func = {
        "resgru_deep": models.rnn3_model,
        "resgru_shallow": models.rnn4_model,
        "convgru_deep": models.rnn5_model,
        "convgru_old": models.rnn2_model,
        "crr_combo_model_old": models.crr_combo_model,
        "crr_combo_model_new": partial(crr_combo_model, model_func=rnn5_model)
    }[model_type]
    return model_func


def regions_for_dir(comp_dir):
    if "core" in comp_dir:
        regions = ["R1", "R2", "R3", "R7", "R8"]
    else:
        regions = ["R4", "R5", "R6", "R9", "R10", "R11"]


def build_model_list(w):
    modif_crr_model = partial(crr_combo_model, model_func=rnn5_model)
    var_models = [
        ("CTTH", "temperature", [
            ("../models/srnn_adabelief_1-temperature.h5", rnn4_model, w["temperature"][0]),
            ("../models/srnn_adabelief_2-temperature.h5", rnn4_model, w["temperature"][1]),
            ("../models/srnn_adabelief_3-temperature.h5", rnn4_model, w["temperature"][2]),
            ("../models/srnn_adabelief_4-temperature.h5", rnn4_model, w["temperature"][3]),
            ("../models/srnn_adabelief_5-temperature.h5", rnn4_model, w["temperature"][4]),
        ]),
        ("CRR", "crr_intensity", [
            ("../models/srnn_adabelief_3-crr_intensity.h5", crr_combo_model, w["crr_intensity"][0]),
            ("../models/srnn_adabelief_4-crr_intensity.h5", modif_crr_model, w["crr_intensity"][1]),
        ]),
        ("ASII", "asii_turb_trop_prob", [
            ("../models/srnn_adabelief_1-asii_turb_trop_prob.h5", rnn4_model, w["asii_turb_trop_prob"][0]),
            ("../models/srnn_adabelief_2-asii_turb_trop_prob.h5", rnn3_model, w["asii_turb_trop_prob"][1]),
            ("../models/srnn_adabelief_3-asii_turb_trop_prob.h5", rnn3_model, w["asii_turb_trop_prob"][2]),
        ]),
        ("CMA", "cma", [
            ("../models/srnn_adabelief_1-cma.h5", rnn4_model, w["cma"][0]),
            ("../models/srnn_adabelief_2-cma.h5", rnn3_model, w["cma"][1]),
        ]),
    ]
    return var_models


def generate_predictions(
    submission_dir,
    comp_dir="w4c-core-stage-1",
    regions=None,
    weights="ridge"
):
    if regions is None:
        regions = regions_for_dir(comp_dir)
    
    batch_gen_valid = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="test",
        augment=False,
        shuffle=False
    )

    w = get_ensemble_weights(weights=weights)
    var_models = build_model_list(w)

    comb_model = models.ensemble_model_with_weights(
        batch_gen_valid, var_models=var_models, logit=(weights!="equal"))

    datasets.generate_submission(
        comb_model,
        submission_dir,
        regions=regions,
        comp_dir=comp_dir
    )


def evaluate(
    comp_dir="w4c-core-stage-1",
    regions=None,
    dataset="CTTH",
    variable="temperature",
    batch_size=32,
    model_type="resgru",
    weight_fn=None
):
    if regions is None:
        regions = regions_for_dir(comp_dir)

    batch_gen_valid = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="validation",
        augment=False,
        shuffle=False
    )
    datasets.setup_univariate_batch_gen(batch_gen_valid, 
        dataset, variable, batch_size=batch_size)
    model_func = get_model(model_type)
    model = models.init_model(batch_gen_valid, model_func=model_func)
    if weight_fn is not None:
        model.load_weights(weight_fn)    

    eval_results = model.evaluate(batch_gen_valid)
    print(eval_results)


def evaluate_ensemble(
    comp_dir="w4c-core-stage-1",
    regions=None,
    dataset="CTTH",
    variable="temperature",
    batch_size=32,
    model_type="resgru",
    weight_fn=None,
    ensemble_weights="ridge"
):
    if regions is None:
        regions = regions_for_dir(comp_dir)

    batch_gen_valid = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="validation",
        augment=False,
        shuffle=False
    )
    datasets.setup_univariate_batch_gen(batch_gen_valid, 
        dataset, variable, batch_size=batch_size)
    
    w = get_ensemble_weights(weights=ensemble_weights)
    var_models = build_model_list(w)
    var_list = [v[1] for v in var_models]  
    ind = var_list.index(variable)
    model_list = var_models[ind][2]

    var_model_list = []
    var_ensemble_weights = []
    for (model_weights, model_func, ensemble_weight) in model_list:            
        model = models.init_model(batch_gen_valid, model_func=model_func, 
            compile=False, init_strategy=False)
        model.load_weights(model_weights)
        var_model_list.append(model)
        var_ensemble_weights.append(ensemble_weight)

    logit = (ensemble_weights != "equal")
    weighted_model = ensemble.weighted_model(
        var_model_list, var_ensemble_weights, variable,
        logit=(logit and (variable=="asii_turb_trop_prob"))
    )

    eval_results = weighted_model.evaluate(batch_gen_valid)
    print(eval_results)


def train(
    comp_dir="w4c-core-stage-1",
    regions=None,
    dataset="CTTH",
    variable="temperature",
    batch_size=32,
    model_type="resgru_shallow",
    weight_fn=None
):
    if regions is None:
        regions = regions_for_dir(comp_dir)

    batch_gen_train = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="training"
    )
    batch_gen_valid = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="validation",
        augment=False,
        shuffle=False
    )
    datasets.setup_univariate_batch_gen(batch_gen_train, 
        dataset, variable, batch_size=batch_size)
    datasets.setup_univariate_batch_gen(batch_gen_valid, 
        dataset, variable, batch_size=batch_size)
    model_func = get_model(model_type)
    model = models.init_model(batch_gen_valid, model_func=model_func)

    models.train_model(model, batch_gen_train, batch_gen_valid,
        weight_fn=weight_fn)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
        help="submit / evaluate / train")
    parser.add_argument('--comp_dir', type=str,
        help="Directory where the data are located")
    parser.add_argument('--regions', type=str, 
        help="Comma-separated list or regions, default all regions for comp_dir")
    parser.add_argument('--submission_dir', type=str, default="",
        help="Directory to save the results in, will be created if needed")
    parser.add_argument('--batch_size', type=int, default=32,
        help="Batch size for training / evaluation")
    parser.add_argument('--dataset', type=str, default="",
        help="Dataset for training / evaluation")
    parser.add_argument('--variable', type=str, default="",
        help="Variable for training / evaluation")
    parser.add_argument('--weights', type=str, default="",
        help="Model weight file for training / evaluation")
    parser.add_argument('--model', type=str, default="resgru",
        help="Model type for training / evaluation, either 'convgru' or 'resgru'")
    parser.add_argument('--ensemble_weights', type=str, default="ridge",
        help="Ensemble weights, either 'ridge', 'equal' or 'ridge_lagrange'")

    args = parser.parse_args()
    mode = args.mode
    regions = args.regions
    if not regions:
        regions = None
    else:
        regions = regions.split(",")
    comp_dir = args.comp_dir

    if mode == "submit":        
        submission_dir = args.submission_dir
        assert(submission_dir != "")
        generate_predictions(submission_dir,
            comp_dir=comp_dir, regions=regions)
    elif mode in ["evaluate", "evaluate_ensemble", "train"]:
        batch_size = args.batch_size
        dataset = args.dataset
        variable = args.variable
        weight_fn = args.weights
        model_type = args.model
        ensemble_weights = args.ensemble_weights
        assert(dataset in ["CTTH", "CRR", "ASII", "CMA"])
        assert(variable in ["temperature", "crr_intensity",
            "asii_turb_trop_prob", "cma"])
        if mode == "evaluate":
            evaluate(comp_dir=comp_dir, regions=regions, dataset=dataset,
                variable=variable, batch_size=batch_size, weight_fn=weight_fn,
                model_type=model_type)
        elif mode == "evaluate_ensemble":
            evaluate_ensemble(comp_dir=comp_dir, regions=regions, 
                dataset=dataset, variable=variable, batch_size=batch_size,
                weight_fn=weight_fn, model_type=model_type,
                ensemble_weights=ensemble_weights)
        else:
            train(comp_dir=comp_dir, regions=regions, dataset=dataset,
                variable=variable, batch_size=batch_size, weight_fn=weight_fn,
                model_type=model_type)
