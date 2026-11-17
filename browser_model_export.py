"""Serialize fitted sklearn classifiers into browser-runnable parameters."""

import numpy as np
from sklearn.pipeline import Pipeline

def floats(values):
    return np.asarray(values).tolist()


def tree_data(tree, classification=False):
    structure = tree.tree_
    data = {
        "left": structure.children_left.tolist(),
        "right": structure.children_right.tolist(),
        "feature": structure.feature.tolist(),
        "threshold": floats(structure.threshold),
    }
    values = structure.value[:, 0, :]
    data["value"] = floats(values)
    data["classification"] = classification
    return data


def export_one(name, pipeline):
    steps = dict(pipeline.named_steps) if isinstance(pipeline, Pipeline) else {}
    classifier = steps.get("classifier", pipeline)
    imputer = steps.get("imputer")
    exported = {"name": name, "classes": [str(x) for x in classifier.classes_]}
    if imputer is not None:
        exported["imputer"] = floats(imputer.statistics_)
    scaler = steps.get("scaler")
    if scaler is not None:
        exported["scaler"] = {"mean": floats(scaler.mean_), "scale": floats(scaler.scale_)}

    kind = type(classifier).__name__
    if kind == "DummyClassifier":
        exported.update(kind="dummy", prediction=str(classifier.classes_[np.argmax(classifier.class_prior_)]) )
    elif kind == "LogisticRegression":
        exported.update(kind="logistic", coef=floats(classifier.coef_), intercept=floats(classifier.intercept_))
    elif kind == "SVC":
        exported.update(kind="svc_rbf", gamma=float(classifier._gamma),
                        support_vectors=floats(classifier.support_vectors_),
                        dual_coef=floats(classifier.dual_coef_), intercept=floats(classifier.intercept_),
                        n_support=classifier.n_support_.tolist())
    elif kind == "KNeighborsClassifier":
        exported.update(kind="knn", k=int(classifier.n_neighbors), X=floats(classifier._fit_X),
                        y=[str(classifier.classes_[int(i)]) for i in classifier._y])
    elif kind == "RandomForestClassifier":
        exported.update(kind="random_forest", trees=[tree_data(estimator, True) for estimator in classifier.estimators_])
    elif kind == "GradientBoostingClassifier":
        exported.update(kind="gradient_boosting", learning_rate=float(classifier.learning_rate),
                        prior=floats(classifier.init_.class_prior_),
                        stages=[[tree_data(classifier.estimators_[stage, cls]) for cls in range(len(classifier.classes_))]
                                for stage in range(classifier.estimators_.shape[0])])
    else:
        raise TypeError(f"Unsupported model for browser inference: {name} ({kind})")
    return exported

