import numpy as np


def get_custom_weights(instance, lam=0.5, **kwargs):
    amounts = instance.t_cash_amounts()
    pw = instance.W[instance.t_priorities()]
    PW = np.sum(pw)
    APW = np.sum(amounts * pw)
    weights = (lam / APW) * amounts * pw + ((1 - lam) / PW) * pw
    return weights


def get_weights(instance, mode=None, **kwargs):
    if mode == "custom":
        return get_custom_weights(instance, **kwargs)
    else:
        return get_custom_weights(instance, **kwargs)


def set_objective_function(instance,
                           model,
                           t_vars,
                           mode="custom",
                           **kwargs):
    weights = get_weights(instance, mode, **kwargs)
    of = model.sum(weights[i] * t_vars[i] for i in range(len(t_vars)))
    model.maximize(of)
