import numpy as np


def compute_cb_indicators(instance, t_vars, cb_inds):
    """
    Bound indicator slack variables in such a way that:
        if aco_need >= 0 then slack=1 else slack=0

    Args:
        instance:
        t_vars:
        cb_inds:
    Returns:
    """
    # initialize dict
    cbc_dict = {cb_inds[cb]: [[], -a] for cb, a in
                enumerate(instance.cb_balances())}
    # compute cash in- and out-flows
    cb_debtors = instance.t_debtors_cb()
    cb_creditors = instance.t_creditors_cb()
    amounts = instance.t_cash_amounts()
    for t, var in enumerate(t_vars):
        cb_db = cb_debtors[t]
        cbc_dict[cb_inds[cb_db]][0].append([-amounts[t], var])
        cb_cd = cb_creditors[t]
        cbc_dict[cb_inds[cb_cd]][0].append([amounts[t], var])
    return cbc_dict


def compute_spl_indicators(instance, spl_inds, cb_inds):
    """
    Bound indicator slack variables in such a way that:
        if aco_need >= 0 then slack=1 else slack=0

    :parameter:
        :param instance:
        :param spl_inds:
        :param cb_inds:
    Returns:
    """
    # initialize dict
    spl_ind_dict = {a: [[], 0] for a in range(len(spl_inds))}
    target_cbs = instance.cmb_cb_receivers()[instance.spl_cmbs()]
    # compute indicator dependency
    for l, var in enumerate(spl_inds):
        cb_ind = cb_inds[target_cbs[l]]
        spl_ind_dict[l][0].append([1, var])
        spl_ind_dict[l][0].append([-1, cb_ind])

    return spl_ind_dict


def compute_max_cash_constraints(instance, cb_inds, spl_vars):
    """
    Compute parameters necessary to enforce max-collateral limit (depending
    on the indicator):
        acol/alim <= indicator
    That is: If the indicator = 0 acol=0 is the only valid solution
             If the indicator = 1 acol<alim else acol/alim>1

    :parameter:
        :param instance:
        :param cb_inds:
        :param spl_vars:
    Returns:
    """
    # initialize dict
    cb_receiver = instance.cmb_cb_receivers()
    if len(cb_receiver) != len(np.unique(cb_receiver)):
        raise ValueError("CB-CMB relation shall be a 1-to-1 relation.")
    max_cash_dict = {cb: [[], 0] for _, cb in enumerate(cb_receiver)}

    # compute collateralized cash
    spl_providers = instance.spl_sp_providers()
    lot_sizes = instance.s_lot_sizes()
    prices = instance.s_prices()
    target_cbs = instance.cmb_cb_receivers()[instance.spl_cmbs()]
    sp_sec = instance.sp_securities()
    aco_limits = instance.cmb_aco_limits()[instance.spl_cmbs()]
    for l, var in enumerate(spl_vars):
        sec = sp_sec[spl_providers[l]]
        cb_r = target_cbs[l]
        max_cash_dict[cb_r][0].append([
            lot_sizes[sec] * prices[sec] / aco_limits[l], var])

    # compute indicator dependency
    for cb in cb_receiver:
        max_cash_dict[cb][0].append([-1, cb_inds[cb]])

    return max_cash_dict


def compute_max_sec_indicator_constraints(instance, t_vars, spl_vars, cb_inds):
    """
    Compute parameters necessary to enforce max-security pledge-able depending
    on the indicator:
        qcol/qin <= indicator
    :param instance:
    :param t_vars:
    :param spl_vars:
    :param cb_inds:
    Returns:
    """
    # initialize dict
    max_sec_ind_dict = {a: [[], 0] for a in range(len(spl_vars))}
    cb_spl_dict = {sp: {} for sp in range(len(instance.sp_securities()))}
    spl_providers = instance.spl_sp_providers()
    target_cbs = instance.cmb_cb_receivers()[instance.spl_cmbs()]
    for spl in range(len(spl_vars)):
        cb_spl_dict[spl_providers[spl]][target_cbs[spl]] = spl

    # compute limit
    spl_limits = np.ones(len(spl_vars))
    sp_creditors = instance.t_creditors_sp()
    cb_debtors = instance.t_debtors_cb()
    quantities = instance.t_security_amounts()
    for t, var in enumerate(t_vars):
        sp_cd = sp_creditors[t]
        cb_db = cb_debtors[t]
        if sp_cd in cb_spl_dict.keys() and cb_db in cb_spl_dict[sp_cd].keys():
            spl = cb_spl_dict[sp_cd][cb_db]
            spl_limits[spl] += quantities[t]

    # compute collateralized securities
    lot_sizes = instance.s_lot_sizes()
    sp_sec = instance.sp_securities()
    for l, var in enumerate(spl_vars):
        sec = sp_sec[spl_providers[l]]
        max_sec_ind_dict[l][0].append([lot_sizes[sec] / spl_limits[l], var])

    # compute indicator dependency
    cb_spl_vars = [cb_inds[t_cb] for t_cb in target_cbs]
    for i, var in enumerate(cb_spl_vars):
        max_sec_ind_dict[i][0].append([-1, var])

    return max_sec_ind_dict


def compute_max_sec_constraints(instance, t_vars, spl_vars):
    """
    Compute parameters necessary to enforce max-security pledge-able
    independently of the indicator:
        qcol <= qlim

    That is the first condition enforce that securities are collateralized only
    if needed.
    The second enforce that limits are respected.

    Args:
        instance:
        t_vars:
        spl_vars:
    Returns:
    """
    # initialize dict
    max_sec_dict = {a: [[], 0] for a in range(len(spl_vars))}
    cb_spl_dict = {sp: {} for sp in range(len(instance.sp_securities()))}
    spl_providers = instance.spl_sp_providers()
    target_cbs = instance.cmb_cb_receivers()[instance.spl_cmbs()]
    for spl in range(len(spl_vars)):
        cb_spl_dict[spl_providers[spl]][target_cbs[spl]] = spl

    # compute collateralized securities
    lot_sizes = instance.s_lot_sizes()
    sp_sec = instance.sp_securities()
    for l, var in enumerate(spl_vars):
        sec = sp_sec[spl_providers[l]]
        max_sec_dict[l][0].append([lot_sizes[sec], var])

    # compute limit
    sp_creditors = instance.t_creditors_sp()
    cb_debtors = instance.t_debtors_cb()
    quantities = instance.t_security_amounts()
    for t, var in enumerate(t_vars):
        sp_cd = sp_creditors[t]
        cb_db = cb_debtors[t]
        if sp_cd in cb_spl_dict.keys() and cb_db in cb_spl_dict[sp_cd].keys():
            spl = cb_spl_dict[sp_cd][cb_db]
            max_sec_dict[spl][0].append([-quantities[t], var])

    return max_sec_dict


def compute_min_sec_constraints(instance, spl_vars, spl_inds):
    # qmin * spl_indicator <= qcol
    # initialize dict
    min_sec_dict = {a: [[], 0] for a in range(len(spl_vars))}
    spl_providers = instance.spl_sp_providers()

    # compute collateralized securities
    lot_sizes = instance.s_lot_sizes()
    sp_sec = instance.sp_securities()
    for l, var in enumerate(spl_vars):
        sec = sp_sec[spl_providers[l]]
        min_sec_dict[l][0].append([-lot_sizes[sec], var])

    # compute indicator dependency
    spl_mins = instance.spl_pledge_mins()
    for i, var in enumerate(spl_inds):
        min_sec_dict[i][0].append([spl_mins[i], var])

    return min_sec_dict


def compute_spl_dep_constraints(instance, spl_vars, spl_inds):
    # qcol / UB <= spl_indicator
    spl_dep_dict = {a: [[], 0] for a in range(len(spl_vars))}
    spl_providers = instance.spl_sp_providers()
    sp_creditors = instance.t_creditors_sp()
    quantities = instance.t_security_amounts()
    col_limits = np.zeros(len(instance.sp_securities()))
    for i in range(len(quantities)):
        col_limits[sp_creditors[i]] += quantities[i]

    # compute collateralized securities
    lot_sizes = instance.s_lot_sizes()
    sp_sec = instance.sp_securities()
    for l, var in enumerate(spl_vars):
        sec = sp_sec[spl_providers[l]]
        spl_dep_dict[l][0].append(
            [lot_sizes[sec] / max(col_limits[spl_providers[l]], 0.01), var])

    # compute indicator dependency
    for i, var in enumerate(spl_inds):
        spl_dep_dict[i][0].append([-1, var])

    return spl_dep_dict


def set_collateral_constraints(instance,
                               model,
                               t_vars,
                               cb_inds,
                               spl_vars,
                               spl_inds,
                               **kwargs):
    # set indicators
    cb_indic_dict = compute_cb_indicators(instance, t_vars, cb_inds)
    for cb in cb_indic_dict.keys():
        pairs, a = cb_indic_dict[cb]
        if len(pairs) == 0:
            continue
        model.add_indicator(
            cb,
            model.sum(p[0] * p[1] for p in pairs) <= a
        )

    spl_indic_dict = compute_spl_indicators(instance, spl_inds, cb_inds)
    for spl, limit in spl_indic_dict.values():
        model.add_constraint(model.sum(p[0] * p[1] for p in spl) <= limit)

    # set upper bounds
    max_cash_dict = compute_max_cash_constraints(instance, cb_inds, spl_vars)
    for cmb, cash_limit in max_cash_dict.values():
        model.add_constraint(model.sum(p[0] * p[1] for p in cmb) <= cash_limit)

    max_sec_dict = compute_max_sec_constraints(instance, t_vars, spl_vars)
    for spl, sec_limit in max_sec_dict.values():
        model.add_constraint(model.sum(p[0] * p[1] for p in spl) <= sec_limit)

    max_sec_ind_dict = compute_max_sec_indicator_constraints(
        instance, t_vars, spl_vars, cb_inds)
    for spl, sec_limit in max_sec_ind_dict.values():
        model.add_constraint(model.sum(p[0] * p[1] for p in spl) <= sec_limit)

    # set lower bounds
    min_sec_dict = compute_min_sec_constraints(instance, spl_vars, spl_inds)
    for spl, sec_min in min_sec_dict.values():
        model.add_constraint(model.sum(p[0] * p[1] for p in spl) <= sec_min)

    # set dependencies between SPL and SPL indicators
    spl_dep_dict = compute_spl_dep_constraints(instance, spl_vars, spl_inds)
    for spl, limit in spl_dep_dict.values():
        model.add_constraint(model.sum(p[0] * p[1] for p in spl) <= limit)
