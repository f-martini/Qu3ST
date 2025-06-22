def compute_cb_constraints(instance, t_vars, spl_vars):
    # initialize dict
    cb_negative_flags = instance.cb_negatives()
    cbs_vals = zip(instance.cb_balances(), cb_negative_flags)
    cbc_dict = {cb: [[], a] for cb, (a, n) in enumerate(cbs_vals) if n == 0}

    # compute cash in- and out-flows
    cb_debtors = instance.t_debtors_cb()
    cb_creditors = instance.t_creditors_cb()
    amounts = instance.t_cash_amounts()
    for t, var in enumerate(t_vars):
        cb_db = cb_debtors[t]
        if cb_negative_flags[cb_db] == 0:
            cbc_dict[cb_db][0].append([amounts[t], var])
        cb_cd = cb_creditors[t]
        if cb_negative_flags[cb_cd] == 0:
            cbc_dict[cb_cd][0].append([-amounts[t], var])

    # compute collateralized cash in-flow
    spl_providers = instance.spl_sp_providers()
    lot_sizes = instance.s_lot_sizes()
    prices = instance.s_prices()
    target_cbs = instance.cmb_cb_receivers()[instance.spl_cmbs()]
    sp_sec = instance.sp_securities()
    for l, var in enumerate(spl_vars):
        sec = sp_sec[spl_providers[l]]
        cb_r = target_cbs[l]
        if cb_negative_flags[cb_r] == 0:
            cbc_dict[cb_r][0].append([-lot_sizes[sec] * prices[sec], var])

    return cbc_dict


def compute_sp_constraints(instance, t_vars, spl_vars):
    # initialize dict
    sp_negative_flags = instance.sp_negatives()
    sps_vals = zip(instance.sp_quantities(), sp_negative_flags)
    spc_dict = {sp: [[], q] for sp, (q, n) in enumerate(sps_vals) if n == 0}

    # compute securities in- and out-flows
    sp_debtors = instance.t_debtors_sp()
    sp_creditors = instance.t_creditors_sp()
    quantities = instance.t_security_amounts()
    for t, var in enumerate(t_vars):
        sp_db = sp_debtors[t]
        if sp_negative_flags[sp_db] == 0:
            spc_dict[sp_db][0].append([quantities[t], var])
        sp_cd = sp_creditors[t]
        if sp_negative_flags[sp_cd] == 0:
            spc_dict[sp_cd][0].append([-quantities[t], var])

    # compute collateral out-flow
    spl_providers = instance.spl_sp_providers()
    lot_sizes = instance.s_lot_sizes()
    sp_sec = instance.sp_securities()
    for l, var in enumerate(spl_vars):
        spl_p = spl_providers[l]
        if sp_negative_flags[spl_p] == 0:
            spc_dict[spl_p][0].append([lot_sizes[sp_sec[spl_p]], var])

    return spc_dict


def set_balance_constraints(instance,
                            model,
                            t_vars,
                            spl_vars,
                            **kwargs):
    cbc_dict = compute_cb_constraints(instance, t_vars, spl_vars)
    for cb, init_amount in cbc_dict.values():
        model.add_constraint(
            model.sum(p[0] * p[1] for p in cb) <= init_amount)

    spc_dict = compute_sp_constraints(instance, t_vars, spl_vars)
    for sp, init_quantity in spc_dict.values():
        model.add_constraint(
            model.sum(p[0] * p[1] for p in sp) <= init_quantity)
