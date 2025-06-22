import numpy as np
from .currencies import generate_currencies


def get_curr_owners_dict(
        currencies,
        participants
):
    curr_owners_dict = {s: [] for s in range(len(currencies))}
    for p in participants["own"].keys():
        for curr in participants["own"][p]["cbs"].keys():
            curr_owners_dict[curr].append(p)
    return curr_owners_dict


def assign_cbs_to_participants(
        cbs,
        participants
):
    n_cbs = len(cbs)
    parts = list(participants["own"].keys())
    c_cbs = [c[0] for c in cbs]
    p_random = np.arange(0, n_cbs) % len(parts)
    for i in range(n_cbs):
        c = c_cbs[i]
        p = participants["own"][p_random[i]]
        if c in p["cbs"].keys():
            p["cbs"][c].append(int(i))
        else:
            p["cbs"][int(c)] = [int(i)]
        # set negative balance flag to true if central bank
        if p_random[i] in participants["central_banks"]:
            cbs[i][2] = 1


def generate_cash_balances_n_currencies(t_c,
                                        participants,
                                        var,
                                        scale,
                                        p_zero,
                                        cb_p_p
                                        ):
    # choose the number of cash balances to generate
    n_par = len(participants["own"])
    max_cbs = n_par + 1 if cb_p_p else max(int(t_c) // 2, n_par + 1)
    n_cbs = np.random.randint(low=n_par, high=max_cbs)

    # generate currencies
    currencies = generate_currencies()

    # generate cash balances
    n_c = len(currencies)
    cbs = []
    c_random = np.random.randint(0, n_c, n_cbs)
    v_random = (
            np.abs(np.random.normal(loc=0, scale=var,
                                    size=n_cbs)) * scale)
    z_random = np.random.random(n_cbs) < p_zero
    # standard cash balances
    for i in range(n_cbs):
        cbs.append([
            int(c_random[i]),
            0 if z_random[i] else max(0., float(format(v_random[i],
                                                       '.2f'))),
            0,
        ])

    # assign cash balances to participants
    assign_cbs_to_participants(cbs, participants)

    # compute currencies ownership dict
    curr_owners_dict = get_curr_owners_dict(currencies, participants)

    return cbs, currencies, curr_owners_dict
