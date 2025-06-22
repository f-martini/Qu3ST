import numpy as np


def get_limit(scale, var):
    v = np.abs(np.random.normal(loc=0, scale=var)) * scale
    return float(format(v, '.2f'))


def get_provider(participants, curr, curr_cbs):
    # select central bank
    eligible_ps = [c for c in participants["central_banks"] if c in curr_cbs]
    if len(eligible_ps) == 0:
        return None, None
    p_p = eligible_ps[np.random.randint(0, len(eligible_ps))]
    # select cash balance
    eligible_cbs = participants["own"][p_p]["cbs"][curr]
    if len(eligible_cbs) == 0:
        return None, None
    cb_p = eligible_cbs[np.random.randint(0, len(eligible_cbs))]
    return cb_p, p_p


def generate_CMBs(curr_owners_dict,
                  participants,
                  scale,
                  var,
                  collateral):
    cmbs = []
    cmbs_participants = {}
    id_cmb = 0
    if collateral:
        for k, p in participants["own"].items():
            if k not in participants["central_banks"]:
                for curr, cbs in p["cbs"].items():
                    for cb in cbs:
                        cb_p, p_p = get_provider(
                            participants,
                            curr,
                            curr_owners_dict[curr])
                        if cb_p is None and p_p is None:
                            continue
                        cmbs.append(
                            [
                                0,
                                cb_p,
                                cb,
                                0,
                                get_limit(scale, var),
                            ]
                        )
                        cmbs_participants[id_cmb] = [p_p, k]
                        id_cmb += 1

    return cmbs, cmbs_participants
