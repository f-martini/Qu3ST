import numpy as np
from .securities import generate_securities


def get_sec_owners_dict(
        securities,
        participants
):
    sec_owners_dict = {s: [] for s in range(len(securities))}
    for p in participants["own"].keys():
        for sec in participants["own"][p]["sps"].keys():
            sec_owners_dict[sec].append(p)
    return sec_owners_dict


def assign_sps_to_participants(
        sps,
        participants
):
    """
    Assign each security position to a
    Compute a dictionary that map each security position to its owner
    (participant).

    Args:
        sps: security positions
        participants:
    Returns
    """
    n_sps = len(sps)
    parts = list(participants["own"].keys())
    s_sps = [s[0] for s in sps]
    p_random = np.arange(0, n_sps) % len(parts)
    for i in range(n_sps):
        s = s_sps[i]
        p = participants["own"][p_random[i]]
        if s in p["sps"].keys():
            p["sps"][s].append(int(i))
        else:
            p["sps"][int(s)] = [int(i)]
        # set negative balance flag to true if central security depository
        if p_random[i] in participants["central_security_depositories"]:
            sps[i][2] = 1


def generate_sec_positions_n_securities(t_c,
                                        participants,
                                        var,
                                        scale,
                                        scale_price,
                                        p_zero
                                        ):
    # choose the number of security position to generate
    n_par = len(participants["own"])
    n_sps = np.random.randint(low=int(n_par * 1.5),
                              high=max(int(t_c), n_par * 3))

    # generate securities
    securities = generate_securities(n_sps, scale_lot=scale,
                                     scale_price=scale_price)

    # generate security positions
    n_s = len(securities)
    sps = []
    s_random = np.random.randint(0, n_s, n_sps)
    v_random = (
            np.abs(np.random.normal(loc=0, scale=var, size=n_sps)) * scale)
    z_random = np.random.random(n_sps) < p_zero
    for i in range(n_sps):
        sps.append([
            int(s_random[i]),
            0 if z_random[i] else max(0, int(v_random[i])),
            0,
        ])

    # assign securities positions to participants
    assign_sps_to_participants(sps, participants)

    # compute securities ownership dict
    sec_owners_dict = get_sec_owners_dict(securities, participants)

    return sps, securities, sec_owners_dict
