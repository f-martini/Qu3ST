import numpy as np


def get_receiver(sps):
    return int(np.random.choice(sps, size=1)[0])


def get_minimum(sec, scale, var=0.01):
    v = np.abs(np.random.normal(loc=0, scale=var)) * scale
    return int(min(sec[0], v))


def generate_SPLs(participants,
                  securities,
                  cmbs_participants,
                  scale,
                  collateral
                  ):
    spls = []
    owners = participants["own"]
    if collateral:
        for id_cmb, pair in cmbs_participants.items():
            for sec, sps in owners[pair[0]]["sps"].items():
                if sec in owners[pair[1]]["sps"].keys():
                    for sp in owners[pair[1]]["sps"][sec]:
                        spls.append(
                            [
                                id_cmb,
                                sp,
                                get_receiver(sps),
                                get_minimum(securities[sec], scale)
                            ]
                        )
    return spls
