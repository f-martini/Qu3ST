import numpy as np


def filter_untradable(d):
    to_pop = []
    for k, v in d.items():
        if len(v) < 2:
            to_pop.append(k)
    for k in to_pop:
        d.pop(k)


def choose_amounts(scale, price, var=10 ** -1):
    sa = (np.abs(np.random.normal(loc=0, scale=var)) * scale)
    ca = (np.abs(np.random.normal(loc=price, scale=var))) * sa
    return float(format(ca, '.2f')), int(sa)


def compute_trading_couples(sec_owners_dict,
                            curr_owners_dict,
                            central_banks):
    couple_dict = {}
    sec_keys = list(sec_owners_dict)
    cash_keys = list(curr_owners_dict)
    for sk in sec_keys:
        for ck in cash_keys:
            key = f"{sk} {ck}"
            p_set = set(sec_owners_dict[sk]).intersection(
                curr_owners_dict[ck])
            couple_dict[key] = [p for p in p_set if p not in central_banks]
    return couple_dict


def generate_transactions(
        t_c,
        curr_owners_dict,
        sec_owners_dict,
        participants,
        securities,
        priorities,
        partial,
        scale_t):
    cbk = list(curr_owners_dict.keys())
    spk = list(sec_owners_dict.keys())
    accounts = participants["own"]

    trading_couples = compute_trading_couples(sec_owners_dict,
                                              curr_owners_dict,
                                              participants["central_banks"])
    filter_untradable(trading_couples)
    tck = list(trading_couples.keys())
    if len(tck) == 0:
        print("Random generator did not create tradable assets pairs.")
        return []

    n_field = 10
    transactions = np.zeros((t_c, n_field))
    for i in range(t_c):
        k = np.random.randint(0, len(tck))
        s, c = [int(num) for num in tck[k].split(" ")]
        transactions[i, 1] = cbk[c]
        transactions[i, 3] = spk[s]
        # compute participants
        sec_buyer, sec_seller = np.random.choice(
            trading_couples[tck[k]],
            size=2,
            replace=False
        )
        transactions[i, 4] = np.random.choice(
            accounts[int(sec_buyer)]["cbs"][c])
        transactions[i, 5] = np.random.choice(
            accounts[int(sec_seller)]["cbs"][c])

        transactions[i, 6] = np.random.choice(
            accounts[int(sec_seller)]["sps"][s])
        transactions[i, 7] = np.random.choice(
            accounts[int(sec_buyer)]["sps"][s])

        transactions[i, 0], transactions[i, 2] = choose_amounts(
            scale=scale_t,
            price=securities[spk[s]][1]
        )
        transactions[i, 8] = np.random.randint(
            0, len(priorities))
        if partial:
            raise NotImplemented
    return [row.tolist() for row in transactions]
