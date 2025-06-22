import numpy as np


def generate_links(transactions, links):
    ls = []
    unique_ls = []
    if links:
        n_t = len(transactions)
        n_links = np.random.randint(low=0, high=(n_t // 2) + 1)
        if n_links == 0:
            return unique_ls
        ts = [i for i in range(n_t)]
        for i in range(n_links):
            link = np.random.choice(
                ts,
                size=2,
                replace=False
            )
            ls.append(link)
        # filter out duplicates
        ls = sorted(ls, key=lambda x: f"{x[0]}_{x[1]}")
        unique_ls.append([int(v) for v in ls[0]])
        for i in range(1, len(ls)):
            if any(ls[i] != ls[i - 1]):
                unique_ls.append([int(v) for v in ls[i]])
    return unique_ls
