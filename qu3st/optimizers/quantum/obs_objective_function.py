import numpy as np
from numba import njit, int32, float64, prange
from numba.experimental import jitclass
from numba.typed.typedlist import List
from qiskit.quantum_info import SparsePauliOp
import logging

logger = logging.getLogger(__name__)


@jitclass([('provider', int32), ('mins', float64), ('lots', float64), ('prices', float64), ('idx', int32)]) # type: ignore
class ColSPs:
    def __init__(self,
                 provider,
                 mins,
                 lots,
                 prices,
                 idx):
        self.provider = provider
        self.mins = mins
        self.lots = lots
        self.prices = prices
        self.idx = idx


@njit
def relu(x, A=4, offset=-1 / 2, p=2):
    if x <= 0:
        return 0
    return A * (x - offset) ** p


@njit
def gen_logistic(x, offset=.0, A=.0, K=1, C=1, Q=1, B=1, v=1):
    return A + (K - A) / (C + Q * np.exp(-B * (x - offset))) ** (1 / v)


def x_inverse_gen_logistic(y, offset=.0, A=.0, K=1, C=1, Q=1, B=1, v=1):
    return - (np.log((((K - A) / (y - A)) ** v - C) / Q) / B) + offset


def B_inverse_gen_logistic(x, y, offset=.0, A=.0, K=1, C=1, Q=1, v=1):
    return - (np.log((((K - A) / (y - A)) ** v - C) / Q) / (x - offset))


def get_Z_string(N, idx):
    return "I" * (N - idx - 1) + "Z" + "I" * idx


@njit
def _of(solution, of_weights):
    ws = 0
    ws += np.sum(solution * of_weights)
    return ws


@njit
def f_cb_balances(solution,
                  sp_bal_weights,
                  col_limits,
                  col_sps,
                  cb_balances,
                  cb_negatives,
                  cb_bal_weights,
                  activation_function_cb):
    ws = 0
    cb_ini = cb_balances
    cb_b = - cb_ini
    for i in range(len(cb_bal_weights)):
        cb_b[i] += np.dot(cb_bal_weights[i, :], solution.astype(float64))
    collateral, collateral_sp = get_collateral(
        sol=solution,
        val_cb=cb_b.copy(),
        sp_bal_weights=sp_bal_weights,
        col_limits=col_limits,
        col_sps=col_sps
    )
    val = cb_b - collateral
    val *= (1 - cb_negatives)  # reset central-banks cb
    for v in val:
        ws += activation_function_cb(v)
    return ws, (cb_b > 0).astype(int32), collateral_sp, collateral


@njit
def f_sp_balances(solution,
                  collateral_sp,
                  sp_quantities,
                  sp_bal_weights,
                  sp_negatives,
                  activation_function_sp):
    ws = 0
    sp_ini = sp_quantities
    val = - sp_ini
    for i in range(len(sp_bal_weights)):
        val[i] += np.dot(sp_bal_weights[i, :], solution.astype(float64))
    val += collateral_sp
    val *= (1 - sp_negatives)  # reset central-banks sp
    for v in val:
        ws += activation_function_sp(v)
    return ws


@njit
def f_links(solution, activation_function, links):
    if len(links) == 0:
        return 0
    ws = 0
    for t1, t2 in links:
        binlist = solution
        val = binlist[int(t2)] - binlist[int(t1)]
        ws += activation_function(val)
    return ws


@njit()
def get_collateral(sol,
                   val_cb,
                   sp_bal_weights,
                   col_limits,
                   col_sps):
    # retrieve data
    val_sp = np.zeros(len(sp_bal_weights), dtype=np.float64)
    for i in range(len(sp_bal_weights)):
        val_sp[i] += np.dot(sp_bal_weights[i, :], sol.astype(np.float64))

    # initialize collateral arrays
    collateral_cb = np.zeros(len(val_cb), dtype=np.float64)
    collateral_sp = np.zeros(len(val_sp), dtype=np.int32)
    for cb, v in enumerate(val_cb):
        tmp_lim = col_limits[cb]
        if v <= 0 or v > tmp_lim:
            continue
        coll = 0
        tmp_sp_call = np.zeros(len(val_sp), dtype=np.int32)

        for c_sps in col_sps[cb]:
            sp = c_sps.provider
            s_min = c_sps.mins
            lot = c_sps.lots
            price = c_sps.prices
            spl = c_sps.idx
            min_pledge = (s_min // lot) * lot
            max_pledge = min(
                (-val_sp[sp] // lot) * lot,  # on-flow sec limit
                ((tmp_lim // price) // lot) * lot  # collateral limit
            )
            if min_pledge <= max_pledge:
                # compute aco approximation
                tmp_sp_call[sp] = max_pledge
                coll += max_pledge * price
                tmp_lim -= max_pledge * price

        if v - coll < 0:
            collateral_cb[cb] = coll
            collateral_sp += tmp_sp_call.astype(np.int32)

    return collateral_cb, collateral_sp


@njit
def evaluate_sample(solution,
                    of_weights,
                    activation_function_links,
                    links,
                    sp_bal_weights,
                    sp_quantities,
                    col_limits,
                    col_sps,
                    cb_balances,
                    cb_negatives,
                    cb_bal_weights,
                    activation_function_cb,
                    sp_negatives,
                    activation_function_sp
                    ):
    wp = -_of(solution, of_weights)
    wp += f_links(solution, activation_function_links, links)
    v, cb_ind, collateral_sp, collateral_cb = f_cb_balances(
        solution=solution,
        sp_bal_weights=sp_bal_weights,
        col_limits=col_limits,
        col_sps=col_sps,
        cb_balances=cb_balances,
        cb_negatives=cb_negatives,
        cb_bal_weights=cb_bal_weights,
        activation_function_cb=activation_function_cb
    )
    wp += v
    wp += f_sp_balances(
        solution=solution,
        collateral_sp=collateral_sp,
        sp_quantities=sp_quantities,
        sp_bal_weights=sp_bal_weights,
        sp_negatives=sp_negatives,
        activation_function_sp=activation_function_sp
    )
    return wp, collateral_sp, collateral_cb, cb_ind


@njit(parallel=True)
def fast_evaluate(samples,
                  probabilities,
                  of_weights,
                  activation_function_links,
                  links,
                  sp_bal_weights,
                  sp_quantities,
                  col_limits,
                  col_sps,
                  cb_balances,
                  cb_negatives,
                  cb_bal_weights,
                  activation_function_cb,
                  sp_negatives,
                  activation_function_sp,
                  ):
    wps = np.zeros(len(samples))
    coll_sps = np.zeros((len(samples), len(sp_quantities)))
    coll_cbs = np.zeros((len(samples), len(cb_balances)))
    cb_ind = np.zeros((len(samples), len(cb_balances)))
    for i in prange(len(samples)):
        wp, coll_sps[i, :], coll_cbs[i, :], cb_ind[i, :] = evaluate_sample(
            solution=samples[i, :],
            of_weights=of_weights,
            activation_function_links=activation_function_links,
            links=links,
            sp_bal_weights=sp_bal_weights,
            sp_quantities=sp_quantities,
            col_limits=col_limits,
            col_sps=col_sps,
            cb_balances=cb_balances,
            cb_negatives=cb_negatives,
            cb_bal_weights=cb_bal_weights,
            activation_function_cb=activation_function_cb,
            sp_negatives=sp_negatives,
            activation_function_sp=activation_function_sp)
        wps[i] = wp
    best_wp = 0
    best_sol = np.zeros(len(samples[0, :])).astype(np.int32)
    for i in range(len(wps)):
        if i == 0 or best_wp > wps[i]:
            best_wp = wps[i]
            best_sol = samples[i, :].astype(int32)
    evaluation = np.dot(wps, probabilities)
    return evaluation, best_wp, best_sol, coll_sps, coll_cbs, cb_ind


class ObsObjectiveFunction:

    def __init__(self,
                 instance,
                 lam=0.5,
                 callback=None,
                 activation_function="relu",
                 gamma=0.5,
                 **kwargs
                 ):
        self.lam = lam
        self.callback = callback
        self.instance = instance
        self.of_weights = self.get_of_weights()
        self.cb_bal_weights = self.get_cb_bal_weights()
        self.sp_bal_weights = self.get_sp_bal_weights()
        self.col_limits, self.col_sps = self.get_col_dict()
        self.activation_functions = self.get_activation_functions(
            activation_function)
        self.gamma = gamma

    def get_activation_functions(self, mode):
        af_dict = {}
        if mode == "logist":
            x_ref_cb = max(min(self.instance.t_cash_amounts()), 1)
            B_cb = int(B_inverse_gen_logistic(x=-x_ref_cb, y=0.1) + 1)
            af_dict["cb"] = njit()(
                lambda x: gen_logistic(x, offset=x_ref_cb, B=B_cb))

            x_ref_sp = max(min(self.instance.t_security_amounts()), 1)
            B_sp = int(B_inverse_gen_logistic(x=-x_ref_sp, y=0.1) + 1)
            af_dict["sp"] = njit()(lambda x: gen_logistic(x, offset=x_ref_sp,
                                                          B=B_sp))

            B_link = int(B_inverse_gen_logistic(x=-1, y=0.1) + 1)
            af_dict["links"] = njit()(lambda x: gen_logistic(x, offset=1,
                                                             B=B_link))
        elif mode == "relu":
            af_dict["cb"] = njit()(lambda x: relu(x))
            af_dict["sp"] = njit()(lambda x: relu(x))
            af_dict["links"] = njit()(lambda x: relu(x))
        else:
            raise ValueError
        return af_dict

    def get_col_dict(self):
        cb_receiver = self.instance.cmb_cb_receivers()[self.instance.spl_cmbs()]
        cb_limits = self.instance.cmb_aco_limits()[self.instance.spl_cmbs()]
        sp_provider = self.instance.spl_sp_providers()
        sp_mins = self.instance.spl_pledge_mins()
        sp_securities = self.instance.sp_securities()[sp_provider]
        sp_lots = self.instance.s_lot_sizes()[sp_securities]
        sp_prices = self.instance.s_prices()[sp_securities]
        # initialize collateral dict
        col_limits = np.array([0 for _ in self.instance.cb_balances()],
                              dtype=float)
        col_dict = {cb: [] for cb, _ in enumerate(
            self.instance.cb_balances())}
        for idx, cb in enumerate(cb_receiver):
            col_dict[cb].append(
                (sp_provider[idx],
                 sp_mins[idx],
                 sp_lots[idx],
                 sp_prices[idx],
                 idx)
            )
            col_limits[cb] = cb_limits[idx]
        col_sps = List()
        for cb in sorted(col_dict.keys()):
            tmp_list = List.empty_list(ColSPs.class_type.instance_type) # type: ignore
            for tp in col_dict[cb]:
                tmp_list.append(ColSPs(
                    tp[0],
                    tp[1],
                    tp[2],
                    tp[3],
                    tp[4]
                ))
            col_sps.append(tmp_list)
        return col_limits, col_sps

    def get_of_weights(self):
        tws = self.instance.W[self.instance.t_priorities()]
        # amounts
        tams = self.instance.t_cash_amounts()
        of_weights = (self.lam * tams * tws) / np.sum(tams * tws)
        # volume
        of_weights += ((1 - self.lam) * tws) / np.sum(tws)
        return of_weights

    def get_cb_bal_weights(self):
        tams = self.instance.t_cash_amounts()
        cred = self.instance.t_creditors_cb()
        deb = self.instance.t_debtors_cb()
        cb_bal_weights = np.zeros((len(self.instance.cb_balances()), len(tams)))
        for idx in range(len(tams)):
            # in-flow
            cb_bal_weights[cred[idx], idx] = -tams[idx]
            # out-flow
            cb_bal_weights[deb[idx], idx] = tams[idx]
        return cb_bal_weights

    def get_sp_bal_weights(self):
        tams = self.instance.t_security_amounts()
        cred = self.instance.t_creditors_sp()
        deb = self.instance.t_debtors_sp()
        sp_bal_weights = np.zeros(
            (len(self.instance.sp_quantities()), len(tams))
        )
        for idx in range(len(tams)):
            # in-flow
            sp_bal_weights[cred[idx], idx] = -tams[idx]
            # out-flow
            sp_bal_weights[deb[idx], idx] = tams[idx]
        return sp_bal_weights

    def get_lot_collateral(self, collateral_sp):
        collateral = np.zeros(len(self.instance.spl_cmbs()))
        spl_ind = np.zeros(len(self.instance.spl_cmbs()))
        for cb in range(len(self.instance.cb_balances())):
            for c_sps in self.col_sps[cb]:
                sp = c_sps.provider
                lot = c_sps.lots
                spl = c_sps.idx
                if collateral_sp[sp] > 0:
                    spl_ind[spl] = 1
                    collateral[spl] += collateral_sp[sp] // lot
        return collateral, spl_ind

    def of(self, solution):
        ws = 0
        ws += np.sum(solution * self.of_weights)
        return ws

    def link_constraint(self, ev_dict):
        ev = 0
        N = len(self.instance.t_cash_amounts())
        for first, second in zip(self.instance.links_first(),
                                 self.instance.links_second()):
            key_first = get_Z_string(N, first)
            key_second = get_Z_string(N, second)
            pair = "".join(["I" if i != first and i != second else "Z" for i in
                            range(N)])
            val = 1 / 4 + (ev_dict[key_second] + 1) / 4 - (
                    ev_dict[key_first] + 1) / 4 - (ev_dict[pair] + 1) / 4
            ev += self.activation_functions["links"](val)
        return ev

    def link_constraint_v2(self, ev_dict):
        ev = 0
        N = len(self.instance.t_cash_amounts())
        for first, second in zip(self.instance.links_first(),
                                 self.instance.links_second()):
            key_first = get_Z_string(N, first)
            key_second = get_Z_string(N, second)
            val = (ev_dict[key_second] + 1) / 2 - (ev_dict[key_first] + 1) / 2
            ev += self.activation_functions["links"](val)
        return ev

    def cb_balance_constraint(self, ev_dict, balance, coll, samples, probs):
        w_coll = probs @ coll
        evs = np.zeros(len(balance))
        vals = np.zeros(len(balance))
        N = len(self.instance.t_cash_amounts())
        cb_debtor = self.instance.t_debtors_cb()
        cb_creditor = self.instance.t_creditors_cb()
        amounts = self.instance.t_cash_amounts()

        # add results related amounts
        for t in range(len(cb_debtor)):
            debtor = cb_debtor[t]
            creditor = cb_creditor[t]
            key = get_Z_string(N, t)
            vals[debtor] += amounts[t] * (ev_dict[key] + 1) / 2
            vals[creditor] += -amounts[t] * (ev_dict[key] + 1) / 2

        # add balance related amounts and compute evaluation
        for b in range(len(balance)):
            if self.instance.cb_negatives()[b] == 0:
                vals[b] += -balance[b] - w_coll[b]
                evs[b] = self.activation_functions["cb"](vals[b])
            else:
                evs[b] = 0

        return sum(evs)

    def sp_balance_constraint(self, ev_dict, balance, coll, samples, probs):
        w_coll = probs @ coll
        evs = np.zeros(len(balance))
        vals = np.zeros(len(balance))
        N = len(self.instance.t_cash_amounts())
        sp_debtor = self.instance.t_debtors_sp()
        sp_creditor = self.instance.t_creditors_sp()
        amounts = self.instance.t_security_amounts()

        # add results related amounts
        for t in range(len(sp_debtor)):
            creditor = sp_creditor[t]
            debtor = sp_debtor[t]
            key = get_Z_string(N, t)
            vals[creditor] += -amounts[t] * (ev_dict[key] + 1) / 2
            vals[debtor] += amounts[t] * (ev_dict[key] + 1) / 2

        # add balance related amounts and compute evaluation
        for b in range(len(balance)):
            if self.instance.sp_negatives()[b] == 0:
                vals[b] += -balance[b] + w_coll[b]
                evs[b] = self.activation_functions["sp"](vals[b])
            else:
                evs[b] = 0

        return sum(evs)

    def of_dict(self, ev_dict):
        of = 0
        N = len(self.of_weights)
        for i in range(len(self.of_weights)):
            s = get_Z_string(N, i)
            of += ((ev_dict[s] + 1) / 2) * self.of_weights[i]
        return of

    def evaluate_ev(self, ev_dict, states, coll=False, call=False):

        # revert ev such that |0^n> is the ground state
        for k, v in ev_dict.items():
            ev_dict[k] = -v

        probabilities = np.zeros(len(states.keys()))
        samples = []
        for n, (k, p) in enumerate(states.items()):
            samples.append(np.fromiter(k, int))
            probabilities[n] = p
        samples = np.array(samples)

        evaluation, best_wp, best_sol, coll_sps, coll_cbs, cb_ind = (
            fast_evaluate(
                samples=samples,
                probabilities=probabilities,
                of_weights=self.of_weights,
                activation_function_links=self.activation_functions["links"],
                links=self.instance.links if len(self.instance.links) > 0 else
                np.empty((0, 2)),
                sp_bal_weights=self.sp_bal_weights,
                sp_quantities=self.instance.sp_quantities(),
                col_limits=self.col_limits,
                col_sps=self.col_sps,
                cb_balances=self.instance.cb_balances(),
                cb_negatives=self.instance.cb_negatives(),
                cb_bal_weights=self.cb_bal_weights,
                activation_function_cb=self.activation_functions["cb"],
                sp_negatives=self.instance.sp_negatives(),
                activation_function_sp=self.activation_functions["sp"],
            )
        )

        if self.callback is not None and callable(self.callback) and call:
            self.callback(best_sol, best_wp)

        wp = -self.of_dict(ev_dict)
        wp += self.link_constraint_v2(ev_dict)

        cb_balance = self.instance.cb_balances()
        wp += self.cb_balance_constraint(
            ev_dict, cb_balance, coll_cbs, samples, probabilities
        )

        sp_balance = self.instance.sp_quantities()
        wp += self.sp_balance_constraint(
            ev_dict, sp_balance, coll_sps, samples, probabilities
        )
        if coll:
            return wp, coll_sps, cb_ind
        return wp

    def call_evaluate_sample(self, sample, coll=False):
        if isinstance(sample, np.ndarray):
            sample = "".join([str(int(i)) for i in sample])
        Nq = len(sample)
        obs = []
        for i in range(Nq):
            obs.append((get_Z_string(Nq, i), 1))
        H = SparsePauliOp.from_list(obs)
        states = {sample: 1}
        ev_dict = {}
        for i, (op, _) in enumerate(H.label_iter()): # type: ignore
            ev_dict[op] = 1 if sample[i] == "0" else -1

        if coll:
            ev, collateral_sp, cb_ind = self.evaluate_ev(ev_dict, states, coll=coll)  # type: ignore
            collateral, spl_ind = self.get_lot_collateral(collateral_sp[0, :])
            return ev, collateral, cb_ind[0, :], spl_ind
        else:
            return self.evaluate_ev(ev_dict, states)
