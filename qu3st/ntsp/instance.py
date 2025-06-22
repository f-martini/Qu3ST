import numpy as np


class Instance:

    def __init__(self, instance):
        """
        Class needed to store and handle NTS problem instances.
        Args:
            instance: dictionary with the instance data.
        """
        # save instance dictionary as-is
        self.instance_dict = instance
        # available securities
        self.currencies = np.array(instance["Currencies"], dtype=float)
        # available securities
        self.securities = np.array(instance["Securities"], dtype=float)
        # # conversion ratios between currencies pairs (matrix)
        # self.conversion_rates = np.array(instance["CurrencyConversionRates"])
        # list of transactions
        self.T = np.array(instance["Transactions"], dtype=float)
        # cash balances initial amounts
        self.CBs = np.array(instance["CBs"], dtype=float)
        # securities positions initial amount
        self.SPs = np.array(instance["SPs"], dtype=float)
        # list of SPLs
        self.SPLs = np.array(instance["SPLs"], dtype=float)
        # list of CMBs
        self.CMBs = np.array(instance["CMBs"], dtype=float)
        # pairs of transaction links
        self.links = np.array(instance["Links"], dtype=float)
        # priority-weights list
        self.W = np.array(instance["PriorityWeights"], dtype=float)
        # optional field participants
        self.participants = None
        if "Participants" in instance.keys():
            self.participants = instance["Participants"]

    def exist_CMBs(self):
        return len(self.CMBs.shape) == 2

    def exist_SPLs(self):
        return len(self.SPLs.shape) == 2

    def exist_links(self):
        return len(self.links.shape) == 2

    def t_cash_amounts(self):
        """
        Returns: Cash amount associated with each transaction.
        """
        return self.T[:, 0] if len(self.T.shape) == 2 else np.empty((0,))

    def t_currencies(self):
        """
        Returns: Currency type with each transaction.
        """
        return self.T[:, 1].astype(int) if len(self.T.shape) == 2 else (
            np.empty((0,)).astype(int))

    def t_security_amounts(self):
        """
        Returns: Security amount associated with each transaction.
        """
        return self.T[:, 2] if len(self.T.shape) == 2 else np.empty((0,))

    def t_securities(self):
        """
        Returns: Security type associated with each transaction.
        """
        return self.T[:, 3].astype(int) if len(self.T.shape) == 2 else (
            np.empty((0,)).astype(int))

    def t_debtors_cb(self):
        """
        Returns: Debtor cash balance associated with each transaction.
        """
        return self.T[:, 4].astype(int) if len(self.T.shape) == 2 else (
            np.empty((0,)).astype(int))

    def t_creditors_cb(self):
        """
        Returns: Creditor cash balance associated with each transaction.
        """
        return self.T[:, 5].astype(int) if len(self.T.shape) == 2 else (
            np.empty((0,)).astype(int))

    def t_debtors_sp(self):
        """
        Returns: Debtor security position associated with each
            transaction.
        """
        return self.T[:, 6].astype(int) if len(self.T.shape) == 2 else (
            np.empty((0,)).astype(int))

    def t_creditors_sp(self):
        """
        Returns: Creditor security position associated with each transaction.
        """
        return self.T[:, 7].astype(int) if len(self.T.shape) == 2 else (
            np.empty((0,)).astype(int))

    def t_priorities(self):
        """
        Returns: Priority associated with each transaction.
        """
        return self.T[:, 8].astype(int) if len(self.T.shape) == 2 else (
            np.empty((0,)).astype(int))

    # def t_partials(self):
    #     """
    #     Returns: Flag associated with each transaction which state if partial
    #         settlement is allowed for that transaction.
    #     """
    #     return self.T[:, 9].astype(int) if len(self.T.shape) == 2 else (
    #         np.empty((0,)).astype(int))

    # def links_types(self):
    #     """
    #     Returns: Links type vectors.
    #     """
    #     return self.links[:, 0].astype(int) if len(self.links.shape) == 2 \
    #         else np.empty((0,)).astype(int)

    def links_first(self):
        """
        Returns: Transactions to be settled before (or together with) the other
            one.
        """
        return self.links[:, 0].astype(int) if len(self.links.shape) == 2 \
            else np.empty((0,)).astype(int)

    def links_second(self):
        """
        Returns: Transactions to be settled after (or together with) the other
            one.
        """
        return self.links[:, 1].astype(int) if len(self.links.shape) == 2 \
            else np.empty((0,)).astype(int)

    def cb_currencies(self):
        """
        Returns: Currencies associated with each cash balance.
        """
        return self.CBs[:, 0].astype(int) if len(self.CBs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def cb_balances(self):
        """
        Returns: Cash amount in each cash balances.
        """
        return self.CBs[:, 1] if len(self.CBs.shape) == 2 else np.empty((0,))

    def cb_negatives(self):
        """
        Returns: Flags associated with each balance which state wheter or not it
            could be negative.
        """
        return self.CBs[:, 2].astype(int) if len(self.CBs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def sp_securities(self):
        """
        Returns: Security associated with each security position.
        """
        return self.SPs[:, 0].astype(int) if len(self.SPs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def sp_quantities(self):
        """
        Returns: Security amount in each security position.
        """
        return self.SPs[:, 1] if len(self.SPs.shape) == 2 else np.empty((0,))

    def sp_negatives(self):
        """
        Returns: Flags associated with security position which state wheter or
        not it could be negative.
        """
        return self.SPs[:, 2].astype(int) if len(self.SPs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def s_lot_sizes(self):
        """
        Returns: Number of securities in each lot of a given security type.
        """
        return self.securities[:, 0] if len(
            self.securities.shape) == 2 else np.empty((0,))

    def s_prices(self):
        """
        Returns: Security prices.
        """
        return self.securities[:, 1] if len(self.securities.shape) == 2 else (
            np.empty((0,)))

    def cmb_types(self):
        """
        Returns: CMB types vector.
        """
        return self.CMBs[:, 0].astype(int) if len(self.CMBs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def cmb_cb_providers(self):
        """
        Returns: Cash balances which provide collateral.
        """
        return self.CMBs[:, 1].astype(int) if len(self.CMBs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def cmb_cb_receivers(self):
        """
        Returns: Cash balances which receive collateral.
        """
        return self.CMBs[:, 2].astype(int) if len(self.CMBs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def cmb_au_limits(self):
        """
        Returns: Authorized usage limits (when applicable, 0 otherwise).
        """
        return self.CMBs[:, 3] if len(self.CMBs.shape) == 2 else np.empty((0,))

    def cmb_aco_limits(self):
        """
        Returns: ACO limits.
        """
        return self.CMBs[:, 4] if len(self.CMBs.shape) == 2 else np.empty((0,))

    def spl_cmbs(self):
        """
        Returns: CMB indexes associated with each SPL.
        """
        return self.SPLs[:, 0].astype(int) if len(self.SPLs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def spl_sp_providers(self):
        """
        Returns: security positions which pledge securities.
        """
        return self.SPLs[:, 1].astype(int) if len(self.SPLs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def spl_sp_receivers(self):
        """
        Returns: security positions which receive pledge securities.
        """
        return self.SPLs[:, 2].astype(int) if len(self.SPLs.shape) == 2 else (
            np.empty((0,)).astype(int))

    def spl_pledge_mins(self):
        """
        Returns: minimum amounts of securities that can be pledge .
        """
        return self.SPLs[:, 3] if len(self.SPLs.shape) == 2 else np.empty((0,))

    def normalize(self):
        """
        Normalize quantities and amounts by scaling them.
        Returns:
        """
        max_value = np.max(np.array([
            np.max(self.t_security_amounts()),
            np.max(self.t_cash_amounts()),
            np.max(self.sp_quantities()),
            np.max(self.cb_balances()),
            np.max(self.cmb_aco_limits()) if len(self.cmb_aco_limits()) > 0
            else 0,
        ]))

        self.T[:, 0] /= max_value
        self.T[:, 2] /= max_value
        self.CBs[:, 1] /= max_value
        self.SPs[:, 1] /= max_value
        self.securities[:, 0] /= max_value
        # self.securities[:, 1] /= max_value
        if len(self.CMBs) > 0:
            self.CMBs[:, 3] /= max_value
            self.CMBs[:, 4] /= max_value
        if len(self.SPLs) > 0:
            self.SPLs[:, 3] /= max_value
        return
