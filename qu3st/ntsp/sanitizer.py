from .instance import Instance
import numpy as np


def check_ref_consistency(references: np.ndarray,
                          common: np.ndarray,
                          referred: np.ndarray,
                          msg=""):
    r_msg = ""
    if len(references) == len(common) == 0:
        return r_msg
    expected = referred[common]
    if not np.all(expected == references):
        r_msg += f"{msg} are not consistent.\n"
    return r_msg


def check_lower_bound_domain(array: np.ndarray,
                             l_bound=0,
                             typecheck=None,
                             msg="",
                             included=False):
    r_msg = ""
    if typecheck is not None:
        r_msg += check_type_domain(array, typecheck, msg)
    if included and not np.all(array >= l_bound):
        r_msg += f"{msg} must be greater or equal than {l_bound}.\n"
    elif not included and not np.all(array > l_bound):
        r_msg += f"{msg} must be greater than {l_bound}.\n"
    return r_msg


def check_values_domain(array: np.ndarray, values: list, msg=""):
    if not np.all(np.isin(array, values)):
        return f"{msg} must be in {values}\n"
    return ""


def check_type_domain(array: np.ndarray, typecheck=None, msg=""):
    if array.dtype != typecheck:
        return f"{msg} must be {str(typecheck)}.\n"
    return ""


def check_reference_domain(array: np.ndarray,
                           min_v: float,
                           max_v: float,
                           typecheck=None,
                           msg=""):
    r_msg = ""
    if typecheck is not None:
        r_msg += check_type_domain(array, typecheck, msg)
    if not np.all((min_v <= array) & (array <= max_v)):
        return f"{msg} must be in [{min_v}, {max_v}].\n"
    return r_msg


def check_same_length(arrays_list: list, msg=""):
    # Convert the list of arrays to a NumPy array
    np_arrays = np.array(arrays_list)
    # Check if all rows have the same length
    if np.all(np.array([len(arr) for arr in np_arrays]) == len(np_arrays[0])):
        return ""
    else:
        return f"{msg}: some entries are incomplete.\n"


class Sanitizer:

    def __init__(self, instance: Instance):
        """
        Performs data sanity checks.
        Args:
             instance: NTSP problem instance
        """
        self.instance = instance
        # currencies range
        self.min_c = 0
        self.max_c = len(self.instance.currencies) - 1
        # securities range
        self.min_s = 0
        self.max_s = len(self.instance.securities) - 1
        # cash balances range
        self.min_cb = 0
        self.max_cb = len(self.instance.CBs) - 1
        # security positions range
        self.min_sp = 0
        self.max_sp = len(self.instance.SPs) - 1
        # cmb range
        self.min_cmb = 0
        self.max_cmb = len(self.instance.CMBs) - 1
        # spl range
        self.min_spl = 0
        self.max_spl = len(self.instance.SPLs) - 1
        # transactions range
        self.min_t = 0
        self.max_t = len(self.instance.T) - 1
        # priorities range
        self.min_p = 0
        self.max_p = len(self.instance.W) - 1
        # bools
        self.bools = [0, 1]
        # link type values
        self.links_types = [0, 1]
        # cmb type values
        self.cmb_types = [0, 1]

    def sanitize(self):
        """
        Performs data sanity checks.
        """
        msg = ""

        msg_d = self.check_domains()
        if msg_d != "":
            msg += "\nDOMAIN CHECKS FAILED: \n" + msg_d

        msg_s = self.check_shapes()
        if msg_s != "":
            msg += "\nSHAPE CHECKS FAILED: \n" + msg_s

        msg_c = self.check_consistencies()
        if msg_c != "":
            msg += "\nCONSISTENCY CHECKS FAILED: \n" + msg_c

        if msg != "":
            raise ValueError(msg)

    def check_domains(self):
        """
        Check if data have proper types and values.
        """
        msg = ""

        # --- weights ---
        msg += check_lower_bound_domain(self.instance.W,
                                        l_bound=0,
                                        msg="Weights")

        # --- currencies ---
        # !!! no checks required !!!

        # --- securities ---
        # check lot sizes
        msg += check_lower_bound_domain(self.instance.s_lot_sizes(),
                                        l_bound=0,
                                        # typecheck=int,
                                        msg="Security lot sizes")

        # check securities prices
        msg += check_lower_bound_domain(self.instance.s_prices(),
                                        l_bound=0,
                                        # typecheck=float,
                                        msg="Security prices",
                                        included=True)

        # --- transactions ---
        # check cash amounts
        msg += check_lower_bound_domain(self.instance.t_cash_amounts(),
                                        l_bound=0,
                                        # typecheck=float,
                                        msg="Transactions' cash amounts",
                                        included=True)
        # check securities amounts
        msg += check_lower_bound_domain(self.instance.t_security_amounts(),
                                        l_bound=0,
                                        # typecheck=float,
                                        msg="Transactions' security amounts",
                                        included=True)
        # check priorities
        msg += check_reference_domain(self.instance.t_priorities(),
                                      min_v=self.min_p,
                                      max_v=self.max_p,
                                      typecheck=int,
                                      msg="Transactions' priorities")

        # check external ids
        msg += check_reference_domain(self.instance.t_currencies(),
                                      min_v=self.min_c,
                                      max_v=self.max_c,
                                      typecheck=int,
                                      msg="Transactions' currency ids")
        msg += check_reference_domain(self.instance.t_securities(),
                                      min_v=self.min_s,
                                      max_v=self.max_s,
                                      typecheck=int,
                                      msg="Transactions' security ids")
        msg += check_reference_domain(self.instance.t_debtors_cb(),
                                      min_v=self.min_cb,
                                      max_v=self.max_cb,
                                      typecheck=int,
                                      msg="Transactions' debtor CB ids")
        msg += check_reference_domain(self.instance.t_creditors_cb(),
                                      min_v=self.min_cb,
                                      max_v=self.max_cb,
                                      typecheck=int,
                                      msg="Transactions' creditor CB ids")
        msg += check_reference_domain(self.instance.t_debtors_sp(),
                                      min_v=self.min_sp,
                                      max_v=self.max_sp,
                                      typecheck=int,
                                      msg="Transactions' debtor SP ids")
        msg += check_reference_domain(self.instance.t_creditors_sp(),
                                      min_v=self.min_sp,
                                      max_v=self.max_sp,
                                      typecheck=int,
                                      msg="Transactions' creditor SP ids")

        # --- CBs ---
        # check negative flag
        msg += check_values_domain(self.instance.cb_negatives(),
                                   self.bools,
                                   msg="CBs' can-be-negative flags")
        # check balances
        msg += check_lower_bound_domain(self.instance.cb_balances()[
                                            self.instance.cb_negatives() == 0
                                            ],
                                        l_bound=0,
                                        # typecheck=float,
                                        msg="Positive CBs' balances",
                                        included=True)
        # check external ids
        msg += check_reference_domain(self.instance.cb_currencies(),
                                      min_v=self.min_c,
                                      max_v=self.max_c,
                                      typecheck=int,
                                      msg="CBs' currency ids")

        # --- SPs ---
        # check negative flag
        msg += check_values_domain(self.instance.sp_negatives(),
                                   self.bools,
                                   msg="SPs' can-be-negative flags")
        # check balances
        msg += check_lower_bound_domain(self.instance.sp_quantities()[
                                            self.instance.sp_negatives() == 0
                                            ],
                                        l_bound=0,
                                        # typecheck=float,
                                        msg="Positive SPs' balances",
                                        included=True)
        # check external ids
        msg += check_reference_domain(self.instance.sp_securities(),
                                      min_v=self.min_s,
                                      max_v=self.max_s,
                                      typecheck=int,
                                      msg="SPs' security ids")

        # --- CMBs ---
        # check CMBs' types
        msg += check_values_domain(self.instance.cmb_types(),
                                   self.cmb_types,
                                   msg="CMBs' types")
        # check au limits
        msg += check_lower_bound_domain(self.instance.cmb_au_limits(),
                                        l_bound=0,
                                        # typecheck=float,
                                        msg="CMBs' au limits",
                                        included=True)
        msg += check_values_domain(self.instance.cmb_au_limits()[
                                       self.instance.cmb_types() == 0
                                       ],
                                   [0],
                                   msg="Primary CMBs' au limits")

        # check aco limits
        msg += check_lower_bound_domain(self.instance.cmb_aco_limits(),
                                        l_bound=0,
                                        # typecheck=float,
                                        msg="CMBs' aco limits",
                                        included=True)

        # check external ids
        msg += check_reference_domain(self.instance.cmb_cb_providers(),
                                      min_v=self.min_cb,
                                      max_v=self.max_cb,
                                      typecheck=int,
                                      msg="CMBs' CB provider ids")
        msg += check_reference_domain(self.instance.cmb_cb_receivers(),
                                      min_v=self.min_cb,
                                      max_v=self.max_cb,
                                      typecheck=int,
                                      msg="CMBs' CB receiver ids")

        # --- SPLs ---
        # check pledge minimum
        msg += check_lower_bound_domain(self.instance.spl_pledge_mins(),
                                        l_bound=0,
                                        # typecheck=float,
                                        msg="CMBs' pledge mins",
                                        included=True)
        # check external ids
        msg += check_reference_domain(self.instance.spl_sp_providers(),
                                      min_v=self.min_sp,
                                      max_v=self.max_sp,
                                      typecheck=int,
                                      msg="SPLs' SP provider ids")
        msg += check_reference_domain(self.instance.spl_sp_receivers(),
                                      min_v=self.min_sp,
                                      max_v=self.max_sp,
                                      typecheck=int,
                                      msg="SPLs' SP receiver ids")

        # --- links ---
        # check external ids
        msg += check_reference_domain(self.instance.links_first(),
                                      min_v=self.min_t,
                                      max_v=self.max_t,
                                      typecheck=int,
                                      msg="Links' first transactions ids")
        msg += check_reference_domain(self.instance.links_second(),
                                      min_v=self.min_t,
                                      max_v=self.max_t,
                                      typecheck=int,
                                      msg="Links' second transactions ids")

        return msg

    def check_consistencies(self):
        msg = ""

        # --- CMBs ---
        # check currency - cash balances
        cmb_curr_cb_rec = self.instance.cb_currencies()[
            self.instance.cmb_cb_receivers()]
        msg += check_ref_consistency(cmb_curr_cb_rec,
                                     self.instance.cmb_cb_providers(),
                                     self.instance.cb_currencies(),
                                     msg="CMB: CB providers and CB receivers "
                                         "currencies")
        # check secondary - cash balances
        cmb_t = self.instance.cmb_types() == 1
        if not np.all(self.instance.cmb_cb_providers()[cmb_t] ==
                      self.instance.cmb_cb_receivers()[cmb_t]):
            msg += "Inconsistency in secondary CMBs."

        # --- SPLs ---
        # check securities - security positions
        spl_sec_sb_rec = self.instance.sp_securities()[
            self.instance.spl_sp_receivers()]
        msg += check_ref_consistency(spl_sec_sb_rec,
                                     self.instance.spl_sp_providers(),
                                     self.instance.sp_securities(),
                                     msg="SPL: SP providers and SP receivers "
                                         "securities")
        # --- Transactions ---
        # check securities - security positions
        msg += check_ref_consistency(self.instance.t_securities(),
                                     self.instance.t_debtors_sp(),
                                     self.instance.sp_securities(),
                                     msg="Transact.: SP debtors' securities")
        msg += check_ref_consistency(self.instance.t_securities(),
                                     self.instance.t_creditors_sp(),
                                     self.instance.sp_securities(),
                                     msg="Transact.: SP creditors' securities")
        # check currency - cash balances
        msg += check_ref_consistency(self.instance.t_currencies(),
                                     self.instance.t_debtors_cb(),
                                     self.instance.cb_currencies(),
                                     msg="Transact.: CB debtors' currencies")
        msg += check_ref_consistency(self.instance.t_currencies(),
                                     self.instance.t_creditors_cb(),
                                     self.instance.cb_currencies(),
                                     msg="Transact.: CB creditors' currencies")

        return msg

    def check_shapes(self):
        msg = ""

        # --- Currencies ---
        # !!! no checks required !!!

        # --- Securities ---
        msg += check_same_length(arrays_list=[self.instance.s_prices(),
                                              self.instance.s_lot_sizes()],
                                 msg="Securities")

        # --- Links ---
        msg += check_same_length(arrays_list=[self.instance.links_first(),
                                              self.instance.links_second()],
                                 msg="Links")

        # --- Transactions ---
        msg += check_same_length(arrays_list=[self.instance.t_cash_amounts(),
                                              self.instance.t_currencies(),
                                              self.instance.t_security_amounts(),
                                              self.instance.t_securities(),
                                              self.instance.t_debtors_cb(),
                                              self.instance.t_creditors_cb(),
                                              self.instance.t_debtors_sp(),
                                              self.instance.t_creditors_sp(),
                                              self.instance.t_priorities(),
                                              self.instance.t_cash_amounts()],
                                 msg="Transaction")

        # --- CBs ---
        msg += check_same_length(arrays_list=[self.instance.cb_currencies(),
                                              self.instance.cb_balances(),
                                              self.instance.cb_negatives()],
                                 msg="CBs")

        # --- SPs ---
        msg += check_same_length(arrays_list=[self.instance.sp_securities(),
                                              self.instance.sp_quantities(),
                                              self.instance.sp_negatives()],
                                 msg="SPs")

        # --- CMBs ---
        msg += check_same_length(arrays_list=[self.instance.cmb_types(),
                                              self.instance.cmb_cb_providers(),
                                              self.instance.cmb_cb_receivers(),
                                              self.instance.cmb_au_limits(),
                                              self.instance.cmb_aco_limits()],
                                 msg="CMBs")

        # --- SPLs ---
        msg += check_same_length(arrays_list=[self.instance.spl_cmbs(),
                                              self.instance.spl_sp_providers(),
                                              self.instance.spl_sp_receivers(),
                                              self.instance.spl_pledge_mins()],
                                 msg="SPLs")

        return msg
