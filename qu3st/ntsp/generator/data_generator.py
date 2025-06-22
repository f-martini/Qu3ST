from .data_saver import DataSaver
from .gen_config import GenConfig
from .object_generators import *
from qu3st.ntsp.instance import Instance
from qu3st.ntsp.sanitizer import Sanitizer


# noinspection PyShadowingNames
class DataGenerator:
    def __init__(self,
                 **kwargs
                 ):
        self.config = GenConfig(**kwargs)
        self.data_saver = DataSaver(self.config)

    def get_instance_dict(self, t_c):
        """
        Generate a NTS problem instance with t_c different transactions and
        randomly generated static data.
        Args:
            t_c: number of transactions in the problem instance.
        Returns: NTS problem instance.
        """

        participants = generate_participants(t_c=t_c)

        cbs, currencies, curr_owners_dict = (
            generate_cash_balances_n_currencies(
                t_c=t_c,
                participants=participants,
                var=self.config.var_cb,
                scale=self.config.scale_cb,
                p_zero=self.config.p_zero_cb,
                cb_p_p=self.config.cb_p_p
            ))

        sps, securities, sec_owners_dict = (
            generate_sec_positions_n_securities(
                t_c=t_c,
                participants=participants,
                var=self.config.var_sp,
                scale=self.config.scale_sp,
                scale_price=self.config.scale_cb / self.config.scale_sp,
                p_zero=self.config.p_zero_sp,
            ))

        priorities = generate_priorities()

        transactions = generate_transactions(
            t_c=t_c,
            curr_owners_dict=curr_owners_dict,
            sec_owners_dict=sec_owners_dict,
            participants=participants,
            securities=securities,
            priorities=priorities,
            partial=self.config.partial,
            scale_t=self.config.scale_t
        )

        cmbs, cmbs_participants = generate_CMBs(
            curr_owners_dict=curr_owners_dict,
            participants=participants,
            var=self.config.var_cb,
            scale=self.config.scale_cmb,
            collateral=self.config.collateral
        )

        spls = generate_SPLs(
            participants=participants,
            securities=securities,
            cmbs_participants=cmbs_participants,
            scale=self.config.scale_spl,
            collateral=self.config.collateral
        )

        links = generate_links(
            transactions=transactions,
            links=self.config.links
        )

        instance = {
            "Currencies": currencies,
            "Securities": securities,
            "Transactions": transactions,
            "SPs": sps,
            "CBs": cbs,
            "SPLs": spls,
            "CMBs": cmbs,
            "Links": links,
            "PriorityWeights": priorities,
            "Participants": participants
        }

        return instance

    def generate(self):
        """
        For each transaction count, it generates, validates,
        and saves n=self.variations instances of the NTS problem.
        """
        self.data_saver.save_config()
        for t_c in self.config.counts:
            for i in range(self.config.variations):
                # generate instance
                instance = self.get_instance_dict(t_c)
                # validate instance
                Sanitizer(Instance(instance)).sanitize()
                # save instance
                self.data_saver.save_data(instance, t=t_c, i=i)
