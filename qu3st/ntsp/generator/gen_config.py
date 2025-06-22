import datetime
import pathlib


class GenConfig:
    def __init__(self,
                 counts: list | None= None,
                 variations: int = 1,
                 data_format: str = 'json',
                 path: str | pathlib.Path | None = None,
                 collateral: bool = True,
                 partial: bool = True,
                 links: bool = True,
                 cb_p_p: bool = False,
                 scale_cb: float = 10 ** 5,
                 var_cb: float = 10 ** -2,
                 p_zero_cb: float = 0.1,
                 scale_sp: float = 10 ** 3,
                 var_sp: float = 10 ** -2,
                 p_zero_sp: float = 0.1,
                 scale_t: float = 10 ** 2,
                 scale_spl: float = 10 ** 1,
                 scale_cmb: float = 10 ** 5
                 ):
        self.counts = [5] if counts is None else counts
        self.cb_p_p = cb_p_p
        self.variations = variations
        self.data_format = data_format
        self.collateral = collateral
        self.partial = partial
        self.links = links
        self.scale_cb = scale_cb
        self.var_cb = var_cb
        self.p_zero_cb = p_zero_cb
        self.scale_sp = scale_sp
        self.var_sp = var_sp
        self.p_zero_sp = p_zero_sp
        self.scale_t = scale_t
        self.scale_spl = scale_spl
        self.scale_cmb = scale_cmb
        current_date_string = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")
        if path is None:
            self.path = pathlib.Path() / "data" / current_date_string
        else:
            self.path = pathlib.Path(path) / current_date_string

    def to_dict(self):
        return {
            name: str(value) for name, value in vars(self).items()
        }
