import json
import pathlib
from .instance import Instance
from .sanitizer import Sanitizer


class Loader:

    def __init__(self, mode, sanitize=False, **kwargs):
        """
        Load an NTS problem instance.
        Args:
            mode: loading mode
            kwargs: additional parameters
        """
        self.sanitize = sanitize
        self.mode = mode
        self.kwargs = kwargs

    def load(self, param=None):
        """
        Load an NTS problem instance.
        Returns: NTS problem instance
        """
        # select proper loading mode
        if self.mode == 'json':
            # load a problem instance from a json file
            return self.load_json(file_name=param)
        else:
            raise ValueError(f"{self.mode} is not a valid parameter. Change "
                             f"loading mode to load data.")

    def load_json(self, file_name=None):
        """
        Loads NTSP instance from json file.
        Returns: [Instance] NTS problem instance
        """
        if file_name is None:
            file_name = self.kwargs["file"]
        # sanitize file_name
        if not file_name.endswith(".json"):
            file_name += ".json"
        # check for file existence
        file_path = pathlib.Path(file_name)
        if not file_path.exists():
            # check also inside base data directory
            file_path = pathlib.Path(__file__).parent / "data" / file_path
            if not file_path.exists():
                raise FileNotFoundError
        # load and parse json file
        with open(file_path, 'r') as json_file:
            instance_dict = json.load(json_file)

        # initialize the problem instance
        instance = Instance(instance_dict)
        if self.sanitize:
            # check data consistency
            Sanitizer(instance).sanitize()

        return instance
