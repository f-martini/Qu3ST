from qu3st.ntsp.generator.data_generator import DataGenerator
import json
import pathlib
import argparse

root_path = pathlib.Path(__file__).parent.parent.parent.resolve()


if __name__ == "__main__":
    def list_of_ints(s):
        try:
            return [int(item) for item in s.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid list of integers: '{}'".format(s))

    # command argument manager
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--counts", type=list_of_ints, default=None)
    parser.add_argument("--variations", type=int, default=None)
    parser.add_argument("--links", type=bool, default=None)
    parser.add_argument("--collateral", type=bool, default=None)
    args = parser.parse_args()

    # load base config
    current_dir = pathlib.Path(__file__).resolve().parent
    configs_dir = current_dir / "configs/data_generation"
    default_file = configs_dir / "default_config.json"
    config = json.loads(default_file.read_text())

    # load config file and substitute default-config field
    if args.config is not None:
        file = pathlib.Path(args.config)
        if file.exists():
            data_config = json.loads(file.read_text())
        elif (configs_dir / args.config).exists():
            data_config = json.loads(
                (configs_dir / args.config).read_text())
        else:
            raise ValueError(
                "The config parameter do not correspond to an "
                "existent .json file.")

        for k in data_config.keys():
            config[k] = data_config[k]

    # substitute custom field
    if args.path is not None:
        config["path"] = args.path

    if args.counts is not None:
        config["counts"] = args.counts

    if args.variations is not None:
        config["variations"] = args.variations

    if args.links is not None:
        config["links"] = args.links

    if args.collateral is not None:
        config["collateral"] = args.collateral

    DataGenerator(**config).generate()
