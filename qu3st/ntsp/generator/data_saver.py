import json
import os
import re
from .gen_config import GenConfig


def get_bool_label(val):
    return "T" if val else "F"


def beautify_json(final_json):
    final_json = final_json.replace('{', '{\n')
    final_json = final_json.replace('}', '\n}')
    final_json = final_json.replace('[[', '[\n[')
    final_json = final_json.replace(']]', ']\n]')
    final_json = final_json.replace('],', '],\n')
    return final_json


class DataSaver:

    def __init__(self,
                 config: GenConfig):
        self.config = config

    def save_config(self):
        self.config.path.mkdir(parents=True, exist_ok=True)
        with (open(self.config.path / "config.json", 'w') as config_file):
            config = self.config.to_dict()
            json.dump(config, config_file, indent=4)
        print(f"Configuration file saved successfully.")

    def save_data(self, instance, **kwargs):
        self.config.path.mkdir(parents=True, exist_ok=True)
        if self.config.data_format == "json":
            self._save_json(instance, **kwargs)
        else:
            raise NotImplementedError("Format different than [json] are not "
                                      "supported.")

    def _save_json(self, instance, t=None, i=None):
        # save on json
        file = (f"{t}_{i}"
                f"_{get_bool_label(self.config.collateral)}"
                f"{get_bool_label(self.config.partial)}"
                f"{get_bool_label(self.config.links)}.json")
        with open(self.config.path / file, 'w') as json_file:
            final_json = json.dumps(instance)
            final_json = beautify_json(final_json)
            json_file.write(final_json)
        print(f"{file} generated successfully.")

    def save_dfs(self, df_dict, marker=""):
        self.config.path.mkdir(parents=True, exist_ok=True)
        for key in df_dict.keys():
            df_dict[key].to_csv(self.config.path / f"{key}{marker}.csv",
                                index=False)
            print(f"{key}.csv generated successfully.")

    def count_files(self, name):
        file_count = 0
        for filename in os.listdir(self.config.path):
            if (os.path.isfile(os.path.join(self.config.path, filename)) and
                    filename.startswith(name)):
                file_count += 1
        return file_count
