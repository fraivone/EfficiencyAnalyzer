import pathlib
import yaml
from shutil import copy


class config:
    """Simple class to load analysis config"""

    def __init__(self, filepath):
        self.filepath = pathlib.Path(filepath)
        self.data_input = dict
        self.parameters = dict
        self.matching_window = list
        self.analysis_label = str
        self.load_config()

    def load_config(self):
        with self.filepath.open() as ymlfile:
            cfg = yaml.full_load(ymlfile)

        self.data_input = cfg["data_input"]
        self.parameters = cfg["parameters"]
        self.matching_window = cfg["matching_window"]
        self.analysis_label = cfg["analysis_label"]

    def dump_config(self, file_dest):
        copy(self.filepath, file_dest)


if __name__ == "__main__":
    # print(confg("config_test.yml").parameters)
    pass
