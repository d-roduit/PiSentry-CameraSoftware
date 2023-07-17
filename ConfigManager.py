import json


class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConfigManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.config = ConfigManager.read_file(filepath)

    @staticmethod
    def __parse_dict(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = ConfigManager.__parse_json(value)
        return result

    @staticmethod
    def __parse_list(data: list):
        result = [ConfigManager.__parse_json(item) for item in data]
        return result

    @staticmethod
    def __parse_json(data):
        if type(data) is dict:
            return ConfigManager.__parse_dict(data)
        elif type(data) is list:
            return ConfigManager.__parse_list(data)
        else:
            return data

    @staticmethod
    def read_file(filepath: str):
        with open(filepath, "r") as file:
            result = ConfigManager.__parse_json(json.loads(file.read()))
        return result

    def write(self):
        with open(self.filepath, "w") as file:
            file.write(json.dumps(self.config))


configManager = ConfigManager('config.json')
