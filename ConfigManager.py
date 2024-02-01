import json
import requests
from urls import cameras_api_endpoint, detectable_objects_actions_api_endpoint, notifications_api_endpoint

class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConfigManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.config = ConfigManager.read_file(filepath)
        self.load_config_from_db()

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

    def load_config_from_db(self):
        """
        Fetches config data from database to insert it into the config object.
        Can be used to insert config data into the config object the first time it is called
        as well as to update the config with new data from the database on subsequent calls.

        :raises requests.exceptions.HTTPError: if a response has a status code between 400 and 599
        :raises ValueError: if a response body does not contain valid json
        :raises KeyError: if a response data does not contain the required fields for the config
        """

        with requests.Session() as http_session:
            http_session.headers = {'Authorization': self.config.user.token}

            # Create base config keys
            self.config.camera = Dict() if 'camera' not in self.config else self.config.camera
            self.config.detection = Dict() if 'detection' not in self.config else self.config.detection
            self.config.notifications = Dict() if 'notifications' not in self.config else self.config.notifications

            # Get camera data
            get_camera_data_response = http_session.get(f'{cameras_api_endpoint}/{self.config.camera.id}', timeout=5)
            get_camera_data_response.raise_for_status()
            camera_json_data = get_camera_data_response.json()
            self.config.camera.name = camera_json_data['name']
            self.config.camera.port = camera_json_data['port']
            self.config.detection.startTime = camera_json_data['detection_start_time']
            self.config.detection.endTime = camera_json_data['detection_end_time']
            self.config.detection.areas = camera_json_data['detection_areas']
            self.config.notifications.startTime = camera_json_data['notifications_start_time']
            self.config.notifications.endTime = camera_json_data['notifications_end_time']

            # Get notifications subscriptions data
            get_notifications_subscriptions_data_response = http_session.get(f'{notifications_api_endpoint}/{self.config.camera.id}', timeout=5)
            get_notifications_subscriptions_data_response.raise_for_status()
            subscriptions_json_data = get_notifications_subscriptions_data_response.json()
            are_notifications_enabled = len(subscriptions_json_data) > 0
            self.config.notifications.enabled = are_notifications_enabled

            # Get notifications subscriptions data
            get_detectable_objects_actions_data_response = http_session.get(f'{detectable_objects_actions_api_endpoint}/{self.config.camera.id}', timeout=5)
            get_detectable_objects_actions_data_response.raise_for_status()
            detectable_objects_actions_json_data = get_detectable_objects_actions_data_response.json()
            detection_objects = {
                object_action['object_type']: {
                    'weight': object_action['object_weight'],
                    'action': object_action['action_name'],
                }
                for object_action
                in detectable_objects_actions_json_data
            }
            self.config.detection.objects = ConfigManager.__parse_json(detection_objects)

configManager = ConfigManager('config.json')
