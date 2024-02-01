class DetectionActions:
    _actions: set[str] = {'ignore', 'record', 'record+notification'}

    @staticmethod
    def _validate_action(action: str) -> None:
        if action not in DetectionActions._actions:
            raise ValueError(f'Action must be one of: {DetectionActions._actions}')

    @staticmethod
    def must_record(action: str) -> bool:
        DetectionActions._validate_action(action)
        return action == 'record' or action == 'record+notification'

    @staticmethod
    def must_send_notification(action: str) -> bool:
        DetectionActions._validate_action(action)
        return action == 'record+notification'
