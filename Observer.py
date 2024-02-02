from abc import ABC, abstractmethod

class Observer(ABC):
    """
    The Observer interface declares the update method, used by observables to flush the latest data to the observers.
    """

    @abstractmethod
    def update(self, observable) -> None:
        """
        Receive update from subject.
        """
        pass
