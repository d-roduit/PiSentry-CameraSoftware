from abc import ABC, abstractmethod

class Observable(ABC):
    """
    The Observable interface declares a set of methods for managing observers.
    """

    @abstractmethod
    def add_observer(self, observer) -> None:
        """
        Add an observer to the observable.
        """
        pass

    @abstractmethod
    def remove_observer(self, observer) -> None:
        """
        Remove an observer from the observable.
        """
        pass

    @abstractmethod
    def _notify_observers(self) -> None:
        """
        Notify all observers about an event.
        """
        pass
