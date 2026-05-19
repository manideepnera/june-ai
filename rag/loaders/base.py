from abc import ABC, abstractmethod

class BaseLoader(ABC):
    """
    Every loader reads a file and returns its text.
    Nothing else. No chunking, no embedding here.
    """

    @abstractmethod
    def load(self, file_path: str) -> str:
        """
        Read file at file_path.
        Return its full text content as a single string.
        """

        pass