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

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        Return True if this loader can handle this file type.
        Ex: PDFLoader.supports("notes.pdf") -> True
        """

        pass