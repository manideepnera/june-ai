from rag.loaders.txt_loader import TXTLoader
from rag.loaders.md_loader import MDLoader
from rag.loaders.pdf_loader import PDFLoader

class LoaderRouter:
    """
    Given any file path, returns the right loader.
    Add new loaders here as you support more formats.
    """

    def __init__(self):
        self._loaders = [
            TXTLoader(),
            MDLoader(),
            PDFLoader(),
        ]

    def get_loader(self, file_path: str):
        for loader in self._loaders:
            if loader.supports(file_path):
                return loader
            
        supported = [".txt", ".md", ".markdown", ".pdf"]

        raise ValueError(
            f"No loader found for: {file_path}\n"
            f"Supported formats: {', '.join(supported)}"
        )
    
    def load(self, file_path: str) -> str:
        """Load a file using the appropriate loader."""

        loader = self.get_loader(file_path)
        return loader.load(file_path)