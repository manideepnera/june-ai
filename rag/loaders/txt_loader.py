from rag.loaders.base import BaseLoader

class TXTLoader(BaseLoader):
    """Loads plain text files."""

    def load(self, file_path: str) -> str:
        # Try UTF-8 first, fall back to UTF-16 if that fails
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-16") as f:
                text = f.read()

        if not text.strip():
            raise ValueError(f"File is empty: {file_path}")
        
        return text.strip()
        
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(".txt")
    
