from rag.loaders.base import BaseLoader

class PDFLoader(BaseLoader):
    """Loads PDF files and extracts all text."""

    def load(self, file_path: str) -> str:
        try:
            import pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf not installed. Run: pip install pymupdf"
            )
        
        reader = pymupdf.open(file_path)

        if len(reader.pages) == 0:
            raise ValueError(f"PDF has no pages: {file_path}")
        
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())

        if not pages:
            raise ValueError(
                f"Could not extract text from PDF: {file_path}\n"
                "The PDF may be scanned/image-based."
            )
        
        return "\n\n".join(pages)
    
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")

