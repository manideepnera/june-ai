from rag.loaders.base import BaseLoader


class MDLoader(BaseLoader):
    """
    Loads Markdown files.
    Strips markdown syntax so the model sees clean text.
    """

    def load(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()


        if not text.strip():
            raise ValueError(f"File is empty: {file_path}")
        
        # remove markdown headers, bold, italic markers
        # Keep the actual words - the meaning is in the text

        import re
        text = re.sub(r"#{1,6}\s*", "", text)                       # remove headers

        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                # remove bold
        
        text = re.sub(r"\*(.+?)\*", r"\1", text)                    # remove italic
        
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text,flags=re.DOTALL) # remove code blocks
        
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)             # remove links

        return text.strip()
    
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith((".md", ".markdown"))