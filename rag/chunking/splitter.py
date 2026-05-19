
class TextSplitter:
    """
    Splites a long text into overlapping chunks.

    chunk_size -> max characters per chunk (~512 tokens = 2000 chars)
    chunk_overlap -> how many characters to repeat between chunks
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap =chunk_overlap

    def split(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.
        Tries to split at paragraph or sentence boundries.
        """
        if not text.strip():
            return []
        
        # Clean up whitespace
        text = " ".join(text.split())

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk - take everything remaining
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to find a clean break point
            # Priority: paragraph > sentaence end > word boundary
            break_point = self._find_break(text, start, end)
            chunk = text[start:break_point].strip()

            if chunk:
                chunks.append(chunk)

            # Move start forward, minus overlap
            start = break_point - self.chunk_overlap

            # Safety: never go backwards
            if start <= 0:
                start = break_point

        return chunks
    
    def _find_break(self, text: str, start: int, end: int) -> int:
        """Find the best position to break the text near 'end'."""
        window = text[start:end]

        # Try paragraph break first
        para_break = window.rfind("\n\n")
        if para_break > self.chunk_size // 2:
            return start + para_break
        
        # Try sentence end
        for punct in [". ", "! ", "? "]:
            sent_break = window.rfind(punct)
            if sent_break > self.chunk_size // 2:
                return start + sent_break + 1
            
        word_break = window.rfind(" ")
        if word_break > 0:
            return start + word_break
        
        # Hard cut as last resort
        return end
    
    def split_with_metadata(self, text: str, source: str) -> list[dict]:
        """
        Split and attach metadata to each chunk.
        Returns list of dicts with text + source info.
        """

        chunks = self.split(text)
        return [
            {
                "text": chunk,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks) 
            }
            for i, chunk in enumerate(chunks)
        ]



