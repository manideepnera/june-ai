from dataclasses import dataclass

@dataclass
class RouterDecision:
    """
    What the orchestrator decided is needed for this request.
    Each field is True/False.
    Phase 1: only rag is active.
    Phase 2: adds memory
    Phase 3: adds internet, tools
    """

    needs_rag: bool = False
    needs_memory: bool = False
    needs_internet: bool = False
    needs_tools: bool = False
    reasoning: str = ""



class Router:
    """
    Analyzes user input and decides what resources to activate.

    Phase 1 logic is simple but effective:
    - If the message looks like a knowledge question -> use RAG
    - If it is casual chat -> Skip RAG (Faster response)

    More Smatter in later phases...    
    """

    # Questions that almost always need knowledge lookup
    KNOWLEDGE_TRIGGERS = [
        "what", "how", "why", "when", "where", "who",
        "explain", "describe", "tell me", "summarize",
        "what is", "what are", "what was", "what were",
        "how does", "how do", "how did", "how can",
        "can you explain", "do you know", "find",
        "search", "look up", "according to", "based on",
        "in my notes", "from my documents", "i wrote",
        "remind me", "what did i", "notes on" 
    ]

    # Casual phrases that do not need knowledge lookup
    CASUAL_TRIGGERS = [
        "hello", "hi", "hey", "thanks", "thank you",
        "ok", "okay", "sure", "got it", "bye", "goodbye",
        "good morning", "good night", "how are you",
        "what's up", "nice", "great", "cool", "awesome",
        "yes", "no", "maybe", "lol", "haha"
    ]

    def analyze(
            self,
            user_input: str,
            has_knowledge_base: bool = True
    ) -> RouterDecision:
        """
        Analyze user input and return a decision.

        Args:
            user_input: the user's message
            has_knowledge_base: whether RAG has any indexed documents
        """

        text = user_input.lower().strip()

        # No point calling RAG if nothing is indexed
        if not has_knowledge_base:
            return RouterDecision(
                needs_rag=False,
                reasoning="Knowledge base is empty"
            )
        
        # Short casual messages - skip RAG
        if len(text.split()) <= 3:
            is_casual = any(phrase in text for phrase in self.CASUAL_TRIGGERS)
            if is_casual:
                return RouterDecision(
                    needs_rag=False,
                    reasoning="Casual short message"
                )
            
        # check for casual phrases
        for phrase in self.CASUAL_TRIGGERS:
            if text == phrase or text.startswith(phrase + " "):
                return RouterDecision(
                    needs_rag=False,
                    reasoning=f"Casual phrase detected: '{phrase}'"
                )
            
        for trigger in self.KNOWLEDGE_TRIGGERS:
            if trigger in text:
                return RouterDecision(
                    needs_rag=True,
                    reasoning=f"Knowledge trigger detected: '{trigger}'"
                )
            
        # Default: if message is long enough, try RAG
        if len(text.split()) > 5:
            return RouterDecision(
                needs_rag=True,
                reasoning="Long message - attempting RAG"
            )
        
        # Default: no RAG for short ambiguous messages
        return RouterDecision(
            needs_rag=False,
            reasoning="Short message, no clear knowledge trigger"
        )
        