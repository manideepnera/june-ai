class ContextBuilder:
    """
    Assembles the final prompt from all available context.

    Prompt sctucture:
    --------------------------------
    [RAG context]  <- from knowledge base (if retrieved)

    [Conversation] <- recent history
    Boss: ...
    June: ...
    Boss: ...

    [current message]
    User: <current input>
    --------------------------------

    The system prompt is passed seprately to the LLM provider.

    """

    SYSTEM_PROMPT = """You are June, a personal AI assistant.
You are helpful, concise, and direct.
You have access to the user's personal knowledge base.
When context is provided, use it to answer accurately.
When no context is provided, answer from your general knowledge.
Never make up information you do not have.
Keep responses clear and to the point."""

    def __init__(self, max_history_turns: int = 6):
        """
        max_history_turns: how many past exchanges to include.
        Each turn = one user message + one assistant response.
        Keep this low to avoid bloating the context window.
        """
        self.max_history_turns = max_history_turns

    def build(
        self,
        user_input: str,
        rag_context: str = "",
        history: list[dict] = None,
        memory_context: str = ""        # reserved for phase 2
    ) -> str:
        """
        Build the final prompt string.

        Args:
            user_input    : the current user message
            rag_context   : formatted RAG chunks (or empty string)
            history       : list of {"user": ..., "assistant": ...} dicts
            memory_context: long-term memory context (phase 2)

        Returns:
            A single prompt string ready to send to the LLM.
        """
        sections = []

        # 1. RAG context block
        if rag_context.strip():
            sections.append(rag_context)
            sections.append("")  # blank line separator

        # 2. Memory context (phase 2 — empty for now)
        if memory_context.strip():
            sections.append(memory_context)
            sections.append("")

        # 3. Conversation history
        if history:
            recent = history[-self.max_history_turns:]
            history_lines = []
            for turn in recent:
                if turn.get("user"):
                    history_lines.append(f"User: {turn['user']}")
                if turn.get("assistant"):
                    history_lines.append(f"June: {turn['assistant']}")
            if history_lines:
                sections.append("\n".join(history_lines))
                sections.append("")

        # 4. Current user message
        sections.append(f"User: {user_input}")
        sections.append("June:")

        return "\n".join(sections)

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT