"""Regenerate the knowledge base (data/knowledge/*.md) and the retrieval eval set."""
from qsr_assistant.data.knowledge import write_eval_set, write_knowledge

if __name__ == "__main__":
    print(f"{write_knowledge()} documents written")
    print(f"{write_eval_set()} retrieval questions written")
