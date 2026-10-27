"""Generate synthetic datasets and the knowledge base."""
from qsr_assistant.data.knowledge import write_knowledge
from qsr_assistant.data.synthetic import generate_all, save_all

if __name__ == "__main__":
    data = generate_all()
    save_all(data)
    write_knowledge()
    for name, df in data.items():
        print(f"{name:20s} {df.shape}")
