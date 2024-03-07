"""Embedding"""

import torch

# Define a function to embed protein sequences
def embed_sequences(tokenizer, embedding_model, sequences, batch_size=32, device="cpu"):
    """
    Embeds a list of sequences using a tokenizer and returns the embeddings.

    Args:
        tokenizer (Tokenizer): The tokenizer used to tokenize the sequences.
        embedding_model (Model): The embedding model used to generate the embeddings.
        sequences (List[str]): The list of sequences to be embedded.
        batch_size (int, optional): The batch size for tokenization. Defaults to 32.
        device (str): The device to use for embedding.

    Returns:
        torch.Tensor: The embeddings of the sequences.
    """
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i : i + batch_size]
        tokenized_seqs = tokenizer(
            batch_sequences, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = embedding_model(**tokenized_seqs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]  # extract the [CLS] token
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0)
    
    return embeddings