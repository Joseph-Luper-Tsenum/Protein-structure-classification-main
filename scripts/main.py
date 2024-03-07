import argparse
import pandas as pd
import torch
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, EsmModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

# Add src/ folder to python path which is in parent folder
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from src.config import ProteinClassifier, CONSTANTS
from src.embedding import embed_sequences

emb_model_tag = CONSTANTS.EMBEDDING_MODEL
batch_size = CONSTANTS.BATCH_SIZE
target_codes_and_names = CONSTANTS.ARCHITECTURE_NAMES

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Protein Classification')
    parser.add_argument('-i', '--input_csv', type=str, required=True, help='Input CSV file containing new sequences')
    parser.add_argument('-m', '--model_checkpoint', type=str, default='data/model_best.pt', help='Path to the model checkpoint')
    parser.add_argument('-o', '--output_csv', type=str, default=None, help='Output CSV file to save predictions')
    args = parser.parse_args()

    # Load DataFrame containing new sequences
    new_sequences_df = pd.read_csv(args.input_csv)  # Adjust the filename as per your data
    print("Loaded new sequences from CSV.")

    # Load sequences from the 'sequence' column
    sequences = new_sequences_df['sequence'].tolist()

    # Load the pre-trained ESM model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(emb_model_tag)
    embedding_model = EsmModel.from_pretrained(emb_model_tag).to(device)
    print("Loaded pre-trained ESM model and tokenizer.")

    # Apply the pipeline to new sequences
    new_embeddings = embed_sequences(tokenizer, embedding_model, sequences, batch_size=batch_size, device=device)
    predictions = []
    print("Applied the embedding to new sequences.")

    # Load the trained model checkpoint
    checkpoint = torch.load(args.model_checkpoint)
    hidden_dim = CONSTANTS.HIDDEN_DIM
    output_dim = len(target_codes_and_names)
    print("Loaded trained model checkpoint.")

    # Create the classification model
    classification_model = ProteinClassifier(input_dim=new_embeddings.shape[-1], hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    classification_model.load_state_dict(checkpoint)
    print("Created the classification model.")

    with torch.no_grad():
        for inputs in new_embeddings.split(32):  # Split into batches of size 32 for inference
            outputs = classification_model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Create inverse mapping
    ints_to_labels = dict(enumerate(target_codes_and_names))

    # Add predictions to DataFrame
    new_sequences_df['predicted_class'] = list(map(lambda x: ints_to_labels[x], predictions))
    new_sequences_df['predicted_tag'] = new_sequences_df['predicted_class'].map(target_codes_and_names)

    # Save the DataFrame with predictions
    output_csv = args.output_csv if args.output_csv else args.input_csv.replace('.csv', '_with_predictions.csv')
    new_sequences_df.to_csv(output_csv, index=False)
    print(f"Predictions saved successfully to {output_csv}:")
    print(new_sequences_df)

if __name__ == "__main__":
    main()