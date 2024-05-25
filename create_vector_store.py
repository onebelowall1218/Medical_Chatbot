import os
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# 1. Load embedding function
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 2. Load text files and preprocess them


def process_text_files(directory):
    # Initialize the text splitter with desired parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )

    all_splits_with_page_no = []

    # Process each .txt file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(directory, filename)

            with open(txt_file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Split the text into chunks using the text splitter
            splits = text_splitter.split_text(text)

            # Add some indexing to the splits
            splits_with_page_no = []
            for i, split in enumerate(splits):
                split = f"page : {i+1}\n\n" + split
                splits_with_page_no.append(split)

            # Add to the list of all splits
            all_splits_with_page_no.extend(splits_with_page_no)

            # For example, print the first 5 pages of each file
            print(f"Processed {filename}:")
            for page in splits_with_page_no[:5]:
                print(page)
                print("-----" * 20)  # Separator between pages for clarity

    return all_splits_with_page_no


# Specify the directory containing the text files
directory = "Books/Texts/"
splits_with_page_no = process_text_files(directory)

# 3. Create vector database
# Persist directory for the vector store
persist_directory = "database/"

# For text files
vectordb = Chroma.from_texts(
    texts=splits_with_page_no,
    embedding=embedding_function,
    persist_directory=persist_directory,
)

print("Database created successfully")
