from concurrent.futures import ThreadPoolExecutor, as_completed
from chromadb.config import Settings
import chromadb
import os

from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    PyMuPDFLoader,
)

LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".md": (TextLoader, {"encoding": "utf8"}),
    ".pdf": (PyMuPDFLoader, {}),
}


def load_document(file_path):
    file_extension = os.path.splitext(os.path.basename(file_path))[1]
    lm = LOADER_MAPPING.get(file_extension)
    if lm is None:
        raise ValueError(f"Unsupported file-type: {file_extension}")
    loader_class, loader_args = lm
    loader = loader_class(file_path, **loader_args)
    return loader.load()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_files",
        help="one or more input files (use - character to read from stdin).",
        nargs="+",
    )

    job_count = 1  # os.cpu_count()
    batch_size = 100

    args = parser.parse_args()

    if len(args.input_files) == 1 and args.input_files[0] == "-":
        args.input_files = sys.stdin

    file_batch = []
    for file_path in args.input_files:
        if len(file_batch) < batch_size:
            file_batch.append(file_path.rstrip())
        else:
            for file_path in file_batch:
                documents = []
                try:
                    documents.extend(load_document(file_path))
                except Exception as ex:
                    print(f"Error occurred: {ex}")

    # print(documents)
