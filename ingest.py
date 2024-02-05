import os

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)

TYPE2LOADER_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
}


def load_document(file_path: str) -> Document:
    file_extension = os.path.splitext(file_path)[1].lower()
    loader_class = TYPE2LOADER_MAP.get(file_extension)
    if loader_class is None:
        raise ValueError(f"Unsupported file-type: {file_extension}")
    loader = loader_class(file_path)
    return loader.load()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_files",
        help="one or more JSON files (use - character to read from stdin).",
        nargs="+",
    )

    args = parser.parse_args()

    if len(args.json_files) == 1 and args.json_files[0] == "-":
        args.json_files = sys.stdin

    documents = []
    for file_path in args.json_files:
        file_extension = os.path.splitext(os.path.basename(file_path))[1]
        if file_extension in TYPE2LOADER_MAP.keys():
            try:
                documents.extend(load_document(file_path))
            except Exception as ex:
                print("Exception caught:" + ex)

    print(documents)
