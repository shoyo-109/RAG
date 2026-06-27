import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader)

from dotenv import load_dotenv
load_dotenv()


def loaf_text_file():
    # create temp file for demo 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, This is Bhubhurv. Thank you for having me ... ")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path, encoding="utf-8")
        documents = loader.load()
        # return documents

        print(f"DOcuments count : {len(documents)}")
        print(f"metadata : {documents[0].metadata}")
        print(f"page content : {documents[0].page_content}")
    finally:
        # Clean up the temp file    
        os.remove(temp_file_path)

def pdf_loader():
    pdf_loader = PyPDFLoader("Documents\Bhubhurv_Resume.pdf")
    pdf = pdf_loader.load()
    print(f"DOcuments count : {len(pdf)}")
    print(f"metadata : {pdf[0].metadata}")
    print(f"page content : {pdf[0].page_content}")

if __name__ == "__main__":
    # docs = loaf_text_file()
    # print("Loaded Documents:", docs)
    pdf_loader()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      