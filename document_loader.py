"""PDF loader"""

from langchain_community.document_loaders import PyMuPDFLoader

FILE_PATH = "./dl-curriculum.pdf"
loader = PyMuPDFLoader(FILE_PATH)
documents = loader.lazy_load()

for doc in documents:
    print(doc.page_content)
