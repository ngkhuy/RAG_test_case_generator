import langchain_classic
from load_data import read_docx
from chunking import chunking_document

def main():
    file_path = r'D:\RAG_testcases\Epic 1 – User Management and Authentication.docx'
    text = read_docx(file_path)
    chunks = chunking_document(text)
    
    # print one chunk
    for i, chunk in enumerate(chunks):
        print(chunk["text"])
        print(chunk["metadata"])
        if i == 5:
            break
    
if __name__ == '__main__':
    main()
