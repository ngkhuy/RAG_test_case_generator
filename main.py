from load_data import read_docx, extract_epic_overview
from chunking import chunking_document

def main():
    
    # Test reading a docx file
    file_path = r'D:\RAG_testcases\Epic 1 – User Management and Authentication.docx'
    text = read_docx(file_path)
    print("\nDocument text loaded.")
    print("-" *40)
    print(text[:200])
    print()
    
    print("Extracting Epic Overview...\n")
    overview = extract_epic_overview(text)
    print(overview)
    
    # Chunking the document
    chunks = chunking_document(text)
    print(f"\nTotal chunks created: {len(chunks)}")
    
    # Display first 2 chunks
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:\n", chunk)
    
if __name__ == '__main__':
    main()
