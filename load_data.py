from docx import Document
import re

def read_docx(file_path):
    """Read a docx file and return the text"""
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def extract_epic_overview(text: str):
    """
    Extracts the Epic Overview section from the raw requirement document text.
    Returns None if no overview is found.
    """
    pattern = r"Epic Overview\s*(.+?)(?=\n(?:Requirements|PSE[0-9.]+|Narrative|Scope|Acceptance Criteria|Priority))"

    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

if __name__ == '__main__':
    file_path = r'D:\RAG_testcases\Epic 1 – User Management and Authentication.docx'
    text = read_docx(file_path)
    
    overview = extract_epic_overview(text)
    print(overview)