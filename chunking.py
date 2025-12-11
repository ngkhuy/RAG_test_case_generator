import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Define list heading in requirements
SECTION_HEADERS = [
    "Narrative",
    "In Scope",
    "Out of Scope",
    "Acceptance Criteria",
    "Priority",
    "Business Rules",
    "Additional Comments"
]

def split_requirements(req_text: str):
    sections = []
    current_header = None
    current_lines = []

    for line in req_text.splitlines():
        stripped = line.strip()
        
        # Check if line matches any header
        header = next((h for h in SECTION_HEADERS if stripped.startswith(h)), None)
        
        if header:
            if current_header:
                sections.append((current_header, None, "\n".join(current_lines)))
            
            current_header = header
            current_lines = []
        else:
            current_lines.append(line)
            
    # flush final block
    if current_header:
        sections.append((current_header, None, "\n".join(current_lines)))

    return sections

def chunking_document(requirement_text: str, max_chunk_size: int = 800):
    """Chunk requirement document into hierarchical chunks (1-pass)."""
    chunks = []

    # Extract epic level
    lang = re.search(r"Language:\s*([a-zA-Z-]+)", requirement_text)
    epic_id = re.search(r"Epic ID:\s*([A-Za-z0-9]+)", requirement_text)

    epic_metadata = {
        "lang": lang.group(1) if lang else None,
        "epic_id": epic_id.group(1) if epic_id else None
    }

    # Detect requirement headers
    req_pattern = r"(PSE[0-9.]+)\s*–\s*(.+?)\n"
    matches = list(re.finditer(req_pattern, requirement_text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    for i, m in tqdm(
        enumerate(matches),
        total=len(matches),
        desc="Chunking requirement",
        ncols=100
    ):
        req_id = m.group(1).strip()
        req_name = m.group(2).strip()

        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(requirement_text)
        req_block = requirement_text[start:end].strip()

        # Extract priority
        priority_match = re.search(r"Priority\s*\n([A-Z]+)", req_block)
        priority = priority_match.group(1).strip() if priority_match else None

        # Requirement-level metadata
        base_metadata = {
            **epic_metadata,
            "requirement_id": req_id,
            "requirement_name": req_name,
            "priority": priority
        }

        # Section-based parsing
        sections = split_requirements(req_block)

        for header, parent, content in sections:

            # Build metadata including optional parent_section
            metadata = {**base_metadata, "section": header}
            if parent:
                metadata["parent_section"] = parent

            # Short enough → 1 chunk
            if len(content) <= max_chunk_size:
                chunks.append({
                    "text": f"{header}\n{content.strip()}",
                    "metadata": metadata
                })
            else:
                # Too long → chunk using LangChain
                parts = splitter.split_text(content)
                for p in parts:
                    chunks.append({
                        "text": f"{header}\n{p}",
                        "metadata": metadata
                    })

    return chunks
