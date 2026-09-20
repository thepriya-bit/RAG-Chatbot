from pypdf import PdfReader
import re


def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=1000, overlap=100):
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding the paragraph stays within the limit
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += "\n" + paragraph

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Split large paragraphs into sentences
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            current_chunk = ""

            for sentence in sentences:

                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence

                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add word-based overlap
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            previous_chunk = chunks[i - 1]
            previous_words = previous_chunk.split()
            overlap_words = previous_words[-20:]
            chunk = " ".join(overlap_words) + " " + chunk
        final_chunks.append(chunk.strip())
    return final_chunks