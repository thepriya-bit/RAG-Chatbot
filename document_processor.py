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


import re


def chunk_text(text, chunk_size=500, overlap=50):

    # Split text using blank lines
    sections = re.split(r"\n\s*\n", text)

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        # If section is small enough, keep it as one chunk
        if len(section) <= chunk_size:
            chunks.append(section)

        else:
            # Split large sections into sentences
            sentences = re.split(
                r"(?<=[.!?])\s+",
                section
            )

            current_chunk = ""

            for sentence in sentences:

                if len(current_chunk) + len(sentence) <= chunk_size:

                    current_chunk += " " + sentence

                else:

                    if current_chunk:
                        chunks.append(
                            current_chunk.strip()
                        )

                    current_chunk = sentence

            if current_chunk:
                chunks.append(
                    current_chunk.strip()
                )

    return chunks