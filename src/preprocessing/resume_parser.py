from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes):
    """
    Extract text from a PDF resume.

    Parameters:
        file_bytes: Uploaded PDF file bytes.

    Returns:
        Extracted text as a string.
    """

    reader = PdfReader(
        BytesIO(file_bytes)
    )

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)


def extract_text_from_txt(file_bytes):
    """
    Extract text from a TXT resume.
    """

    return file_bytes.decode(
        "utf-8",
        errors="ignore"
    )


def extract_resume_text(uploaded_file):
    """
    Extract resume text based on file type.
    """

    file_name = uploaded_file.name.lower()

    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)

    if file_name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)

    raise ValueError(
        "Unsupported file type. "
        "Please upload a PDF or TXT file."
    )