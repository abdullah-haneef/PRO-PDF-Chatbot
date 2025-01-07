import PyPDF2
from io import BytesIO

def extract_text_from_pdf(file) -> str:
    """
    Extracts text from a PDF file.
    """
    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text() + "\n"
    return extracted_text
