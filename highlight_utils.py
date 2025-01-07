import pdfplumber
import fitz

def get_text_coordinates(pdf_file, text_to_highlight):
    coordinates = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words()
            for word in words:
                if text_to_highlight in word['text']:
                    coordinates.append((page_num, word['x0'], word['top'], word['x1'], word['bottom']))
    return coordinates

def highlight_text_in_pdf(input_file, output_file, highlights):
    pdf_document = fitz.open(input_file)
    for page_num, x0, y0, x1, y1 in highlights:
        page = pdf_document[page_num]
        rect = fitz.Rect(x0, y0, x1, y1)
        page.add_highlight_annot(rect)
    pdf_document.save(output_file)
