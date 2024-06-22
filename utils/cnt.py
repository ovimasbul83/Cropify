# Let's count the number of words in the PDF.
from PyPDF2 import PdfReader

# Path to the PDF file
pdf_path = './data/sodapdf-converted.pdf'

# Read the PDF file
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text()

# Count the number of words
word_count = len(text.split())
print(word_count)
