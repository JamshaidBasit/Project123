import pdfplumber
import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize SQLite database and create the table
def init_db():
    conn = sqlite3.connect("test0001.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_title TEXT,
            page_number INTEGER,
            chunk_number INTEGER,
            chunk_text TEXT
        )
    """)
    conn.commit()
    return conn

# Configuration
BOOK_TITLE = "History Book"
PDF_FILE = r"C:\Users\USER\Desktop\Mcs_Project\MCS_Project\testing.pdf" 
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Initialize LangChain text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

# Main logic to extract and store PDF chunks
def process_pdf_to_db():
    conn = init_db()
    cursor = conn.cursor()

    with pdfplumber.open(PDF_FILE) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                print(f"\n--- Page {page_num} ---")
                chunks = text_splitter.split_text(page_text)
                for chunk_num, chunk in enumerate(chunks, start=1):
                    print(f"Chunk {chunk_num}:\n{chunk}\n")
                    # Insert into database
                    cursor.execute("""
                        INSERT INTO pdf_chunks (book_title, page_number, chunk_number, chunk_text)
                        VALUES (?, ?, ?, ?)
                    """, (BOOK_TITLE, page_num, chunk_num, chunk))
    
    conn.commit()
    conn.close()
    print("✅ PDF content saved to database 'test001.db'")

# Run the script
if __name__ == "__main__":
    process_pdf_to_db()
