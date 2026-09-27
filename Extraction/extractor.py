import os
import csv
import re
import pdfplumber
import time

# --- CONFIGURATION ---
PDF_DIR = "all_the_pdf_from_gov_database"  
OUTPUT_CSV = "model_ready_mospi_complete.csv"
CHECKPOINT_LOG = "processed_pdfs_log.txt"  # Tracks finished PDFs to prevent duplicates if restarted
DATA_COLUMNS_COUNT = 15  

# Core infrastructure sectors to identify from page headers
KNOWN_SECTORS = [
    "RAILWAY", "ROAD", "HIGHWAY", "POWER", "PETROLEUM", "COAL", "STEEL", 
    "TELECOMMUNICATION", "URBAN", "WATER", "ATOMIC", "AVIATION", 
    "FERTILIZER", "MINE", "PORT", "SHIPPING", "HEALTH", "EDUCATION"
]

GARBAGE_HEADERS = [
    "SL. NO", "ANTICIPATED", "EXPENDITURE", "AGENCY", "ORIGINAL", 
    "REVIEW", "SANCTION", "PROJECT", "COST", "TOTAL", "OVERVIEW",
    "STATUS", "SCHEDULE", "MILESTONE", "REMARKS"
]

def clean_cell(cell):
    if cell is None: return ""
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', str(cell))
    cleaned = cleaned.replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r'\s+', ' ', cleaned)

def extract_sector(text, current_sector):
    if not text: return current_sector
    text_upper = text.upper()
    for sector in KNOWN_SECTORS:
        if sector in text_upper: return sector
    return current_sector

def is_valid_project_row(row_data):
    if len(row_data) < 2: return False
    project_name = str(row_data[1]).upper()
    if len(project_name) <= 5: return False
    for garbage in GARBAGE_HEADERS:
        if garbage in project_name: return False
    filled_cells = sum(1 for cell in row_data if cell.strip() != "")
    if filled_cells < 4: return False
    return True

def run_extractor():
    print("🚀 Starting 20-Year MoSPI PDF Smart Extractor (2006-2026)...\n" + "-"*60)
    
    if not os.path.exists(PDF_DIR):
        print(f"❌ Error: Folder '{PDF_DIR}' not found.")
        return

    headers = ['File_Name', 'Page', 'Sector'] + [f"Column_{i}" for i in range(1, DATA_COLUMNS_COUNT + 1)]

    # 1. Initialize CSV only if it doesn't exist (prevents overwriting on restart)
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)

    # 2. Load the Checkpoint Log to see which PDFs are already done
    completed_pdfs = set()
    if os.path.exists(CHECKPOINT_LOG):
        with open(CHECKPOINT_LOG, "r") as f:
            completed_pdfs = set(f.read().splitlines())

    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')])
    total_files = len(pdf_files)
    print(f"📁 Found {total_files} PDF files. ({len(completed_pdfs)} already completed).")

    total_valid_rows = 0
    start_time = time.time()

    for file_idx, filename in enumerate(pdf_files, 1):
        # SMART RESUME: Skip this PDF if it's already in the log
        if filename in completed_pdfs:
            print(f"⏭️  [{file_idx}/{total_files}] Skipping {filename} (Already Processed)")
            continue

        filepath = os.path.join(PDF_DIR, filename)
        file_valid_rows = 0
        current_sector = "UNKNOWN"
        
        print(f"⚙️ [{file_idx}/{total_files}] Parsing: {filename}...")
        
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    current_sector = extract_sector(page.extract_text(), current_sector)
                    
                    tables = page.extract_tables(table_settings={
                        "vertical_strategy": "lines", "horizontal_strategy": "lines"
                    })
                    
                    for table in tables:
                        for row in table:
                            cleaned_row = [clean_cell(cell) for cell in row]
                            
                            if len(cleaned_row) < DATA_COLUMNS_COUNT:
                                cleaned_row += [""] * (DATA_COLUMNS_COUNT - len(cleaned_row))
                            else:
                                cleaned_row = cleaned_row[:DATA_COLUMNS_COUNT]
                                
                            if is_valid_project_row(cleaned_row):
                                final_csv_row = [filename, page_num, current_sector] + cleaned_row
                                
                                # INSTANT DISK STREAMING (Zero Data Loss)
                                with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as f:
                                    csv.writer(f).writerow(final_csv_row)
                                    
                                file_valid_rows += 1
                                total_valid_rows += 1

            # 3. SUCCESSFUL CHECKPOINT: Mark PDF as finished
            with open(CHECKPOINT_LOG, mode='a', encoding='utf-8') as log_file:
                log_file.write(filename + "\n")
                
            print(f"   ✅ Kept {file_valid_rows} pristine project rows.")
            
        except Exception as e:
            print(f"   ❌ Error reading {filename} (File might be corrupted): {e}")

    elapsed_mins = (time.time() - start_time) / 60
    print("\n" + "="*60)
    print("🏆 20-YEAR SMART EXTRACTION COMPLETED")
    print("="*60)
    print(f"📁 Model-Ready File : '{OUTPUT_CSV}'")
    print(f"📊 New Rows Added   : {total_valid_rows:,}")
    print(f"⏱️ Time Taken       : {elapsed_mins:.1f} minutes")
    print("="*60)

if __name__ == "__main__":
    run_extractor()
