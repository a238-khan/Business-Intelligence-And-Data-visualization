import pandas as pd
import time

# Test loading performance
print("Testing Excel file loading performance...")

# Test Institution Level Data
start_time = time.time()
try:
    inst_data = pd.read_excel('Guardian_Cleaned_Final.xlsx', sheet_name='Institution Level Data')
    load_time = time.time() - start_time
    print(f"Institution Level Data - Load time: {load_time:.2f}s, Shape: {inst_data.shape}")
    print(f"Columns: {list(inst_data.columns)}")
    print(f"Memory usage: {inst_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"Error loading Institution Level Data: {e}")

# Test Subject Level Data
start_time = time.time()
try:
    subj_data = pd.read_excel('Guardian_Cleaned_Final.xlsx', sheet_name='Subject Level Data')
    load_time = time.time() - start_time
    print(f"Subject Level Data - Load time: {load_time:.2f}s, Shape: {subj_data.shape}")
    print(f"Memory usage: {subj_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"Error loading Subject Level Data: {e}")

# Test all sheets
try:
    excel_file = pd.ExcelFile('Guardian_Cleaned_Final.xlsx')
    print(f"Available sheets: {excel_file.sheet_names}")
except Exception as e:
    print(f"Error reading Excel file info: {e}")
