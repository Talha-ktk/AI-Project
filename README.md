Laptop Price Prediction: Data Cleaning & Feature Engineering
💻 Project Overview
This project focuses on the Data Preprocessing and Feature Engineering phase of a Machine Learning pipeline. The goal was to take a raw dataset containing laptop specifications and prices and transform it into a clean, numeric, and structured format suitable for predictive modeling.

📊 Dataset Description
The dataset contains various technical specifications of laptops from different brands.

Original Data: 1,303 rows and 11 columns.

Cleaned Data: 1,242 rows and 18 optimized features.

Key Challenges Identified:
Missing Values: Approximately 30 missing values were found across almost every column.

Duplicates: The dataset contained 58 duplicate entries that needed removal.

Incorrect Data Types: Columns like Ram and Weight were stored as objects (text) due to units like "GB" and "kg".

Complex Formatting: Columns like ScreenResolution, Cpu, and Memory contained multiple pieces of information in a single string.

Data Typos: The Inches column contained invalid entries like "?" and physically impossible sizes like 35.6.

🛠️ Data Cleaning Steps
The following steps were performed using Python and the Pandas library:

Integrity Check: Dropped duplicate rows and removed completely empty rows.

Unit Stripping: Removed "GB" from Ram and "kg" from Weight, converting both to numeric types.

Typos & Error Correction: Replaced incorrect Inches values based on a logical mapping (e.g., 35.6 → 15.6) and dropped rows with unfixable errors.

Feature Engineering:

Screen: Extracted Touchscreen and Ips as binary flags. Parsed X_res and Y_res from the resolution strings.

Processor: Extracted Cpu_Speed_GHz and Cpu_Vendor (Intel, AMD, etc.).

Storage: Decomposed the Memory column into four separate numeric columns: SSD, HDD, Flash_Storage, and Hybrid.

OS: Grouped various operating systems into broader categories like Windows, Mac, and Linux.

📈 Final Cleaned Schema
The final dataset consists of 18 columns, all prepared for analysis: | Column | Description | | :--- | :--- | | Company | Laptop Brand | | TypeName | Category (Ultrabook, Notebook, etc.) | | Inches | Screen Size | | Ram | Memory (Numeric) | | Weight | Weight in kg (Numeric) | | Price | Target Variable | | Touchscreen/Ips | Display Technology flags | | X_res / Y_res | Pixel Dimensions | | Cpu_Speed_GHz | Processor Clock Speed | | SSD / HDD | Storage Capacity per type | | OpSys_Category | Categorized Operating System |
