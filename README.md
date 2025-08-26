🧠 NLP Analysis of Patient Journals for Anxiety Trends
📌 Project Overview

This project uses Natural Language Processing (NLP) techniques to analyze patient journals and identify anxiety-related trends over time. Synthetic data (>100 entries) is generated to simulate real-world scenarios. The project performs text preprocessing, anxiety scoring, topic modeling (LDA), and temporal trend analysis.

📂 Features

Generates synthetic patient journals with anxiety indicators.

Preprocesses and vectorizes text using TF-IDF.

Detects hidden patterns with Latent Dirichlet Allocation (LDA) topic modeling.

Calculates and visualizes average anxiety trends over time.

Exports both raw and processed datasets to Excel.

📊 Dataset

The dataset is synthetically generated with:

PatientID – Unique patient identifier

Date – Journal entry date

JournalEntry – Text entry

AnxietyScore – Numeric score based on anxiety word frequency

CleanedText – Preprocessed journal entry

⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/YourUsername/NLP-Analysis-of-Patient-Journals-for-Anxiety-Trends.git
cd NLP-Analysis-of-Patient-Journals-for-Anxiety-Trends
pip install -r requirements.txt

▶️ Usage

Run the script:

python nlp_anxiety_trends.py


Outputs:

patient_journals_anxiety_synthetic.xlsx → Synthetic dataset

patient_journals_anxiety_processed.xlsx → Processed dataset

anxiety_trends.png → Visualization of anxiety trends

📈 Results

Topics Identified: Key terms from patient journals grouped into latent themes.

Trend Plot: Displays changes in anxiety scores across months.

Processed Dataset: Ready for deeper statistical or ML-based analysis.

📌 Applications

Mental health monitoring

Clinical NLP research

Early detection of patient anxiety trends

Prototyping healthcare AI models

👨‍💻 Author

Okes Imoni
🔗 GitHub: Okes2024
