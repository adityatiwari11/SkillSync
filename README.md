# SkillSync: AI-Powered Resume Matcher (Detailed Analysis)

SkillSync is a sophisticated web application built with Flask and powered by NLP techniques to intelligently score and rank candidate resumes against a job description. It moves beyond simple keyword matching by implementing a custom, multi-faceted scoring algorithm to provide a more accurate and human-like assessment of resume relevance.

![SkillSync Screenshot](https://user-images.githubusercontent.com/39327825/213904085-3b772c65-859a-4122-8d76-e575e92c63d5.png)
*(This is a placeholder image. Replace it with a screenshot of your running application for best results.)*

---

## 1. Technical Deep Dive: The Enhanced Scoring Algorithm

The core of SkillSync is the `ResumeJobMatcher` class, which calculates a nuanced similarity score. This isn't just a single metric; it's a weighted combination of several analytical factors. Here’s a breakdown of the process for each resume:

### Step 1: Advanced Text Preprocessing (`preprocess_text`)

Before any comparison, the text from both the job description and the resume undergoes a rigorous cleaning process to standardize the content for accurate analysis:

1.  **Normalization:** All text is converted to lowercase.
2.  **Character Cleaning:** Special characters are removed, but crucial symbols like `+` (for C++), `#` (for C#), and `-` are preserved.
3.  **Abbreviation Expansion:** Common technical abbreviations are expanded to ensure they match their full-form counterparts (e.g., `'ml'` is expanded to `'ml machine learning'`). This prevents mismatches and improves keyword recognition.
4.  **Tokenization:** The text is broken down into individual words (tokens) using `nltk.word_tokenize`.
5.  **Lemmatization & Stopword Removal:** Each token is reduced to its root form (lemma) using `WordNetLemmatizer` (e.g., "running" becomes "run"). Common English stopwords (like "the", "a", "is") are discarded to reduce noise.

### Step 2: Multi-Factor Similarity Calculation (`calculate_enhanced_similarity`)

The final score is a composite metric derived from five key components:

1.  **TF-IDF Cosine Similarity (Baseline Score):**
    * **What it is:** This forms the foundation of the score. Term Frequency-Inverse Document Frequency (TF-IDF) is used to vectorize the preprocessed texts, giving higher weight to words that are important to a document but not common across all documents. Cosine similarity then measures the angle between these two vectors, providing a score of how similar their content and context are.
    * **Implementation:** A `TfidfVectorizer` is configured with `ngram_range=(1, 2)` to consider both single words and two-word phrases (bigrams), which is crucial for capturing technical terms like "machine learning".

2.  **Skills Matching Bonus:**
    * **What it is:** A direct, high-value bonus is awarded for each predefined skill found in both the job description and the resume.
    * **Implementation:** The code iterates through a hard-coded dictionary of `skill_keywords` (categorized into programming, data science, design, etc.). Every common skill adds a fixed bonus (`+0.08`) to the final score, directly rewarding candidates who possess the required hard skills.

3.  **Keyword Density Bonus:**
    * **What it is:** This metric rewards resumes that have a high overlap of significant words with the job description.
    * **Implementation:** It calculates the ratio of common words between the two documents relative to the total number of words in the job description. This provides a bonus of up to `+0.2`, rewarding resumes that are highly focused on the topics mentioned in the job post.

4.  **Exact Phrase Matching Bonus:**
    * **What it is:** Certain multi-word phrases carry significant meaning (e.g., "natural language processing", "full stack developer"). This component awards a bonus for each of these exact phrases found in both texts.
    * **Implementation:** A predefined list of `important_phrases` is checked against both documents. Each match provides a `+0.05` bonus, ensuring that candidates with experience in key technologies or methodologies are ranked higher.

5.  **Length Penalty:**
    * A minor penalty (`-0.05`) is applied if a resume is extremely short (under 30 words), flagging potentially incomplete or poorly detailed submissions.

### Final Score Combination

The final score is calculated with a weighted formula that boosts the foundational cosine similarity and adds the bonuses:

`Final Score = (Cosine Similarity * 1.3) + Skills Bonus + Keyword Bonus + Phrase Bonus - Length Penalty`

The result is then clamped between 0.0 and 1.0 and converted to a percentage (e.g., 85) for a user-friendly display.

---

## 2. Application Architecture & Workflow

The application is built using a simple but powerful Flask backend.

### Backend (`main.py`)

* **Server:** A Flask web server handles all incoming requests.
* **Routing:**
    * `@app.route("/")`: Renders the main `skillsync.html` page.
    * `@app.route("/matcher", methods=['POST'])`: This is the core endpoint. It receives the job description and resume files via a POST request from the HTML form.
* **File Handling:**
    1.  Uploaded resume files are received via `request.files.getlist('resumes')`.
    2.  Each file is temporarily saved to the `/uploads` directory with a sanitized filename.
    3.  The appropriate text extraction function (`extract_text_from_pdf`, `extract_text_from_docx`, etc.) is called based on the file extension.
    4.  After text extraction, the temporary file is immediately deleted to ensure privacy and clean up server space.
* **Error Handling:** The application gracefully handles common errors:
    * `413 Request Entity Too Large`: If the user uploads files exceeding the 16MB limit.
    * `500 Internal Server Error`: For any unexpected exceptions in the backend.
    * Custom validation for missing job descriptions or resume files.
* **NLTK Data:** The script automatically checks for and downloads the required NLTK datasets (`punkt`, `stopwords`, `wordnet`) on first launch.

### Frontend (`skillsync.html`)

* **UI Framework:** The interface is built with **Bootstrap** for a clean, responsive layout. Custom CSS is used for the modern, gradient-based theme.
* **Templating:** **Jinja2** (natively supported by Flask) is used to dynamically render the results. The backend passes variables like `top_resumes` and `similarity_scores` to the HTML, which are then displayed in a list.
* **User Interaction:**
    * A simple HTML form (`enctype="multipart/form-data"`) is used to submit the data.
    * A small JavaScript snippet provides user feedback, showing the names and count of selected files before uploading.

---

## 3. Setup and Deployment Guide

Follow these steps to get a local instance of SkillSync running.

### Prerequisites

* Python 3.8+
* `pip` package installer

### Step 1: Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
