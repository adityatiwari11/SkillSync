from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import logging

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeJobMatcher:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Job-specific keywords that should have higher weight
        self.skill_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 'react', 'angular', 'node'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy',
                             'scikit-learn'],
            'design': ['photoshop', 'illustrator', 'figma', 'sketch', 'ui', 'ux', 'adobe', 'creative'],
            'management': ['leadership', 'project management', 'agile', 'scrum', 'team lead', 'manager'],
            'marketing': ['seo', 'sem', 'social media', 'content marketing', 'digital marketing', 'analytics']
        }

    def preprocess_text(self, text):
        """Enhanced text preprocessing for better matching"""
        try:
            if not text or text.strip() == "":
                return ""

            # Convert to lowercase
            text = text.lower()

            # Remove special characters but keep important symbols
            text = re.sub(r'[^\w\s+#-]', ' ', text)

            # Handle common abbreviations and variations
            replacements = {
                'javascript': 'javascript js',
                'artificial intelligence': 'ai artificial intelligence',
                'machine learning': 'ml machine learning',
                'user interface': 'ui user interface',
                'user experience': 'ux user experience',
                'database': 'db database sql',
                'full stack': 'fullstack full stack',
                'front end': 'frontend front end',
                'back end': 'backend back end'
            }

            for original, replacement in replacements.items():
                text = text.replace(original, replacement)

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)

            return ' '.join(processed_tokens)

        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return text.lower() if text else ""

    def extract_skills_and_keywords(self, text):
        """Extract important skills and keywords with weights"""
        try:
            skills_found = {}
            text_lower = text.lower()

            for category, keywords in self.skill_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        skills_found[keyword] = skills_found.get(keyword, 0) + 1

            return skills_found
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return {}

    def extract_important_phrases(self, text):
        """Extract important multi-word phrases"""
        try:
            text_lower = text.lower()
            important_phrases = [
                'machine learning', 'deep learning', 'data science', 'artificial intelligence',
                'neural networks', 'computer vision', 'natural language processing',
                'big data', 'data analysis', 'statistical analysis', 'predictive modeling',
                'python programming', 'sql database', 'cloud computing', 'aws', 'azure',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
                'data visualization', 'business intelligence', 'project management',
                'software development', 'full stack', 'front end', 'back end'
            ]

            found_phrases = []
            for phrase in important_phrases:
                if phrase in text_lower:
                    found_phrases.append(phrase)

            return found_phrases
        except Exception as e:
            logger.error(f"Error extracting phrases: {str(e)}")
            return []

    def calculate_enhanced_similarity(self, job_description, resume_text):
        """Calculate similarity with multiple factors"""
        try:
            # Preprocess texts
            processed_job = self.preprocess_text(job_description)
            processed_resume = self.preprocess_text(resume_text)

            if not processed_job or not processed_resume:
                return 0.0

            # Debug: Print processed texts (remove in production)
            print(f"DEBUG - Processed Job (first 200 chars): {processed_job[:200]}...")
            print(f"DEBUG - Processed Resume (first 200 chars): {processed_resume[:200]}...")

            # 1. TF-IDF Cosine Similarity (main component) - MORE LENIENT SETTINGS
            vectorizer = TfidfVectorizer(
                max_features=10000,  # Increased from 5000
                ngram_range=(1, 2),  # Reduced to 1,2 for better matching
                max_df=0.95,  # More lenient - was 0.85
                min_df=1,  # Keep rare terms
                sublinear_tf=False,  # Don't use log scaling
                lowercase=True,
                stop_words=None  # Don't remove stopwords here (already done)
            )

            tfidf_matrix = vectorizer.fit_transform([processed_job, processed_resume])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            print(f"DEBUG - Base cosine similarity: {cosine_sim:.4f}")

            # 2. Skills matching bonus (ENHANCED)
            job_skills = self.extract_skills_and_keywords(job_description)
            resume_skills = self.extract_skills_and_keywords(resume_text)

            common_skills = set(job_skills.keys()) & set(resume_skills.keys())
            skills_bonus = len(common_skills) * 0.08  # Increased from 0.05 to 0.08

            print(f"DEBUG - Job skills found: {list(job_skills.keys())}")
            print(f"DEBUG - Resume skills found: {list(resume_skills.keys())}")
            print(f"DEBUG - Common skills: {list(common_skills)}")
            print(f"DEBUG - Skills bonus: {skills_bonus:.4f}")

            # 3. Keyword density bonus (ENHANCED)
            job_words = set(processed_job.split())
            resume_words = set(processed_resume.split())
            common_words = job_words & resume_words
            keyword_overlap = len(common_words) / max(len(job_words), 1)
            keyword_bonus = min(keyword_overlap * 0.2, 0.2)  # Up to 20% bonus (was 10%)

            print(f"DEBUG - Keyword overlap: {keyword_overlap:.4f}, bonus: {keyword_bonus:.4f}")
            print(f"DEBUG - Common words count: {len(common_words)} out of {len(job_words)} job words")

            # 4. Exact phrase matching bonus
            job_phrases = self.extract_important_phrases(job_description)
            resume_phrases = self.extract_important_phrases(resume_text)
            phrase_matches = len(set(job_phrases) & set(resume_phrases))
            phrase_bonus = phrase_matches * 0.05  # 5% per exact phrase match

            print(f"DEBUG - Phrase matches: {phrase_matches}, bonus: {phrase_bonus:.4f}")

            # 5. Length penalty (reduced)
            length_penalty = 0
            if len(resume_text.split()) < 30:  # Reduced threshold from 50 to 30
                length_penalty = 0.05  # Reduced penalty from 0.1 to 0.05

            # 6. WEIGHTED COMBINATION - Give more weight to base similarity
            weighted_cosine = cosine_sim * 1.3  # Boost base similarity

            # Combine all factors
            final_score = weighted_cosine + skills_bonus + keyword_bonus + phrase_bonus - length_penalty

            print(
                f"DEBUG - Final components: cosine={weighted_cosine:.4f}, skills={skills_bonus:.4f}, keywords={keyword_bonus:.4f}, phrases={phrase_bonus:.4f}, penalty={length_penalty:.4f}")
            print(f"DEBUG - Raw final score: {final_score:.4f}")

            # Ensure score is between 0 and 1, but allow higher ceiling
            final_score = min(max(final_score, 0.0), 1.0)

            print(f"DEBUG - Clamped final score: {final_score:.4f}")
            print("-" * 50)

            return final_score

        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0


def extract_text_from_pdf(file_path):
    """Extract text from PDF with error handling"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text from {file_path}: {str(e)}")
        return ""


def extract_text_from_docx(file_path):
    """Extract text from DOCX with error handling"""
    try:
        text = docx2txt.process(file_path)
        return text if text else ""
    except Exception as e:
        logger.error(f"Error extracting DOCX text from {file_path}: {str(e)}")
        return ""


def extract_text_from_txt(file_path):
    """Extract text from TXT with error handling"""
    try:
        encodings = ['utf-8', 'latin-1', 'cp1252']  # Try multiple encodings
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        return ""
    except Exception as e:
        logger.error(f"Error extracting TXT text from {file_path}: {str(e)}")
        return ""


def extract_text(file_path):
    """Extract text from various file formats with comprehensive error handling"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""

        if file_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            return extract_text_from_docx(file_path)
        elif file_path.lower().endswith('.txt'):
            return extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""


# Initialize the matcher
matcher = ResumeJobMatcher()


@app.route("/")
def skillsync():
    return render_template('skillsync.html')


@app.route('/matcher', methods=['POST'])
def matcher_route():
    try:
        if request.method == 'POST':
            job_description = request.form.get('job_description', '').strip()
            resume_files = request.files.getlist('resumes')

            # Validation
            if not job_description:
                return render_template('skillsync.html',
                                       message="Please enter a job description.",
                                       error=True)

            if not resume_files or not any(file.filename for file in resume_files):
                return render_template('skillsync.html',
                                       message="Please upload at least one resume file.",
                                       error=True)

            # Process resumes
            resume_data = []
            valid_files = 0

            for resume_file in resume_files:
                if resume_file.filename:
                    try:
                        # Secure filename handling
                        filename = resume_file.filename
                        safe_filename = re.sub(r'[^\w\s.-]', '', filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

                        # Save file
                        resume_file.save(file_path)

                        # Extract text
                        resume_text = extract_text(file_path)

                        if resume_text and len(resume_text.strip()) > 10:  # Minimum content check
                            resume_data.append({
                                'filename': filename,
                                'text': resume_text,
                                'file_path': file_path
                            })
                            valid_files += 1
                        else:
                            logger.warning(f"No valid text extracted from {filename}")

                        # Clean up file after processing
                        try:
                            os.remove(file_path)
                        except:
                            pass

                    except Exception as e:
                        logger.error(f"Error processing file {resume_file.filename}: {str(e)}")
                        continue

            if valid_files == 0:
                return render_template('skillsync.html',
                                       message="No valid resume content could be extracted. Please check your files.",
                                       error=True)

            # Calculate similarities using enhanced model
            results = []
            for resume_info in resume_data:
                try:
                    similarity = matcher.calculate_enhanced_similarity(
                        job_description,
                        resume_info['text']
                    )

                    # Convert to percentage (0.53 -> 53)
                    percentage_score = round(similarity * 100)

                    results.append({
                        'filename': resume_info['filename'],
                        'score': percentage_score,
                        'similarity': similarity  # Keep original for sorting
                    })
                except Exception as e:
                    logger.error(f"Error calculating similarity for {resume_info['filename']}: {str(e)}")
                    continue

            if not results:
                return render_template('skillsync.html',
                                       message="Could not calculate similarities. Please try again.",
                                       error=True)

            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)

            # Get top 5 results
            top_results = results[:5]
            top_resumes = [result['filename'] for result in top_results]
            similarity_scores = [result['score'] for result in top_results]

            return render_template('skillsync.html',
                                   message=f"Top matching resumes (processed {valid_files} files):",
                                   top_resumes=top_resumes,
                                   similarity_scores=similarity_scores,
                                   job_description=job_description)

    except Exception as e:
        logger.error(f"Unexpected error in matcher route: {str(e)}")
        return render_template('skillsync.html',
                               message="An unexpected error occurred. Please try again.",
                               error=True)

    return render_template('skillsync.html')


@app.errorhandler(413)
def too_large(e):
    return render_template('skillsync.html',
                           message="File too large. Please upload smaller files.",
                           error=True)


@app.errorhandler(500)
def internal_error(e):
    return render_template('skillsync.html',
                           message="Internal server error. Please try again.",
                           error=True)


if __name__ == '__main__':
    try:
        # Create upload folder if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")

        # Set maximum file size (16MB)
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

        logger.info("Starting Enhanced Resume Matcher Flask App...")
        app.run(debug=True, host='0.0.0.0', port=5000)

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error starting app: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)