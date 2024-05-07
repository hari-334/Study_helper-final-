import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import pytesseract
from pytesseract import Output
from PIL import Image
from flask import Flask, render_template, request ,redirect , url_for , session 
from flask_session import Session
# from flask_session import FileSystemSessionInterface
import PyPDF2
from io import BytesIO
import uuid
import os

nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')

UPLOAD_FOLDER = 'uploads'



def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    english_vocab = set(word.lower() for word in nltk.corpus.words.words())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

def divide_documents(text):
    paragraphs = text.split('\n\n')
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))
    return sentences

def generate_summaries(sentences):
    summaries = []
    for sentence in sentences:
        summary = sentence
        summaries.append(summary)
    return summaries

def readable_summary(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


app = Flask(__name__)

# nltk.download('words')
# nltk.download('punkt')
# nltk.download('stopwords')
app.secret_key ='supra_me'

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    for page_number in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_number)
        text += page.extractText()
    return text

@app.route('/')
def index():
    return render_template('index.ejs')

@app.route('/process_file', methods=['POST'])
def process_file():
    session['questions']=[]
    session['answers']=[]
    
    # Get the choice of the user (image or pdf)
    file_type = request.form['file_type']
    session['file_type'] = file_type
    
    if file_type == 'image':
        # Example: Get the uploaded image file
        uploaded_file = request.files['file']
        name=uploaded_file.filename
        if uploaded_file.filename == '':
            return 'No selected file', 400
        file_id = str(uuid.uuid4())  # Generate a unique ID for the file
        file_path = os.path.join(UPLOAD_FOLDER, file_id)
        uploaded_file.save(file_path)
        session['file_path'] = file_path
        # print(uploaded_file)

        

    if file_type == 'pdf':
        # Example: Get the uploaded PDF file
        uploaded_file = request.files['file']
        name=uploaded_file.filename
        if uploaded_file.filename == '':
            return 'No selected file', 400
        file_id = str(uuid.uuid4())  # Generate a unique ID for the file
        file_path = os.path.join(UPLOAD_FOLDER, file_id)
        uploaded_file.save(file_path)
        session['file_path'] = file_path
        print(name)
    return redirect(url_for('quest',filename=name))
       
@app.route('/quest/<filename>',methods=['GET'])
def quest(filename):
        # Extract text from the PDF using PyPDF2
        # print("Meowwww")

        return render_template("question.ejs",filename=filename)
        # uploaded_file

@app.route('/quest/<filename>',methods=['POST'])
def upload(filename):
        print("broo")
        items=session.get('questions')
        print("Session before appending:", items)
        print('hello')
        file_path = session.get('file_path')

        if file_path is None or not os.path.exists(file_path):
            return "File not found", 400
        
        if session.get('file_type') == 'pdf':
            with open(file_path, 'rb') as file:
                file_data = file.read()
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_data))

            # Get the number of pages in the PDF
            num_pages =num_pages = len(pdf_reader.pages)

            # Initialize an empty string to store the extracted text
            txt = ""

            # Iterate through all pages
            for page_num in range(num_pages):
                # Get the page
                page = pdf_reader.pages[page_num]

                # Extract text from the page
                etext = page.extract_text()

                # Append the text to the result string
                txt+=etext
        
        if session.get('file_type') == 'image':
            # Perform OCR on the image
            with Image.open(file_path) as image:
                txt = pytesseract.image_to_string(image)

        sample_text = txt
        if sample_text == '':
            sample_text = "No answers found"
        # Preprocess text using the loaded function
        preprocessed_text = preprocess_text(sample_text)

        # Divide documents into sentences using the loaded function
        sentences = divide_documents(preprocessed_text)

        # Generate summaries using the loaded function
        summaries = generate_summaries(sentences)

        # Check if the user wants to view the summary
        show_summary = request.form.get('show_summary')

        if show_summary:
            # Use the loaded function to generate a readable summary
            summary = readable_summary(sample_text)
        else:
            summary = None

        # Example: Get the question from the user
        user_question = request.form['question']
        session['questions'].append(user_question)
        session.modified = True
        print("Session after appending:", session.get('questions'))
        # Use the loaded QA pipeline
        answer = qa_pipeline(question=user_question, context=sample_text)
        session['answers'].append(answer['answer'])

        # Render the result on the webpage
        return render_template('question.ejs', question=session.get('questions'), answer=session.get('answers'), summary=summary,filename=filename,len=len(session.get('questions')))

if __name__ == '__main__':
    app.run(debug=True)

