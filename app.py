from flask import Flask, render_template, request,session,flash,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, current_user, login_required, logout_user,UserMixin
from models.models import User, Patient, Doctor
import datetime
from datetime import datetime
from uuid import uuid4
from flask import Flask, render_template, request, redirect,url_for,flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
import re
import enchant
import joblib
from werkzeug.utils import secure_filename
import medspacy
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import subprocess
import matplotlib.pyplot as plt
from medspacy.ner import TargetRule
from spacy.tokens import Span
import os
import ast
from pydub import AudioSegment
import nltk
from nltk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from flask_login import LoginManager
import io
import base64
import locationtagger
from uuid import uuid4
from joblib import Parallel, delayed,load
from twilio.rest import Client
from dotenv import load_dotenv
import secrets
email_g = ""
password_g =""

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RECORDINGS_FOLDER'] = 'recordings'
app.config['static_folder'] = 'static'
app.config['css_folder']='css'
app.config['SESSION_TYPE'] = 'filesystem'
secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] =secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/MT'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
try:
    class User(db.Model, UserMixin):
        __tablename__ = 'user'
        id = db.Column(db.Integer, primary_key=True)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password = db.Column(db.String(120), nullable=False)
        patients = db.relationship('Patient', backref='user', lazy=True)
        def is_active(self):
            return True  # You can implement your own logic here to determine if the user is active

        def get_id(self):
            return str(self.id)  # Assuming the user ID is an integer

    class Doctor(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password = db.Column(db.String(120), nullable=False)
        # Define any relationships involving doctors here

    class Patient(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
        preprocessed_text = db.Column(db.Text, nullable=False)
        personal_detail = db.Column(db.Text, nullable=False)
        derma_detail = db.Column(db.Text, nullable=False)
        clinical_detail = db.Column(db.Text, nullable=False)
        image = db.Column(db.Text, nullable=False)
        status = db.Column(db.Text, nullable=False)
        def to_dict(self):
            return {
                'id': self.id,
                'user_id': self.user_id,
                'preprocessed_text': self.preprocessed_text,
                'personal_detail':self.personal_detail,
                'derma_detail': self.derma_detail,
                'clinical_detail': self.clinical_detail,
                'image': self.image,
                'status': self.status
            }
        def __init__(self,user_id, preprocessed_text, personal_detail,derma_detail, clinical_detail, image,status):
            # self.id =id
            self.user_id = user_id
            self.preprocessed_text = preprocessed_text
            self.personal_detail = personal_detail
            self.derma_detail = derma_detail
            self.clinical_detail = clinical_detail
            self.image = image
            self.status = status

    class RegistrationForm(FlaskForm):
        email = StringField('Email', validators=[InputRequired(), Email()])
        password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
        submit = SubmitField('Register')
    class LoginForm(FlaskForm):
        email = StringField('Email', validators=[InputRequired(), Email()])
        password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
        submit = SubmitField('Login')
    
    class DocLoginForm(FlaskForm):
        email = StringField('Email', validators=[InputRequired() ,Email()])
        password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
        submit = SubmitField('Login')


    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')
    @app.route('/service')
    def service():
        return render_template('service.html')
    @app.route('/index')
    def index1():
        return render_template('index.html')
  
    @app.route('/about')
    def about():
        return render_template('about.html')
    @app.route('/feature')
    def feature():
        return render_template('feature.html')
    @app.route('/appointment', methods=['GET', 'POST'])
    def appointment():
        print("haiiiiiiiiiiiiiiiiii")
        if request.method == 'POST':
            name = request.form['name']
            email = request.form['email']
            mobile = request.form['number']
            doc = request.form['doc']
            date = request.form['date']
            time = request.form['time']
            # Send SMS message
            message = client.messages.create(
                body=f'Concern Submitted From :\nName: {name}\nEmail: {email}\nDoctor: {doc}\n\n*****Appointment Scheduled On *****\n Date: {date}\nTime: {time}',
                from_='+19078917970',
                to=f'+91 {mobile}'
            )

            print(f"Message sent successfully! SID: {message.sid}")
            return redirect(url_for('index'))
        else:
            return render_template('appointment.html')
    @app.route('/testimonial')
    def testimonial():
        return render_template('testimonial.html')
    @app.route('/user_success')
    def user_success():
        return render_template('user_success.html')
    ################################REGISTRATION FOR USER################################
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        form = RegistrationForm()
        if form.validate_on_submit():
            with app.app_context():
                user = User(email=form.email.data, password=form.password.data)
                db.session.add(user)
                db.session.commit()
            flash('Registration successful', 'success')
            return redirect(url_for('login'))
        return render_template('register.html', form=form)
    ################################LOGIN OF USER################################
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.password == form.password.data:
                login_user(user) 
                flash('Login successful', 'success')
                return redirect(url_for('service'))
            else:
                flash('Invalid email or password', 'error')
        return render_template('login.html', form=form)

 ################################DOCTOR LOGIN################################
    @app.route('/doclogin', methods=['GET', 'POST'])   
    def doclogin():
        form = DocLoginForm()
        if form.validate_on_submit():
            doctor = Doctor.query.filter_by(email=form.email.data, password=form.password.data).first()
            if doctor:
                flash('Welcome Doctor', 'success')
                patients = Patient.query.all()
                session['patients'] = [patient.to_dict() for patient in patients]  # Convert patients to dictionaries
                # user_id = doctor.id
                # Query the Patient table for all entries with the user_id
                # patients = Patient.query.filter_by(user_id=user_id).all()
                return redirect(url_for('team'))
            else:
                flash('Invalid userid or password', 'error')
        return render_template('doclogin.html', form=form)
  ################################CONTACT################################
    # Your Twilio account SID and auth token]#
    load_dotenv()
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    client = Client(account_sid, auth_token)
    @app.route('/contact', methods=['GET', 'POST'])
    def contact():
        print("hai")
        if request.method == 'POST':
            name = request.form['name']
            mobile = request.form['number']
            subject = request.form['subject']
            message_body = request.form['message']
            # Send SMS message
            message = client.messages.create(
                body=f'Concern Submitted From :\nName: {name}\nSubject: {subject}\nMessage: {message_body}',
                from_='+19078917970',
                to=f'+91 {mobile}'
            )

            print(f"Message sent successfully! SID: {message.sid}")

            return redirect(url_for('index'))
        else:
            return render_template('contact.html')
    ################################DOCTOR VIEWS################################
    @app.route('/team')
    def team():
        patients = session.get('patients')  # Retrieve the patients list from the session
        if patients is None:
            flash('No patients found', 'error')
            patients = []
        return render_template('team.html',patients=patients)
    ###############################DOCTOR VIEWS########################
    @app.route('/process_details', methods=['POST'])
    def process_details():
        preprocessed_text = request.form.get('preprocessedText')
        clinical_detail = request.form.get('clinicalDetail')
        derma_detail = request.form.get('dermaDetail')
        personal_detail = request.form.get('personalDetail')
        image=request.form.get('image')
        preprocessed_text_words = preprocessed_text.split()
        html = "<div style='font-family: Georgia, serif; text-align: center,font-weight: bold, color: black;'>"
        print("derma",derma_detail)    
        derma_detail = derma_detail[1:-1].split(', ')
        clinical_detail = clinical_detail[1:-1].split(', ')
        personal_detail = personal_detail[1:-1].split(', ')
        print("derma",type(derma_detail))
        for word in preprocessed_text_words:
            if word.lower() in clinical_detail:
                print("hai")
                html +=f"<span style='background-color:#C68375'>{word}</span> "
                print(html)
                
            elif word.lower() in derma_detail:
                html +=f"<span style='background-color:#3E9385'>{word}</span> "
            elif word.lower() in personal_detail:
                html +=f"<span style='background-color:red'>{word}</span> "
            else:
                html += f"{word} "
        html+="</div>"
        wordcloud = f"<img src='{image}' alt='Word Cloud'>"
        cluster_text = clustering(preprocessed_text)
        print("cluster_text",cluster_text)
        print(type(cluster_text))
        print("html",html)
        print(type(html))
        print(wordcloud)
        print(type(wordcloud))

        return jsonify({'html': html, 'wordcloud': wordcloud, 'cluster_text': cluster_text})

    ################################UPLOAD TEXT################################
    @app.route('/upload_text', methods=['POST'])
    @login_required  # This decorator ensures the user is logged in
    def upload_text():
        text = request.form['preprocess_text']
        personal_detail = patient_info(text)
        preprocessed_text = preprocess_text(text)
        # Calculate TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        # Convert TF-IDF matrix to a dictionary for word cloud
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_scores)
        img_folder = os.path.join(os.getcwd(), 'img')
        os.makedirs(img_folder, exist_ok=True)
        # Convert the word cloud image to a base64-encoded string

        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        wordcloud_image_path = os.path.join('static/img/', f'wordcloud_image_{current_user.get_id()}_{current_datetime}.png')
        wordcloud.to_file(wordcloud_image_path)
        # Read the image file and encode it to base64
        with open(wordcloud_image_path, "rb") as img_file:
            wordcloud_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        image_data = base64.b64decode(wordcloud_image_base64)
        clinical_detail = clinical_condition_info(preprocessed_text)
        derma_detail = derma_info(preprocessed_text)
        
        clinical_detail=str(clinical_detail)
        derma_detail=str(derma_detail)
        personal_detail=str(personal_detail)
        if current_user.is_authenticated:
            user_id = current_user.get_id()  # Get the current user's ID
            # Use the user_id to create a new patient entry
            if request.method == 'POST':
                patient = Patient(
                    user_id=user_id,
                    preprocessed_text=preprocessed_text,
                    derma_detail=derma_detail,
                    clinical_detail=clinical_detail,
                    personal_detail=personal_detail,
                    image=wordcloud_image_path,
                    status ="pending"
                )
                db.session.add(patient)
                db.session.commit()
        else:
            # Handle the case where the user is not logged in
            flash('You must be logged in to perform this action', 'error')
            return redirect(url_for('login'))
        return render_template('user_success.html', processed_text=preprocessed_text, wordcloud_image=wordcloud_image_base64)

        
    ################################PREPROCESSING&PATIENT TABLE################################
    def preprocess_text(text):
        # Tokenization & Lowercasing
        # abbreviation_df = pd.read_csv('C:/Users/dell/INhouse/uploads/abbreviation.csv',encoding='latin-1')
        abbreviation_df = pd.read_csv('E:/6/IN-HOUSE/sample 2/static/uploads/abbreviation.csv',encoding='latin-1')
        abbreviation_mapping = dict(zip(abbreviation_df['Abbreviation'], abbreviation_df['Stands for']))
        has_abbreviations = any(abbreviation in text for abbreviation in abbreviation_mapping.keys())  
        if has_abbreviations:
            # Replace abbreviations with their full forms
            for abbreviation, full_form in abbreviation_mapping.items():
                text = text.replace(abbreviation, full_form)
        tokens = word_tokenize(text.lower())
        # Remove Stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back into a single string
        preprocessed_text = ' '.join(tokens)
        # Create a dictionary mapping abbreviations to their full forms
        return preprocessed_text
    def patient_info(text):
        preprocessed_text = text
        nlp = medspacy.load()
        Span.set_extension("icd10", default="", force=True)
        Span.set_extension("is_personal_info", default=False, force=True)
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
        race_country_date_time_rules = [
            TargetRule(pattern=[{"LOWER": {"IN": ["hispanic","indian","india","caucasian", "african american", "asian"]}}],
               category="PERSONAL_INFO",
               literal="personal_info",
               attributes={"is_personal_info": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["name", "patient"]}}],
               category="PERSONAL_INFO",
               literal="personal_info",
               attributes={"is_personal_info": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["approximately", "every", "several", "past", "few"]}},
                                {"LOWER": {"IN": ["weeks", "months", "years"]}}],
                   category="PERSONAL_INFO",
               literal="personal_info",
               attributes={"is_personal_info": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["old"]}}],
                     category="PERSONAL_INFO",
               literal="personal_info",
               attributes={"is_personal_info": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["age", "years", "yo", "y/o"]}}],
               category="PERSONAL_INFO",
               literal="personal_info",
               attributes={"is_personal_info": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["gender", "male", "female", "man", "woman","white female","white male","black male","black female"]}}],
               category="PERSONAL_INFO",
               literal="personal_info",
               attributes={"is_personal_info": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["phone", "number", "cell", "contact", "email"]}}],
               category="PERSONAL_INFO",
               literal="personal_info",
               attributes={"is_personal_info": True}),
            # Add more rules for identifying personal information
            TargetRule(pattern=[{"LOWER": {"IN": ["name", "age", "gender"]}}],
                    category="PERSONAL_INFO",
                    literal="personal_info",
                    attributes={"is_personal_info": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["address", "home", "residence", "location"]}}],
               category="PERSONAL_INFO",
               literal="personal_info",
               attributes={"is_personal_info": True}),
            # Medical record number or ID
            TargetRule(pattern=[{"LOWER": {"IN": ["mrn", "id", "identifier", "medical record"]}}],
                        category="PERSONAL_INFO",
                        literal="personal_info",
                        attributes={"is_personal_info": True}),
            # Health insurance information
            TargetRule(pattern=[{"LOWER": {"IN": ["insurance", "policy", "payer", "plan"]}}],
                        category="PERSONAL_INFO",
                        literal="personal_info",
                        attributes={"is_personal_info": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["address", "phone", "email"]}}],
                    category="PERSONAL_INFO",
                    literal="personal_info",
                    attributes={"is_personal_info": True})
        ]
        target_matcher.add(race_country_date_time_rules)
        doc = nlp(preprocessed_text)
        target_words = [rule.pattern[0]["LOWER"]["IN"] for rule in race_country_date_time_rules]
        tt=[]
        for sent in nltk.sent_tokenize(preprocessed_text):
            words = nltk.word_tokenize(sent)
            tags = nltk.pos_tag(words)
            chunks = ne_chunk(tags)
            for chunk in chunks:
                if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                    target_words.append(' '.join(c[0] for c in chunk.leaves()))
                elif isinstance(chunk, nltk.Tree):
                    if all(c[1] != 'NNP' for c in chunk.leaves()):
                        tt.append(' '.join(c[0] for c in chunk.leaves()))
        t=[]
        result_list=[]
        for item in tt:
            if isinstance(item, str): 
                result_list.extend(item.split())
        t.append(result_list)
        place_entity = locationtagger.find_locations(text = preprocessed_text)
        if place_entity.countries:
            t.append(place_entity.countries)
        if place_entity.cities:
            t.append(place_entity.cities)
        if place_entity.regions:
            t.append(place_entity.regions)
        flattened_list = [item for sublist in t for item in sublist]
        unique_list = list(set(flattened_list))
        target_words.append(unique_list)   
        print(target_words)
        highlighted_personal,personal = visualize_ent_fam(doc, target_words)
        return personal


    
    def clinical_condition_info(text):
        preprocessed_text = text
        nlp = medspacy.load()

        Span.set_extension("icd10", default="", force=True)
        Span.set_extension("is_medical_term", default=False, force=True)
        
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
        medical_condition_rules = [
            TargetRule(pattern=[{"LOWER": {"IN": ["chronic", "acute", "severe", "moderate", "mild","severe","rash","flaring", "Flaring of acne","Folliculitis lesions","fever","high fever","sinus","hives","eczema","photo facials","acne","folliculitis", "lesions"]}}],
                    category="MEDICAL_CONDITION_DESCRIBER",
                    literal="medical_condition_describer",
                    attributes={"is_medical_term": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["headache","cancer", "nausea", "vomiting", "fatigue", "dizziness", "shortness of breath", "chest pain", "back pain", "abdominal pain", "joint pain", "muscle pain", "swelling", "rash"]}}],
                    category="SYMPTOM",
                    literal="symptom",
                    attributes={"is_medical_term": True}),
            TargetRule(pattern=[{"LOWER": {"IN": ["diabetes","diabetic", "hypertension", "asthma", "cancer", "arthritis", "depression", "anxiety", "stroke", "heart disease", "chronic obstructive pulmonary disease", "kidney disease", "liver disease", "thyroid disorder"]}}],
                    category="DISEASE",
                    literal="disease",
                    attributes={"is_medical_term": True})
        ]
    
        target_matcher.add(medical_condition_rules)
        doc = nlp(preprocessed_text)
    
        target_words = [rule.pattern[0]["LOWER"]["IN"] for rule in medical_condition_rules]
        highlighted_personal,clinical = visualize_ent(doc, target_words)
        print(clinical)
        return clinical


   
    def derma_info(text):
        preprocessed_text = text
        
        nlp = medspacy.load()

        Span.set_extension("icd10", default="", force=True)
        Span.set_extension("is_medical_term", default=False, force=True)
        
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
        medical_condition_rules = [
        TargetRule(pattern=[{"LOWER": {"IN": [
                'inflammation', 'papulovesicular lesions', 'plaques', 'non-tender', 'blistering rash', 'erythematous', 'discoloration of the nails', 'bald patches', 'asymptomatic','pigmentation','pigment','pigmented', 'hyperpigmented papules', 'onycholysis', 'hypopigmented patches', 'thickened nails', 'erythema', 'macule', 'nail plate', 'dark patches', 'pruritic', 'lichenification', 'skin excoriations', 'rash', 'erythematous plaques', 'excoriated papules', 'hair loss', 'hyperpigmented macules', 'acne', 'hypopigmented macules', 'eczema', 'hyperpigmented plaques', 'hypopigmentation', 'excoriations', 'nail bed', 'hyperkeratotic papule', 'vesicles', 'crusted plaques', 'erythematous scaly', 'erythematous scaly plaques', 'well-demarcated', 'hyperpigmented patches', 'hyperkeratosis', 'nail pitting', 'induration', 'itching', 'thinning of the hair', 'nodules', 'scaling', 'scales', 'crusts', 'hyperpigmentation', 'nail changes', 'skin discoloration', 'lichenified plaques', 'hyperpigmented', 'papules']}}],
                category="MEDICAL_CONDITION_DESCRIBER",
                literal="medical_condition_describer",
                attributes={"is_medical_term": True})
    ]
    
        target_matcher.add(medical_condition_rules)
        doc = nlp(preprocessed_text)
    
        target_words = medical_condition_rules[0].pattern[0]["LOWER"]["IN"]

        highlighted_personal,derma = visualize_ent_derma(doc, target_words)
        print(derma)
        return derma
    def visualize_ent_derma(doc, target_words):
        html = "<div>"
        derma=[]
        for word in doc:
            if word.text.lower() in target_words:
                html += f"<span style='background-color: blue'>{word}</span> "
                derma.append(word)
            else:
                html += f"{word} "
        html += "</div>"
        return html,derma

    def visualize_ent(doc, target_words):
        html = "<div>"
        clinical =[]
        flattened_target_words = [word for sublist in target_words for word in sublist]
        for word in doc:
            if word.text.lower() in flattened_target_words:
                clinical.append(word)
                html += f"<span style='background-color: yellow'>{word}</span> "
            else:
                html += f"{word} "
        html += "</div>"
        return html,clinical

    def visualize_ent_fam(doc, target_words):
        html = "<div>"
        text = doc.text  # Convert SpaCy Doc object to string
        ages = re.findall(r'\b(\d{1,3})\s*year(?:s?)\s*old\b', text, flags=re.IGNORECASE)
        for age in ages:
            target_words.append(age.lower())  # Append lowercased age to target_words
        lowercase_list = [[element.lower() for element in sublist] for sublist in target_words]   
        personal=[]
        for word in doc:
            present = any(word.text.lower() in sublist for sublist in lowercase_list)
            if present:
                personal.append(word)
                html += f"<span style='background-color: red'>{word}</span> "
            else:
                html += f"{word} "
        html += "</div>"
        print(personal)
        return html,personal
    ################################CLUSTERING################################
    # @app.route('/clustering', methods=['POST'])
    def clustering(preprocessed_text):
        # preprocessed_text = request.form['personal_text']
        df = pd.read_csv('C:/Users/dell/INhouse/uploads/clustered_data.csv')
        # Load the KMeans model
        #old kmeans = load('E:/6/IN-HOUSE/sample 2/static/uploads/model.pkl')
        kmeans = load('C:/Users/dell/INhouse/uploads/model.pkl')
        #old tfidf_vectorizer = load('E:/6/IN-HOUSE/sample 2/static/uploads/tf_idf.pkl')
        tfidf_vectorizer = load('C:/Users/dell/INhouse/uploads/tfidf_vectorizer.pkl')
        print(preprocessed_text)
        print(type(preprocessed_text))
        # Transform the input text using the loaded TfidfVectorizer
        tfidf_matrix = tfidf_vectorizer.transform([preprocessed_text])


        # Predict the cluster
        cluster = kmeans.predict(tfidf_matrix)[0]
        if cluster ==0:
            cluster_analyse ="Facial Problem, acne symptoms"
        elif cluster ==1:
            cluster_analyse ="Dry Skin Symptoms"
        else:
            cluster_analyse ="Negative Phrase. Patient is confused"
        
        cluster_phrases = df[df['cluster'] == cluster]['text'].tolist()
        cluster = cluster 
        cluster_phrases = cluster_analyse
        return cluster_phrases

except Exception as e:
    print(f"Error: {e}")
if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
    app.run(debug=True)
