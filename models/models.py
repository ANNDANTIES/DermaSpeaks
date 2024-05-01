from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    patients = db.relationship('Patient', backref='user', lazy=True)

class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    # Define any relationships involving doctors here

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    preprocessed_text = db.Column(db.Text, nullable=False)
    derma_detail = db.Column(db.Text, nullable=False)
    clinical_detail = db.Column(db.Text, nullable=False)
    image = db.Column(db.Text, nullable=False)
    status = db.Column(db.Text, nullable=False)
