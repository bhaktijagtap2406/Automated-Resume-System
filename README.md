# Automated Resume Screening System

## Overview

The Automated Resume Screening System is a web-based application that automates the process of analyzing, filtering, and ranking resumes based on job requirements. It reduces manual effort and improves the efficiency and accuracy of candidate shortlisting.

---

## Features

* Resume parsing (PDF/DOCX)
* Keyword-based matching with job descriptions
* Score-based ranking of candidates
* Bulk resume processing
* Job description input
* Candidate data storage in database
* Basic dashboard for results display

---

## Tech Stack

**Frontend**

* HTML
* CSS
* JavaScript

**Backend**

* Python (Flask)

**Database**

* MySQL

**Libraries / Tools**

* spaCy / NLTK
* Pandas
* PyPDF2 / python-docx

---

## Project Structure

```
├── api/                # API routes and endpoints
├── data/               # Training and dataset files
├── frontend/           # Frontend files (HTML, CSS, JS)
├── models/             # Trained ML models
├── notebooks/          # Jupyter notebooks for experimentation
├── src/                # Core backend logic and SQL integration
├── .gitignore
├── requirements.txt
├── setup_mysql.sql
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/automated-resume-system.git
cd Automated-Resume-System-main
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup database

* Open MySQL
* Run `setup_mysql.sql` to create required tables

### 5. Run the application

```bash
python app.py
```

### 6. Open in browser

```
http://127.0.0.1:5000/
```

---

## How It Works

1. Upload resumes
2. Provide job description
3. System extracts relevant information
4. Matches resumes with job criteria
5. Assigns scores
6. Displays ranked candidates

---

## Use Cases

* HR recruitment automation
* Campus placement systems
* Recruitment agencies
* High-volume hiring workflows

---

## Future Improvements

* Advanced ML-based ranking models
* Integration with external job platforms
* Real-time resume feedback
* Cloud deployment support
