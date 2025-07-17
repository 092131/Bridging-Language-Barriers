# 🌐 Bridging Language Barriers with GEN AI

A multilingual AI translation system designed to bridge communication gaps across Indian languages — with a focus on **Telugu, Tamil, and Marathi**. This project uses **mBART** (Multilingual BART) and **Flask** to create a scalable and user-friendly web application for real-time translation.

## 🚀 Project Objective

- Enable seamless and high-quality translations between English and multiple Indian languages.
- Support input via **text, URL, and PDF** with **automatic language detection**.
- Provide a web interface using **Flask** for easy access.
- Evaluate translation quality using advanced NLP metrics.

## 🔧 Tech Stack

| Category            | Tools / Technologies                            |
|---------------------|--------------------------------------------------|
| Language            | Python 3.7+                                      |
| Framework           | Flask                                            |
| Translation Model   | mBART (via Hugging Face Transformers)           |
| Model Optimization  | LoRA (Low-Rank Adaptation), PEFT, SFTTrainer    |
| Language Detection  | `langdetect`                                     |
| PDF Processing      | `PyPDF2`                                         |
| Web Scraping        | `BeautifulSoup` + `requests`                     |

## 🧠 Features

- ✅ Text, PDF, and URL translation
- ✅ Automatic language detection
- ✅ Context-aware translation using mBART
- ✅ Fine-tuning via LoRA for Indian languages
- ✅ Evaluation Metrics: BLEU, METEOR, ROUGE, Cosine Similarity
- ✅ Translation history tracker
- ✅ Scalable and modular architecture

## 📂 Folder Structure (sample)

```
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── styles.css
├── models/
│   └── fine-tuned-mbart-models
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/genai-translation.git
cd genai-translation

# 2. Create and activate virtual environment
python -m venv env
source env/bin/activate    # or env\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Flask app
set FLASK_APP=app.py       # or export FLASK_APP=app.py
flask run
```

## 📊 Evaluation Metrics

| Language | METEOR | ROUGE Precision | ROUGE Recall | F1 Score | Cosine Similarity |
|----------|--------|------------------|---------------|----------|-------------------|
| Telugu   | 0.69   | 0.75             | 0.63          | 0.68     | 0.73              |
| Tamil    | 0.58   | 0.62             | 0.61          | 0.61     | 0.62              |
| Marathi  | 0.65   | 0.69             | 0.68          | 0.68     | 0.72              |

## 🛡️ Limitations

- High GPU/compute requirements
- Limited to 3 Indian languages initially
- Domain-specific language challenges in translation

## 🔭 Future Scope

- Add support for more Indian languages (e.g., Bengali, Odia, Assamese)
- Integrate **speech-to-text** and **text-to-speech**
- Real-time translation previews
- Cloud deployment for scalability



