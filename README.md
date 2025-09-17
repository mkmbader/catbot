# CatBot
Meet CatBot - the c(h)atbot who know's all about Cats, based on [simple wikipedia](https://simple.wikipedia.org/wiki/Cat). Catbot's gives a quick start guide to RAG evaluation and was developped to fuel a presentation at PyData Amsterdam 2025. 

Find the slides to the presentation 👉 here.
Connect to the developer 👉 [here](https://www.linkedin.com/in/mkmbader/).


### Overview
This `main.py` notebook contains the following workflow:

- **Database Creation**: Initializes a database using ChromaDB to efficiently store and retrieve data.
- **Retrieval-Augmented Generation (RAG) QA System**: Builds a question-answering system that queries the database using RAG techniques.
- **Synthetic Dataset Generation**: Utilizes Ragas with a custom prompt to generate a synthetic dataset tailored for QA tasks.
- **Automated Evaluation**: Assesses the QA system’s responses using Ragas-provided evaluation metrics such as Faithfulness, Answer Correctness, Answer Relevancy

### Setup
Environment
* create virtual environment: `python -m venv .catbot`
* activate environment: `source .catbot/bin/activate`
* install dependencies: `pip install -r requirements.txt`

Create database
* add your api keys and other variables to .env