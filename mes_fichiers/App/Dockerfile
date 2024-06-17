FROM python:3.12

# Définir le répertoire de travail dans le conteneur Docker
WORKDIR /application_streamlit

# Copier le fichier requirements.txt dans le conteneur Docker
COPY requirements.txt .

# Installer les dépendances Python à partir du fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de App/ dans le répertoire de travail du conteneur
COPY . /application_streamlit

# Exposer le port Streamlit
EXPOSE 8501

# Commande pour exécuter l'application Streamlit lorsque le conteneur démarre
CMD ["streamlit", "run", "--server.enableCORS", "false", "dashboard.py"]