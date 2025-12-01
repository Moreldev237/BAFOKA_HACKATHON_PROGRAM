
backend_flask/
│
├─ app.py                # fichier principal Flask
├─ transcribe_clean.py   # notre module STT clean
├─ custom_scaler.py      # ton module déjà fourni
├─ requirements.txt
└─ uploads/              # dossier pour stocker temporairement les fichiers audio uploadés




2️⃣ Créer un environnement Python
python -m venv venv  ou py -3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows ou  source ./venv/Scripts/activate

3️⃣ Installer les dépendances
pip install -r requirements.txt


curl -X POST http://localhost:5000/transcribe \
  -F "file=@./B.ogg"
