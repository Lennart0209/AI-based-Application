Voraussetzungen
---------------
- Python 3.8 oder neuer
- API-Schlüssel von https://chat-ai.academiccloud.de
- Webbrowser

Installation
------------
1. Projekt entpacken oder in ein Verzeichnis kopieren

2. Öffne ein Terminal oder eine Eingabeaufforderung **im Projektordner** (dort, wo sich `requirements.txt` befindet)

3. Abhängigkeiten installieren:

   pip install -r requirements.txt

4. API-Schlüssel setzen:

   Die Anwendung benötigt einen kompatiblen API-Key von Academic Cloud. Setze ihn entweder als Umgebungsvariable:

   export OPENAI_API_KEY=sk-...     # Linux/macOS
   set OPENAI_API_KEY=sk-...        # Windows

   Alternativ (nicht empfohlen): Du kannst den Schlüssel direkt in app.py eintragen:

   openai.api_key = "sk-..."

Starten
-------
python app.py

Danach im Browser öffnen:
http://localhost:5000
