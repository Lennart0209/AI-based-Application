# PDF-Chat Anwendung (Academic Cloud)

## Voraussetzungen
- Python 3.8 oder neuer  
- API-Schlüssel von [https://chat-ai.academiccloud.de](https://chat-ai.academiccloud.de)  
- Webbrowser  

## Installation

1. Projekt entpacken oder in ein Verzeichnis kopieren

2. Öffne ein Terminal oder eine Eingabeaufforderung **im Projektordner** (dort, wo sich `requirements.txt` befindet)

3. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   ```

4. API-Schlüssel setzen:

   Die Anwendung benötigt einen kompatiblen API-Key von Academic Cloud. Setze ihn entweder als Umgebungsvariable:

   ```bash
   export OPENAI_API_KEY=sk-...     # Linux/macOS
   set OPENAI_API_KEY=sk-...        # Windows
   ```

   Alternativ (nicht empfohlen): Du kannst den Schlüssel direkt in `app.py` eintragen:

   ```python
   openai.api_key = "sk-..."
   ```

## Starten

```bash
python app.py
```

Dann im Browser öffnen:  
[http://localhost:5000](http://localhost:5000)
