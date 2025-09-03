# HH-Bildungsplan BM25-Suche

FastAPI-Server für die Suche in den digitalisierten Hamburger Bildungsplänen (Stand 2024) mit einem BM25-Index.

## Installation

Klonen:
```bash
git clone https://github.com/christophschuhmann/HH-Bildungsplan-Suche-BM25.git
cd HH-Bildungsplan-Suche-BM25
````

Benötigte Pakete installieren:

```bash
pip install fastapi uvicorn bm25s pydantic
```

## Nutzung

Server starten:

```bash
python curriculum-serverbm25.py
```

Der Server läuft dann auf `http://localhost:8020`.

## API

* **POST /query**
  Body (JSON):

  ```json
  { "query": "Informatik Algorithmus", "top_n": 5 }
  ```

  Antwort:

  ```json
  {
    "results": [
      { "text": "...", "score": 12.34 },
      { "text": "...", "score": 11.78 }
    ]
  }
  ```


👉 Soll ich dir auch noch gleich die **kurze GitHub-Projektbeschreibung (einen Satz)** so formulieren, dass sie genau zu diesem README passt?
```
