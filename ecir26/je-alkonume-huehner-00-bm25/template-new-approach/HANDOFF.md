# Handoff-Dokument: IR-Lab WiSe 2025 – Team je-alkonume-huehner

## Übersicht

Dieses Dokument beschreibt alle Arbeiten, die für das IR-Lab-Projekt durchgeführt wurden. Ziel war es, mehrere Retrieval-Ansätze aufzusetzen, auszuführen, zu evaluieren und bei TIRA hochzuladen.

**Repository:** `https://github.com/Niklas2021/wows-code.git`  
**Branch:** `je-alkonüme-hühner-mvp`  
**TIRA Team:** `fsu-alkonueme-huener`  
**TIRA Task:** `ir-lab-wise-2025`  
**Dataset:** `radboud-validation-20251114-training` (63.621 Dokumente, 28 Queries, 1.311 Qrels)  
**Maschine:** macOS, Apple M4 Max, 128 GB RAM, Python 3.14 (bzw. 3.13 für 02-rerank)

---

## 1. Durchgeführte Runs (alle auf TIRA hochgeladen)

### 1.1 BM25 Baseline (`00-bm25`)
- **Pfad:** `ecir26/je-alkonume-huehner-00-bm25/template-new-approach/`
- **Script:** `retrieve.py`
- **Methode:** Standard-BM25 über PyTerrier
- **TIRA System-Name:** `pyterrier-BM25`
- **Run-Datei:** `output/runs/radboud-validation-20251114-training/pyterrier-BM25-on-default_text/run.txt.gz`

### 1.2 BM25 + RM3 Query Expansion (`01-rm3`)
- **Pfad:** `ecir26/je-alkonume-huehner-01-rm3/template-new-approach/`
- **Script:** `retrieve.py`
- **Methode:** BM25 mit anschließender RM3 Query Expansion
- **TIRA System-Name:** `pyterrier-BM25-rm3`
- **Run-Datei:** `output/runs/radboud-validation-20251114-training/pyterrier-BM25-rm3-on-default_text/run.txt.gz`

### 1.3 BM25 + MonoT5 Reranking (`02-rerank`)
- **Pfad:** `ecir26/je-alkonume-huehner-02-rerank/template-new-approach/`
- **Script:** `retrieve.py`
- **Methode:** BM25 Top-100 → MonoT5 (`castorini/monot5-base-msmarco`) Reranking
- **Python:** 3.13 (wegen SIGSEGV-Crash von `tokenizers` auf Python 3.14)
- **Wichtig:** `transformers<4.40` nötig (neuere Versionen haben `batch_encode_plus` entfernt)
- **TIRA System-Name:** `pyterrier-BM25-monot5d100`
- **Run-Datei:** `output/runs/radboud-validation-20251114-training/pyterrier-BM25-monot5d100-on-default_text/run.txt.gz`

### 1.4 BM25 + LLM Reranking – Llama 3.1:70b (`00-bm25`, erste Version)
- **Pfad:** `ecir26/je-alkonume-huehner-00-bm25/template-new-approach/`
- **Script:** `rerank_llm.py` (erste Version: Scale 0-10, Rerank-Depth 50, kein Tiebreaker)
- **Methode:** BM25 Top-50 → Ollama Llama 3.1:70b bewertet Relevanz per Prompt
- **Laufzeit:** ~2h 46min
- **TIRA System-Name:** `pyterrier-BM25-llm-llama3-1-70b`
- **Run-Datei:** `output/runs/radboud-validation-20251114-training/pyterrier-BM25-llm-llama3.1-70b-d50-on-default_text/run.txt.gz`

### 1.5 BM25 + LLM Reranking – Gemma2:9b (`00-bm25`, verbesserte Version)
- **Pfad:** `ecir26/je-alkonume-huehner-00-bm25/template-new-approach/`
- **Script:** `rerank_llm.py` (aktuelle Version mit 3 Verbesserungen)
- **Methode:** BM25 Top-200 → Ollama Gemma2:9b bewertet Relevanz per Prompt
- **Laufzeit:** ~1h 43min
- **TIRA System-Name:** `pyterrier-BM25-llm-gemma2-9b-d200`
- **Run-Datei:** `output/runs/radboud-validation-20251114-training/pyterrier-BM25-llm-gemma2-9b-d200-on-default_text/run.txt.gz`

---

## 2. Evaluierungsergebnisse

Alle Runs evaluiert mit `evaluate.py` gegen die offiziellen Qrels:

| Approach | MAP | nDCG | nDCG@10 | P@10 | RR@10 |
|---|---|---|---|---|---|
| **BM25 (Baseline)** | **0.3814** | **0.6147** | **0.4516** | **0.4321** | 0.7765 |
| BM25 + RM3 | 0.3087 | 0.5674 | 0.3887 | 0.3786 | 0.6668 |
| BM25 + LLM Llama3.1:70b (d50) | 0.3369 | 0.5150 | 0.4474 | 0.4250 | **0.7808** |
| BM25 + LLM Gemma2:9b (d200) | 0.3615 | 0.5709 | 0.4404 | 0.3964 | 0.7712 |

**Fazit:** BM25-Baseline ist bei MAP immer noch am besten. Gemma2:9b ist durch die Verbesserungen (größere Rerank-Tiefe, feineres Scoring, Tiebreaker) deutlich besser als der erste Llama-70b-Versuch, kommt aber nicht ganz an BM25 heran.

---

## 3. Erstellte/Modifizierte Dateien

### 3.1 `rerank_llm.py` (NEU ERSTELLT)
Komplett neues Script für zweistufiges Retrieval: BM25 → LLM Reranking via Ollama.

**Kernkomponenten:**
- `score_with_ollama()`: Sendet Query+Document an Ollama API, parst JSON-Score aus der Antwort
- `rerank_with_llm()`: Iteriert per Query über alle Kandidaten-Dokumente, sammelt LLM-Scores
- `get_index()`: Baut PyTerrier-Index mit Text-Meta (`meta={'docno': 100, 'text': 8192}`)

**Drei Verbesserungen gegenüber erster Version:**
1. **Scoring-Skala 0-100** statt 0-10 (weniger Ties)
2. **Rerank-Depth 200** statt 50 (mehr Dokumente werden vom LLM bewertet)
3. **BM25-Tiebreaker:** BM25-Score wird pro Query auf 0-0.99 normalisiert und zum LLM-Score addiert → bei gleichen LLM-Scores gewinnt das von BM25 höher gerankte Dokument

**LLM-Prompt:** Fordert das Modell auf, die volle Skala zu nutzen und runde Zahlen zu vermeiden.

**Aufruf:**
```bash
cd ecir26/je-alkonume-huehner-00-bm25/template-new-approach
.venv/bin/python rerank_llm.py \
    --dataset radboud-validation-20251114-training \
    --output output \
    --ollama-model gemma2:9b \
    --text-field-to-retrieve default_text
```

### 3.2 `evaluate.py` (NEU ERSTELLT)
Evaluiert alle Runs gegen Qrels mit MAP, nDCG, nDCG@10, P@10, RR@10.

**Aufruf:**
```bash
cd ecir26/je-alkonume-huehner-00-bm25/template-new-approach
.venv/bin/python evaluate.py
```

### 3.3 `requirements.txt` (MODIFIZIERT in 00-bm25)
Hinzugefügt: `tqdm`, `requests` (für Ollama-API und Fortschrittsbalken).

### 3.4 `ir-metadata.yml` (MANUELL ERGÄNZT bei jedem Run)
TIRA erfordert `actor.name`, `actor.team` und `data.test collection.name`. Diese Felder werden nicht automatisch vom `tirex-tracker` generiert und müssen manuell ans Ende der YAML-Datei angehängt werden:
```yaml
actor:
  name: je-alkonume-huehner
  team: fsu-alkonueme-huener
data:
  test collection:
    name: radboud-validation-20251114-training
```

### 3.5 Ranking-Fix in `02-rerank/retrieve.py`
MonoT5 überschreibt Scores, aber nicht die Rank-Spalte. Fix: Nach Reranking wird nach `qid` + `score` sortiert und `rank` per `cumcount()` neu vergeben.

---

## 4. Umgebungs-Setup

### Virtual Environments
Jeder Approach hat ein eigenes `.venv` im jeweiligen `template-new-approach/`-Ordner:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Bekannte Probleme & Workarounds

| Problem | Workaround |
|---|---|
| `tirex-tracker` braucht `gcc@14` | `brew install gcc@14` |
| `tokenizers` crasht auf Python 3.14 (SIGSEGV) | Python 3.13 venv für `02-rerank` verwenden |
| `transformers>=4.40` entfernt `batch_encode_plus` | `transformers<4.40` in requirements pinnen |
| `.tirex-tracker`-Ordner blockiert erneuten Run | Run-Ordner löschen vor erneutem Ausführen |
| TIRA lehnt Upload ab bei `run.txt` + `run.txt.gz` | Nur eine der beiden Dateien behalten |

### Ollama
- Installiert via `brew install ollama`, läuft als Brew Service
- Heruntergeladene Modelle: `llama3.1:70b` (42 GB), `gemma2:9b` (5.4 GB)
- API: `http://localhost:11434`
- Status prüfen: `brew services list | grep ollama`

### TIRA Login
```bash
.venv/bin/tira-cli login --token <TOKEN>
```

### TIRA Upload
```bash
.venv/bin/tira-cli upload \
    --directory output/runs/radboud-validation-20251114-training/<RUN-TAG> \
    --system <SYSTEM-NAME>
```

---

## 5. Nächste mögliche Schritte

- **Prompt Engineering:** Anderen Prompt testen (z.B. Chain-of-Thought, oder Relevanz-Kriterien spezifizieren)
- **Größere Modelle via GPU:** Falls eine GPU (z.B. RTX 5090) verfügbar ist, Llama 3.1:70b mit Depth 200 testen
- **Hybrid-Scoring:** LLM-Score mit BM25-Score gewichteter kombinieren (nicht nur als Tiebreaker)
- **Andere Modelle:** Mistral, Phi3, Qwen etc. via Ollama testen
- **03-custom:** `je-alkonume-huehner-03-custom/template-new-approach/retrieve.py` hat BM25-Tuning-Parameter (k1, b) – könnte noch ausgeführt werden
