# MonoT5 GPU Reranking – Anleitung für Windows (5090)

## Voraussetzungen auf Windows

1. **NVIDIA Treiber** – sollte für die 5090 schon installiert sein
2. **Docker Desktop** – installieren von https://www.docker.com/products/docker-desktop/
   - Bei der Installation: **WSL2-Backend** aktiviert lassen (ist Default)
   - Nach Installation: Docker Desktop starten

### Docker GPU-Support prüfen
```powershell
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
Wenn das die 5090 anzeigt, ist alles ready.

---

## Schritt 1: Repository klonen

```powershell
git clone https://github.com/Niklas2021/wows-code.git
cd wows-code\ecir26\je-alkonume-huehner-03-custom\template-new-approach
```

## Schritt 2: Docker Image bauen

```powershell
docker build -f Dockerfile.gpu -t monot5-gpu .
```

Das dauert beim ersten Mal ~5-10 Minuten (lädt PyTorch + CUDA + MonoT5-Modell herunter).
Danach ist alles im Image gecached.

## Schritt 3: Reranking starten

### MonoT5 mit Top-1000 Reranking (empfohlen):
```powershell
docker run --gpus all -v "%cd%\output:/app/output" monot5-gpu --dataset radboud-validation-20251114-training --rerank-depth 1000 --monot5-batch-size 64
```

### MonoT5 mit ALLEN Dokumenten (experimentell):
```powershell
docker run --gpus all -v "%cd%\output:/app/output" monot5-gpu --dataset radboud-validation-20251114-training --rerank-depth 65000 --monot5-batch-size 128
```

### Geschätzte Laufzeiten mit 5090:
| Depth | Docs total | Geschätzte Zeit |
|-------|-----------|----------------|
| 100   | 2.800     | ~5 Sekunden    |
| 1000  | 28.000    | ~45 Sekunden   |
| 65000 | 1.780.000 | ~30-45 Minuten |

## Schritt 4: Ergebnis kopieren

Das Ergebnis liegt danach unter:
```
output\runs\radboud-validation-20251114-training\pyterrier-BM25-monot5d1000-gpu-on-default_text\
├── run.txt.gz          ← Der eigentliche Run
└── ir-metadata.yml     ← Metadaten
```

Diesen Ordner auf den Mac kopieren (USB-Stick, Cloud, etc.), dann dort:

### Auf dem Mac evaluieren und hochladen:
```bash
cd ecir26/je-alkonume-huehner-00-bm25/template-new-approach

# Evaluieren (evaluate.py muss den neuen Pfad kennen)
.venv/bin/python evaluate.py

# TIRA Upload (erst ir-metadata.yml ergänzen!)
# Am Ende von ir-metadata.yml anhängen:
#   actor:
#     name: je-alkonume-huehner
#     team: fsu-alkonueme-huener
#   data:
#     test collection:
#       name: radboud-validation-20251114-training

.venv/bin/tira-cli upload \
    --directory <pfad-zum-run-ordner> \
    --system pyterrier-BM25-monot5d1000-gpu
```

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| `docker: Error response from daemon: could not select device driver` | NVIDIA Container Toolkit fehlt. Wird normalerweise mit Docker Desktop + aktuellem NVIDIA-Treiber automatisch installiert. |
| `nvidia-smi` zeigt keine GPU | NVIDIA Treiber aktualisieren: https://www.nvidia.com/Download/index.aspx |
| Out of Memory | `--monot5-batch-size` reduzieren (z.B. 32 oder 16) |
| Langsam trotz GPU | Prüfe ob `Device: cuda` in der Ausgabe steht. Wenn `cpu` → Docker hat keinen GPU-Zugriff. |
