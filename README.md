# DrawRecognizer
[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue?logo=github)](https://november7.github.io/DrawRecognizer/)
[![Language-English-success](https://img.shields.io/badge/Language-English-success)](#english)
[![J%C4%99zyk-Polski-red](https://img.shields.io/badge/J%C4%99zyk-Polski-red)](#polski)

## English

DrawRecognizer is a browser demo for handwritten symbol recognition.
The current app runs with ONNX Runtime Web and loads `.onnx` models directly from the `models/` directory.

Legacy TensorFlow.js implementation (old app + old TFJS models) is kept in `tensorflow_js/`.

## How It Works

After selecting a model:

1. the app loads a `.onnx` file,
2. it reads class labels from metadata (if available),
3. it clears the canvas and resets prediction bars,
4. the user draws a symbol,
5. after drawing, the app:
   - reads pixels from canvas,
   - resizes input to model shape,
   - normalizes values to 0..1,
   - runs ONNX inference in the browser,
   - displays predicted class and class probabilities.

## Models And Metadata

Current models are listed in `models/index.json` and point to ONNX files, for example:

- `cyfry-mnist-test1.onnx`
- `cyfry-mnist-test2.onnx`

Optional metadata supported by the app:

- `metadata.json` (same directory as model),
- `<model-name>.metadata.json`.

If metadata is missing, labels are generated as numeric indexes (`0`, `1`, `2`, ...).

## Repository Structure

```text
.
├── digits.ipynb                      # notebook for training/experiments
├── index.html                        # ONNX web demo entry page
├── LICENSE
├── README.md
├── models/                           # active ONNX models
│   ├── index.json
│   ├── cyfry-mnist-test1.onnx
│   └── cyfry-mnist-test2.onnx
├── src/
│   ├── app.js                        # ONNX Runtime Web logic
│   └── style.css
└── tensorflow_js/                    # archived legacy TensorFlow.js app
    ├── index.html
    ├── models/
    │   ├── index.json
    │   ├── Cyfry-Mnist/
    │   ├── Cyfry-TM/
    │   └── XO-TM/
    └── src/
        ├── app.js
        └── style.css
```

## Run Local Preview

Use any local HTTP server (required for model fetch):

```bash
npx serve . --listen 5500
```

Open:

```text
http://localhost:5500/index.html
```

## Training And Export

`digits.ipynb` contains training experiments for digit recognition.
For the current app, export models to ONNX and place them in `models/`, then update `models/index.json`.

## Migration Notes (TFJS -> ONNX)

To add a new ONNX model to the current app:

1. copy `<name>.onnx` to `models/`,
2. add `<name>.onnx` to `models/index.json`,
3. optionally add labels file as:
    - `models/metadata.json` or
    - `models/<name>.metadata.json`.

The app version in the repository root is the active ONNX version.
Legacy TensorFlow.js code and models are archived in `tensorflow_js/`.

---

## Polski

DrawRecognizer to demonstracyjna aplikacja webowa do rozpoznawania ręcznie rysowanych symboli.
Aktualna wersja działa na ONNX Runtime Web i ładuje modele `.onnx` bezpośrednio z katalogu `models/`.

Starsza implementacja TensorFlow.js (stara aplikacja + stare modele TFJS) została zachowana w `tensorflow_js/`.

## Opis Działania

Po wybraniu modelu:

1. aplikacja ładuje plik `.onnx`,
2. odczytuje etykiety klas z metadanych (jeśli istnieją),
3. czyści canvas i resetuje paski predykcji,
4. użytkownik rysuje symbol,
5. po zakończeniu rysowania aplikacja:
   - pobiera piksele z canvasu,
   - skaluje wejście do kształtu modelu,
   - normalizuje wartości do zakresu 0..1,
   - uruchamia inferencję ONNX w przeglądarce,
   - wyświetla rozpoznaną klasę i prawdopodobieństwa klas.

## Modele I Metadane

Aktualne modele są trzymane w `models/index.json` i wskazują pliki ONNX, np.:

- `cyfry-mnist-test1.onnx`
- `cyfry-mnist-test2.onnx`

Opcjonalne metadane obsługiwane przez aplikację:

- `metadata.json` (w tym samym katalogu co model),
- `<nazwa-modelu>.metadata.json`.

Jeżeli metadanych brak, aplikacja generuje etykiety numeryczne (`0`, `1`, `2`, ...).

## Struktura Repozytorium

```text
.
├── digits.ipynb                      # notatnik treningowy / eksperymenty
├── index.html                        # wejście do demo ONNX
├── LICENSE
├── README.md
├── models/                           # aktywne modele ONNX
│   ├── index.json
│   ├── cyfry-mnist-test1.onnx
│   └── cyfry-mnist-test2.onnx
├── src/
│   ├── app.js                        # logika ONNX Runtime Web
│   └── style.css
└── tensorflow_js/                    # archiwum starej wersji TensorFlow.js
    ├── index.html
    ├── models/
    │   ├── index.json
    │   ├── Cyfry-Mnist/
    │   ├── Cyfry-TM/
    │   └── XO-TM/
    └── src/
        ├── app.js
        └── style.css
```

## Uruchomienie Lokalnie

Użyj dowolnego serwera HTTP (wymagane dla fetch modeli):

```bash
npx serve . --listen 5500
```

Otwórz:

```text
http://localhost:5500/index.html
```

## Trening I Eksport

`digits.ipynb` zawiera eksperymenty treningowe dla rozpoznawania cyfr.
Dla obecnej aplikacji eksportuj modele do ONNX, umieść je w `models/` i zaktualizuj `models/index.json`.

## Notatki Migracyjne (TFJS -> ONNX)

Aby dodać nowy model ONNX do obecnej aplikacji:

1. skopiuj `<nazwa>.onnx` do `models/`,
2. dopisz `<nazwa>.onnx` do `models/index.json`,
3. opcjonalnie dodaj plik etykiet jako:
    - `models/metadata.json` albo
    - `models/<nazwa>.metadata.json`.

Wersja aplikacji w głównym katalogu repo to aktywna wersja ONNX.
Stary kod i modele TensorFlow.js są zarchiwizowane w `tensorflow_js/`.
