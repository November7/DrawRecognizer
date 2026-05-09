# DrawRecognizer
[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue?logo=github)](https://november7.github.io/DrawRecognizer/)
[![Language-English-success](https://img.shields.io/badge/Language-English-success)](#english)
[![J%C4%99zyk-Polski-red](https://img.shields.io/badge/J%C4%99zyk-Polski-red)](#polski)

## English

DrawRecognizer is a simple demo project that combines training image recognition models in TensorFlow with running them directly in the browser via TensorFlow.js. The repository includes a Jupyter notebook for neural network experiments and an HTML page where you can draw a symbol and see the classification result.

## How It Works

After selecting a model:

1. the app loads `model.json` and model weights,
2. it reads class labels from `metadata.json` (if available),
3. the canvas is cleared and prediction values are reset,
4. the user draws a symbol,
5. after drawing, the app:
   - reads the image from the canvas,
   - resizes it to the selected model input size,
   - normalizes data to the range from 0 to 1,
   - runs prediction in TensorFlow.js,
   - displays the recognized class and a probability chart for all classes.

You can switch between available models in the interface:

- **Cyfry-Mnist** - convolutional model trained on MNIST, 10 classes (digits 0-9),
- **Cyfry-TM** - model from Google Teachable Machine, 10 classes (digits 0-9),
- **XO-TM** - model from Google Teachable Machine, 2 classes (X and O).

## Models And Metadata

Each model is stored in a separate directory and contains:

- `model.json` - TensorFlow.js model architecture and configuration,
- `*.bin` - model weights file,
- `metadata.json` (optional) - class labels in JSON format.

For static hosting (for example GitHub Pages), the model list is loaded from `models/index.json`.
This allows the app to work even when directory listing is not available on the server.

If a model does not include `metadata.json`, the app generates default numeric labels (0, 1, 2, ...).

TM models (Cyfry-TM and XO-TM) were trained in Google Teachable Machine:
https://teachablemachine.withgoogle.com/

They were then exported to TensorFlow.js format and stored in the `models` directory.

## Repository Structure

```text
.
├── digits.ipynb               # notebook for training and model experiments
├── index.html                 # browser demo interface
├── src/
│   ├── app.js                # app logic (loading, prediction, visualization)
│   └── style.css             # interface styling
└── models/
    ├── index.json            # model list for static hosting
    ├── Cyfry-Mnist/          # digit model (MNIST)
    │   ├── model.json
    │   ├── group1-shard1of1.bin
    │   └── metadata.json     # labels: ["0", "1", "2", ...]
    ├── Cyfry-TM/             # Teachable Machine model
    │   ├── model.json
    │   └── metadata.json
    └── XO-TM/                # Teachable Machine model (X, O)
        ├── model.json
        └── metadata.json
```

## Run A Local Preview

The project requires a local HTTP server because the browser fetches models and metadata.

Node.js example:

```
npx serve . --listen 5500
```

Then open this URL in your browser:

```text
http://localhost:5500/index.html
```

## Training And Model Export

The `digits.ipynb` notebook includes experiments with different neural network architectures for MNIST digit recognition, including:

- dense models,
- convolutional models,
- a variant with BatchNormalization and Dropout.

The notebook also contains cells related to installing compatible package versions and exporting a model to TensorFlow.js. The project assumes a Python/Jupyter environment with TensorFlow, matplotlib, scikit-learn, and tensorflowjs.

To export a model to TensorFlow.js and add metadata:

1. train a model in TensorFlow/Keras,
2. export the model using `tensorflowjs_converter`,
3. place files in the proper directory under `models/`,
4. create `metadata.json` with class labels.

Additionally, the repository includes sample models created in Teachable Machine (Google) and exported as TensorFlow.js (directories with the `TM` suffix).

---

## Polski

DrawRecognizer to prosty projekt demonstracyjny łączący trenowanie modeli rozpoznawania obrazów w TensorFlow z ich uruchamianiem bezpośrednio w przeglądarce przez TensorFlow.js. Repozytorium zawiera notatnik Jupyter do eksperymentów z sieciami neuronowymi oraz stronę HTML, na której można ręcznie narysować znak i zobaczyć wynik klasyfikacji.

## Opis działania

Po wybraniu modelu:

1. aplikacja ładuje plik `model.json` i wagi modelu,
2. pobiera etykiety klas z pliku `metadata.json` (jeśli istnieje),
3. canvas jest czyszczony i wartości predykcji są zerowane,
4. użytkownik rysuje symbol,
5. po rysunku aplikacja:
   - pobiera obraz z canvasu,
   - skaluje go do rozmiaru wejściowego wybranego modelu,
   - normalizuje dane do zakresu od 0 do 1,
   - wykonuje predykcję w TensorFlow.js,
   - wyświetla rozpoznaną klasę oraz wykres prawdopodobieństw dla wszystkich klas.

W interfejsie można przełączać dostępne modele:

- **Cyfry-Mnist** — model konwolucyjny trenowany na zbiorze MNIST, 10 klas (cyfry 0–9),
- **Cyfry-TM** — model z Google Teachable Machine, 10 klas (cyfry 0–9),
- **XO-TM** — model z Google Teachable Machine, 2 klasy (X i O).

## Modele i metadane

Każdy model znajduje się w oddzielnym katalogu i zawiera:

- `model.json` — architektura i konfiguracja modelu TensorFlow.js,
- `*.bin` — plik wag modelu,
- `metadata.json` (opcjonalnie) — etykiety klas w formacie JSON.

Dla hostingu statycznego (np. GitHub Pages) lista modeli jest odczytywana z pliku `models/index.json`.
Brak listingu katalogów na serwerze nie blokuje wtedy uruchomienia aplikacji.

Jeśli model nie posiada pliku `metadata.json`, aplikacja generuje domyślne etykiety numeryczne (0, 1, 2, ...).

Modele oznaczone jako TM (Cyfry-TM i XO-TM) zostały wytrenowane w Google Teachable Machine:
https://teachablemachine.withgoogle.com/

Następnie zostały wyeksportowane w formacie TensorFlow.js i umieszczone w katalogu models.

## Struktura repozytorium

```text
.
├── digits.ipynb               # notatnik do treningu i eksperymentów z modelami
├── index.html                 # interfejs demonstracyjny w przeglądarce
├── src/
│   ├── app.js                # logika aplikacji (ładowanie, predykcja, wizualizacja)
│   └── style.css             # stylowanie interfejsu
└── models/
    ├── index.json            # lista dostępnych modeli dla hostingu statycznego
    ├── Cyfry-Mnist/          # model cyfr (MNIST)
    │   ├── model.json
    │   ├── group1-shard1of1.bin
    │   └── metadata.json     # etykiety: ["0", "1", "2", ...]
    ├── Cyfry-TM/             # model Teachable Machine
    │   ├── model.json
    │   └── metadata.json
    └── XO-TM/                # model Teachable Machine (X, O)
        ├── model.json
        └── metadata.json
```

## Uruchomienie podglądu

Projekt wymaga lokalnego serwera HTTP, ponieważ przeglądarka pobiera modele i metadane przez fetch.

Przykład z Node.js:

```
npx serve . --listen 5500
```

Następnie otwórz w przeglądarce adres:

```text
http://localhost:5500/index.html
```

## Trenowanie i eksport modeli

Notatnik digits.ipynb zawiera eksperymenty z różnymi architekturami sieci dla rozpoznawania cyfr MNIST, w tym:

- modele gęste,
- modele konwolucyjne,
- wariant z BatchNormalization i Dropout.

W notatniku znajdują się również komórki związane z instalacją zgodnych wersji pakietów oraz eksportem modelu do TensorFlow.js. Projekt zakłada środowisko Python/Jupyter z bibliotekami TensorFlow, matplotlib, scikit-learn oraz tensorflowjs.

Aby wyeksportować model do TensorFlow.js i dodać metadane:

1. wytrenuj model w TensorFlow/Keras,
2. eksportuj model za pomocą tensorflowjs_converter,
3. umieść pliki w odpowiednim katalogu w `models/`,
4. utwórz plik `metadata.json` z etykietami klas.

Dodatkowo repozytorium zawiera przykładowe modele przygotowane w Teachable Machine (Google) i wyeksportowane jako TensorFlow.js (katalogi z sufiksem TM).