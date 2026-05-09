# DrawRecognizer

DrawRecognizer to prosty projekt demonstracyjny łączący trenowanie modeli rozpoznawania obrazów w TensorFlow z ich uruchamianiem bezpośrednio w przeglądarce przez TensorFlow.js. Repozytorium zawiera notatnik Jupyter do eksperymentów z sieciami neuronowymi oraz stronę HTML, na której można ręcznie narysować znak i zobaczyć wynik klasyfikacji.

## Opis działania

Plik index.html udostępnia dwa obszary canvas:

- pierwszy służy do rysowania znaku myszą,
- drugi pokazuje rozkład prawdopodobieństw przewidywanych przez model.

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

## Zawartość repozytorium

- interfejs przeglądarkowy do rysowania na canvasie,
- ładowanie modeli TensorFlow.js z katalogu models,
- wizualizacja prawdopodobieństw klas na wykresie słupkowym,
- notatnik z treningiem modeli dla rozpoznawania cyfr,
- dodatkowe modele: Cyfry-TM i XO-TM z etykietami z Teachable Machine,
- wsparcie dla metadanych modeli (metadata.json).

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

## Technologie

- TensorFlow,
- TensorFlow.js,
- Jupyter Notebook,
- HTML i JavaScript,
- Canvas API.

## Zastosowanie

Projekt nadaje się jako:

- demonstracja działania klasyfikacji obrazów w przeglądarce,
- materiał edukacyjny do eksperymentów z modelami MNIST,
- punkt wyjścia do rozpoznawania własnych, ręcznie rysowanych symboli,
- przykład integracji TensorFlow.js z interfejsem użytkownika.
