# DrawRecognizer

DrawRecognizer to prosty projekt demonstracyjny łączący trenowanie modeli rozpoznawania obrazów w TensorFlow z ich uruchamianiem bezpośrednio w przeglądarce przez TensorFlow.js. Repozytorium zawiera notatnik Jupyter do eksperymentów z sieciami neuronowymi oraz stronę HTML, na której można ręcznie narysować znak i zobaczyć wynik klasyfikacji.

## Co znajduje się w projekcie

- interfejs przeglądarkowy do rysowania na canvasie,
- ładowanie modeli TensorFlow.js z katalogu models,
- wizualizacja prawdopodobieństw klas na wykresie kołowym,
- notatnik z treningiem modeli dla rozpoznawania cyfr,
- dodatkowy model XO z etykietami X i O.

## Jak to działa

Plik drawView.html udostępnia dwa obszary canvas:

- pierwszy służy do rysowania znaku myszą,
- drugi pokazuje rozkład prawdopodobieństw przewidywanych przez model.

Po zakończeniu rysowania aplikacja:

1. pobiera obraz z canvasu,
2. skaluje go do rozmiaru wejściowego wybranego modelu,
3. normalizuje dane do zakresu od 0 do 1,
4. wykonuje predykcję w TensorFlow.js,
5. wyświetla rozpoznaną klasę oraz wykres prawdopodobieństw.

W interfejsie można przełączać dostępne modele:

- digits1,
- digits2,
- xo.

Dla modelu xo etykiety klas są pobierane z pliku metadata.json. Jeśli model nie posiada metadanych, aplikacja używa domyślnych etykiet numerycznych.

## Struktura repozytorium

```text
.
├── digits.ipynb          # notatnik do treningu i eksperymentów z modelami
├── drawView.html         # interfejs demonstracyjny w przeglądarce
└── models/
	├── digits1/          # wyeksportowany model TensorFlow.js
	├── digits2/          # wyeksportowany model TensorFlow.js
	└── xo/               # model TensorFlow.js oraz metadata.json z etykietami
```

## Uruchomienie podglądu

Najprościej uruchomić projekt przez lokalny serwer HTTP, ponieważ przeglądarka pobiera modele i metadane przez fetch.

Przykład z Node.js:

```
npx serve . --listen 5500
```

Następnie otwórz w przeglądarce adres:

```text
http://localhost:5500/drawView.html
```

## Trenowanie i eksport modeli

Notatnik digits.ipynb zawiera eksperymenty z różnymi architekturami sieci dla rozpoznawania cyfr MNIST, w tym:

- modele gęste,
- modele konwolucyjne,
- wariant z BatchNormalization i Dropout.

W notatniku znajdują się również komórki związane z instalacją zgodnych wersji pakietów oraz eksportem modelu do TensorFlow.js. Projekt zakłada środowisko Python/Jupyter z bibliotekami TensorFlow, matplotlib, scikit-learn oraz tensorflowjs.

## Technologie

- TensorFlow,
- TensorFlow.js,
- Jupyter Notebook,
- Chart.js,
- HTML i JavaScript.

## Zastosowanie

Projekt nadaje się jako:

- demonstracja działania klasyfikacji obrazów w przeglądarce,
- materiał edukacyjny do eksperymentów z modelami MNIST,
- punkt wyjścia do rozpoznawania własnych, ręcznie rysowanych symboli.