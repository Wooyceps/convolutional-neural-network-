# Klasyfikacja obrazów K-MNIST

Notatnik Jupter o nazwie `main_2.ipynb` służy do klasyfikacji obrazów z zestawu danych Kuzushiji-MNIST (KMNIST) za pomocą sieci konwolucyjnej (CNN) zbudowanej za pomocą TensorFlow.

## Importowanie bibliotek

```python
import numpy as np
import tensorflow_datasets as tfds
```

Importowane są niezbędne biblioteki. `numpy` jest używane do operacji numerycznych, a `tensorflow_datasets` do wczytania zestawu danych KMNIST.

## Wczytywanie zestawu danych

```python
dataset, info = tfds.load('kmnist', split=['train', 'test'], shuffle_files=True, with_info=True)
```

Zestaw danych KMNIST jest wczytywany i dzielony na zestawy treningowe i testowe.

Funkcja `tfds.load` jest używana do wczytania zestawu danych. Przyjmuje ona kilka argumentów:

- `'kmnist'` to nazwa zestawu danych, który ma zostać wczytany. KMNIST to zestaw danych zawierający obrazy ręcznie pisanych znaków japońskiego pisma Kuzushiji.

- `split=['train', 'test']` określa, jakie podzbiory zestawu danych mają zostać wczytane. W tym przypadku wczytywane są zestawy treningowe i testowe.

- `shuffle_files=True` oznacza, że pliki zestawu danych są mieszane przed wczytaniem. Jest to przydatne, gdy zestaw danych jest zbyt duży, aby zmieścić się w pamięci, a dane są rozłożone na wiele plików. Mieszanie plików pomaga zapewnić, że dane są dobrze wymieszane podczas treningu, nawet jeśli są wczytywane z wielu plików.

- `with_info=True` oznacza, że funkcja zwraca również obiekt `tfds.core.DatasetInfo`, który zawiera metadane na temat wczytanego zestawu danych. Metadane te mogą obejmować informacje takie jak liczba przykładów w każdym podzbiorze, nazwy klas, kształt i typ danych wejściowych itp.

Wynikiem tej linii kodu są dwie zmienne: `dataset` i `info`. 

`dataset` to `tf.data.Dataset` zawierający wczytane dane, a `info` to `tfds.core.DatasetInfo` zawierający metadane zestawu danych.

```python
train_dataset, test_dataset = dataset
```

Zmienna `dataset` zawiera dwa zestawy danych: treningowy i testowy. Aby rozdzielić te zestawy na dwie zmienne, używamy przypisania wielokrotnego.

```python
train_df = tfds.as_dataframe(train_dataset, info)
test_df = tfds.as_dataframe(test_dataset, info)

train_images = [row['image'] for _, row in train_df.iterrows()]
train_labels = [row['label'] for _, row in train_df.iterrows()]

test_images = [row['image'] for _, row in test_df.iterrows()]
test_labels = [row['label'] for _, row in test_df.iterrows()]
```

Ten fragment kodu przekształca dane z formatu TensorFlow Dataset do formatu pandas DataFrame, który jest łatwiejszy do manipulowania i analizy. 

1. **`train_df = tfds.as_dataframe(train_dataset, info)`**: Ta linia kodu konwertuje `train_dataset` (zbiór treningowy) na DataFrame za pomocą funkcji `as_dataframe` z biblioteki TensorFlow Datasets (`tfds`). Argument `info` jest używany do określenia struktury danych.

2. **`test_df = tfds.as_dataframe(test_dataset, info)`**: Podobnie, ta linia kodu konwertuje `test_dataset` (zbiór testowy) na DataFrame.

3. **`train_images = [row['image'] for _, row in train_df.iterrows()]`**: Ta linia kodu tworzy listę obrazów treningowych, iterując przez każdy wiersz DataFrame `train_df` i wybierając kolumnę 'image'. 

4. **`train_labels = [row['label'] for _, row in train_df.iterrows()]`**: Podobnie, ta linia kodu tworzy listę etykiet treningowych, iterując przez każdy wiersz DataFrame `train_df` i wybierając kolumnę 'label'.

5. **`test_images = [row['image'] for _, row in test_df.iterrows()]`**: Ta linia kodu tworzy listę obrazów testowych, analogicznie do tego, jak zostało to zrobione dla obrazów treningowych.

6. **`test_labels = [row['label'] for _, row in test_df.iterrows()]`**: Na koniec, ta linia kodu tworzy listę etykiet testowych, analogicznie do tego, jak zostało to zrobione dla etykiet treningowych.

W skrócie, ten fragment kodu przekształca dane z formatu TensorFlow Dataset na format pandas DataFrame, a następnie tworzy listy obrazów i etykiet dla zbiorów treningowego i testowego.

```python
train_images = np.array(train_images) / 255.0
test_images = np.array(test_images) / 255.0
```

Dane obrazu (kolor) są normalizowane do wartości między 0 a 1.

```python
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
```

Ten fragment kodu używa funkcji `np.expand_dims` z biblioteki NumPy, aby dodać dodatkowy wymiar do tablic `train_images` i `test_images`. Parametr `axis=-1` oznacza, że nowy wymiar jest dodawany na końcu kształtu tablicy.

Przygotowujemy tutaj dane wejściowe do użycia w modelu sieci neuronowej, dodając wymiar kanału do obrazów.

```python
num_classes = 10
```

`num_classes = 10` definiuje liczbę klas, które model ma nauczyć się rozpoznawać. W tym przypadku mamy 10 różnych klas.

```python
train_labels = np.eye(num_classes)[train_labels]
test_labels = np.eye(num_classes)[test_labels]
```

Funkcja `np.eye(num_classes)` tworzy macierz jednostkową o rozmiarze równym liczbie klas. Następnie, dla każdej etykiety w `train_labels` i `test_labels`, wybierany jest odpowiedni wiersz z tej macierzy. W efekcie, każda etykieta jest zamieniana na wektor "one-hot" odpowiadający danej klasie.

Format "one-hot" to sposób reprezentacji kategorii, w którym każda kategoria jest reprezentowana jako wektor, gdzie wszystkie elementy są równe 0, z wyjątkiem jednego, który jest równy 1. Indeks tego jedynkowego elementu odpowiada klasie, której dotyczy dany wektor.

## Budowanie modelu

```python
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
```

Z modułów `tensorflow.keras.layers` i `tensorflow.keras.models` importowane są klasy warstw i modelu, które są używane do zbudowania modelu CNN.

```python
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model
```

Model sieci konwolucyjnej (CNN) zdefiniowany w tym kodzie składa się z kilku warstw:

1. **Warstwa konwolucyjna (Conv2D)**: Pierwsza warstwa modelu to warstwa konwolucyjna z 32 filtrami o rozmiarze 3x3 i funkcją aktywacji ReLU. Ta warstwa służy do wykrywania lokalnych wzorców w obrazach, takich jak krawędzie, kształty itp.

2. **Warstwa poolingowa (MaxPooling2D)**: Następnie mamy warstwę poolingową, która redukuje wymiarowość danych poprzez wybór maksymalnej wartości z każdego okna poolingowego. W tym przypadku rozmiar okna to 2x2.

3. **Warstwa konwolucyjna (Conv2D)**: Kolejna warstwa konwolucyjna z 64 filtrami o rozmiarze 3x3 i funkcją aktywacji ReLU. Ta warstwa pozwala na wykrywanie bardziej skomplikowanych wzorców na podstawie tych wykrytych przez poprzednią warstwę.

4. **Warstwa poolingowa (MaxPooling2D)**: Kolejna warstwa poolingowa, podobnie jak poprzednia, redukuje wymiarowość danych.

5. **Warstwa konwolucyjna (Conv2D)**: Trzecia warstwa konwolucyjna z 64 filtrami o rozmiarze 3x3 i funkcją aktywacji ReLU. Ta warstwa pozwala na wykrywanie jeszcze bardziej skomplikowanych wzorców.

6. **Warstwa spłaszczająca (Flatten)**: Ta warstwa spłaszcza dane wejściowe, przekształcając je z formatu 2D do formatu 1D. Jest to wymagane, ponieważ następne warstwy (gęste) oczekują danych w formacie 1D.

7. **Warstwa gęsta (Dense)**: Ta warstwa gęsta składa się z 64 neuronów i używa funkcji aktywacji ReLU. Warstwy gęste są standardowymi warstwami sieci neuronowych, które uczą się globalnych wzorców w ich wejściowych danych.

8. **Warstwa wyjściowa (Dense)**: Ostatnia warstwa to warstwa gęsta, która składa się z tylu neuronów, ile jest klas (w tym przypadku 10). Ta warstwa używa funkcji aktywacji softmax, która jest często używana w warstwach wyjściowych modeli klasyfikacji, ponieważ jej wyniki można interpretować jako prawdopodobieństwa przynależności do każdej z klas.

```python
input_shape = (28, 28, 1)
num_classes = 10

model = build_model(input_shape, num_classes)
```

`input_shape` to kształt obrazów w zestawie danych KMNIST. Obrazy w KMNIST mają rozmiar 28x28 pikseli, więc kształt ten to `(28, 28, 1)`, gdzie 1 oznacza, że obrazy są w skali szarości i informacja o kolorze zawiera się w dodatkowym wymiarze.

`num_classes` to liczba kategorii w zestawie danych KMNIST. W tym przypadku jest to 10, ponieważ KMNIST zawiera 10 różnych znaków.

`model = build_model(input_shape, num_classes)` tworzy model CNN, korzystając z funkcji `build_model` zdefiniowanej wcześniej.

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

Model jest kompilowany z optymalizatorem Adam, funkcją straty `categorical_crossentropy` (która jest typowym wyborem dla problemów klasyfikacji wieloklasowej), i metryką 'accuracy', która jest procentem poprawnie sklasyfikowanych obrazów.

`model.summary()` wyświetla podsumowanie modelu, które zawiera informacje o każdej warstwie, liczbie parametrów w modelu itp.

## Trenowanie modelu

```python
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

checkpoint_callback = ModelCheckpoint('model.keras', save_best_only=True)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
```

W tym fragmencie kodu importowane są dwie klasy z modułu `tensorflow.keras.callbacks`, które są używane do tworzenia funkcji zwrotnych (callbacks) podczas trenowania modelu sieci neuronowej.

1. **ModelCheckpoint**: `ModelCheckpoint` jest funkcją zwrotną, która pozwala na zapisywanie modelu w trakcie i na końcu treningu. W tym przypadku, `ModelCheckpoint('model.keras', save_best_only=True)` tworzy funkcję zwrotną, która zapisuje model do pliku `model.keras` tylko wtedy, gdy model na danym etapie treningu jest "najlepszy" do tej pory, czyli ma najniższą stratę na zbiorze walidacyjnym.

2. **ReduceLROnPlateau**: `ReduceLROnPlateau` jest funkcją zwrotną, która zmniejsza szybkość uczenia się (learning rate), gdy metryka (w tym przypadku `val_loss`) przestaje się poprawiać. Konkretnie, `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)` tworzy funkcję zwrotną, która monitoruje stratę na zbiorze walidacyjnym (`val_loss`) i jeśli strata nie poprawiła się przez 5 epok (`patience=5`), szybkość uczenia się jest mnożona przez 0.2 (`factor=0.2`). Szybkość uczenia nie spadnie poniżej 0.001 (`min_lr=0.001`).

```python
hist = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2,
                 callbacks=[checkpoint_callback, reduce_lr_callback])
```

Model jest trenowany na danych treningowych przez 10 epok z rozmiarem batcha 64. Wskaźnik uczenia jest redukowany, jeśli strata walidacji nie zmniejsza się po 5 epokach.
### ***Epoka***
**Epoka**, znana również jako era, to termin używany w uczeniu maszynowym, szczególnie w kontekście uczenia sieci neuronowych, do opisania jednego przejścia przez cały zestaw treningowy. 

Podczas jednej epoki, algorytm uczenia maszynowego próbuje nauczyć się wzorców w danych treningowych, a następnie na koniec epoki, błąd (strata) jest obliczany na całym zestawie treningowym. 

Na przykład, jeśli mamy 1000 próbek treningowych i ustawimy rozmiar batcha na 100, to oznacza, że sieć neuronowa będzie się uczyć na podzbiorach 100 próbek na raz, a cały zestaw treningowy będzie podzielony na 10 batchy. Po przetworzeniu wszystkich 10 batchy, kończy się jedna epoka.

Liczba epok jest jednym z hiperparametrów, które muszą być ustawione przed rozpoczęciem procesu uczenia. Zbyt mała liczba epok może prowadzić do niedouczenia modelu, podczas gdy zbyt duża liczba epok może prowadzić do przeuczenia.

### ***Batch***

**Rozmiar batcha**, znany również jako wielkość batcha lub batch size, to liczba próbek treningowych używanych w jednej iteracji, czyli jednym kroku uczenia sieci neuronowej. 

Na przykład, jeśli mamy 1000 próbek treningowych i ustawimy rozmiar batcha na 100, to oznacza, że sieć neuronowa będzie się uczyć na podzbiorach 100 próbek na raz, a cały zestaw treningowy będzie podzielony na 10 batchy. Każdy batch jest przetwarzany niezależnie przez sieć neuronową, a parametry sieci są aktualizowane po każdym batchu.

Ustalenie odpowiedniego rozmiaru batcha jest istotnym aspektem uczenia sieci neuronowych. Zbyt mały rozmiar batcha może prowadzić do niestabilnego uczenia, podczas gdy zbyt duży rozmiar batcha może prowadzić do niewystarczającego uczenia.

## Ocena modelu

```python
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
```

W tych dwóch liniach kodu, model sieci neuronowej, który został wcześniej wytrenowany, jest używany do przewidywania etykiet dla obrazów testowych.

1. **`predictions = model.predict(test_images)`**: Metoda `predict` jest używana do generowania przewidywań modelu na podstawie danych wejściowych (`test_images`). Wynikiem tej metody jest tablica, gdzie każdy element tablicy jest wektorem prawdopodobieństw dla każdej klasy. Długość tego wektora jest równa liczbie klas, a suma prawdopodobieństw w wektorze wynosi 1. Każde prawdopodobieństwo w wektorze wskazuje, jak pewny jest model, że dany obraz należy do danej klasy.

2. **`predicted_labels = np.argmax(predictions, axis=1)`**: Następnie, funkcja `np.argmax` jest używana do wyznaczenia indeksu największego elementu w każdym wektorze prawdopodobieństw. W tym kontekście, indeks ten odpowiada etykiecie klasy, dla której model jest najbardziej pewny. Parametr `axis=1` oznacza, że operacja jest wykonywana wzdłuż osi oznaczającej różne klasy. Wynikiem tej linii kodu jest tablica `predicted_labels`, która zawiera przewidywane etykiety dla obrazów testowych.

## Wizualizacja wyników

```python
def plot_images(images, true_labels, predicted_labels, class_names, num_images=25):
    true_label_indices = np.argmax(true_labels, axis=1)
    plt.figure(figsize=(15, 18))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = np.squeeze(images[i])
        plt.imshow(img, cmap=plt.cm.binary)
        color = 'green' if predicted_labels[i] == true_label_indices[i] else 'red'
        plt.xlabel(f"True: {class_names[true_label_indices[i]]}\nPred: {class_names[predicted_labels[i]]}", color=color)
    plt.show()
```

Funkcja `plot_images` jest używana do wyświetlania obrazów wraz z ich prawdziwymi etykietami i etykietami przewidzianymi przez model. 

1. **`def plot_images(images, true_labels, predicted_labels, class_names, num_images=25):`**: Definiuje funkcję `plot_images` z pięcioma argumentami: `images` (obrazy do wyświetlenia), `true_labels` (prawdziwe etykiety obrazów), `predicted_labels` (etykiety przewidziane przez model), `class_names` (nazwy klas odpowiadające etykietom) i `num_images` (liczba obrazów do wyświetlenia, domyślnie 25).

2. **`true_label_indices = np.argmax(true_labels, axis=1)`**: Konwertuje etykiety z formatu one-hot na indeksy klas.

3. **`plt.figure(figsize=(15, 18))`**: Tworzy nowy rysunek o określonym rozmiarze.

4. **`for i in range(num_images):`**: Iteruje przez obrazy do wyświetlenia.

5. **`plt.subplot(5, 5, i + 1)`**: Dodaje podwykres do rysunku. 

6. **`plt.xticks([]), plt.yticks([]), plt.grid(False)`**: Usuwa osie x i y oraz siatkę z podwykresu.

7. **`img = np.squeeze(images[i])`**: Usuwa niepotrzebne wymiary z obrazu.

8. **`plt.imshow(img, cmap=plt.cm.binary)`**: Wyświetla obraz w podwykresie.

9. **`color = 'green' if predicted_labels[i] == true_label_indices[i] else 'red'`**: Ustala kolor etykiety na zielony, jeśli przewidywana etykieta jest prawidłowa, w przeciwnym razie na czerwony.

10. **`plt.xlabel(f"True: {class_names[true_label_indices[i]]}\nPred: {class_names[predicted_labels[i]]}", color=color)`**: Dodaje etykiety do obrazu, pokazując prawdziwą i przewidywaną klasę.

11. **`plt.show()`**: Wyświetla rysunek z podwykresami.

```python
class_names = ['O', 'KI', 'SU', 'TSU', 'NA',
               'HA', 'MA', 'YA', 'RE', 'WO']

plot_images(test_images[:25], test_labels[:25], predicted_labels[:25], class_names)
```

Po definicji funkcji `plot_images`, następuje definicja `class_names`, która jest listą nazw klas. Następnie funkcja `plot_images` jest wywoływana, aby wyświetlić pierwsze 25 obrazów z `test_images` wraz z ich prawdziwymi i przewidywanymi etykietami.