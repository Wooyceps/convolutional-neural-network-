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

## Przetwarzanie zestawu danych

```python
train_images = np.array(train_images) / 255.0
test_images = np.array(test_images) / 255.0
```

Dane obrazu (kolor) są normalizowane do wartości między 0 a 1.

## Budowanie modelu

```python
def build_model(input_shape, num_classes):
    model = Sequential()
    ...
    return model
```

Definiowana jest funkcja do budowy modelu CNN. Model składa się z kilku warstw konwolucyjnych, warstw poolingowych i gęstych.

## Trenowanie modelu

```python
hist = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2,
                 callbacks=[checkpoint_callback, reduce_lr_callback])
```

Model jest trenowany na danych treningowych przez 10 epok z rozmiarem batcha 64. Wskaźnik uczenia jest redukowany, jeśli strata walidacji nie zmniejsza się po 5 epokach.

## Ocena modelu

```python
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
```

Wytrenowany model jest używany do przewidywania etykiet obrazów testowych.

## Wizualizacja wyników

```python
def plot_images(images, true_labels, predicted_labels, class_names, num_images=25):
    ...
    plt.show()
```

Definiowana jest funkcja do wyświetlania obrazów wraz z ich prawdziwymi i przewidywanymi etykietami. Funkcja jest następnie wywoływana do wizualizacji pierwszych 25 obrazów ze zbioru testowego.