---
author:
- |
  \
  Konrad Karpiuk, Miłosz Więcyk
date: 2025
title: "**Aktywne uczenie dla zestawu danych o niezbalansowanych
  klasach.**"
---

:::::::: titlepage
::: center
**Wydział Elektroniki i Technik Informacyjnych**\
**Politechnika Warszawska**\
:::

::: center
**Uczenie Maszynowe**
:::

::: center
:::

::: center
:::

::: center
**Warszawa, **
:::
::::::::

# Wprowadzenie

Główne gałęzie uczenia maszynowego objemują uczenie nadzorowane, uczenie
nienadzorowane oraz uczenie przez wzmacnianie. W przypadku uczenia
nadzorowanego w zadaniu klasyfikacji model potrzebuje wielu danych z
przypisanymi do nich etykietami informujacymi o przynależności do jednej
z klas. Etykietowanie danych potrafi być najbardziej kosztowną częścią
procesu uczenia, gdyż zwykle jest wykonywane przez człowieka, często
eksperta w dziedzinie. Zmniejszenie liczby potrzebnych danych do
wytrenowania dobrze działającego modelu pozwoliłoby znacznie
przyspieszyć proces wdrażania modelu do środowiska produkcyjnego oraz
zmniejszyć koszt jego uzyskania. Istnieją przesłanki, że poprzez
wykorzystanie aktywnego uczenia można uzyskać właśnie ten efekt. Proces
aktywnego uczenia przebiega w pętli w następujący sposób:

1.  Wytrenowanie modelu na zbiorze trenującym.

2.  Wybranie przykładu bez etykiety, dla którego model ma największy
    współczynnik niepewności.

3.  Etykietowanie przykładu i dodanie go do zbioru trenującego.

Drugi rodzaj uczenia, uczenie pasywne, polega na wybraniu losowego
przykładu do etykietowania, zamiast tego o największym współczynniku
niepewności. Projekt ten powstał, aby porównać uczenie aktywne z
uczeniem pasywnym dla zbioru danych o niezbalansowanych klasach.

## Zestaw danych

Wykorzystany zestaw danych pochodzi z serwisu Kaggle i można go znaleźć
pod URL:\
\
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data\
\
Zbiór dotyczy nieautoryzowanego użycia karty kredytowej (ang. credit
card fraud). Zawiera 284807 danych, z czego 492 wierszy posiada klasę 1,
oznaczającą, że doszło do oszustwa z użyciem karty kredytowej, a reszta
wierszy posiada klasę 0, wskazującą na brak oszustwa. Oznacza to, że
zbiór ten cechuje się znacznym niezbalansowaniem klas - IR ratio wynosi
577.88. Kolumny V1, V2, \..., V28, powstałe prawdopodobnie przez
zastosowanie PCA, zawierają dane numeryczne i nie posiadają znaczenia
semantycznego prawdopodobnie w celu ochrony danych klientów banku.
Oprócz nich w zbiorze znajdują się jeszcze dwie kolumny: Time - czas w
sekundach, który minął od pierwszej transakcji w zbiorze, oraz Amount -
wartość transakcji.

# Badania

## Model

Model, który wybrano do badań, to Support Vector Machine. Jest to
algorytm, który w trakcie treningu dąży do uzyskania hiperpłaszczyzny
najlepiej separującej klasy. Wyjściem takiego modelu są
prawdopodobieństwa przynależności danego punktu w przestrzeni do klasy,
na podstawie odległości tego punktu od hiperpłaszczyzny. Konsekwencją
tej cechy jest to, że przypisanie przez model jednej z dwóch klas jest
funkcją przyjętej wartości progowej. Manipulując tą wartością można
wykreślić krzywe PR oraz ROC oraz wyliczać wartość g-means, które
posłużą za metryki do porównania modelu trenowanego aktywnie i modelu
trenowanego pasywnie.

## Przygotowanie danych

Na podstawie danych utworzono obiekt typu DataFrame z bilbioteki pandas.
Ze względu na to, że SVM wykorzystuje odległości w przestrzeni,
niezwykle istotne jest, aby żaden wymiar nie posiadał znaczenie
większych wartości od pozostałych, gdyż może to wprowadzić (po testach
wiadomo nawet, że wprowadza) obciążenie modelu w kierunku tego wymiaru.
Z tego powodu, zanim przystępiono do właściwego uczenia, ustandaryzowano
dane z każdej kolumny wykorzysując narzędzie z biblioteki **sklearn** -
**StandardScaler**. Odrzucono również kolumnę Time, ponieważ uznano, że
nie jest to kluczowa cecha do poprawnego działania modelu. Operację
zrealizowano przez kod w języku Python (listing
[\[standardScaler\]](#standardScaler){reference-type="ref"
reference="standardScaler"}).

``` {#standardScaler label="standardScaler" caption="Przygotowanie danych." style="python"}
df = df.drop(columns="Time")
scaler = StandardScaler()
for col in df.columns:
    if col == "Class":
        continue
    df[col] = scaler.fit_transform(df[[col]])
```

Aby porównać oba podejścia do uczenia dla różnego współczynnika
niezbalansowania klas (IR), wykorzystano metodę znaną jako **under
sampling**, polegającą na losowym odrzucaniu instancji klasy
większościowej, dopóki IR nie osiągnie wymaganego poziomu. Realizację
tego zadania przedstawia listing
[\[underSampler\]](#underSampler){reference-type="ref"
reference="underSampler"}. Na koniec dokonano podziału na zbiór
trenujący i testowy. Zbiór trenujący zawiera pulę przykładów, z których
model będzie mógł wybrać przykład do poetykietowania, a następnie dodać
go do zbioru trenującego poetykietowanego, z którego może się uczyć.

``` {#underSampler label="underSampler" caption="Under sampling." style="python"}
class_counts = y.value_counts()
minority_class = class_counts.idxmin()
minority_count = class_counts.min()
majority_class = class_counts.idxmax()

desired_ratio = 2
majority_count = minority_count * desired_ratio
sampling_strategy = {
    minority_class: minority_count,
    majority_class: majority_count
    }
rus = RandomUnderSampler(
    sampling_strategy=sampling_strategy, 
    random_state=17)
X_resampled, y_resampled = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=17)
```

Z tak przygotowanego zbioru danych wybrano losowo 1 przykład klasy
mniejszościowej oraz IR przykładów klasy większościowej do początkowego
zbioru trenującego poetykietowanego, imitując w ten sposób losowe
wybieranie przykładów z rozkładu całego zbioru trenującego z gwarancją
obecności reprezentantów obu klas. Zatem dla $IR = 4$ w początkowym
zbiorze poetykietowanych danych znajdzie się 1 przykład klasy
mniejszościowej i 4 przykłady klasy większościowej
([\[labeled_set\]](#labeled_set){reference-type="ref"
reference="labeled_set"}).

``` {#labeled_set label="labeled_set" caption="Wybieranie pierwszych przykładów do etykietowania." style="python"}
labeled_X = pd.DataFrame()
labeled_y = pd.Series()
chosen_0_class = 0  # majority class
chosen_1_class = 0  # minority class
for i in y_train.index.tolist():
    class_value = y_train.loc[i]
    if class_value == 0 and chosen_0_class < desired_ratio:
        chosen_0_class += 1
    elif class_value == 1 and chosen_1_class < 1:
        chosen_1_class += 1
    elif chosen_0_class == desired_ratio and chosen_1_class == 1:
        break
    else:
        continue
    labeled_X = pd.concat([labeled_X, X_train.loc[[i]]])
    X_train = X_train.drop(i)
    labeled_y = pd.concat([labeled_y, y_train.loc[[i]]])
    y_train = y_train.drop(i)
```

## Uczenie

Najpierw tworzone są dwa modele SVM z wykorzystaniem jądra RBF (Radial
Basis Function), potrafiącego dostosować się do dużej nieliniowości i
złożoności naszych danych. Oba trenowane są na przygotowanym przed
chwilą zbiorze trenującym poetykietowanych danych. Model uczony aktywnie
po wytrenowaniu oblicza prawdopodobieństwa wyboru klas dla wszystkich
danych znajdujących się w niepoetykietowanym zbiorze trenującym i
wybiera ten przykład, dla którego prawdopodobieństwo przypisania klasy 0
jest najbardziej zbliżone do $50\%$, natomiast model trenowany pasywnie
wybiera losowy przykład. Wybrany przykład zostaje dodany do zbioru
treningowego poetykietowanego, a proces jest powtarzany do momentu
wyczerpania zbioru treningowego. Po każdej iteracji modele są oceniane
na zbiorze testowym, a wyniki zapisywane do pliku. Na listingu
[\[active_learning\]](#active_learning){reference-type="ref"
reference="active_learning"} pokazano jak wygląda trening dla modelu
uczonego aktywnie.

``` {#active_learning label="active_learning" caption="Uczenie aktywne." style="python"}
active_model = SVC(kernel="rbf", probability=True, cache_size=1000)
ITER = len(unlabeled_y)
for i in range(ITER):
    active_model.fit(labeled_X, labeled_y)
    
    probabilities = active_model.predict_proba(unlabeled_X)
    uncertainty = np.abs(probabilities[:, 0] - probabilities[:, 1])
    least_confident_index = np.argmin(uncertainty)
    least_confident_sample = unlabeled_X.iloc[least_confident_index]
    df_index_number = least_confident_sample.name
    
    labeled_X = pd.concat(
        [labeled_X, least_confident_sample.to_frame().T])
    labeled_y = pd.concat(
        [labeled_y, unlabeled_y.loc[[df_index_number]]])
    unlabeled_X = unlabeled_X.drop(df_index_number)
    unlabeled_y = unlabeled_y.drop(df_index_number)
```

Proces ten przeprowadzono dla współczynników niezbalanosowania
$IR = 2, 4, 8, 16, ..., 256$.

## Wyniki

### Wybieranie klasy mniejszościowej

Pierwszym ciekawym zagadnieniem wartym przeanalizowania jest to, czy
aktywne uczenie sprzyja częstszemu wybieraniu klasy mniejszościowej jako
następny przykład do etykietowania niż uczenie pasywne. Wykresy
przedstawiące liczbę użytych przykładów klasy mniejszościowej w funkcji
liczby wszystkich użytych przykładów przedstawiono na rysunkach
[2.1](#fig:svm_plots){reference-type="ref" reference="fig:svm_plots"}.
Począwszy od $IR = 32$ uczenie było zatrzymywane po około $500$ cyklach
ze względu na jego czasochłonność, dlatego wykresy te przedstawiają
tylko początkowy wycinek pełnego treningu. Na wszystkich rysunkach
wykres modelu pasywnego przypomina linię prostą, czego należałoby się
spodziewać przy losowym wybieraniu przykładów do etykietowania. Jej
nachylenie powinno być równe $\frac{1}{1+IR}$, gdyż taki jest %
przykładów mniejszościowych w zbiorze trenującym. Z poniższych rysunków
rzeczywiście da się zaobserwować istnienie preferencji modelu uczonego
aktywnie do wyboru przykładów klasy mniejszościowej na początku procesu
uczenia. Efekt ten wydaje się być tym silniejszy, im większe IR
zastosowano. Ciekawa jest obserwacja, że po początkowym chętnym
wykorzystywaniu przykładów klasy mniejszościowej, tempo to znacznie
zwalnia i jest nawet niższe niż przy losowym wybieraniu przykładów. Może
to być oznaką dobrego odseparowania hiperpłaszyzną przykładów klasy
mniejszościowej w przestrzeni i brakiem wątpliwości przy przypiswaniu im
klasy (pewność znacznie powyżej 50%). Szczególnie dla przypadku
$IR = 256$ liczba przykładów klasy mniejszościowej dodanych do zbioru
etykietowanego wynosi ponad 200, podczas gdy dokonano zaledwie 300
iteracji algorytmu. Oznacza to, że ok. 66% razy wybrany został przykład
klasy mniejszościowej do etykietowania, mimo że klasa ta stanowi
zaledwie 0.39% zbioru trenującego.

<figure id="fig:svm_plots">
<figure>
<embed src="pictures/svm/svm_plot_IR2.svg" />
</figure>
<figure>
<embed src="pictures/svm/svm_plot_IR4.svg" />
</figure>
<figure>
<embed src="pictures/svm/svm_plot_IR8.svg" />
</figure>
<figure>
<embed src="pictures/svm/svm_plot_IR16.svg" />
</figure>
<figure>
<embed src="pictures/svm/svm_plot_IR32.svg" />
</figure>
<figure>
<embed src="pictures/svm/svm_plot_IR64.svg" />
</figure>
<figure>
<embed src="pictures/svm/svm_plot_IR128.svg" />
</figure>
<figure>
<embed src="pictures/svm/svm_plot_IR256.svg" />
</figure>
<figcaption>Użycie przykładów klasy mniejszościowej w zależności od
IR.</figcaption>
</figure>

### PR, ROC, g-means

Dla każdego zbioru danych (różne IR) narysowano wykresy przedstawiające
wartość pola pod krzywą ROC i PR oraz wartość g-means w zależności od
liczby przykładów, na których trenowany był model.

#### IR = 2

Dla $IR = 2$ nie zaobserwowano znaczącej różnicy między uczeniem
aktywnym, a uczeniem pasywnym jeśli chodzi o metryki, a nawet uczenie
pasywne poradziło sobie trochę lepiej. Model wykorzystujący uczenie
aktywne miał bardzo niskie wyniki dla pierwszych itracji uczenia. Jest
to o tyle dziwne, że nawet model strzelający klasę z 50%
prawdopodobieństwem powinien mieć ROC AUC na poziomie 0.5. Nie udało się
dotrzeć do źródła tej anomalii, jednak zakłożono, że jest ona związana z
niefortunnym wyborem początkowych przykładów trenujących, przez co model
miał \"odwrócone\" postrzeganie klas. Po dotarciu do danych niosących
wartościowe informacje, model ten zaczął szybko działać tak samo dobrze
jak model uczenia pasywnego.

<figure>
<figure>
<embed src="pictures/svm_metrics/roc_IR2.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/pr_IR2.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/g_mean_IR2.svg" />
</figure>
<figcaption>Metryki dla SVM, IR = 2</figcaption>
</figure>

#### IR = 4

Tutaj sytuacja wygląda bardzo podobnie do $IR = 2$.

<figure>
<figure>
<embed src="pictures/svm_metrics/roc_IR4.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/pr_IR4.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/g_mean_IR4.svg" />
</figure>
<figcaption>Metryki dla SVM, IR = 4</figcaption>
</figure>

#### IR = 8

Przy ośmiokrotnej przewadze występowania klasy 0 nad klasą 1 dochodzi
już do ciekawej obserwacji. Analizując wszystkie 3 metryki można
zauważyć, iż krzywa reprezentująca model trenowany aktywnie jest zawsze
nad krzywą modelu trenowanego pasywnie. Daje do przesłanki, co do
prawdziwości twierdzenia postawionego na początku.

<figure>
<figure>
<embed src="pictures/svm_metrics/roc_IR8.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/pr_IR8.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/g_mean_IR8.svg" />
</figure>
<figcaption>Metryki dla SVM, IR = 8</figcaption>
</figure>

#### IR $\geq$ 16

Na rysunkach dla danych o znacznym niezbalansowaniu klas (stosunek 1 do
16 lub więcej) widać już znacznie lepsze działanie modelu trenowanego
aktywnie (rys. [2.2](#svm_metrics_IR16){reference-type="ref"
reference="svm_metrics_IR16"} -
[2.6](#svm_metrics_IR256){reference-type="ref"
reference="svm_metrics_IR256"}). Oba modele mają podobne osiągnięcia na
krzywej przedstawiającej pole pod krzywą ROC w zależności od liczby
instancji zbioru trenującego, różnice są naprawdę niewielkie, rzędu
kilku dziesiątych. Model trenowany pasywnie znacząco odpada przy
pozostałych metrykach: PR oraz g-means. W problemie niezbalansowania
klas PR wydaje się być ważniejszą metryką od ROC, gdyż najważniejsze
jest wykrywanie klasy mniejszościowej (pozytywnej), a zarówno Precision
jak i Recall odnoszą się bezpośrednio do liczby TP (true positive)
przypadków. Precision można interpretować jako ufność, że dana
wytypowana jako pozytywna rzeczywiście jest pozytywna, natomiast Recall
jako procent odnalezionych danych o pozytywnej klasie. Czy to w
przypadku wykrywania oszustw kredytowych, czy w diagnozie złośliwego
nowotworu, znajdowanie pozytywnych przypadków jest najważniejsze. Warto
zauważyć, że model trenowany aktywnie bardzo szybko osiąga wysokie
wyniki i stabilnie się na nich utrzymuje.

<figure id="svm_metrics_IR16">
<figure>
<embed src="pictures/svm_metrics/roc_IR16.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/pr_IR16.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/g_mean_IR16.svg" />
</figure>
<figcaption>Metryki dla SVM, IR = 16</figcaption>
</figure>

<figure id="svm_metrics_IR32">
<figure>
<embed src="pictures/svm_metrics/roc_IR32.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/pr_IR32.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/g_mean_IR32.svg" />
</figure>
<figcaption>Metryki dla SVM, IR = 32</figcaption>
</figure>

<figure id="svm_metrics_IR64">
<figure>
<embed src="pictures/svm_metrics/roc_IR64.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/pr_IR64.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/g_mean_IR64.svg" />
</figure>
<figcaption>Metryki dla SVM, IR = 64</figcaption>
</figure>

<figure id="svm_metrics_IR128">
<figure>
<embed src="pictures/svm_metrics/roc_IR128.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/pr_IR128.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/g_mean_IR128.svg" />
</figure>
<figcaption>Metryki dla SVM, IR = 128</figcaption>
</figure>

<figure id="svm_metrics_IR256">
<figure>
<embed src="pictures/svm_metrics/roc_IR256.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/pr_IR256.svg" />
</figure>
<figure>
<embed src="pictures/svm_metrics/g_mean_IR256.svg" />
</figure>
<figcaption>Metryki dla SVM, IR = 256</figcaption>
</figure>

# Sieci neuronowe

Model SVM okazał się być bardzo dobry do analizowanego problemu i już po
niewielkiej liczbie iteracji osiągał przyzwoite wyniki. Można było też
zaobserwować skokowy wzrost jakości działania algorytmu po trafieniu na
odpowiednie przykłady trenujące. Postanowiono sprawdzić, czy aktywne
uczenie będzie miało równie pozytywny wpływ przy wykorzystaniu innego
modelu - perceptronu dwuwarstwowego, posiadającego po 10 neuronów w
każdej warstwie. Do uzyskania modelu wykorzystano klasę
**MLPClassifier** z pakietu **sklearn**. Przygotowanie danych polegało
na identycznych operacjach, co dla SVM. Pętla uczenia również
przebiegała w ten sam sposób.

## Wybieranie klasy mniejszościowej

Dla sieci neuronowej zachodzi podobna zależność, co dla SVM: im większe
IR, tym więcej przykładów klasy mniejszościowej zostaje wytypowanych do
etykietowania w początkowej fazie treningu.

<figure id="fig:nn_plots">
<figure>
<embed src="pictures/nn/nn_plot_IR2.svg" />
</figure>
<figure>
<embed src="pictures/nn/nn_plot_IR4.svg" />
</figure>
<figure>
<embed src="pictures/nn/nn_plot_IR8.svg" />
</figure>
<figure>
<embed src="pictures/nn/nn_plot_IR16.svg" />
</figure>
<figure>
<embed src="pictures/nn/nn_plot_IR32.svg" />
</figure>
<figure>
<embed src="pictures/nn/nn_plot_IR64.svg" />
</figure>
<figure>
<embed src="pictures/nn/nn_plot_IR128.svg" />
</figure>
<figure>
<embed src="pictures/nn/nn_plot_IR256.svg" />
</figure>
<figcaption>Użycie przykładów klasy mniejszościowej w zależności od IR
dla sieci neuronowej.</figcaption>
</figure>

## Metryki

W przeciwieństwie do SVM, zmiana jakości klasyfikacji w przypadku sieci
neuronowej nie jest tak gwałtowna, a różnice między uczeniem aktywnym, a
uczeniem pasywnym są lepiej widoczne. Dla małych IR oba modele działają
podobnie, z lekką przewagą modelu uczonego aktywnie. Podobnie jak
wcześniej od $IR = 16$ model uczący się aktywnie utrzymuje wysoki poziom
we wszystkich metrykach, natomiast model uczony pasywnie mocno odpada.
Tutaj nawet na krzywej AUC ROC różnice są znaczne. Tak znaczącą przewagę
modelu trenowanego aktywnie da się w prosty sposób wytłumaczyć. Przy
$IR = 256$ i losowym próbkowaniu, oczekiwana liczba przykładów klasy
pozytywnej w 512 pierwszych próbkach wynosi 2. Jest to bardzo mało, na
pewno za mało, aby można było odnaleźć jakieś wzorce rozróżniające klasę
pozytywną od negatywnej. W przypadku uczenia aktywnego, jak można
zaobserwować na wykresie [3.1](#fig:nn_plots){reference-type="ref"
reference="fig:nn_plots"}, liczba przykładów klasy pozytywnej wynosi
około 100, co daje już solidne podstawy do odnajdywania wzorców.

<figure>
<figure>
<embed src="pictures/nn_metrics/roc_IR2.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/pr_IR2.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/g_mean_IR2.svg" />
</figure>
<figcaption>Metryki dla sieci nauronowej, IR = 2</figcaption>
</figure>

<figure>
<figure>
<embed src="pictures/nn_metrics/roc_IR4.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/pr_IR4.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/g_mean_IR4.svg" />
</figure>
<figcaption>Metryki dla sieci nauronowej, IR = 4</figcaption>
</figure>

<figure>
<figure>
<embed src="pictures/nn_metrics/roc_IR8.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/pr_IR8.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/g_mean_IR8.svg" />
</figure>
<figcaption>Metryki dla sieci nauronowej, IR = 8</figcaption>
</figure>

<figure id="nn_metrics_IR16">
<figure>
<embed src="pictures/nn_metrics/roc_IR16.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/pr_IR16.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/g_mean_IR16.svg" />
</figure>
<figcaption>Metryki dla sieci nauronowej, IR = 16</figcaption>
</figure>

<figure id="nn_metrics_IR32">
<figure>
<embed src="pictures/nn_metrics/roc_IR32.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/pr_IR32.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/g_mean_IR32.svg" />
</figure>
<figcaption>Metryki dla sieci nauronowej, IR = 32</figcaption>
</figure>

<figure id="nn_metrics_IR64">
<figure>
<embed src="pictures/nn_metrics/roc_IR64.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/pr_IR64.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/g_mean_IR64.svg" />
</figure>
<figcaption>Metryki dla sieci nauronowej, IR = 64</figcaption>
</figure>

<figure id="nn_metrics_IR128">
<figure>
<embed src="pictures/nn_metrics/roc_IR128.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/pr_IR128.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/g_mean_IR128.svg" />
</figure>
<figcaption>Metryki dla sieci nauronowej, IR = 128</figcaption>
</figure>

<figure id="nn_metrics_IR256">
<figure>
<embed src="pictures/nn_metrics/roc_IR256.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/pr_IR256.svg" />
</figure>
<figure>
<embed src="pictures/nn_metrics/g_mean_IR256.svg" />
</figure>
<figcaption>Metryki dla sieci nauronowej, IR = 256</figcaption>
</figure>

# Trening pełny vs aktywny

Na zakończenie pozostało sprawdzić, czy model trenowany aktywnie może
osiągać znacznie lepsze wyniki klasyfikacji niż model nauczony na pełnym
zbiorze. Przesłanką do prawdziwości tej tezy jest dużo mniejszy
procentowy udział klasy pozytywnej w pełnym zbiorze trenującym w
porównaniu do zbioru po kilku iteracjach treningu aktywnego, przez co
model trenowany na pełnym zbiorze może być obciążony w kierunku klasy
większościowej. Jednak porównując wyniki uzyskane przez modele
wytrenowane na pełnych zbiorach (rys.
[4.1](#SVM_full_training){reference-type="ref"
reference="SVM_full_training"} i rys.
[4.2](#NN_full_training){reference-type="ref"
reference="NN_full_training"}) z wynikami modeli trenowanymi aktywne,
nie zaobserwowano, aby model wytrenowany na pełnym zbiorze osiągał
gorsze wyniki. Natomiast zauważono, że model trenowany aktywnie bardzo
szybko osiąga podobny poziom jakości klasyfikacji, jak model trenowany
na pełnym zbiorze, zatem aktywne uczenie z powodzeniem może być
stosowane do osiągania wysokiej jakości modeli przy minimalnym koszcie
potrzebnym na etykietowanie danych.

<figure id="SVM_full_training">
<figure>
<embed src="pictures/full_training/svm/ROC.svg" />
</figure>
<figure>
<embed src="pictures/full_training/svm/PR.svg" />
</figure>
<figcaption>SVM wytrenowany na pełnym zbiorze. Najlepsze g-means:<br />
IR = 2: 0.946, IR = 4: 0.966, IR = 8: 0.954, IR = 16: 0.948,<br />
IR = 32: 0.900, IR = 64: 0.946, IR = 128: 0.952, IR = 256:
0.917</figcaption>
</figure>

<figure id="NN_full_training">
<figure>
<embed src="pictures/full_training/nn/ROC.svg" />
</figure>
<figure>
<embed src="pictures/full_training/nn/PR.svg" />
</figure>
<figcaption>Sieć neuronowa wytrenowana na pełnym zbiorze. Najlepsze
g-means:<br />
IR = 2: 0.937, IR = 4: 0.963, IR = 8: 0.933, IR = 16: 0.948,<br />
IR = 32: 0.941, IR = 64: 0.961, IR = 128: 0.964, IR = 256:
0.940</figcaption>
</figure>
