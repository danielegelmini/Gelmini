import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.spatial.distance import hamming
from imageio import imread
import random
from PIL import Image, ImageDraw, ImageFont
import os


def IsScalar(x):
    """
    Verifica se il valore x è uno scalare.

    Input:
        x: Valore da verificare

    Output:
        True se x è uno scalare, False altrimenti
    """
    if isinstance(x, (list, np.ndarray)):
        return False
    else:
        return True


def Thresh(x):
    """
    Imposta i valori degli elementi di x a 1 se positivi, a -1 se negativi,
    o crea un array di 1 e -1 in base ai valori di x se x è un array.

    Input:
        x: Valore o array di valori

    Output:
        Valore o array di valori modificati secondo la logica di Thresh
    """
    if IsScalar(x):
        val = 1 if x > 0 else -1
    else:
        val = np.ones_like(x)
        val[x < 0] = -1
    return val


def Energy(W, b, X):
    """
    Calcola l'energia di un insieme di pattern utilizzando i pesi e il bias dati.

    Input:
        W: Matrice dei pesi
        b: Bias
        X: Insieme di pattern (righe: pattern, colonne: attributi)

    Output:
        Lista delle energie calcolate per ogni pattern in X
    """
    E = []
    for xx in X:
        energy = -0.5 * (xx @ W) @ xx.T + b @ xx.T
        E.append(energy)
    return E


def Update(W, x, b):
    """
    Calcola l'update di un pattern dato i pesi e il bias.

    Input:
        W: Matrice dei pesi
        x: Pattern da aggiornare
        b: Bias

    Output:
        Nuovo pattern aggiornato
    """
    xnew = x @ W - b
    return Thresh(xnew)


def Perturb(x, p):
    """
    Perturba un pattern con una certa probabilità p.

    Input:
        x: Pattern da perturbare
        p: Probabilità di perturbazione

    Output:
        Nuovo pattern perturbato
    """
    y = copy.deepcopy(x)
    for yy in y:
        for k in range(len(yy)):
            if np.random.rand() < p:
                yy[k] = Thresh(np.random.randint(2) * 2 - 1)
    return y


def format_mem_similarity(idx, similarity, is_correct):
    """
    Formatta l'output della percentuale di assomiglianza tra pattern.

    Input:
        idx: Indice del pattern
        similarity: Percentuale di assomiglianza
        is_correct: True se il pattern è corretto, False altrimenti

    Output:
        Stringa formattata con l'informazione sulla similarità
    """
    if is_correct:
        return '\033[1m\033[2m\033[3mLa memoria {:2d} è corretta al {:3.0f}%\033[0m'.format(idx, similarity)
    else:
        return 'La memoria {:2d} è corretta al {:3.0f}%'.format(idx, similarity)

### Binary Code

# Introduco le variabili del problema
n_b = 10  # Numero di memorie
N_b = 100  # Dimensioni delle memorie
X_b = Thresh(np.random.normal(size=(n_b, N_b)))  # Matrice formata da n pattern di lunghezza N casuali

b_b = np.zeros((1, N_b))  # Bias
b_b = np.sum(X_b, axis=0) / n_b
W_b = (X_b.T @ X_b) / n_b - np.eye(N_b)  # Pesi

# Modifico una memoria e mi calcolo di quanto differisce dall'originale con la funzione di hamming
k_b = np.random.randint(n_b)
Y_b = Perturb(X_b, p=0.4)
x_b = Y_b[k_b:k_b + 1, :]  # memoria perturbata
err_b = hamming(x_b[0], X_b[k_b]) * len(x_b[0])
print('Classe ' + str(k_b) + ' con ' + str(err_b) + ' errori')

# Definisco delle quantità da poter modificare e il numero di iterazioni da svolgere
xs_b = copy.deepcopy(x_b)
xa_b = copy.deepcopy(x_b)
n_iters_b = 150

# Update sincrono
Es_b = []
for _ in range(n_iters_b):  # per ogni ciclo si aggiorna ogni neurone
    xs_b = Update(W_b, xs_b, b_b)
    Es_b.append(Energy(W_b, b_b, xs_b))

# Update asincrono
xa_b = np.copy(x_b)
Ea_b = []
for count_b in range(n_iters_b):
    node_idx_b = list(range(N_b))
    np.random.shuffle(node_idx_b)
    for i_b in node_idx_b:  # per ogni iterazione scelgo un nodo i-esimo e calcolo l'update per esso
        ic_b = xa_b @ W_b[:, i_b] - b_b[
            i_b]  # prodotto tra xa e colonna i-esima della matrice pesi meno il bias associato al nodo i-esimo
        xa_b[0, i_b] = Thresh(ic_b)  # assegna il risultato al nodo i-esimo nell'array xa.
        Ea_b.append(Energy(W_b, b_b, xa_b))

# Grafico dell'andamento dell'energia per ogni iterazione
Ea_graph_b = np.array(Ea_b[:n_iters_b])
Es_graph_b = np.array([e_b[0] for e_b in Es_b])
n_iters_vett_b = np.arange(0, n_iters_b)

plt.figure(figsize=(10, 6))
plt.plot(n_iters_vett_b, Ea_graph_b, marker='o', label='Energia update asincrono')
plt.plot(n_iters_vett_b, Es_graph_b, marker='x', label='Energia update sincrono')
plt.xlabel('Numero di Iterazioni')
plt.ylabel('Energia')
plt.title('Variazione dell\'energia in funzione del numero di iterazioni')
plt.grid(True)
plt.legend()
plt.show()


# Stampo la percentuale per cui x differisce da ogni singola memoria
print('La classe corretta è la ' + str(k_b))
print('-' * 67)
print('         Update Sincrono                   Update Asincrono')
print('-' * 67)
for idx, (t_sync, t_async) in enumerate(zip(X_b, X_b)):
    ds_sync_b = hamming(xs_b[0], t_sync) * len(xs_b[0])  # calcolo il numero di errori
    da_async_b = hamming(xa_b[0], t_async) * len(xa_b[0])

    similarity_sync_b = ((len(xs_b[0]) - ds_sync_b) / len(xs_b[0])) * 100  # calcolo la percentuale di errore
    similarity_async_b = ((len(xa_b[0]) - da_async_b) / len(xa_b[0])) * 100

    formatted_text_sync_b = format_mem_similarity(idx, similarity_sync_b, idx == k_b)
    formatted_text_async_b = format_mem_similarity(idx, similarity_async_b, idx == k_b)

    print(f'{formatted_text_sync_b} | {formatted_text_async_b}')



### Letters

# Definisco le dimensioni dell'immagine da creare
width, height = 6, 6

# Lista delle lettere dell'alfabeto
letters = 'ABCDEFGHIJKLMNOPRSTUVWXYZ'  # non considero la Q in quanto risulta identica alla O

# Crea la cartella "images" se non esiste
if not os.path.exists('images'):
    os.makedirs('images')

# Creo e salvo le immagini
for letter in letters:
    image = Image.new('L', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    # Posiziono il testo al centro dell'immagine
    font = ImageFont.load_default()  # Usa il font di default
    text_width, text_height = draw.textsize(letter, font=font)
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), letter, fill='black')
    image = image.crop((0, 0, width, height))  # Riduco l'immagine alle dimensioni corrette
    image.save(f'images/{letter}.png')  # Salvo l'immagine

# Visualizzo le immagini create
plt.figure(figsize=(12, 10))
for idx, letter in enumerate(letters):
    image_path = f'images/{letter}.png'
    img = plt.imread(image_path)
    plt.subplot(6, 8, idx + 1)
    plt.imshow(img, cmap='tab20c')
    plt.title(letter)
    plt.axis('off')

plt.tight_layout()
plt.show()

letters = []

# seleziono alcune lettere
letters.append(imread('images/A.png'))
letters.append(imread('images/U.png'))
letters.append(imread('images/W.png'))
letters.append(imread('images/T.png'))

# Introduco le variabili del problema

n = len(letters)  # Numero di memorie
N = len(letters[0].flatten())  # Dimensioni delle memorie
X = np.zeros((n, N))

b = np.zeros((1, N))  # bias
b = np.sum(X, axis=0) / n
W = (X.T @ X) / n - np.eye(N)  # Pesi

for idx, img in enumerate(letters):  # trasformo le immagini in matrice X
    X[idx, :] = Thresh(np.array([img.flatten() - 0.5]))

plt.figure(figsize=(16, 4))
for k in range(n):
    plt.subplot(1, n, k + 1)
    plt.imshow(np.reshape(X[k], (width, height)), cmap='tab20c')
    plt.axis('off')

# Modifico una memoria e mi calcolo di quanto differisce dall'originale con la funzione di hamming
k = np.random.randint(n)
Y = Perturb(X, p=0.2)
x = Y[k:k + 1, :]  # memoria perturbata
x[0, round(
    width * height * 2 / 3):] = -1.  # imposto un terzo dell'immagine pari a -1, ovvero è come se tagliassi il fondo dell'immagine
err = hamming(x[0], X[k]) * len(x[0])
print('Classe ' + str(k) + ' con ' + str(err) + ' errori')
plt.imshow(np.reshape(x, [width, height]), cmap='tab20c')
plt.gca().set_axis_off()

n_iters = 200

# Update sincrono
xs = copy.deepcopy(x)
xs = xs.reshape(1, -1)  # mi assicuro che xs sia una matrice unidimensionale con una sola riga

Es = []
for _ in range(n_iters):  # per ogni ciclo si aggiorna ogni neurone
    Es.append(Energy(W, b, xs))
    xs = Update(W, xs, b)

Es_graph = np.array([e[0] for e in Es])

# Update asincrono
xa = np.copy(x)
xa = xa.reshape(1, -1)

Ea = []
for count in range(n_iters):
    node_idx = list(range(N))
    random.shuffle(node_idx)
    for i in node_idx:  # per ogni iterazione scelgo un nodo i-esimo e calcolo l'update per esso
        Ea.append(Energy(W, b, xa))
        ic = xa @ W[:, i] - b[
            i]  # prodotto tra xa e colonna i-esima della matrice pesi meno il bias associato al nodo i-esimo
        xa[0, i] = Thresh(ic)  # assegna il risultato al nodo i-esimo nell'array xa.

Ea_graph = np.array(Ea[:n_iters])

# Grafico dell'andamento dell'energia per ogni iterazione
n_iters_vett = np.arange(0, n_iters)
plt.figure(figsize=(10, 6))
plt.plot(n_iters_vett, Ea_graph, marker='o', label='Energia update asincrono')
plt.plot(n_iters_vett, Es_graph, marker='x', label='Energia update sincrono')
plt.xlabel('Numero di Iterazioni')
plt.ylabel('Energia')
plt.title('Variazione dell\'energia in funzione del numero di iterazioni')
plt.grid(True)
plt.legend()
plt.show()

# Stampo la percentuale per cui x differisce da ogni singola memoria
print('La classe corretta è la ' + str(k))
print('-' * 67)
print('         Update Sincrono                   Update Asincrono')
print('-' * 67)

for idx, (t_sync, t_async) in enumerate(zip(X, X)):
    ds_sync = hamming(xs[0], t_sync) * len(xs[0])
    da_async = hamming(xa[0], t_async) * len(xa[0])

    similarity_sync = ((len(xs[0]) - ds_sync) / len(xs[0])) * 100
    similarity_async = ((len(xa[0]) - da_async) / len(xa[0])) * 100

    formatted_text_sync = format_mem_similarity(idx, similarity_sync, idx == k)
    formatted_text_async = format_mem_similarity(idx, similarity_async, idx == k)

    print(f'{formatted_text_sync} | {formatted_text_async}')

# Grafico la memoria originale, quella modificata e quella ottenuta con i due update
plt.subplot(1, 4, 1)
plt.imshow(np.reshape(X[k], [width, height]), cmap='tab20c')
plt.title('Originale')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(np.reshape(x, [width, height]), cmap='tab20c')
plt.title('Danneggiato')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(np.reshape(xs, [width, height]), cmap='tab20c')
plt.title('Sincrono')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(np.reshape(xa, [width, height]), cmap='tab20c')
plt.title('Asincrono')
plt.axis('off')

plt.tight_layout()
plt.show()