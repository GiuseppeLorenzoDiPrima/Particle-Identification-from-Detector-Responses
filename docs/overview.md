# Panoramica del progetto

## Contesto fisico

### Particle Identification (PID)

La **Particle Identification** (identificazione di particelle) è uno dei problemi fondamentali della fisica sperimentale delle particelle. In un esperimento di scattering (urto) tra particelle, le sotto-particelle prodotte dalla collisione attraversano una serie di rivelatori che ne misurano proprietà fisiche. Il compito del sistema è determinare con precisione la **specie di ogni particella** a partire da queste misurazioni.

Questo progetto si concentra sull'identificazione di 4 specie di particelle cariche prodotte in scattering inelastico elettrone-protone:

| Particella | Simbolo | PDG ID | Carica |
|---|---|---|---|
| Positrone | e⁺ | −11 | +1 |
| Pione positivo | π⁺ | 211 | +1 |
| Kaone positivo | K⁺ | 321 | +1 |
| Protone | p | 2212 | +1 |

### I rivelatori simulati

Il dataset contiene le risposte simulate con metodo **Monte Carlo** di 6 rivelatori. Ogni evento (particella prodotta) è descritto da 6 variabili:

| Feature | Simbolo | Descrizione fisica |
|---|---|---|
| `p` | $p$ | Momento della particella [GeV/c] |
| `theta` | $\theta$ | Angolo polare rispetto al fascio [rad] |
| `beta` | $\beta$ | Velocità ridotta $v/c$ (da Time-of-Flight) |
| `nphe` | $n_{phe}$ | Numero di fotoelettroni rivelati (rivelatore Cherenkov) |
| `ein` | $E_{in}$ | Energia depositata nel calorimetro in ingresso [GeV] |
| `eout` | $E_{out}$ | Energia depositata nel calorimetro in uscita [GeV] |

#### Diagramma di Bethe-Bloch

La relazione tra l'energia depositata e il momento è descritta dalla **formula di Bethe-Bloch**, che fornisce il principale discriminante tra particelle di massa diversa. A parità di momento, una particella più pesante ha velocità $\beta$ minore e deposita più energia nel materiale. Questo effetto genera le caratteristiche "bande" nel piano $E_{dep}$ vs $p$ che consentono la separazione delle specie.

### Il dataset

- **Fonte:** [Kaggle — Particle Identification from Detector Responses](https://www.kaggle.com/datasets/naharrison/particle-identification-from-detector-responses)
- **Autore del dataset:** Harrison Nathan
- **Dimensione:** ~5 milioni di eventi
- **Formato:** CSV (`pid-5M.csv`)
- **Classe target:** Colonna `id` (PDG ID numerico, mappato in nome fisico)

---

## Obiettivi del progetto

Il progetto confronta **tre approcci** all'identificazione di particelle, in ordine crescente di complessità:

### 1. Metodo tradizionale a tagli (Fase 2)

Il metodo classico usato in fisica sperimentale. Si definiscono **intervalli di accettazione** (tagli) nelle variabili del rivelatore per ogni specie. Una particella viene assegnata alla classe per cui il maggior numero di feature rientra nell'intervallo atteso.

- **Vantaggio:** Interpretabile, nessuna "black box"
- **Svantaggio:** Non sfrutta correlazioni tra variabili, limiti arbitrari

### 2. Machine Learning classico (Fase 3)

Cinque modelli di ML supervisionato, tutti addestrati sulle stesse feature:

- **Logistic Regression:** modello lineare come riferimento
- **K-Nearest Neighbors:** classificazione per prossimità
- **Decision Tree:** regole interpretabili a struttura ad albero
- **Random Forest:** ensemble di alberi con bagging
- **XGBoost:** gradient boosting con regolarizzazione

Per robustezza sperimentale, tutti i modelli sono stati anche valutati con **cross-validation stratificata a 5-fold**.

### 3. Deep Learning (Fase 4)

Una **Multi-Layer Perceptron (MLP)** implementata in PyTorch con:
- Architettura configurabile
- Batch Normalization per stabilità del training
- Dropout per regolarizzazione
- Early stopping per prevenire overfitting
- Supporto CUDA per accelerazione GPU

---

## Analisi avanzate

### Interpretabilità — SHAP (Fase 5a)

L'analisi **SHAP (SHapley Additive exPlanations)** quantifica il contributo di ogni feature alla predizione di ogni singolo evento. Consente di rispondere a domande come: *"Perché il modello ha classificato questo evento come un kaone?"*

- **TreeExplainer** per i modelli ad albero (RF, XGBoost, DT): esatto e veloce
- **KernelExplainer** per la MLP: basato su perturbazioni campionate

### Uncertainty Quantification — MC Dropout (Fase 5b)

L'**MC Dropout** (Monte Carlo Dropout) è una tecnica per stimare l'**incertezza epistemica** di una rete neurale senza modificarne l'architettura. Durante l'inferenza, il dropout viene mantenuto attivo e si eseguono $N$ forward pass indipendenti. La **varianza delle predizioni** misura quanto il modello è incerto su un dato evento.

L'**entropia predittiva** $H = -\sum_c p_c \log p_c$ quantifica l'incertezza globale sulla predizione:
- Bassa entropia → il modello è sicuro
- Alta entropia → il modello è incerto (evento in zona di sovrapposizione tra classi)

La **rejection curve** mostra come l'accuracy aumenta al diminuire della percentuale di eventi accettati (rifiutando quelli ad alta incertezza).

---

## Architettura del codice

Il progetto segue una struttura **flat e modulare**. Ogni pacchetto ha una responsabilità ben definita:

```
main.py              ← Orchestrazione delle 6 fasi
config/              ← Configurazione centralizzata YAML
data_classes/        ← I/O e preprocessing
models/              ← Implementazione dei modelli
utils/               ← Valutazione, interpretabilità, uncertainty
plot/                ← Visualizzazioni
```

Il flusso di dati segue una singola **pipeline lineare**: i dati vengono caricati una volta nella Fase 1 e passati come dizionario Python (`data`) attraverso tutte le fasi successive. I risultati di ogni modello vengono accumulati nel dizionario `all_results`, poi usato nella Fase 6 per la valutazione comparativa.
