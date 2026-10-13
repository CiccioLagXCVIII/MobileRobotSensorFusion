# 🤖 Fusione Sensoriale EKF per la Stima di Posa di un Robot Mobile: Mobile Robot Localization

Questo progetto propone uno studio metodologico e un'implementazione ingegneristica di un **Filtro di Kalman Esteso (EKF)** per la stima accurata e in tempo reale della posa planare di una piattaforma robotica mobile (UGV). Il lavoro è stato sviluppato per il corso di *Autonomous Robotics* presso l'Università degli Studi di Perugia.

A differenza dei sistemi di navigazione basati su pura odometria o integrazione sincrona, questo lavoro affronta le problematiche di **asincronia multisensoriale**, disallineamento temporale delle frequenze di campionamento e gestione selettiva delle incertezze sensoriali (deriva odometrica, slittamento delle ruote e rumore di misura), realizzando un'architettura modulare ispirata al framework `robot_localization`.

---

## 🔬 La Sfida Metodologica: Fusione Multisensoriale Asincrona e Controllo del Drift

La principale criticità nell'odometria di veicoli terrestri risiede nell'integrazione di sorgenti informative eterogenee caratterizzate da dinamiche di errore profondamente discordanti:

*   **Deriva Cumulativa (Dead Reckoning Drift):** L'odometria calcolata tramite encoder di bordo soffre di errori sistematici e non sistematici dovuti a slittamenti meccanici (*wheel slip*) su terreni irregolari.
*   **Deriva Locale Lidar Odometry:** Algoritmi di scan-matching ad alta frequenza (DLIO) offrono stime locali ad altissima reattività su orientamento e velocità, ma degradano progressivamente sulla traslazione globale a lungo termine.
*   **Frequenze Eterogenee e Disallineamento:** I sensori trasmettono dati a frequenze asimmetriche (da $10\text{ Hz}$ a $100\text{ Hz}$). L'architettura non può dipendere da una barriera di sincronizzazione (*lock-step*), ma deve propagare e correggere lo stato in modalità **event-driven**.

### Strategia di Isotropia e Reiezione Misure
Per prevenire la sovra-stima della confidenza algoritmica e l'insorgere di divergenze numeriche, sono stati adottati i seguenti criteri:
*   **Reiezione Messaggi Duplicati:** Scarto proattivo dei pacchetti con $\Delta t < 0$ o timestamp identico già processato per evitare la contrazione fittizia della matrice di covarianza $P_t$.
*   **Mascheramento Selettivo delle Misure:** Isolamento chirurgico dei gradi di libertà su cui ciascun sensore ha la massima affidabilità metrologica, neutralizzando le componenti instabili tramite l'impostazione di varianze di rumore asintoticamente elevate ($R_{ii} \gg 0$).

---

## 🛠️ Stack Tecnologico e Ambiente di Esecuzione

*   **Linguaggio Core:** Python 3.8+ / NumPy, SciPy
*   **Middleware Robotico:** ROS Noetic Ninjemys (Ubuntu 20.04 LTS / Docker)
*   **Data Ingestion & Playback:** `rosbag` API per la deserializzazione dei flussi di telemetria
*   **Data Visualization & Diagnostica:** Matplotlib per l'analisi comparativa dei residui e delle traiettorie
*   **Piattaforma Target:** Unmanned Ground Vehicle (UGV) *Warthog* operante in ambiente outdoor

---

## 📐 Formulazione Matematica e Modello Cinematico

Il sistema opera su uno spazio di stato quintidimensionale ($n = 5$):

$$x_t = \begin{bmatrix} x & y & \theta & v & \omega \end{bmatrix}^T$$

dove:
*   $(x, y, \theta) \in \mathbb{R}^2 \times [-\pi, \pi]$: Posa globale del centro del robot rispetto al frame cartesiano di riferimento (`map`).
*   $(v, \omega) \in \mathbb{R}^2$: Velocità traslazionale lineare e velocità angolare rispetto al frame solidale del veicolo (`base_link`).

### 1. Modello di Transizione dello Stato (Prediction Phase)
Assumendo un modello cinematico unicycle a velocità costante nell'intervallo temporale discreto $\Delta t$:

$$x_t^- = f(x_{t-1}, \Delta t) = \begin{bmatrix}
x_{t-1} + v_{t-1} \cos(\theta_{t-1}) \Delta t \\
y_{t-1} + v_{t-1} \sin(\theta_{t-1}) \Delta t \\
\text{wrapToPi}\big(\theta_{t-1} + \omega_{t-1} \Delta t\big) \\
v_{t-1} \\
\omega_{t-1}
\end{bmatrix}$$

La linearizzazione locale del moto attorno alla stima precedente è calcolata tramite la **matrice Jacobiana di transizione** $F_t = \left. \frac{\partial f}{\partial x} \right|_{\hat{x}_{t-1}}$:

$$F_t = \begin{bmatrix}
1 & 0 & -v_{t-1} \sin(\theta_{t-1})\Delta t & \cos(\theta_{t-1})\Delta t & 0 \\
0 & 1 &  v_{t-1} \cos(\theta_{t-1})\Delta t & \sin(\theta_{t-1})\Delta t & 0 \\
0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

La propagazione a priori della covarianza d'errore è regolata da:

$$P_t^- = F_t P_{t-1} F_t^T + Q_t$$

con $Q_t \in \mathbb{R}^{5 \times 5}$ matrice di covarianza del rumore di processo, tarata in funzione della flessibilità dinamica concessa al modello continuo.

### 2. Modello di Misura e Aggiornamento (Correction Phase)
All'arrivo di una misura $z_t \in \mathbb{R}^m$, la matrice di osservazione $H_t \in \mathbb{R}^{m \times 5}$ seleziona linearmente le variabili direttamente osservabili dal sensore corrente ($z_t = H_t x_t + v_t$ con $v_t \sim \mathcal{N}(0, R_t)$):

$$\tilde{y}_t = z_t - H_t x_t^- \quad (\text{con normalizzazione dell'innovazione angolare } \tilde{y}_\theta \in [-\pi, \pi])$$

$$S_t = H_t P_t^- H_t^T + R_t$$

$$K_t = P_t^- H_t^T S_t^{-1}$$

$$\hat{x}_t = x_t^- + K_t \tilde{y}_t$$

$$P_t = (I - K_t H_t) P_t^-$$

---

## 📡 Pipeline di Sensing & Tuning delle Covarianze

Il filtro gestisce in maniera asincrona tre flussi sensoriali indipendenti, discriminando rigorosamente quali variabili fondere in base al dominio di affidabilità del dispositivo.

| Sorgente Sensoriale | Topic ROS di Riferimento | Rate Stimato | Componenti Fuse | Varianza di Tuning ($R$) | Motivazione Metodologica & Ingegneristica |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GPS Globale** | `/odometry/gps` | $\approx 10\text{ Hz}$ | $x, y$ | Bassa / Nominale | **Ancoraggio Assoluto**: Elimina la deriva a lungo termine. Orientamento e velocità vengono esclusi per rumore d'antenna e ritardi di fase. |
| **Lidar Odometry (DLIO)** | `/robot/dlio/odom_node/odom` | $\approx 100\text{ Hz}$ | $\theta, v, \omega$ | Nominale ($\theta, v, \omega$)<br>$999.0$ ($x, y$) | **Dinamica Rapida & Heading**: Ottima stima dell'angolo e delle derivate d'assetto. La posizione traslazionale è inibita ($R_{pos}=999$) per non importare il drift dell'algoritmo di scan matching. |
| **Odometria Ruote** | `/warthog_velocity_controller/odom` | $\approx 20\text{ Hz}$ | $v, \omega$ | Nominale ($v, \omega$)<br>$999.0$ ($x, y, \theta$) | **Velocità Istantanee**: Lettura diretta degli encoder di trazione. Le coordinate spaziali e l'angolo vengono rigettati per via dei macro-slittamenti delle ruote del Warthog UGV. |

---

## 🏛️ Architettura del Software e Flusso di Esecuzione

```
                       ┌───────────────────────────────┐
                       │     ROS Bag Dataset (.bag)    │
                       └───────────────┬───────────────┘
                                       │
                         Lettura Asincrona Topics
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  /odometry/gps              /robot/dlio/odom_node       /warthog_velocity...
  [x, y] (~10 Hz)            [θ, v, ω] (~100 Hz)         [v, ω] (~20 Hz)
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     MainSensorFusion.py       │
                       │   - Calcolo Δt & Time Sync    │
                       │   - Reiezione Duplicati       │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │      EKFSensorFusion.py       │
                       │   - Predict (se Δt > 0)       │
                       │   - Dynamic H_t Selection     │
                       │   - State & Covariance Update │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     datiSensorFusion.npz      │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │    AnalisiSensorFusion.py     │
                       │   - Curve Traiettoria 2D      │
                       │   - Analisi Covarianza/Errori │
                       └───────────────────────────────┘
```

### Moduli del Repository:
*   `MainSensorFusion.py`: Engine di orchestrazione. Estrae i messaggi dal bagfile, gestisce il timing asincrono, invoca le fasi dell'algoritmo e serializza i risultati su file compresso `.npz`.
*   `EKFSensorFusion.py`: Core matematico del filtro. Implementa la cinematica non lineare, le Jacobiane simboliche linearizzate, la proiezione di covarianza e le correzioni con normalizzazione d'angolo.
*   `AnalisiSensorFusion.py`: Modulo di diagnostica e benchmark. Rigenera graficamente le traiettorie stimate, traccia l'andamento delle varianze diagonali di $P_t$ ed esegue l'analisi dei residui rispetto ai topic originali.

---

## 🚀 Guida all'Installazione e Riproducibilità

### 1. Prerequisiti di Sistema
- Distribuzione **Ubuntu 20.04 LTS** con **ROS Noetic** installato (Desktop-Full raccomandato).
- Python 3.8+ con le dipendenze scientifiche standard:
  ```bash
  pip install numpy scipy matplotlib rospkg
  ```

### 2. Configurazione dell'Ambiente e Dati
Sourcing dell'ambiente ROS ed organizzazione dello storage locale:
```bash
# Inizializza l'ambiente ROS
source /opt/ros/noetic/setup.bash

# Posiziona il dataset ROSBag nella directory target (ignorata da git)
mkdir -p Dati
cp /percorso/del/tuo/file.bag Dati/
```

### 3. Pipeline di Elaborazione
Esecuzione del ciclo di fusione sensoriale:
```bash
# Esecuzione del filtro e salvataggio delle serie temporali
python3 MainSensorFusion.py
```
Il completamento dello script genera il file compresso binario `datiSensorFusion.npz`.

### 4. Diagnostica Grafica e Validazione
Avvio del motore di plotting per l'estrazione visiva delle metriche:
```bash
python3 AnalisiSensorFusion.py
```

Lo script produce a video e su disco:
1. **Confronto Traiettoria 2D:** Proiezione piana $(x, y)$ tra pura odometria, GPS grezzo e traiettoria stimata dall'EKF.
2. **Benchmark con Filtro di Riferimento:** Valutazione comparativa tra la stima implementata e l'output dell'odometria integrata originaria.
3. **Analisi dello Stato d'Errore:** Profilo temporale dell'incertezza $\sigma_x, \sigma_y, \sigma_\theta$ derivato dagli autovalori della matrice $P_t$.

---

## 📈 Considerazioni Sperimentali e Risultati

La validazione sperimentale evidenzia il comportamento complementare del filtro:

*   **Soppressione del Rumore ad Alta Frequenza:** La traiettoria stimata smorza le oscillazioni casuali del GPS mantenendo una continuità differenziabile grazie all'alta reattività del modulo DLIO ($100\text{ Hz}$).
*   **Contenimento della Deriva:** L'integrazione del GPS azzera la divergenza quadratica tipica della dead reckoning su percorsi estesi, confinando l'incertezza posizionale asintotica all'interno dell'ellissoide di confidenza nominale del ricevitore satellitare.
*   **Stabilità dell'Innovazione:** L'adozione del wrapping dell'angolo $\theta$ previene singolarità o discontinuità artificiali nel calcolo del guadagno di Kalman in prossimità del cambio di quadrante ($\pm \pi$), garantendo stabilità numerica lungo l'intera sequenza di test.

---
*Progetto realizzato per l'esame di Autonomous Robotics - Corso di Laurea in Ingegneria Informatica - Curriculum Robotics*
