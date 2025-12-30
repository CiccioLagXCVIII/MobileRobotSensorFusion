
# Mobile Robot Localization - Autonomous Robotics
Questo repository contiene l'implementazione di un **Extended Kalman Filter (EKF)** per la localizzazione di un robot mobile planare, sviluppato come progetto per il corso di *Autonomous Robotics* (Università degli Studi di Perugia).

**Studente:** Francesco Lo Verde  
**Anno Accademico:** 2025-2026

---

## 1. Modello di Sistema
Il sistema stima la posa del robot utilizzando uno stato a 5 dimensioni:
$$x_t = [x, y, \theta, v, \omega]^T$$
- $(x, y, \theta)$: Posa nel frame `map`.
- $(v, \omega)$: Velocità lineare e angolare nel frame `robot`.

### Caratteristiche principali:
- **Modello cinematico:** A velocità costante durante l'intervallo $\Delta t$.
- **Linearizzazione:** Calcolo della matrice Jacobiana $F_t$ ad ogni passo di predizione.
- **Normalizzazione:** L'angolo $\theta$ viene mantenuto nell'intervallo $[-\pi, \pi]$.

## 2. Strategia di Integrazione Sensori
L'architettura segue un **approccio asincrono** (ispirato a `robot_localization`). Il filtro non attende tutti i sensori, ma aggiorna lo stato non appena riceve un messaggio, gestendo frequenze diverse:
- **GPS:** $\approx 10$ Hz
- **LIDAR:** $\approx 100$ Hz
- **Odometria:** $\approx 20$ Hz

### Logica di aggiornamento:
- **$\Delta t > 0$:** Esegue la fase di **Predizione** e poi la Correzione.
- **$\Delta t = 0$:** Salta la predizione (stesso timestamp) ed esegue direttamente la fusione delle misure.
- **Messaggi duplicati:** Vengono scartati per evitare sovra-stima della confidenza (riduzione artificiale della covarianza).

## 3. Configurazione Sensori (Tuning)
Il filtro è stato tarato per sfruttare i punti di forza di ogni sorgente dati:

1. **GPS (`/odometry/gps`):** Utilizzato esclusivamente per la **Posizione Assoluta (x, y)** per contrastare il drift sul lungo periodo.
2. **Lidar Odometry (`/robot/dlio/odom_node/odom`):** Utilizzato per **Orientamento ($\theta$) e Velocità ($v, \omega$)**. La posizione è ignorata (varianza impostata a 999) a causa del drift locale.
3. **Odometria Ruote (`/warthog_velocity_controller/odom`):** Utilizzata solo per le **Velocità ($v, \omega$)**, ignorando la posizione per via degli slittamenti.

---

## 4. Struttura del Progetto
Il progetto è organizzato come segue:

```text
MobileRobotSensorFusion/
├── MainSensorFusion.py       # Script principale (gestione ROS e ciclo EKF)
├── EKFSensorFusion.py        # Modulo contenente le funzioni core del filtro
├── AnalisiSensorFusion.py    # Script per la visualizzazione grafica dei risultati
├── .gitignore                # Esclusioni (Dati, PDF, cache)
└── README.md                 # Documentazione del progetto
```

---

## 5. Guida per l'Esecuzione

### Requisiti
- **ROS Noetic** installato (o ambiente Docker compatibile).
- Pacchetto `rosbag` per la lettura dei dati.

### Istruzioni
1. Assicurarsi di aver "richiamato" l'ambiente ROS nel terminale:
   ```bash
   source /opt/ros/noetic/setup.bash
   ```
2. Posizionare i file `.bag` nella cartella `Dati/` (come indicato nello script).
3. Eseguire lo script principale per processare i dati:
   ```bash
   python3 MainSensorFusion.py
   ```
4. Una volta generato il file `datiSensorFusion.npz`, visualizzare i grafici:
   ```bash
   python3 AnalisiSensorFusion.py
   ```

### Analisi Grafica
Lo script di analisi produce i seguenti confronti:
- Performance del filtro EKF implementato.
- Confronto tra il filtro custom e il filtro presente nel topic originale.
- Validazione del filtro su dati di test.

---

### Note sul file .gitignore
Per mantenere la repository pulita, sono stati esclusi:
- La cartella pesante `Dati/` contenente i file `.bag`.
- Le cartelle di cache di Python (`__pycache__`).
- Il report in formato PDF (i cui contenuti sono riassunti in questo README).
