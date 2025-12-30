import numpy as np

# SS: FASE DI INIZIALIZZAZIONE FILTRO
# La fase di inizializzazione ha lo scopo di definire la conoscenza iniziale che abbiamo sul sistema.
# Nell'EKF l'incertezza è modellata da una distribuzione Gaussiana, definita da due elementi della stima iniziale dello stato
# cioè la migliore ipotesi su dove si trova il robot: POSA [x, y, theta] e MATRICE DI COVARIANZA, cioè l'incertezza della stima.
# In questa matrice i valori sulla diagonale quantificano questa incertezza
# (valori alti (gaussiana larga) = incertezza elevata, valori bassi (gaussiana stretta) = siamo abbastanza sicuri)
# AA DEFINIZIONE FUNZIONE INIZIALIZZAZIONE
def initEKF(stima_stato_iniziale, covarianza_iniziale):
    stima_stato = stima_stato_iniziale
    covarianza = covarianza_iniziale
    return stima_stato, covarianza

# SS: FASE DI PREDIZIONE (A PRIORI)
# In questa fase, l'algoritmo proietta in avanti nel tempo la stima corrente dello stato e la sua incertezza,
# basandosi esclusivamente sul modello dinamico (cinematico) del sistema.
#
# 1. PREDIZIONE DELLO STATO A PRIORI
# La stima dello stato \hat{x} viene predetta applicando la funzione di moto non lineare alla stima precedente:
#    \hat{x}_{t|t-1} = f(\hat{x}_{t-1|t-1}, 0)
#
# Utilizziamo il metodo di Eulero assumendo che le velocità (lineare v e angolare omega) rimangano costanti
# nell'intervallo dt ("Constant Velocity Model"). Questo comporta che l'accelerazione è considerata nulla o rumore.
#
# Le equazioni di evoluzione per lo stato [x, y, theta, v, omega] sono:
#    x_{t|t-1}     = x_{t-1|t-1} + v_{t-1} * cos(theta_{t-1|t-1}) * dt  ->  x_pred[0] = x[0] + x[3] * np.cos(x[2]) * dt
#    y_{t|t-1}     = y_{t-1|t-1} + v_{t-1} * sin(theta_{t-1|t-1}) * dt  ->  x_pred[1] = x[1] + x[3] * np.sin(x[2]) * dt
#    theta_{t|t-1} = theta_{t-1|t-1} + omega_{t-1} * dt                 ->  x_pred[2] = x[2] + x[4] * dt
#    v_{t|t-1}     = v_{t-1|t-1}       (Ipotesi velocità costante)      ->  x_pred[3] = x[3]
#    omega_{t|t-1} = omega_{t-1|t-1}   (Ipotesi velocità costante)      ->  x_pred[4] = x[4]
#
# 2. CALCOLO DELLA COVARIANZA A PRIORI
# Per propagare l'incertezza, il modello di stato (che è non lineare a causa di seno e coseno) deve essere linearizzato.
# Si calcola la matrice Jacobiana F_x rispetto allo stato, valutata nel punto corrente (la stima precedente).
#    F_x = \frac{\partial f}{\partial x}\bigg|_{\hat{x}_{t-1|t-1}}
#
# La covarianza dell'errore di stima viene quindi proiettata in avanti usando la formula di propagazione lineare:
#    P_{t|t-1} = F_x * P_{t-1|t-1} * F_x^T + Q_t  ->  P_pred = F @ P @ F.T + Q
#
# Analisi Termini:
# - P_{t-1|t-1} (Matrice Covarianza Stima Passo Precedente):
#   Rappresenta quanto eravamo incerti sulla posizione del robot prima di questa predizione.
#
# - F_x (Jacobiana Dello Stato):
#   Linearizza il modello cinematico e descrive geometricamente come le incertezze sulle componenti dello stato
#   precedente si trasformano e si propagano al nuovo stato.
#   Per esempio, ci dice come un errore sull'angolo theta al tempo t-1 si trasforma in un errore sulla posizione X e Y al tempo t.
#
# - Q_t (Matrice Di Rumore Di Processo):
#   Modella l'incertezza intrinseca del modello di moto stesso. Poiché il "Constant Velocity Model" è una
#   semplificazione della realtà (il robot in realtà accelera, frena, subisce slittamenti), Q_t rappresenta
#   l'errore additivo introdotto da queste discrepanze non modellate.
#   Senza Q_t, la covarianza tenderebbe a diventare troppo piccola (il filtro diventerebbe "troppo sicuro di dove si trova il robot")
#   impedendo di accettare le correzioni future dei sensori.
#
# Analisi Operazioni:
# - (F @ P @ F.T): Proietta l'incertezza precedente nel futuro. Immaginando l'incertezza come un'ellisse,
#   la matrice F la deforma (la stira o la ruota) seguendo la dinamica del robot.
#   Usiamo F e F.T per trasformare correttamente la matrice P.
#
# - (+ Q_t): Aggiunge il "Rumore di Processo". Anche se conoscessimo perfettamente lo stato precedente, il mondo
#   reale introduce errori (ad esempio slittamenti o imperfezioni del terreno) nell'intervallo dt.
#   La matrice Q rappresenta questa incertezza che si accumula ad ogni passo impedendo alla covarianza
#   di diventare piccola e far aumentare troppo la fiducia in se stesso del robot
#
# Al termine di questa fase, otteniamo la stima "a priori" (\hat{x}_{t|t-1}, P_{t|t-1}) pronta per essere corretta.
# Ritorno x_pred e P_pred

# AA DEFINIZIONE FUNZIONE PREDIZIONE
def predict(stato_stimato_ekf, P_stimata_ekf, Q_t, dt):

    # AA Estrazione Dati Per Leggibilità
    x_tm1 = stato_stimato_ekf[0]
    y_tm1 = stato_stimato_ekf[1]
    theta = stato_stimato_ekf[2]
    v = stato_stimato_ekf[3]
    omega = stato_stimato_ekf[4]

    # AA Predizione Posa
    x_pred = np.zeros(5)
    # x_k = x_{k-1} + v * cos(theta) * dt
    # y_k = y_{k-1} + v * sin(theta) * dt
    # theta_k = theta_{k-1} + omega * dt
    # v_k = v_{k-1}
    # omega_k = omega_{k-1}

    x_pred[0] = float(x_tm1 + v * np.cos(theta) * dt)
    x_pred[1] = float(y_tm1 + v * np.sin(theta) * dt)
    x_pred[2] = float(theta + omega * dt)
    x_pred[3] = float(v)
    x_pred[4] = float(omega)

    # AA Normalizzazione angolo theta tra -pi e pi
    # È fondamentale mantenere l'angolo tra -pi e pi. Se il robot fa molti giri su se stesso,
    # theta potrebbe crescere indefinitamente (es. 400 gradi), causando problemi numerici
    # o errori nel calcolo delle differenze angolari successive.
    x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

    # AA Propagazione Covarianza
    # La Matrice Jacobiana È Una Matrice 5x5
    # Inizializzo F come matrice Identità. Questo imposta automaticamente a 1 tutti gli elementi sulla diagonale, infatti
    # ogni variabile di stato dipende da se stessa al passo precedente con coefficiente 1.
    # Per esempio, la derivata di x_t = x_{t-1} +  v * np.cos(theta) * dt è uguale a 1.
    F = np.eye(5)

    # La Jacobiana F serve a linearizzare il modello cinematico di transizione. Ogni elemento F[i, j] è la derivata parziale
    # della funzione f_i rispetto alla variabile di stato x_j.
    #
    # Le equazioni di moto per lo stato [x, y, theta, v, omega] sono:
    # f_0   = x + v * cos(theta) * dt
    # f_1   = y + v * sin(theta) * dt
    # f_2   = theta + omega * dt
    # f_3   = v
    # f_4   = omega
    # Poiché il moto reale è "curvo" (non lineare) e l'EKF lavora con trasformazioni lineari, la Jacobiana calcola la retta
    # tangente alla traiettoria nello stato attuale. Questa operazione "riporta" l'errore delle variabili al passo precedente
    # (es. incertezza su angolo e velocità) in errore sulle nuove variabili (es. incertezza sulla posizione x, y), permettendo
    # di propagare correttamente la covarianza P. La struttura della matrice F è:
    #      Col 0    Col 1      Col 2            Col 3          Col 4
    #      (x)      (y)       (theta)            (v)          (omega)
    #     [ dx/dx   dx/dy    dx/dtheta         dx/dv          dx/domega    ]  Row 0 (x)
    #     [ dy/dx   dy/dy    dy/dtheta         dy/dv          dy/domega    ]  Row 1 (y)
    #     [ dth/dx  dth/dy   dth/dtheta        dth/dv         dth/domega   ]  Row 2 (theta)
    #     [ dv/dx   dv/dy    dv/dtheta         dv/dv          dv/domega    ]  Row 3 (v)
    #     [ dom/dx  dom/dy   dom/dtheta        dom/dv         dom/domega   ]  Row 4 (omega)

    # Derivate Parziali Rispetto A Theta
    F[0, 2] = -v * np.sin(theta) * dt   # dx/dtheta
    F[1, 2] = v * np.cos(theta) * dt    # dy/dtheta

    # Derivate  Parziali Rispetto Alla Velocità
    F[0, 3] = np.cos(theta) * dt        # dx/dv
    F[1, 3] = np.sin(theta) * dt        # dy/dv

    # Derivate Parziali Rispetto A Omega (indice 4)
    F[2, 4] = dt                        # dth/domega

    # Tutte le altre derivate fuori dalla diagonale (es. dx/dy, dv/dtheta) sono 0 perché le variabili non dipendono
    # l'una dall'altra (per esempio x_{t} non dipende da y_{t-1}). Essendo F inizializzata come Matrice Identità,
    # questi valori sono già 0.

    # Propagazione Della Covarianza E Calcolo Covarianza Predetta
    # La formula standard dell'EKF è: P_{t|t-1} = F_t * P_{t-1|t-1} * F_{t}^{T} + Q_
    P_pred = F @ P_stimata_ekf @ F.T + Q_t
    # F_x @ covarianza_ekf @ F_x.T + F_u @ Q_t @ F_u.T
    # # 5. Propagazione della Covarianza
    #     # P_pred = F * P * F^T + Q
    #     P_pred = F @ P @ F.T + Q

    return x_pred, P_pred

# SS: FASE DI CORREZIONE (A POSTERIORI)
# In questa fase, l'algoritmo mette insieme la misura rumorosa (z) che arriva dai sensori con la stima a priori calcolata
# nella fase di predizione. L'obiettivo è quello di correggere la stima predetta riducendo l'incertezza, "bilanciando"
# la fiducia che il robot ha nel modello matematico (predizione) e il dato reale (sensore).
#
# 1. CALCOLO DELLA PROIEZIONE DELLO STATO NELLO SPAZIO DELLA MISURA
# Prima di poter confrontare la nostra predizione con il dato del sensore, dobbiamo convertire lo stato predetto (x_pred)
# nello stesso "formato" del sensore. Ad esempio, se lo stato ha 5 variabili ma il sensore legge solo la posizione x,y,
# dobbiamo estrarre solo x,y dalla predizione.
#       z_pred = H * \hat{x}_{t|t-1}    ->   z_pred = H @ x_pred
#
# 2. CALCOLO DELL'INNOVAZIONE (ERRORE)
# L'innovazione è la differenza tra la misura reale ottenuta dai sensori (z) e la misura che ci saremmo aspettati (z_pred).
#       innovazione = z - z_pred
#
# IMPORTANTE (Normalizzazione Angoli):
# Se la misura include un orientamento theta (come nel caso dei dati provenienti dal topic del LIDAR) potrebbero generarsi errori.
# Per risolvere questo problema, è molto importante normalizzare l'errore su theta nell'intervallo [-pi, pi] usando atan2:
#       innovazione[theta] = atan2(sin(diff), cos(diff))
#
# 3. CALCOLO COVARIANZA DELL'INNOVAZIONE (S)
# Si calcola la covarianza dell'innovazione S, che rappresenta l'incertezza che ci aspettiamo abbia la misura. In pratica,
# sarebbe la somma dell'incertezza accumulata fino all'istante della predizione proiettata all'istante t e del rumore del sensore stesso.
# È la somma dell'incertezza della predizione proiettata nello spazio misura e del rumore intrinseco del sensore.
#       S = H * P_{t|t-1} * H^T + R_t   ->   S = H @ P_pred @ H.T + R
#
# 4. CALCOLO DEL GUADAGNO DI KALMAN (K)
# Il Guadagno di Kalman K serve per minimizzare la covarianza (l'incertezza accumulata fino all'istante corrente)
# dell'errore a posteriori.
# In pratica, è il "peso" (tra 0 e 1) che diamo all'innovazione (errore) per correggere lo stato.
#       K = P_{t|t-1} * H^T * S^{-1}    ->   K = P_pred @ H.T @ np.linalg.inv(S)
#
# 5. AGGIORNAMENTO DELLO STATO E DELLA COVARIANZA (STIMA A POSTERIORI)
# Correggiamo la predizione aggiungendo l'innovazione pesata dal guadagno K.
#    \hat{x}_{t|t} = \hat{x}_{t|t-1} + K * innovazione    ->   x_post = x_pred + K @ innovazione
#
# Poiché abbiamo integrato una nuova informazione (sensore), l'incertezza totale diminuisce.
# Di conseguenza, si deve aggiorniamo la matrice di covarianza che è il nostro "indicatore" di incertezza accumulata fino all'istante t,
# sottraendo la quantità di informazione guadagnata.
#    P_{t|t} = (I - K * H) * P_{t|t-1}          ->   P_post = (np.eye(n) - K @ H) @ P_pred
#
# Analisi Termini:
# - z (Vettore Misura): Sono i dati reali provenienti dallo specifico topic
#
# - H (Matrice di Misura): È una matrice che mappa lo spazio dello stato (5 variabili) nello spazio della misura.
#   Serve a selezionare quali variabili dello stato vengono osservate dal sensore corrente.
#   Ad esempio, per i topic di Odometria Ruote prendiamo solo le velocità (v, omega), quindi la matrice H sarà una matrice
#   con 0 ovunque e 1 in corrispondenza degli indici di v e omega nel vettore di stato e 0 altrove, in modo da prendere solo
#   quelle componenti dai dati ottenuti dalla predizione
#
# - R_t (Matrice di Covarianza del Rumore di Misura): Rappresenta l'incertezza intrinseca del sensore.
#   Valori grandi sulla diagonale di R indicano che il sensore è molto rumoroso/impreciso.
#
# - R_t (Matrice di Covarianza del Rumore di Misura): Rappresenta quanto ci fidiamo del sensore.
#   Dei valori grandi sulla diagonale di R indicano che il sensore è molto rumoroso/impreciso, quindi il filtro tenderà
#   a ignorare la misura e fidarsi di più della predizione
#
# - K (Guadagno di Kalman): Determina di chi ci fidiamo di più.
#   - Se K è alto (~1): Mi fido molto del SENSORE.
#     Succede quando il sensore è preciso (R Bassa) oppure quando la mia predizione era molto incerta (P Alta).
#     Il filtro, si può fidare della misura e correggerà pesantemente la posizione verso la misura z.
#   - Se K è basso (~0): Mi fido molto della PREDIZIONE (Modello).
#     Succede quando il sensore è rumoroso (R Alta) oppure quando sono già molto sicuro della mia posizione (P Bassa).
#     Il filtro può vedere questo errore come rumore e ignorarlo, mantenendo i dati ottenuti dalla predizione

# Analisi Operazioni:
# - (K * innovazione): È il termine di correzione. In pratica, è il vettore che viene sommato alla stima a priori per "spostarla" verso la realtà
# Se la nostra predizione era perfetta o quasi (innovazione ~ 0), e quindi il termine di correzione è nullo (o quasi)
# Di conseguenza, lo stato non cambia (o cambia in modo impercettibile), confermando l'ipotesi della predizione.
# Se la nostra predizione è lontana dalla misura reale, il filtro interviene, correggendo lo stato "spostandolo" per avvicinarlo alla misura.
# Di conseguenza, lo stato cambia in maniera direttamente proporzionale a K:
# - K alto:     Grande correzione verso la misura e quindi mi fido principalmente del sensore
# - K basso:    Piccola correzione verso la misura e quindi rimango vicino allo stato predetto
#
# - (I - K * H): È il fattore di riduzione dell'incertezza. Ogni volta che arriva una misura valida, l'ellisse di incertezza attorno alla
#   posizione del robot si "restringe" rendendo anche la matrice di covarianza (l'incertezza accumulata fino all'istante t) diminuisce e di
#   conseguenza il filtro diventa più sicuro di dove si trova
#   Più alto è K (più mi fido del sensore), più l'incertezza P si riduce drasticamente.
#
# Al termine di questa fase, otteniamo la stima "a posteriori" (\hat{x}_{t|t}, P_{t|t}) che diventa
# il punto di partenza (stato_tm1, P_tm1) per la predizione all'istante successivo.
# return x_post e P_post

# AA DEFINIZIONE FUNZIONE CORREZIONE
def update(x_pred, P_pred, z, H_t, R_t):
    # L’operatore @ è la sintassi per la moltiplicazione matriciale in Python.
    # Sostituisce np.dot(), ma rende il codice più leggibile e coerente con la notazione dell’algebra lineare.
    # A differenza di '*', che esegue una moltiplicazione elemento per elemento*e può attivare il
    # "broadcasting" (ossia l’espansione automatica di un array per adattarlo alle dimensioni dell’altro),
    # @ esegue sempre e solo un prodotto matriciale, impedendo interpretazioni ambigue.
    # Questa modifica riduce il rischio di errori ad esempio quando un’operazione dovrebbe
    # restituire un vettore ma produce invece una matrice

    # Conversione In Array Numpy Per Evitare Errori Di Calcolo
    x_pred = np.array(x_pred)
    z = np.array(z)
    H_t = np.array(H_t)
    R_t = np.array(R_t)

    # La matrice R_t viene costruita usando le covarianze fornite dai messaggi ROS (msg.pose.covariance).
    # Tuttavia, questi valori grezzi possono essere piccoli o grandi. Si deve verificare che abbiano senso.
    # Se il filtro tende a fidarsi eccessivamente (o troppo poco) di un sensore,
    # potrebbe essere necessario sovrascrivere questi valori o applicare delle correzioni.
    #
    # TODO: Valutare l'aggiunta di un termine epsilon sulla diagonale per garantire che R sia sempre invertibile
    #       epsilon = 1
    #       np.fill_diagonal(R_t, R_t.diagonal() + epsilon)

    # AA CALCOLO DELLA PROIEZIONE DELLO STATO NELLO SPAZIO DELLA MISURA
    # Proiettiamo lo stato predetto nello spazio delle osservazioni tramite la matrice H
    # z_pred = H * \hat{x}_{t|t-1}
    z_pred = H_t @ x_pred

    # A CALCOLO DELL'INNOVAZIONE (RESIDUO)
    # Calcoliamo la differenza tra la misura reale e quella predetta (attesa)
    # innovazione = z - z_pred
    innov = z - z_pred

    # AA NORMALIZZAZIONE DEGLI ANGOLI NEL VETTORE DI INNOVAZIONE
    # Se la misura include l'orientamento (theta), potrebbero verificarsi errori. Per evitarlo si deve normalizzare
    # theta in modo che sia compreso tra [-pi, pi].
    # - TOPIC GPS   (len=2): [x, y]                     -> Non Contiene Theta
    # - TOPIC Odom  (len=2): [v, omega]                 -> Non Contiene Theta
    # - TOPIC Lidar (len=5): [x, y, theta, v, omega]    -> Contiene Theta (3 Elemento Vettore Di Stato [Indice 2])
    # - TOPIC Lidar (len=3): [x, y, theta] -            -> Contiene Theta (3 Elemento Vettore Di Stato [Indice 2])
    #
    # Di conseguenza, indipendentemente dal fatto che usiamo il LIDAR per correggere solo la posa o anche le velocità,
    # la presenza dell'angolo è garantita se il vettore di innovazione ha almeno 3 elementi.
    if len(innov) >= 3:
        sinValue_innov = np.sin(innov[2])
        cosValue_innov = np.cos(innov[2])
        innov[2] = np.arctan2(sinValue_innov, cosValue_innov)

    # AA CALCOLO DELLA COVARIANZA DELL'INNOVAZIONE (S)
    # Rappresenta l'incertezza totale attesa nella misura, combinando l'incertezza dello stato e il rumore del sensore.
    # S = H * P_{t|t-1} * H^T + R_t
    S = H_t @ P_pred @ H_t.T + R_t

    # AA CALCOLO DEL GUADAGNO DI KALMAN (K)
    # Il guadagno determina quanto "pesare" l'innovazione per correggere la stima a priori.
    # K = P_{t|t-1} * H^T * S^{-1}

    # TODO: Valutare l'uso della pseudo-inversa per maggiore robustezza numerica
    #       try:
    #           # Calcolo Inversa standard
    #       except np.linalg.LinAlgError:
    #           print("Matrice S singolare. Uso Pseudo-Inversa per il calcolo di K")
    #           K = P_pred @ H_t.T @ np.linalg.pinv(S)

    K = P_pred @ H_t.T @ np.linalg.inv(S)

    # AA AGGIORNAMENTO DELLO STATO E DELLA COVARIANZA (STIMA A POSTERIORI)

    # Calcolo dello Stato A Posteriori
    # Correggiamo la predizione sommando l'innovazione pesata dal guadagno K
    # \hat{x}_{t|t} = \hat{x}_{t|t-1} + K * innovazione
    x_post = x_pred + K @ innov

    # È fondamentale normalizzare nuovamente l'orientamento nel vettore di stato aggiornato.
    # Poiché x_post è il risultato di una somma vettoriale, l'angolo theta potrebbe essere uscito
    # dall'intervallo [-pi, pi], anche se l'innovazione era stata corretta in precedenza.
    sinValue_post = np.sin(x_post[2])
    cosValue_post = np.cos(x_post[2])
    x_post[2] = np.arctan2(sinValue_post, cosValue_post)

    # Creiamo una matrice identità della dimensione dello stato per l'aggiornamento della covarianza
    lenState = len(x_pred)
    I = np.eye(lenState)

    # Calcolo della Covarianza A Posteriori
    # Riduciamo l'incertezza della stima in base all'informazione acquisita
    # P_{t|t} = (I - K * H) * P_{t|t-1}
    P_post = (I - K @ H_t) @ P_pred

    return x_post, P_post