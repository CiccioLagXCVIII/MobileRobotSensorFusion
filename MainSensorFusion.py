#!/usr/bin/env python3
import rosbag
import os
import sys
import numpy as np
# Pacchetto Per Gestire Le Trasformazioni Quaternione <-> Eulero
import tf.transformations as tf_trans

import EKFSensorFusion as ekf
import plottingSensorFusion
import matplotlib.pyplot as plt

# AA CONFIGURAZIONE INIZIALE
# ------------------------------------------------------------------------------
NOME_CARTELLA_DATI = 'Dati'
NOME_SOTTOCARTELLA_BAG = 'Rosbag_3_Topic'
NOME_FILE_BAG = 'Rosbag_Warthog_PercorsoVerde_AR_2526-46-26_3_topics_map_drift.bag'
bag_file = os.path.join(NOME_CARTELLA_DATI, NOME_SOTTOCARTELLA_BAG, NOME_FILE_BAG)

topic_fusion = [
    '/odometry/gps',
    '/robot/dlio/odom_node/odom',
    '/warthog_velocity_controller/odom'
]
# Pose(Posizione / Orientamento) Rispetto A map(header.frame_id)
# Twist(Velocità) Rispetto A base_link(child_frame_id)

# Definizione Vettore Di Stato Iniziale E Covarianza Iniziale
# I Seguenti Valori Numerici Sono Presi Da 'EstrazionePrimoMessaggioBag.py'
# Vettore Di Stato Inziale
posa_iniziale = np.array([0.2596, 0.0099, 0.8985, 0.0000, 0.0000])

# P_t (Matrice Di Covarianza Dello Stato): Rappresenta l'incertezza accumulata fino all'istante t.
# "Quanto non sono sicuro della mia posizione ADESSO?"
# All'inizio è fissa, ma poi cambia continuamente:
# - Cresce quando il robot si muove (Fase di Predizione): muoversi alla cieca aumenta l'incertezza.
# - Diminuisce quando guardiamo i dati dai sensori (Fase di Correzione): osservare il mondo riduce l'incertezza.
#
# Più i valori sulla diagonale della matrice P sono alti, più il filtro "ammette" di non sapere dove si trova.
# - Per Valori BASSI: La curva Gaussiana è stretta e alta. Quindi, sono abbastanza sicuro della mia stima (mi sbaglio di poco).
# - Per Valori ALTI: La curva Gaussiana è larga e bassa (piatta). Non sono per niente sicuro della mia stima (mi posso sbagliare anche di tanto).
#
# La matrice P definisce un'"ellisse di incertezza" attorno alla posizione stimata
# - Per valori ALTI: L'ellisse è molto grande e il robot potrebbe trovarsi in un punto qualsiasi al suo interno.
# - Per valori BASSI: L'ellisse è molto piccola (o potrebbe anche essere un punto) e quindi l'area in cui il robot si può trovarsi è molto ridotta.
# ---------------------------------VALORI OLD 1 ---------------------------------
# var_x = 0.001
# var_y = 0.001
# var_theta = 0.001
# var_v = 0.1
# var_omega = 0.001
# ---------------------------------VALORI NEW ---------------------------------
var_x = 0.002
var_y = 0.002
var_theta = 0.001
var_v = 1
var_omega = 0.5
P_iniziale = np.diag([var_x, var_y, var_theta, var_v, var_omega])

# Q_t (Matrice Di Rumore Di Processo): Rappresenta l'incertezza "nuova" che si aggiunge al sistema ad ogni passo dt a
# causa delle imperfezioni del modello di moto
# Quanto errore commetto in questo intervallo dt assumendo che la velocità sia costante, visto che nella realtà il robot
# subisce accelerazioni, decelerazioni e cambi di direzione?
#
# Poiché il nostro modello assume che accelerazione lineare e angolare siano zero, qualsiasi variazione reale
# di velocità (accelerazione) viene vista dal filtro come "rumore". La matrice Q serve a quantificare questo rumore.
#
# - Valori BASSI per la Posa (x, y, theta): L'errore di posizione in un singolo dt è trascurabile,
#   poiché la posizione è l'integrale della velocità.
# - Valori PIÙ ALTI per le Velocità (v, omega): L'errore di posizione in un singolo dt è più alto,
#   poiché è qui che "agisce" l'accelerazione ignota.

# Quindi, dobbiamo dire al filtro di aspettarsi variazioni significative (alta incertezza) sulla velocità rispetto alla
# previsione del modello dove consideriamo la velocità costante
# ---------------------------------VALORI OLD 1---------------------------------
# q_var_x = 0.01
# q_var_y = 0.01
# q_var_theta = 0.01
# q_var_v = 0.75
# q_var_omega = 0.75
# q_var_v = 1
# q_var_omega = 1
# q_var_v = 0.5
# q_var_omega = 0.5
# ---------------------------------VALORI OLD 2 ---------------------------------
# q_var_x = 0.001
# q_var_y = 0.001
# q_var_theta = 0.001
# q_var_v = 0.75
# q_var_omega = 0.25
# ---------------------------------VALORI NEW ---------------------------------
# Diciamo al filtro che il modello a velocità costante è poco affidabile
# Il robot può accelerare e cambiare direzione più liberamente, questo rende il filtro più incline ad ascoltare i sensori
q_var_x = 0.01
q_var_y = 0.01
q_var_theta = 0.01
q_var_v = 1.0               # Accelerazioni Maggiori
q_var_omega = 1.0           # Cambi Di Velocità Angolari Bruschi
# Incertezza Sul Modello Bassa
Q_iniziale = np.diag([q_var_x, q_var_y, q_var_theta, q_var_v, q_var_omega])

# print(f"Posa Iniziale:\n {posa_iniziale}")
# print(f"Matrice Covarianza Iniziale:\n {P_iniziale}")
# print(f"Matrice Di Rumore Di Processo:\n {Q_iniziale}")

try:
    # AA Definizione Liste Per Salvare Dati
    # percorsoGPS Cresce Solo Quando Entra Nell'If Del GPS (~10Hz)
    percorsoGPS = []          # Lista di [t, x_gps, y_gps]
    # orientamentiLidar e percorsoLidar Crescono Solo Quando Entra Nell'If Del Lidar (~100Hz)
    orientamentiLidar = []      # Lista di [t, theta_lidar]
    percorsoLidar = []          # Lista di [t, x_lidar, y_lidar]
    # velocitaOdometria e percorsoOdometria Cresce Solo Quando Entra Nell'If Dell'Odometria (~20Hz)
    velocitaOdometria = []      # Lista di [t, v_odom, omega_odom]
    percorsoOdometria = []      # Lista di [t, x_odom, y_odom]
    # percorsoStimato Cresce Ad Ogni Iterazione Del For (GPS + Lidar + Odometria = ~130Hz)
    percorsoStimato = []        # Lista di [t, x_stim, y_stim, theta_stim, v_stim, omega_stim]
    # metricheCovarianza Cresce Ogni Volta Che Vengono Eseguite Entrambe Le Fasi (Predizione - Correzione)
    metricheCovarianza = []     # Lista di [t, trace, tracePos, rmse_pos, diag_index, devStd_theta, devStd_v, devStd_omega]

    with rosbag.Bag(bag_file, 'r') as bag:

        tm1 = None
        stato_tm1 = None    # Stato x_{t-1 | t-1}
        P_tm1 = None        # Covarianza P_{t-1 | t-1}

        for topic, msg, t in bag.read_messages(topics=topic_fusion):

            # AA INIZIALIZZAZIONE VARIABILI EKF CORRENTI
            if tm1 is None:
                # Entra In Questo Ciclo Solo Alla Prima IterazioneQuindi Lo Stato E La Covarianza All'Istante
                # Precedente Sono Lo Stato E La Covarianza Iniziali
                tm1 = t
                topic_tm1 = topic
                stato_tm1 = posa_iniziale
                P_tm1 = P_iniziale

                # Aggiungo Lo Stato Corrente (Iniziale) Alla Lista percorsoStimato
                percorsoStimato.append([t.to_sec(), posa_iniziale[0], posa_iniziale[1], posa_iniziale[2], posa_iniziale[3],posa_iniziale[4]])
                percorsoLidar.append([t.to_sec(), posa_iniziale[0], posa_iniziale[1]])
                percorsoOdometria.append([t.to_sec(), posa_iniziale[0], posa_iniziale[1]])

                # Calcolo Metriche Covarianza
                # 1. Traccia Posizione
                # 1.1 Traccia Totale
                trace = np.trace(P_iniziale)
                # 1.2 Traccia Posizione
                tracePos = P_iniziale[0, 0] + P_iniziale[1, 1]
                rmse_pos = np.sqrt(tracePos)
                # 2. Indice Diagonalità
                sum_abs = np.sum(np.abs(P_iniziale))
                sum_abs_diag = np.sum(np.abs(np.diag(P_iniziale)))
                diag_index = sum_abs_diag / sum_abs
                # 3. Dev Std Orientamento
                # Moltiplicare Per (180 / np.pi) Serve Per Convertirlo In Gradi
                devStd_theta = np.sqrt(P_iniziale[2, 2]) * (180 / np.pi)
                # 4. Dev Std Velocità Lineare
                devStd_v = np.sqrt(P_iniziale[3, 3])
                # 5. Dev Std Velocità Angolare
                # Moltiplicare Per (180 / np.pi) Serve Per Convertirlo In Gradi
                devStd_omega = np.sqrt(P_iniziale[4, 4]) * (180 / np.pi)

                # Salvo Le Metriche Per Ogni Istante Nella Lista metricheCovarianza
                metricheCovarianza.append([t.to_sec(), trace, tracePos, rmse_pos, diag_index, devStd_theta, devStd_v, devStd_omega])

                # Vai Alla Prossima Iterazione Dopo Aver Inizializzato
                continue

            # SS Da Qui Parte Il Codice Per L'Iterazione Generica

            # AA FASE PREDIZIONE FILTRO EKF
            # Creazione Variabili Temporanee Iterazione Corrente
            stato_prior = None
            P_prior = None

            # Calcolo dt
            dt = (t - tm1).to_sec()

            # Q_t è il rumore di processo legato al movimento nel tempo.
            # Se dt=0 allora Q_t = 0. La predizione sarebbe P_{k|k-1} = P_{k-1|k-1} + 0, quindi è meglio saltarla

            # CASO 1: Stesso istante, topic diverso             -> Sensor Fusion
            if dt == 0 and topic != topic_tm1:
                # Abbiamo due "occhi" diversi che guardano il robot nello stesso istante.
                # Si salta la fase di predizione perché il robot non si è spostato tra le due misure quindi
                # facciamo direttamente la correzione. In questo modo "miglioriamo" la stima sando entrambe le informazioni.

                # Se il tempo non avanza (stesso timestamp --> dt = 0) lo stato a priori per "l'istante attuale" è esattamente
                # lo stato a posteriori ottenuto dalla correzione "dell'istante precedente"

                # Non C'è Evoluzione Temporale, Quindi Lo Stato Predetto È Lo Stato A Posteriori All'Istante Precedente
                stato_prior = stato_tm1
                P_prior = P_tm1

            # CASO 2: Stesso Istante Stesso Topic              -> Duplicato
            elif dt == 0 and topic == topic_tm1:
                # Se lo stesso sensore ci da due dati nello stesso istante, è un errore di registrazione (duplicato) o è una raffica
                # di dati anomala e quindi andiamo direttamente al messaggio successivo.
                # Facendo la correzione due volte con lo stesso dato, il filtro diventerebbe "troppo sicuro di sé"
                # (la covarianza scende troppo)
                continue

            # CASO 3: Istante Successivo Timestamp Aumentato    -> Predizione Standard
            elif dt > 0:
                # Aggiorno Matrice Del Rumore (Q_t)
                # Q_t scala con il tempo: più tempo passa, più incertezza accumulo
                Q_t = Q_iniziale * dt

                # Calcolo Stato Predetto E Covarianza Predetta
                # Chiamo La Funzione Predizione EKF Che:
                #   Predice Posa Con Modello A Velocità Costante
                #   Propaga La Covarianza Predetta (P_prior)

                # La Predizione Prende In Input Lo Stato All' Istante Precedente (t-1) E Restituisce Lo Stato All'Istante Corrente (t)
                stato_prior, P_prior = ekf.predict(stato_tm1, P_tm1, Q_t, dt)

                # A Questo Punto Lo Stato E La Covarianza All' Istante Corrente Del Filtro Sono Lo Stato E La
                # Covarianza Predetti (Stato E Covarianza A Priori [CONOSCENZA A PRIORI])
                # Che Utilizzeremo Per La Fase Di Correzione

            # Prima Della Correzione, Lo Stato All' Istante Corrente Coincide Con La Predizione
            stato_t = stato_prior
            P_t = P_prior

            # AA FASE CORREZIONE DELL'EKF

            # CASO 1: GPS (/odometry/gps)
            if topic == topic_fusion[0]:
                # NOTE: CORREZIONE POSIZIONE

                # 1. ESTRAZIONE PARAMETRI MISURA
                # Estraggo I Dati Di Posizione Reali Che Derivano Dal GPS
                x_gps = msg.pose.pose.position.x
                y_gps = msg.pose.pose.position.y
                # Costruisco Il Vettore Di Misura Rumorosa [x_gps, y_gps].
                misura_rumorosa_gps = np.array([x_gps, y_gps])

                # Aggiungo I Dati Di Posizione Nella Lista percorsoGPS [t.to_sec(), x_gps, y_gps]
                percorsoGPS.append([t.to_sec(), x_gps, y_gps])

                # 2. COSTRUZIONE MATRICE H_t (Matrice di Misura)
                # Costruisco La Matrice Di Misura H E Siccome Stiamo Considerando Solo
                # La Posizione Nel Piano Sarà 2x5 Perchè Vogliamo Correggere
                # Solo L'Indice 0 E L'Indice 1
                H_t_gps = [[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]]

                # Rispetto A Quella Che Ho Fatto Nell'Altro Assignment
                # Gli Passo Direttamente H Quindi Non Mi Servono I Beacon

                # 3. COSTRUZIONE MATRICE R_t (Covarianza Rumore Misura)
                # Estraggo La Matrice Di Covarianza R_t Che Rappresenta L'Incertezza
                # Per Costruzione In Questo Caso msg.pose.covariance È Una Lista Di 36 Elementi
                # In Questo Caso Per La Posizione Ci Servono I Dati Sulla "Diagonale", Cioè

                # var_x_gps = msg.pose.covariance[0] + 0.5
                # var_y_gps = msg.pose.covariance[7] + 0.5

                # Aumento La Fiducia Nel GPS
                # Varianza bassa significa che quando arriva una misura GPS, la usiamo per correggere la posizione
                var_x_gps = msg.pose.covariance[0]
                var_y_gps = msg.pose.covariance[7]

                if var_x_gps < 0.0001: cov_x = 0.1
                if var_y_gps < 0.0001: cov_y = 0.1

                R_t_gps = [[var_x_gps, 0],
                           [0,var_y_gps]]

                # 4. CHIAMATA ALLA FUNZIONE DI UPDATE
                # Chiamo La Funzione Che Fa La Fase Di Correzione Dell'EKF
                # La Correzione Prende In Input Lo Stato All'Istante Corrente (A Priori) Restituisce Lo Stato All'Istante Corrente (A Posteriori)
                stato_post, P_post = ekf.update(stato_prior, P_prior, misura_rumorosa_gps, H_t_gps, R_t_gps)


            # CASO B: LIDAR ODOMETRY (/robot/dlio/odom_node/odom)
            elif topic == topic_fusion[1]:
                # NOTE: CORREZIONE POSA E VELOCITÀ
                # Il Lidar è fluido e potrebbe driftare meno dell' Odometria Ruote
                # Se si usa la POSIZIONE del Lidar come "verità", e questa drifta rispetto al GPS, il filtro andrà in confusione.
                # Di solito si usano solo le VELOCITÀ e l'ORIENTAMENTO (theta) del Lidar, ignorando la posizione se c'è il GPS

                # 1. ESTRAZIONE PARAMETRI DAL MESSAGGIO
                # Estaggo Il Quaternione Dell'Orientamento
                q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

                # Converto Il Quaternione In Angoli Di Eulero
                # Roll  --> Angolo Attorno X
                # Pitch --> Angolo Attorno Y
                # Yaw   --> Angolo Attorno Z

                # Poiché stiamo lavorando con un robot che si muove in un ambiente planare 2D, il robot ha solo 3 gradi di libertà cinematici:
                # 1.  Muoversi Avanti/Indietro Lungo L'Asse X
                # 2.  Muoversi Lateralmente Lungo L'Asse Y (Se Non Si Considerano Gli Slittamenti È Trascurabile, Ma Li Consideriamo).
                # 3.  Ruotare Su Se Stesso Lungo L'Asse Z (Yaw).
                # Non Può Muoversi Lungo Z (Volare) Nè Ribaltarsi (Roll E Pitch)

                # E Prendo L'Angolo Di Yaw Che È Theta
                (roll, pitch, yaw) = tf_trans.euler_from_quaternion(q)

                # Estraggo I Dati Di Posizione Reali Che Derivano Dal LIDAR
                x_lidar = msg.pose.pose.position.x
                y_lidar = msg.pose.pose.position.y
                # Estraggo Orientamento Del Lidar
                theta_lidar = yaw
                # Estraggo Velocità Lineare Lungo X v_x: msg.twist.twist.linear.x
                v_lidar = msg.twist.twist.linear.x
                # Estraggo Velocità Angolare Omega: msg.twist.twist.linear.z
                omega_lidar = msg.twist.twist.angular.z

                # Possiamo Ignorare:
                #   Velocità Lineare Lungo Y v_y: msg.twist.twist.linear.y. Di Solito Nulla O Rumore (A Meno Che Non Stia Slittando)
                #   Velocità Lineare Lungo Z v_z: msg.twist.twist.linear.z
                #   Velocità Angolare Roll: roll
                #   Velocità Angolare Pitch: pitch

                # Costruisco Il Vettore Di Misura Rumorosa [x_lidar, y_lidar, theta_lidar, v_lidar, omega_lidar]
                misura_rumorosa_lidar = np.array([x_lidar, y_lidar, theta_lidar, v_lidar, omega_lidar])

                # Aggiungo Posizione Alla Lista percorsoLidar [t.to_sec(), x_lidar, y_lidar]
                percorsoLidar.append([t.to_sec(), x_lidar, y_lidar])

                # Aggiungo L'Angolo Yaw Alla Lista orientamentiLidar [t.to_sec(), theta_lidar]
                orientamentiLidar.append([t.to_sec(), theta_lidar])

                # 2. COSTRUZIONE MATRICE H_t (Matrice di Misura)
                # TODO: Se Non Si Vogliono Usare Le Velocità Del Lidar La Matrice H Sarà 3x5 E Il Vettore Di Misura Ha Dimensione 3
                # Costruisco La Matrice Di Misura H E Siccome Stiamo Considerando Tutti I Dati Sarà
                # Una Matrice Identità 5x5 Perchè Vogliamo Correggere Tutti E 5 Gli Indici

                # CASO POSA E VELOCITÀ: MATRICE H (5x5)
                H_t_lidar = [[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]]

                # CASO SOLO POSA: MATRICE H (3x5)
                # H_t_lidar = [[1, 0, 0, 0, 0],
                #              [0, 1, 0, 0, 0],
                #              [0, 0, 1, 0, 0]]

                # 3. COSTRUZIONE MATRICE R_t (Covarianza Rumore Misura)
                # Estraggo La Matrice Di Covarianza R_t Che Rappresenta L'Incertezza
                # Per Costruzione In Questo Caso msg.pose.covariance E msg.twist.covariance Sono Liste Di 36 Elementi
                # In Questo Caso Per Posizione, Yaw, Velocità Ci Servono I Dati
                # var_x_lidar = msg.pose.covariance[0] + 999    -> Errore 1000 Cosi Non Si Fida Per Posizione
                # var_y_lidar = msg.pose.covariance[7] + 999
                # var_yaw_lidar = msg.pose.covariance[35]       -> Errore 57°
                # var_v_lin_lidar = msg.twist.covariance[0]     -> Errore 1 m/s
                # var_v_ang_lidar = msg.twist.covariance[35]    -> 1 deg/s

                # Posizione Del LIDAR Non È Affidabile
                var_x_lidar = msg.pose.covariance[0] + 999
                var_y_lidar = msg.pose.covariance[7] + 999

                # Mi Fido Molto Del LIDAR Per Orientamento E Per Velocità (Lineare E Angolare)

                # # sqrt(0.01) = 0.1 rad = ~5.7 gradi
                # var_yaw_lidar = 0.01        # Errore stimato: ± 5.7 Gradi

                # Fiducia Elevata Nell' Orientamento Del LIDAR
                # sqrt(0.05) = 0.2236 rad = ~12.81 gradi
                var_yaw_lidar = 0.05  # Errore stimato: ± 12.81 Gradi

                #  Fiducia Elevata Nelle Velocità (Lineare E Angolare)
                # sqrt(0.1) = ~0.316
                var_v_lin_lidar = 0.1       # Errore stimato: ± 0.32 m/s
                # sqrt(0.1) = ~0.316 rad/s = ~18.1 deg/s
                var_v_ang_lidar = 0.1       # Errore stimato: ± 18.1 deg/s

                # CASO POSA E VELOCITÀ: MATRICE R (5x5)
                R_t_lidar = [[var_x_lidar, 0, 0, 0, 0],
                             [0, var_y_lidar, 0, 0, 0],
                             [0, 0, var_yaw_lidar, 0, 0 ],
                             [0, 0, 0, var_v_lin_lidar, 0],
                             [0, 0, 0, 0, var_v_ang_lidar]]

                # CASO SOLO POSA: MATRICE R (3x5)
                # R_t_lidar = [[var_x_lidar, 0, 0, 0, 0],
                #              [0, var_y_lidar, 0, 0, 0],
                #              [0, 0, var_yaw_lidar, 0, 0 ]]

                # NOTE: Provare Con La Matrice R_t_lidar Costruita Con I Dati Presi Da msg.pose.covariance E Osservare I Grafici
                #       Poichè il LIDAR ha un drifta (poco) può succedere che la posizione che restituisce sia lontana da quella
                #       del GPS (che usiamo come "verità" in quanto più precisa)
                #       Se la stima finale segue troppo il LIDAR, significa che il filtro si fida troppo di quest'ultimo
                #       Per risolvere, modifichiamo manualmente i valori R_t_lidar[0,0] e R_t_lidar[1,1] (varianza di x e y) in
                #       modo che il filtro segua il LIDAR localmente ma in generale tenga conto anche del GPS

                # 4. CHIAMATA ALLA FUNZIONE DI UPDATE
                # Chiamo La Funzione Che Fa La Fase Di Correzione Dell'EKF
                # La Correzione Prende In Input Lo Stato All'Istante Corrente (A Priori) Restituisce Lo Stato All'Istante Corrente (A Posteriori)

                stato_post, P_post = ekf.update(stato_prior, P_prior, misura_rumorosa_lidar, H_t_lidar, R_t_lidar)


            # CASO C: WHEEL ODOMETRY (/warthog_velocity_controller/odom)
            elif topic == topic_fusion[2]:
                # NOTE: CORREZIONE VELOCITÀ
                # Questo topic drifta non usiamo la posizione ma solo le velocità

                # Non servono conversioni di frame, perché lo stato è definito nel frame del robot. Stessa cosa per la misura.

                # 1. ESTRAZIONE PARAMETRI DAL MESSAGGIO
                # Estraggo Velocità Lineare Lungo X (Rispetto Al Frame Del Robot): msg.twist.twist.linear.x
                v_odom = msg.twist.twist.linear.x
                # Estraggo Velocità Angolare Omega (Rispetto Al Frame Del Robot): msg.twist.twist.linear.z
                omega_odom = msg.twist.twist.angular.z

                # Costruisco Il Vettore Di Misura Rumorosa [t.tosec(), v_odom, omega_odom]
                misura_rumorosa_odom = np.array([v_odom, omega_odom])

                # Aggiungo I Dati Di Velocità Nella Lista velocitaOdometria [t.to_sec(), v_odom, omega_odom]
                velocitaOdometria.append([t.to_sec(), v_odom, omega_odom])

                # Aggiungo Posizione Alla Lista percorsoLidar [t.to_sec(), x_odom, y_odom]
                x_odom = msg.pose.pose.position.x
                y_odom = msg.pose.pose.position.y
                percorsoOdometria.append([t.to_sec(), x_odom, y_odom])

                # 2. COSTRUZIONE MATRICE H_t (Matrice di Misura)
                # Costruisco La Matrice Di Misura H E Siccome Stiamo Considerando Solo
                # La Velocità Lineare E La Velocità Angolare Sarà 2x5 Perchè Vogliamo Correggere
                # Solo L'Indice 3 E L'Indice 4
                H_t_odom = [[0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]]

                # 3. COSTRUZIONE MATRICE R_t (Covarianza Rumore Misura)
                # Estraggo La Matrice Di Covarianza R_t Che Rappresenta L'Incertezza
                # Per Costruzione In Questo Caso msg.twist.covariance È Una Lista Di 36 Elementi
                # In Questo Caso Per Le Velocità Ci Servono I Dati
                # var_v_lin_odom = msg.twist.covariance[0]       # Errore stimato: ± 1 m/s
                # var_v_ang_odom = msg.twist.covariance[35]      # Errore stimato: ±  deg/s

                # Mi Fido Abbastanza Degli Encoder, Ma Devo Considerare Lo Slittamento
                # Siccome Devo Considerare Lo Slittamento Mi Fido Delle Ruote Per Capire
                # Quanto Gira, Ma Mi Fido Di Più Del LIDAR Per Questo

                # # sqrt(0.2) = ~0.447
                # var_v_lin_odom = 0.2  # Errore stimato: ± 0.45 m/s
                # # sqrt(0.5) = ~0.707 rad/s = ~40.5 deg/s
                # var_v_ang_odom = 0.5  # Errore stimato: ± 40.5 deg/s

                # Aumento La Fiducia Nelle Velocità In Modo Che Contribuiscano Significativamente
                # sqrt(0.5) = ~0.7071
                var_v_lin_odom = 0.5 # Errore stimato: ± 0.71 m/s #0.15
                # sqrt(0.5) = ~0.7071 rad/s = ~40.51 deg/s
                var_v_ang_odom = 0.5  # Errore stimato: ± 40.51 deg/s

                R_t_odom = [[var_v_lin_odom, 0],
                            [0, var_v_ang_odom]]

                # 4. CHIAMATA ALLA FUNZIONE DI UPDATE
                # Chiamo La Funzione Che Fa La Fase Di Correzione Dell'EKF
                # La Correzione Prende In Input Lo Stato All'Istante Corrente (A Priori) Restituisce Lo Stato All'Istante Corrente (A Posteriori)
                stato_post, P_post = ekf.update(stato_prior, P_prior, misura_rumorosa_odom, H_t_odom, R_t_odom)

            # A Questo Punto Lo Stato E La Covarianza All' Istante Corrente Del Filtro Sono Lo Stato E La
            # Covarianza Corretti (Stato E Covarianza A Posteriori [CONOSCENZA A POSTERIORI])

            # AA Aggiorno Stato Istante t
            # Dopo La Correzione, Lo Stato All' Istante Corrente Coincide Con La Correzione
            stato_t = stato_post
            P_t = P_post

            # AA SALVATAGGIO RISULTATO
            # Aggiungo Lo Stato Corrente Alla Lista percorsoStimato
            # [t, x, y, theta, v, omega]
            percorsoStimato.append([t.to_sec(), stato_t[0], stato_t[1], stato_t[2], stato_t[3], stato_t[4]])

            # AA Calcolo Metriche Covarianza Istante t
            # 1. Traccia Posizione
            # 1.1 Traccia Totale
            trace = np.trace(P_t)
            # 1.2 Traccia Posizione
            tracePos = P_t[0,0] + P_t[1,1]
            rmse_pos = np.sqrt(tracePos / 2)
            # 2. Indice Diagonalità
            sum_abs = np.sum(np.abs(P_t))
            sum_abs_diag = np.sum(np.abs(np.diag(P_t)))
            diag_index = sum_abs_diag / sum_abs
            # 3. Dev Std Orientamento
            # Moltiplicare Per (180 / np.pi) Serve Per Convertirlo In Gradi
            devStd_theta = np.sqrt(P_t[2,2]) * (180 / np.pi)
            # 4. Dev Std Velocità Lineare
            devStd_v = np.sqrt(P_t[3,3])
            # 5. Dev Std Velocità Angolare
            # Moltiplicare Per (180 / np.pi) Serve Per Convertirlo In Gradi
            devStd_omega = np.sqrt(P_t[4, 4]) * (180 / np.pi)

            # Salvo Le Metriche Per Ogni Istante Nella Lista metricheCovarianza
            metricheCovarianza.append([t.to_sec(), trace, tracePos, rmse_pos, diag_index, devStd_theta, devStd_v, devStd_omega])

            # AA Aggiornamento Stato Prossima Iterazione
            # Lo Stato All' Istante Corrente Diventa Lo Stato All' Istante Precedente Per La Prossima Iterazione
            tm1 = t
            topic_tm1 = topic

            stato_tm1 = stato_t
            P_tm1 = P_t

        # Conversione In Array Numpy
        percorsoGPS_np = np.array(percorsoGPS)
        percorsoLidar_np = np.array(percorsoLidar)
        percorsoOdometria_np = np.array(percorsoOdometria)
        percorsoStimato_np = np.array(percorsoStimato)
        orientamentiLidar_np = np.array(orientamentiLidar)
        velocitaOdometria_np = np.array(velocitaOdometria)

        metricheCovarianza_np = np.array(metricheCovarianza)

        # Normalizzazione Asse Temporale
        # I timestamp della Rosbag sono in secondi a partire dal 1° Gennaio 1970
        # Per migliorare la visualizzazione sottraggo il primo istante di tempo (t0)
        # dalla prima colonna. In questo modo ogni grafico parte da zero e gli istanti saranno in secondi
        t0 = percorsoStimato_np[0, 0]

        percorsoGPS_np[:, 0] -= t0
        percorsoLidar_np[:, 0] -= t0
        percorsoOdometria_np[:, 0] -= t0
        percorsoStimato_np[:, 0] -= t0
        orientamentiLidar_np[:, 0] -= t0
        velocitaOdometria_np[:, 0] -= t0

        metricheCovarianza_np[:, 0] -= t0

        # AA Esportazione Dati
        print("\nSalvataggio Dati Sensor Fusion 'datiSensorFusion.npz'...")
        np.savez(
            'datiSensorFusion.npz',
            percorsoGPS = percorsoGPS_np,
            percorsoLidar = percorsoLidar_np,
            percorsoOdometria = percorsoOdometria_np,
            percorsoStimato = percorsoStimato_np,
            orientamentiLidar = orientamentiLidar_np,
            velocitaOdometria = velocitaOdometria_np,
            metricheCovarianza =  metricheCovarianza_np
        )
        print("Salvataggio Dati (Train) Completato.")

except Exception as e:
    # Blocco Gestione Eccezioni Che Stampa L'Errore
    print(f"\n\nErrore: {e}")