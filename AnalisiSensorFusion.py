#!/usr/bin/env python3
import rosbag
import os
import sys
import numpy as np
# Pacchetto Per Gestire Le Trasformazioni Quaternione <-> Eulero
import tf.transformations as tf_trans

import plottingSensorFusion

id_fig = 0

# CASO ================================================= DATI DI TRAIN =================================================

# SS ANALISI FILTRO EKF IMPLEMENTATO SU DATI DI TRAIN

# AA Caricamento Dati Implementazione EKF (Dati Train)
# Tenta Di Caricare Il File .npz Che Contiene I Risultati Della Simulazione EKF Sui Dati Di Train
try:
    dataSF = np.load('datiSensorFusion.npz')
    percorsoGPS_SF = dataSF['percorsoGPS']
    percorsoLidar_SF = dataSF['percorsoLidar']
    percorsoOdometria_SF = dataSF['percorsoOdometria']
    percorsoStimato_SF = dataSF['percorsoStimato']
    orientamentiLidar_SF = dataSF['orientamentiLidar']
    velocitaOdometria_SF = dataSF['velocitaOdometria']
    metricheCovarianza_SF = dataSF['metricheCovarianza']
    print("Dati Fusione (EKF) Su Dati Di Train Caricati Correttamente!")

# Se Il File Non Viene Trovato, Stampa Un Messaggio Di Errore Chiaro E Termina L'Esecuzione.
except FileNotFoundError:
    print("File 'datiSensorFusion.npz' Non Trovato. Eseguire Lo Script \'MainSensorFusion.py\' E Riprovare")
    exit()

# NOTE -----------------------------------------------------------------------------------------------------------------
# AA Plotting Grafici Implemaentazione EKF Su Dati Di Train

# Performance Qualitative Filtro Implementato
# Errore Qualitativo
titolo = "Confronto Traiettorie: GPS, LIDAR, Odometria, Filtro  (Dati Train)"
id_fig += 1
plottingSensorFusion.plotAllTrajectories(percorsoGPS_SF, percorsoLidar_SF, percorsoOdometria_SF, percorsoStimato_SF, titolo, id_fig)
titolo = "Confronto Traiettorie: GPS, Filtro  (Dati Train)"
id_fig += 1
plottingSensorFusion.plotGpsVsFilteredTrajectory(percorsoGPS_SF, percorsoStimato_SF, titolo, id_fig)

# Performance Numeriche Del Filtro
# Errore Quantitativo
titolo = "Analisi Errore di Posizione EKF vs GPS  (Dati Train)"
id_fig += 1
plottingSensorFusion.plotPoseErrorVsGps(percorsoStimato_SF, percorsoGPS_SF, titolo, id_fig)
# Evoluzione Dello Stato
titolo = "Evoluzione Posa Nel Tempo EKF Implementato (Dati Train)"
id_fig += 1
plottingSensorFusion.plotPoseEvolution(percorsoGPS_SF, percorsoStimato_SF, orientamentiLidar_SF, titolo, id_fig)
titolo = "Confronto Velocità: Stima EKF vs Odometria Ruote (Dati Train)"
id_fig += 1
plottingSensorFusion.plotVelocityEvolution(percorsoStimato_SF, velocitaOdometria_SF, titolo, id_fig)
# Analisi Incertezza
titolo = "Metriche Matrice Di Covarianza (Dati Train)"
id_fig += 1
plottingSensorFusion.plotCovarianceMetrics(metricheCovarianza_SF, titolo, id_fig)
titolo = "Deviazione Standard Velocità L'Orientamento (Dati Train)"
id_fig += 1
plottingSensorFusion.plotStandardDeviations(metricheCovarianza_SF, titolo, id_fig)
# NOTE -----------------------------------------------------------------------------------------------------------------
# SS CONFRONTO FILTRO EKF IMPLEMENTATO CON FILTRO ISARLAB

# AA Definizione Variabili Per Dati Confronto

# Percorso ROS Bag Con 4 Topic
NOME_CARTELLA_DATI = 'Dati'
NOME_SOTTOCARTELLA_BAG = 'Rosbag_4_Topic'
NOME_FILE_BAG = 'Rosbag_Warthog_PercorsoVerde_AR_2526-46-26_4_topics_map_drift.bag'
bag_file_confronto = os.path.join(NOME_CARTELLA_DATI, NOME_SOTTOCARTELLA_BAG, NOME_FILE_BAG)

# Definisci il nome del topic che vuoi estrarre
topicEKF = '/isarlab/odometry/ekf_utm'

# percorsoEKF (Risultati Per Confronto)
percorsoEKF = []        # Lista di [t, x_stim, y_stim, theta_stim, v_stim, omega_stim]

# AA Estrazione Dati Topic /isarlab/odometry/ekf_utm (Filtro Di Confronto Su Dati Di Train)
try:
    with rosbag.Bag(bag_file_confronto, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topicEKF]):
            # Estrazione Posizione
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y

            # Estrazione Orientamento
            q = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            (roll, pitch, yaw) = tf_trans.euler_from_quaternion(q)
            theta = yaw

            # Aggiungi i dati alla lista
            percorsoEKF.append([t.to_sec(), x, y, theta])

except Exception as e:
    # Blocco Gestione Eccezioni Che Stampa L'Errore
    print(f"\n\nErrore: {e}")

# Conversione In Array Numpy
percorsoEKF_np = np.array(percorsoEKF)

# Normalizzazione Asse Temporale
# I timestamp della Rosbag sono in secondi a partire dal 1° Gennaio 1970
# Per migliorare la visualizzazione sottraggo il primo istante di tempo (t0)
# dalla prima colonna. In questo modo ogni grafico parte da zero e gli istanti saranno in secondi
t0 = percorsoEKF_np[0, 0]
percorsoEKF_np[:, 0] -= t0

# NOTE -----------------------------------------------------------------------------------------------------------------
# AA Confronto Tra Filtro Implementato E Filtro Isarlab
# Confronto Filtro Implementato Con Filtro Di Riferimento

# Confronto Traiettorie
titolo = "Confronto Traiettorie Dei Due Filtri (Implementazione -  Topic)"
id_fig += 1
plottingSensorFusion.plotMultipleFilterTrajectories(percorsoGPS_SF, percorsoStimato_SF, percorsoEKF_np, titolo, id_fig)

# Analisi Errore Relativo Tra I Due Filtri
titolo = "Errore Tra Due Filtri Diversi (Implementazione -  Topic)"
id_fig += 1
plottingSensorFusion.plotFilterError(percorsoStimato_SF, percorsoEKF_np, titolo, id_fig)

# Confronto Evoluzione Stato Nel Tempo
titolo = "Confronto Evoluzione Posa Nel Tempo Tra I Due Filtri (Implementazione -  Topic)"
id_fig += 1
plottingSensorFusion.plotFiltersPoseComparison(percorsoGPS_SF,orientamentiLidar_SF, percorsoStimato_SF, percorsoEKF_np, titolo, id_fig)
# NOTE -----------------------------------------------------------------------------------------------------------------

# AA Estrazioni Dati Per Analisi Specifica
# Definizione Intervallo
inizio = 100
fine = 150

# Estrazione Dati Implementazione (percorsoGPS_SF)
gpsFilter = (percorsoGPS_SF[:, 0] >= inizio) & (percorsoGPS_SF[:, 0] <= fine)
percorsoGPSFiltered = percorsoGPS_SF[gpsFilter]

# Estrazione Dati Implementazione (percorsoGPS_SF)
thetaFilter = (orientamentiLidar_SF[:, 0] >= inizio) & (orientamentiLidar_SF[:, 0] <= fine)
orientamentiLidarFiltered = orientamentiLidar_SF[thetaFilter]

# Estrazione Dati Implementazione (percorsoStimato_SF)
stimatoFilter = (percorsoStimato_SF[:, 0] >= inizio) & (percorsoStimato_SF[:, 0] <= fine)
percorsoStimatoFiltered = percorsoStimato_SF[stimatoFilter]

# Estrazione Dati Riferimento (percorsoEKF)
ekfFilter = (percorsoEKF_np[:, 0] >= inizio) & (percorsoEKF_np[:, 0] <= fine)
percorsoEKFFiltered = percorsoEKF_np[ekfFilter]

# AA Confronto Specifico Tra Filtro Implementato E Filtro Isarlab
# Confronto Filtro Implementato Con Filtro Di Riferimento

# Confronto Traiettorie
titolo = "Confronto Traiettorie Dei Due Filtri t = [100, 150] (Implementazione -  Topic)"
id_fig += 1
plottingSensorFusion.plotMultipleFilterTrajectories(percorsoGPSFiltered, percorsoStimatoFiltered, percorsoEKFFiltered, titolo, id_fig)

# Analisi Errore Relativo Tra I Due Filtri
titolo = "Errore Tra Due Filtri Diversi t = [100, 150] (Implementazione -  Topic)"
id_fig += 1
plottingSensorFusion.plotFilterError(percorsoStimatoFiltered, percorsoEKFFiltered, titolo, id_fig)

# Confronto Evoluzione Stato Nel Tempo
titolo = "Confronto Evoluzione Posa Nel Tempo Tra I Due Filtri t = [100, 150] (Implementazione -  Topic)"
id_fig += 1
plottingSensorFusion.plotFiltersPoseComparison(percorsoGPSFiltered,orientamentiLidarFiltered, percorsoStimatoFiltered, percorsoEKFFiltered, titolo, id_fig)
# CASO ================================================= DATI DI TEST =================================================

# AA Caricamento Dati Implementazione EKF (Dati Test)

# SS ANALISI FILTRO EKF IMPLEMENTATO SU DATI DI TEST

# Tenta Di Caricare Il File .npz Che Contiene I Risultati Della Simulazione EKF Sui Dati Di Test
try:
    dataDT = np.load('datiSensorFusion_DatiTest.npz')
    percorsoGPS_DT = dataDT['percorsoGPS']
    percorsoLidar_DT = dataDT['percorsoLidar']
    percorsoOdometria_DT = dataDT['percorsoOdometria']
    percorsoStimato_DT = dataDT['percorsoStimato']
    orientamentiLidar_DT = dataDT['orientamentiLidar']
    velocitaOdometria_DT = dataDT['velocitaOdometria']
    metricheCovarianza_DT = dataDT['metricheCovarianza']
    print("Dati Fusione (EKF) Su Dati Di Test Caricati Correttamente!")

# Se Il File Non Viene Trovato, Stampa Un Messaggio Di Errore Chiaro E Termina L'Esecuzione.
except FileNotFoundError:
    print("File 'datiSensorFusion_DatiTest.npz' Non Trovato. Eseguire Lo Script \'SensorFusionDatiTest.py\' E Riprovare")
    exit()

# NOTE -----------------------------------------------------------------------------------------------------------------
# AA Plotting Grafici Implemaentazione EKF Su Dati Di Test
# Performance Qualitative Filtro Implementato
# Errore Qualitativo
titolo = "Confronto Traiettorie: GPS, LIDAR, Odometria, Filtro  (Dati Test)"
id_fig += 1
plottingSensorFusion.plotAllTrajectories(percorsoGPS_DT, percorsoLidar_DT, percorsoOdometria_DT, percorsoStimato_DT, titolo, id_fig)
titolo = "Confronto Traiettorie: GPS, Filtro  (Dati Test)"
id_fig += 1
plottingSensorFusion.plotGpsVsFilteredTrajectory(percorsoGPS_DT, percorsoStimato_DT, titolo, id_fig)

# Performance Numeriche Del Filtro
# Errore Quantitativo
titolo = "Analisi Errore di Posizione EKF vs GPS  (Dati Test)"
id_fig += 1
plottingSensorFusion.plotPoseErrorVsGps(percorsoStimato_DT, percorsoGPS_DT, titolo, id_fig)
# Evoluzione Dello Stato
titolo = "Evoluzione Posa Nel Tempo EKF Implementato (Dati Test)"
id_fig += 1
plottingSensorFusion.plotPoseEvolution(percorsoGPS_DT, percorsoStimato_DT, orientamentiLidar_DT, titolo, id_fig)
titolo = "Confronto Velocità: Stima EKF vs Odometria Ruote (Dati Test)"
id_fig += 1
plottingSensorFusion.plotVelocityEvolution(percorsoStimato_DT, velocitaOdometria_DT, titolo, id_fig)
# Analisi Incertezza
titolo = "Metriche Matrice Di Covarianza (Dati Test)"
id_fig += 1
plottingSensorFusion.plotCovarianceMetrics(metricheCovarianza_DT, titolo, id_fig)
titolo = "Deviazione Standard Velocità L'Orientamento (Dati Test)"
id_fig += 1
plottingSensorFusion.plotStandardDeviations(metricheCovarianza_DT, titolo, id_fig)
# NOTE -----------------------------------------------------------------------------------------------------------------
