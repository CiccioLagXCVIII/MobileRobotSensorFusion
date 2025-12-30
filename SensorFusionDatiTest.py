#!/usr/bin/env python3
import rosbag
import os
import sys
import numpy as np
import EKFSensorFusion as ekf
import tf.transformations as tf_trans
import plottingSensorFusion
import matplotlib.pyplot as plt

# AA CONFIGURAZIONE INIZIALE
NOME_CARTELLA_DATI = 'Dati'
NOME_SOTTOCARTELLA_BAG = 'DatiTest'
NOME_FILE_BAG = 'Rosbag_Warthog_PercorsoVerde_AR_2526-01-22_4_topics_map_drift.bag'
bag_file = os.path.join(NOME_CARTELLA_DATI, NOME_SOTTOCARTELLA_BAG, NOME_FILE_BAG)

topic_fusion = [
    '/odometry/gps',
    '/robot/dlio/odom_node/odom',
    '/warthog_velocity_controller/odom'
]
posa_iniziale = np.array([0.2596, 0.0099, 0.8985, 0.0000, 0.0000])

# AA P_t (Matrice Di Covarianza Dello Stato)
var_x = 0.002
var_y = 0.002
var_theta = 0.001
var_v = 1
var_omega = 0.5
P_iniziale = np.diag([var_x, var_y, var_theta, var_v, var_omega])

# AA Q_t (Matrice Di Rumore Di Processo)
q_var_x = 0.01
q_var_y = 0.01
q_var_theta = 0.01
q_var_v = 1.0
q_var_omega = 1.0
Q_iniziale = np.diag([q_var_x, q_var_y, q_var_theta, q_var_v, q_var_omega])

try:
    # AA Definizione Liste Per Salvare Dati
    percorsoGPS = []
    orientamentiLidar = []
    percorsoLidar = []
    velocitaOdometria = []
    percorsoOdometria = []
    percorsoStimato = []
    metricheCovarianza = []

    with rosbag.Bag(bag_file, 'r') as bag:

        tm1 = None
        stato_tm1 = None
        P_tm1 = None

        for topic, msg, t in bag.read_messages(topics=topic_fusion):

            # AA INIZIALIZZAZIONE VARIABILI EKF CORRENTI
            if tm1 is None:
                tm1 = t
                topic_tm1 = topic
                stato_tm1 = posa_iniziale
                P_tm1 = P_iniziale

                percorsoStimato.append([t.to_sec(), posa_iniziale[0], posa_iniziale[1], posa_iniziale[2], posa_iniziale[3],posa_iniziale[4]])
                percorsoLidar.append([t.to_sec(), posa_iniziale[0], posa_iniziale[1]])
                percorsoOdometria.append([t.to_sec(), posa_iniziale[0], posa_iniziale[1]])

                # SS Calcolo Metriche Covarianza
                trace = np.trace(P_iniziale)
                tracePos = P_iniziale[0, 0] + P_iniziale[1, 1]
                rmse_pos = np.sqrt(tracePos)
                sum_abs = np.sum(np.abs(P_iniziale))
                sum_abs_diag = np.sum(np.abs(np.diag(P_iniziale)))
                diag_index = sum_abs_diag / sum_abs
                devStd_theta = np.sqrt(P_iniziale[2, 2]) * (180 / np.pi)
                devStd_v = np.sqrt(P_iniziale[3, 3])
                devStd_omega = np.sqrt(P_iniziale[4, 4]) * (180 / np.pi)

                metricheCovarianza.append([t.to_sec(), trace, tracePos, rmse_pos, diag_index, devStd_theta, devStd_v, devStd_omega])

                continue

            # AA FASE PREDIZIONE FILTRO EKF
            stato_prior = None
            P_prior = None

            dt = (t - tm1).to_sec()

            # CASO 1: Stesso istante, topic diverso
            if dt == 0 and topic != topic_tm1:
                stato_prior = stato_tm1
                P_prior = P_tm1

            # CASO 2: Stesso Istante Stesso Topic
            elif dt == 0 and topic == topic_tm1:
                continue

            # CASO 3: Istante Successivo Timestamp Aumentato
            elif dt > 0:
                Q_t = Q_iniziale * dt

                stato_prior, P_prior = ekf.predict(stato_tm1, P_tm1, Q_t, dt)

            stato_t = stato_prior
            P_t = P_prior

            # AA FASE CORREZIONE DELL'EKF

            # CASO 1: GPS (/odometry/gps)
            if topic == topic_fusion[0]:
                # NOTE: CORREZIONE POSIZIONE

                x_gps = msg.pose.pose.position.x
                y_gps = msg.pose.pose.position.y
                misura_rumorosa_gps = np.array([x_gps, y_gps])

                percorsoGPS.append([t.to_sec(), x_gps, y_gps])

                H_t_gps = [[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]]

                var_x_gps = msg.pose.covariance[0]
                var_y_gps = msg.pose.covariance[7]

                if var_x_gps < 0.0001: cov_x = 0.1
                if var_y_gps < 0.0001: cov_y = 0.1

                R_t_gps = [[var_x_gps, 0],
                           [0,var_y_gps]]

                stato_post, P_post = ekf.update(stato_prior, P_prior, misura_rumorosa_gps, H_t_gps, R_t_gps)


            # CASO B: LIDAR ODOMETRY (/robot/dlio/odom_node/odom)
            elif topic == topic_fusion[1]:
                # NOTE: CORREZIONE POSA E VELOCITÀ
                
                q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                (roll, pitch, yaw) = tf_trans.euler_from_quaternion(q)

                x_lidar = msg.pose.pose.position.x
                y_lidar = msg.pose.pose.position.y
                theta_lidar = yaw
                v_lidar = msg.twist.twist.linear.x
                omega_lidar = msg.twist.twist.angular.z

                misura_rumorosa_lidar = np.array([x_lidar, y_lidar, theta_lidar, v_lidar, omega_lidar])

                percorsoLidar.append([t.to_sec(), x_lidar, y_lidar])

                orientamentiLidar.append([t.to_sec(), theta_lidar])

                H_t_lidar = [[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]]

                var_x_lidar = msg.pose.covariance[0] + 999
                var_y_lidar = msg.pose.covariance[7] + 999
                var_yaw_lidar = 0.05
                var_v_lin_lidar = 0.1
                var_v_ang_lidar = 0.1

                R_t_lidar = [[var_x_lidar, 0, 0, 0, 0],
                             [0, var_y_lidar, 0, 0, 0],
                             [0, 0, var_yaw_lidar, 0, 0 ],
                             [0, 0, 0, var_v_lin_lidar, 0],
                             [0, 0, 0, 0, var_v_ang_lidar]]

                stato_post, P_post = ekf.update(stato_prior, P_prior, misura_rumorosa_lidar, H_t_lidar, R_t_lidar)


            # CASO C: WHEEL ODOMETRY (/warthog_velocity_controller/odom)
            elif topic == topic_fusion[2]:
                # NOTE: CORREZIONE VELOCITÀ

                v_odom = msg.twist.twist.linear.x
                omega_odom = msg.twist.twist.angular.z

                misura_rumorosa_odom = np.array([v_odom, omega_odom])

                velocitaOdometria.append([t.to_sec(), v_odom, omega_odom])

                x_odom = msg.pose.pose.position.x
                y_odom = msg.pose.pose.position.y
                percorsoOdometria.append([t.to_sec(), x_odom, y_odom])

                H_t_odom = [[0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]]

                var_v_lin_odom = 0.5
                var_v_ang_odom = 0.5

                R_t_odom = [[var_v_lin_odom, 0],
                            [0, var_v_ang_odom]]

                stato_post, P_post = ekf.update(stato_prior, P_prior, misura_rumorosa_odom, H_t_odom, R_t_odom)

            # AA Aggiornamento Stato Istante t
            stato_t = stato_post
            P_t = P_post

            percorsoStimato.append([t.to_sec(), stato_t[0], stato_t[1], stato_t[2], stato_t[3], stato_t[4]])

            # AA Calcolo Metriche Covarianza Istante t
            trace = np.trace(P_t)
            tracePos = P_t[0,0] + P_t[1,1]
            rmse_pos = np.sqrt(tracePos / 2)
            sum_abs = np.sum(np.abs(P_t))
            sum_abs_diag = np.sum(np.abs(np.diag(P_t)))
            diag_index = sum_abs_diag / sum_abs
            devStd_theta = np.sqrt(P_t[2,2]) * (180 / np.pi)
            devStd_v = np.sqrt(P_t[3,3])
            devStd_omega = np.sqrt(P_t[4, 4]) * (180 / np.pi)

            metricheCovarianza.append([t.to_sec(), trace, tracePos, rmse_pos, diag_index, devStd_theta, devStd_v, devStd_omega])

            # AA Aggiornamento Stato Prossima Iterazione
            tm1 = t
            topic_tm1 = topic

            stato_tm1 = stato_t
            P_tm1 = P_t

        percorsoGPS_np = np.array(percorsoGPS)
        percorsoLidar_np = np.array(percorsoLidar)
        percorsoOdometria_np = np.array(percorsoOdometria)
        percorsoStimato_np = np.array(percorsoStimato)
        orientamentiLidar_np = np.array(orientamentiLidar)
        velocitaOdometria_np = np.array(velocitaOdometria)

        metricheCovarianza_np = np.array(metricheCovarianza)

        # Normalizzazione Asse Temporale
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
            'datiSensorFusion_DatiTest.npz',
            percorsoGPS = percorsoGPS_np,
            percorsoLidar = percorsoLidar_np,
            percorsoOdometria = percorsoOdometria_np,
            percorsoStimato = percorsoStimato_np,
            orientamentiLidar = orientamentiLidar_np,
            velocitaOdometria = velocitaOdometria_np,
            metricheCovarianza =  metricheCovarianza_np
        )
        print("Salvataggio Dati (Test) Completato.")

except Exception as e:
    # Blocco Gestione Eccezioni Che Stampa L'Errore
    print(f"\n\nErrore: {e}")