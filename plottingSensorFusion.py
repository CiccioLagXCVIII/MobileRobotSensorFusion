import matplotlib.pyplot as plt
import numpy as np


# SS ANALISI FILTRO IMPLEMENTATO
# AA # Confronta Traiettorie Dei Sensori Con La Stima Del Filtro
def plotAllTrajectories(percorsoGPS, percorsoLidar, percorsoOdometria, percorsoStimato, titolo, id_fig):
    fig, ax = plt.subplots(figsize=(14, 9), num=id_fig)
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # SS GPS
    # Percorso GPS
    ax.scatter(percorsoGPS[:, 1], percorsoGPS[:, 2], label='Percorso GPS (Reale)', color='orange', s=5, alpha=0.5)
    # Fine
    ax.plot(percorsoGPS[-1, 1], percorsoGPS[-1, 2], marker='p', markersize=12, markerfacecolor='orange', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (GPS)')

    # SS LIDAR
    # Percorso GPS
    ax.scatter(percorsoLidar[:, 1], percorsoLidar[:, 2], label='Percorso LIDAR', color='green', s=5, alpha=0.5)
    # Fine
    ax.plot(percorsoLidar[-1, 1], percorsoLidar[-1, 2], marker='s', markersize=12, markerfacecolor='green', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (LIDAR)')

    # SS ODOMETRIA
    # Percorso GPS
    ax.scatter(percorsoOdometria[:, 1], percorsoOdometria[:, 2], label='Percorso Odometria', color='red', s=5,alpha=0.5)
    # Fine
    ax.plot(percorsoOdometria[-1, 1], percorsoOdometria[-1, 2], marker='^', markersize=12, markerfacecolor='red', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (Odometria)')

    # SS EKF
    # Percorso EKF
    ax.plot(percorsoStimato[:, 1], percorsoStimato[:, 2], color='blue', linewidth=2, label='Percorso Filtro EKF (Stimato)')
    # Fine
    ax.plot(percorsoStimato[-1, 1], percorsoStimato[-1, 2], marker='d', markersize=12, markerfacecolor='blue', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (Stimato)')

    # SS Assi
    ax.set_title("Traiettoria Robot (Piano XY)", fontsize=12)
    ax.set_facecolor('#f0f0f0')
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.set_aspect('equal')
    # ax.set_xlim(-5, 90)
    # ax.set_ylim(-5, 160)
    ax.legend(loc='best', fontsize=9)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()


# AA Confronto Tra La Traiettoria GPS E Quella Stimata Dal Filtro
def plotGpsVsFilteredTrajectory(percorsoGPS, percorsoStimato, titolo, id_fig):
    fig, ax = plt.subplots(figsize=(14, 9), num=id_fig)
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # SS GPS
    # Percorso GPS
    ax.scatter(percorsoGPS[:, 1], percorsoGPS[:, 2], label='Percorso GPS (Reale)', color='orange', s=5, alpha=0.5)
    # Fine
    ax.plot(percorsoGPS[-1, 1], percorsoGPS[-1, 2], marker='p', markersize=12, markerfacecolor='orange', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (GPS)')

    # SS EKF
    # Percorso EKF
    ax.plot(percorsoStimato[:, 1], percorsoStimato[:, 2], color='blue', linewidth=2, label='Percorso Filtro EKF (Stimato)')
    # Fine
    ax.plot(percorsoStimato[-1, 1], percorsoStimato[-1, 2], marker='d', markersize=12, markerfacecolor='blue', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (Stimato)')

    # SS Assi
    ax.set_title("Traiettoria Robot (Piano XY)", fontsize=12)
    ax.set_facecolor('#f0f0f0')
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    # ax.set_aspect('equal')
    # ax.set_xlim(-5, 50)
    # ax.set_ylim(-5, 160)

    ax.legend(loc='best', fontsize=9)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()


# AA Calcola Errore Di Posizione Euclideo Tra Stima Filtro E GPS
def plotPoseErrorVsGps(percorsoStimato, percorsoGPS, titolo, id_fig):
    # Estraggo Timestamp E Posizioni Stimato
    t_stimato = percorsoStimato[:, 0]
    pos_stimata = percorsoStimato[:, 1:3]

    # Estraggo Timestamp E Posizioni GPS
    t_gps = percorsoGPS[:, 0]
    pos_gps = percorsoGPS[:, 1:3]

    # Per calcolare l'errore, le stime del filtro (~130Hz) devono essere confrontate con le misure GPS (~10Hz) agli stessi istanti t
    # La funzione np.interp "stima" i valori (x, y) di percorsoStimato negli istanti t  esatti in cui sono disponibili le misure GPS
    # Dato che np.interp opera su array 1D, devono essere calcolate le coordinate in modo indipendente per ciascuna coordinata spaziale.
    x_stimato_calc = np.interp(t_gps, t_stimato, pos_stimata[:, 0])
    y_stimato_calc = np.interp(t_gps, t_stimato, pos_stimata[:, 1])

    # Calcolo Distanza Euclidea
    distEuclidea = np.sqrt((x_stimato_calc - pos_gps[:, 0]) ** 2 + (y_stimato_calc - pos_gps[:, 1]) ** 2)

    # Calcolo Statistiche Errore
    erroreMedio = np.mean(distEuclidea)
    stdDevErrore = np.std(distEuclidea)

    plt.style.use('seaborn-v0_8-colorblind')

    fig, ax = plt.subplots(figsize=(14, 9), num=id_fig)
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(titolo, fontsize=13, fontweight='bold')
    ax.set_title(f"Errore Euclideo nel Tempo (Media: {erroreMedio:.2f} m)", fontsize=12)
    ax.set_xlabel("t [s]", fontsize=12)
    ax.set_ylabel("Errore [m]", fontsize=12)
    ax.tick_params(labelsize=10)

    # SS Plot Errore
    ax.plot(t_gps, distEuclidea, color='purple', linewidth=1.5, label='Errore Euclideo')

    # SS Plot Linee Riferimento
    ax.axhline(y=erroreMedio, color='red', linestyle='--', linewidth=2, label=f'Errore Medio: {erroreMedio:.2f} m')
    ax.axhline(y=erroreMedio + stdDevErrore, color='green', linestyle=':', linewidth=2, label=f'Dev. Std: {stdDevErrore:.2f} m')
    ax.axhline(y=max(0, erroreMedio - stdDevErrore), color='blue', linestyle=':', linewidth=2)

    ax.set_xlim(0, t_gps[-1])
    ax.set_ylim(0)
    ax.legend(loc='best', fontsize=10)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()


# AA Evoluzione Delle Componenti Di Posa (x, y, theta) Nel Tempo
def plotPoseEvolution(percorsoGPS, percorsoStimato, orientamentiLidar, titolo, id_fig):
    plt.style.use('seaborn-v0_8-colorblind')

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), num=id_fig)
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # NOTE: [0] X vs Tempo
    ax1 = axes[0]
    ax1.set_facecolor('#f0f0f0')
    ax1.set_title("Confronto Evoluzione Coordinata X Nel Tempo", fontsize=12)

    # SS GPS
    # Percorso GPS
    ax1.scatter(percorsoGPS[:, 0], percorsoGPS[:, 1], color='orange', s=10, alpha=0.75, marker='p', linewidths=0.5, label='Percorso GPS (Reale)')

    # SS EKF
    # Percorso EKF
    ax1.plot(percorsoStimato[:, 0], percorsoStimato[:, 1], color='blue', linewidth=2, linestyle='-',label='Percorso Stimato (Filtro)')

    # SS Assi
    ax1.set_xlabel("t [s]", fontsize=12)
    ax1.set_ylabel("X [m]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best', fontsize=10)
    ax1.tick_params(labelsize=10)

    # NOTE: [1] Y vs Tempo
    ax2 = axes[1]
    ax2.set_facecolor('#f0f0f0')
    ax2.set_title("Confronto Evoluzione Coordinata Y Nel Tempo", fontsize=12)

    # SS GPS
    # Percorso GPS
    ax2.scatter(percorsoGPS[:, 0], percorsoGPS[:, 2], color='orange', s=8, alpha=0.75, marker='p', linewidths=0.75, label='Percorso GPS (Reale)')

    # SS EKF
    # Percorso EKF
    ax2.plot(percorsoStimato[:, 0], percorsoStimato[:, 2], color='blue', linewidth=2, linestyle='-', label='Percorso Stimato (Filtro)')

    # SS Assi
    ax2.set_xlabel("t [s]", fontsize=12)
    ax2.set_ylabel("Y [m]", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best', fontsize=9)

    # NOTE: [2] Theta vs Tempo
    ax3 = axes[2]
    ax3.set_facecolor('#f0f0f0')
    ax3.set_title("Confronto Evoluzione Orientamento Theta Nel Tempo", fontsize=12)

    # SS LIDAR
    # Orientamento LIDAR
    ax3.scatter(orientamentiLidar[:, 0], orientamentiLidar[:, 1], color='orange', s=8, alpha=0.75, marker='p', linewidths=0.5, label='Orientamento (LIDAR)')
    # Orientamento EKF
    ax3.plot(percorsoStimato[:, 0], percorsoStimato[:, 3], color='blue', linewidth=1.5, linestyle='-', label='Orientamento (EKF)')

    # SS Assi
    ax3.set_xlabel("t [s]", fontsize=12)
    ax3.set_ylabel("Theta [rad]", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='best', fontsize=9)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()


# AA Evoluzione Velocità (Lineare E Angolare) Nel Tempo
def plotVelocityEvolution(percorsoStimato, velocitaOdometria, titolo, id_fig):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), num=id_fig)
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # NOTE: [0] Velocità Lineare vs Tempo
    ax1 = axes[0]
    ax1.set_facecolor('#f0f0f0')
    ax1.set_title("Velocità Lineare (v)", fontsize=12)
    ax1.set_ylabel("Velocità [m/s]", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # SS Odometria
    ax1.scatter(velocitaOdometria[:, 0], velocitaOdometria[:, 1], label='Odometria Ruote (Misura)', color='green', s=10,alpha=0.6)
    # SS Stimata
    ax1.plot(percorsoStimato[:, 0], percorsoStimato[:, 4], color='red', linewidth=2, label='Filtro EKF (Stima)')

    ax1.legend(loc='best', fontsize=10)

    # NOTE: [0] Velocità Angolare vs Tempo
    ax2 = axes[1]
    ax2.set_facecolor('#f0f0f0')
    ax2.set_title("Velocità Angolare (ω)", fontsize=12)
    ax2.set_xlabel("t [s]", fontsize=10)
    ax2.set_ylabel("Velocità Angolare [rad/s]", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    ## SS Odometria
    ax2.scatter(velocitaOdometria[:, 0], velocitaOdometria[:, 2], label='Odometria Ruote (Misura)', color='green', s=10, alpha=0.6)
    # SS Stimata
    ax2.plot(percorsoStimato[:, 0], percorsoStimato[:, 5], color='red', linewidth=2, label='Filtro EKF (Stima)')

    ax2.legend(loc='best', fontsize=10)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()


## AA Metriche Della Matrice Di Covarianza: Traccia e Indice di Diagonalità.
def plotCovarianceMetrics(metricheCovarianza, titolo, id_fig):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), num=id_fig)
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # NOTE: [0] Variazione Traccia Covarianza Posizione Nel Tempo
    # La Traccia Indica L'Incertezza Totale Istante Per Istante
    ax1 = axes[0]
    ax1.set_facecolor('#f0f0f0')
    ax1.set_title("Variazione Traccia Covarianza Posizione Nel Tempo", fontsize=12)
    # SS Traccia Covarianza Posizione
    ax1.plot(metricheCovarianza[:, 0], metricheCovarianza[:, 2], color='purple', linewidth=1, linestyle='-', label='Traccia Covarianza Posizione')
    # RSE Posizione
    # Usando RMSE (Radice Media) Facciamo La Media E Quindi Avremo La Media Dell'Errore Sul Singolo Asse
    # Usando RSE (Radice Traccia) Avremo La Distana (Euclidea) E Quindi L'Errore Totale
    # ax1.plot(metricheCovarianza[:, 0], metricheCovarianza[:, 3], color='green', linewidth=2, linestyle='-', label='RSE Posizione')

    # SS Assi
    ax1.set_xlabel("t [s]", fontsize=10)
    ax1.set_ylabel("Traccia Covarianza Posizione", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    yLimit_Trace = np.min(metricheCovarianza[:, 2])
    xLimit_Max = np.max(metricheCovarianza[:, 0])
    ax1.set_xlim(0, xLimit_Max)
    ax1.set_ylim(bottom=yLimit_Trace)
    ax1.legend(loc='best', fontsize=10)

    # NOTE: [1] Variazione Indice Diagonalità Nel Tempo
    # L'Indice di Diagonalità Indica Il Grado Di Indipendenza (Disaccoppiamento) Tra Le Variabili Di Stato Istante Per Istante
    # \frac{\sum |P_{ii}|}{\sum |P_{ij}|}
    ax2 = axes[1]
    ax2.set_facecolor('#f0f0f0')
    ax2.set_title("Variazione Indice Diagonalità P_t Nel Tempo", fontsize=12)
    # SS Indice Diagonalità
    ax2.plot(metricheCovarianza[:, 0], metricheCovarianza[:, 4], color='red', linewidth=1, linestyle='-', label='Indice Diagonalità')
    # Linea Orizzontale A Y = 1 (Idealità)
    ax2.plot([metricheCovarianza[0, 0], metricheCovarianza[-1, 0]], [1.0, 1.0], color='green', linewidth=1, linestyle='--', label='Ideale (Indice = 1)')

    # SS Assi
    ax2.set_xlabel("t [s]", fontsize=10)
    ax2.set_ylabel("Indice Diagonalità", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    yLimit_Diag = np.min(metricheCovarianza[:, 4])
    xLimit_Max = np.max(metricheCovarianza[:, 0])
    ax2.set_xlim(0, xLimit_Max)
    ax2.set_ylim(bottom=yLimit_Diag)
    ax2.legend(loc='best', fontsize=10)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()


## AA Deviazione Standard Per Le Velocità  (Lineare E Angolare) E L'Orientamento
def plotStandardDeviations(metricheCovarianza, titolo, id_fig):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), num=id_fig)
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # NOTE: [0] Deviazione Standard Orientamento (Indice 5)
    ax1 = axes[0]
    ax1.set_facecolor('#f0f0f0')
    ax1.set_title("Variazione Nel Tempo Deviazione Standard Orientamento", fontsize=12)
    ax1.plot(metricheCovarianza[:, 0], metricheCovarianza[:, 5], color='blue', linewidth=1, linestyle='-', label='DevStd Orientamento ($\sigma_\\theta$)')

    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylabel('Incertezza [Deg]', fontsize=10)
    yLimit_Theta = np.min(metricheCovarianza[:, 5])
    ax1.set_ylim(bottom=yLimit_Theta)
    ax1.legend(fontsize=10, loc='upper right')

    # NOTE: [1] Deviazione Standard Velocità Lineare (Indice 6)
    ax2 = axes[1]
    ax2.set_facecolor('#f0f0f0')
    ax2.set_title("Variazione Nel Tempo Deviazione Standard Velocità Lineare", fontsize=12)
    ax2.plot(metricheCovarianza[:, 0], metricheCovarianza[:, 6], color='orange', linewidth=1, linestyle='-', label='DevStd Velocità Lineare ($\sigma_v$)')

    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylabel('Incertezza [m/s]', fontsize=10)
    yLimit_V = np.min(metricheCovarianza[:, 6])
    ax2.set_ylim(bottom=yLimit_V)
    ax2.legend(fontsize=10, loc='upper right')

    # NOTE: [2] Deviazione Standard Velocità Angolare (Indice 7)
    ax3 = axes[2]
    ax3.set_facecolor('#f0f0f0')
    ax3.set_title("Variazione Nel Tempo Deviazione Standard Velocità Angolare", fontsize=12)
    ax3.plot(metricheCovarianza[:, 0], metricheCovarianza[:, 7], color='green', linewidth=1, linestyle='-', label='DevStd Velocità Angolare ($\sigma_\omega$)')

    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylabel('Incertezza [Deg/s]', fontsize=10)
    ax3.set_xlabel('t [s]', fontsize=10)
    yLimit_Omega = np.min(metricheCovarianza[:, 7])
    ax3.set_ylim(bottom=yLimit_Omega)
    ax3.legend(fontsize=10, loc='upper right')

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()


# SS CONFRONTO CON DATI FILTRO DA TOPIC

# AA Confronto Traiettorie Di Due Filtri Diversi Con Il GPS
def plotMultipleFilterTrajectories(percorsoGPS, percorsoStimato, percorsoEKF, titolo, id_fig):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, axes = plt.subplots(1, 3, figsize=(14, 9), num=id_fig)
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # --- CORREZIONE PRINCIPALE ---
    # Calcolo Dinamico Limiti Su GPS
    padding = 2.0
    x_min = min(percorsoGPS[:, 1]) - padding
    x_max = max(percorsoGPS[:, 1]) + padding
    y_min = min(percorsoGPS[:, 2]) - padding
    y_max = max(percorsoGPS[:, 2]) + padding

    # NOTE: [0] GPS - EKF Implementato
    ax1 = axes[0]

    # SS GPS
    ax1.plot(percorsoGPS[:, 1], percorsoGPS[:, 2], '.', color='orange', markersize=3, label='Misure GPS (Ground Truth)')
    ax1.plot(percorsoGPS[-1, 1], percorsoGPS[-1, 2], marker='p', markersize=12, markerfacecolor='orange', markeredgecolor='k', linestyle='None', label='Fine (GPS)')

    # SS EKF (Filtro Implementato)
    ax1.plot(percorsoStimato[:, 1], percorsoStimato[:, 2], color='blue', linewidth=3, label='Filtro EKF (Implementato)')
    ax1.plot(percorsoStimato[-1, 1], percorsoStimato[-1, 2], marker='*', markersize=15, markerfacecolor='blue', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (EKF Implementato)')

    # SS Assi
    ax1.set_title("GPS - EKF", fontsize=12)
    ax1.set_facecolor('#f0f0f0')
    ax1.set_xlabel("Posizione X [m]", fontsize=12)
    ax1.set_ylabel("Posizione Y [m]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_aspect('equal')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='best', fontsize=10)

    # NOTE: [1] GPS - EKF ISARLAB
    ax2 = axes[1]
    # SS GPS
    ax2.plot(percorsoGPS[:, 1], percorsoGPS[:, 2], '.', color='orange', markersize=3, label='Misure GPS (Ground Truth)')
    ax2.plot(percorsoGPS[-1, 1], percorsoGPS[-1, 2], marker='p', markersize=12, markerfacecolor='orange', markeredgecolor='k', linestyle='None', label='Fine (GPS)')

    # SS EKF ISARLAB (Filtro Confronto)
    ax2.plot(percorsoEKF[:, 1], percorsoEKF[:, 2], color='green', linestyle='-', linewidth=2.5, label='Filtro EKF (ISARLAB)')
    ax2.plot(percorsoEKF[-1, 1], percorsoEKF[-1, 2], marker='d', markersize=12, markerfacecolor='green', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (ISARLAB)')

    # SS Assi
    ax2.set_title("GPS - TOPIC", fontsize=12)
    ax2.set_facecolor('#f0f0f0')
    ax2.set_xlabel("Posizione X [m]", fontsize=12)
    ax2.set_ylabel("Posizione Y [m]", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_aspect('equal')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.legend(loc='best', fontsize=10)

    # NOTE: [2] GPS - EKF ISARLAB - EKF Implementato
    ax3 = axes[2]

    # SS GPS
    ax3.plot(percorsoGPS[:, 1], percorsoGPS[:, 2], '.', color='orange', markersize=3, label='Misure GPS (Ground Truth)')
    ax3.plot(percorsoGPS[-1, 1], percorsoGPS[-1, 2], marker='p', markersize=12, markerfacecolor='orange', markeredgecolor='k', linestyle='None', label='Fine (GPS)')

    # SS EKF ISARLAB (Filtro Confronto)
    ax3.plot(percorsoEKF[:, 1], percorsoEKF[:, 2], color='red', linestyle='--', linewidth=2.5, label='Filtro EKF (ISARLAB)')
    ax3.plot(percorsoEKF[-1, 1], percorsoEKF[-1, 2], marker='d', markersize=12, markerfacecolor='red', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (ISARLAB)')

    # SS EKF (Filtro Implementato)
    ax3.plot(percorsoStimato[:, 1], percorsoStimato[:, 2], color='blue', linewidth=3, label='Filtro EKF (Implementato)')
    ax3.plot(percorsoStimato[-1, 1], percorsoStimato[-1, 2], marker='*', markersize=15, markerfacecolor='blue', markeredgecolor='k', markeredgewidth=0.5, linestyle='None', label='Fine (EKF Implementato)')

    # SS Assi
    ax3.set_title("Tutti i Filtri", fontsize=12)
    ax3.set_facecolor('#f0f0f0')
    ax3.set_xlabel("Posizione X [m]", fontsize=12)
    ax3.set_ylabel("Posizione Y [m]", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_aspect('equal')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.legend(loc='best', fontsize=10)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.05,
        right=0.98,
        hspace=0.4,
        wspace=0.15
    )
    plt.show()

# AA  Calcola Errore (Posizione E Orientamento) Tra Due Filtri Diversi
def plotFilterError(percorsoStimato, percorsoEKF, titolo, id_fig):
    # Estraggo Timestamp, Posizioni e Orientamento Stima (Mio Filtro)
    t = percorsoStimato[:, 0]
    pos = percorsoStimato[:, 1:3]
    theta = percorsoStimato[:, 3]

    # Estraggo Timestamp, Posizioni e Orientamento Reference (Filtro Isarlab)

    t_EKF = percorsoEKF[:, 0]
    pos_EKF = percorsoEKF[:, 1:3]
    theta_EKF = percorsoEKF[:, 3]

    # Per calcolare l'errore, le stime del filtro (~130Hz) devono essere confrontate con le misure GPS (~10Hz) agli stessi istanti t
    # La funzione np.interp "stima" i valori (x, y, theta) di percorsoEKF negli istanti t esatti in cui sono disponibili le misure in percorsoStimato.
    # Dato che np.interp opera su array 1D, devono essere calcolate le coordinate in modo indipendente per ciascuna coordinata spaziale.

    # Dato che np.interp opera su array 1D, calcoliamo le coordinate indipendentemente.
    x_EKF_calc = np.interp(t, t_EKF, pos_EKF[:, 0])
    y_EKF_calc = np.interp(t, t_EKF, pos_EKF[:, 1])
    theta_EKF_calc = np.interp(t, t_EKF, theta_EKF)

    # Calcolo Distanza Euclidea (Errore Posizione)
    errore_x = pos[:, 0] - x_EKF_calc
    errore_y = pos[:, 1] - y_EKF_calc
    distEuclidea = np.sqrt(errore_x ** 2 + errore_y ** 2)

    # Calcolo Errore Orientamento (Errore Theta) E Normalizzazione In [-pi, pi]
    erroreTheta = theta - theta_EKF_calc
    erroreTheta = np.arctan2(np.sin(erroreTheta), np.cos(erroreTheta))

    # Calcolo Statistiche Errore
    erroreMedio_pos = np.mean(distEuclidea)
    stdDevErrore_pos = np.std(distEuclidea)
    erroreMedio_theta = np.mean(np.abs(erroreTheta))
    stdDevErrore_theta = np.std(np.abs(erroreTheta))

    # Calcolo RMSE (Root Mean Square Error) Coordinate X, Y, Posizione e Theta
    rmse_x = np.sqrt(np.mean(errore_x ** 2))
    rmse_y = np.sqrt(np.mean(errore_y ** 2))
    rmse_pos = np.sqrt(np.mean(distEuclidea ** 2))
    rmse_theta = np.sqrt(np.mean(erroreTheta ** 2))

    print("\nStatistiche Errore Tra Filtro EKF e Filtro EKF Isarlab:")
    print(f"Errore Medio Posizione: {erroreMedio_pos:.2f} m")
    print(f"Deviazione Standard Errore Posizione: {stdDevErrore_pos:.2f} m")
    print(f"Errore Medio Orientamento Theta: {np.degrees(erroreMedio_theta):.2f} Gradi")
    print(f"Deviazione Standard Errore Orientamento: {np.degrees(stdDevErrore_theta):.2f} Gradi")
    print(f"RMSE X: {rmse_x:.2f} m")
    print(f"RMSE Y: {rmse_y:.2f} m")
    print(f"RMSE Posizione: {rmse_pos:.2f} m")
    print(f"RMSE Theta: {np.degrees(rmse_theta):.2f} Gradi")

    # Plotting Errore
    plt.style.use('seaborn-v0_8-colorblind')
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), num=id_fig)  # sharex=True
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # NOTE: [0] Errore X vs Tempo
    ax1 = axes[0]
    ax1.set_facecolor('#f0f0f0')
    ax1.set_title("Evoluzione Errore Coordinata X Nel Tempo (Implementazione EKF - EKF Isarlab)", fontsize=12)

    # SS Errore X
    ax1.plot(t, errore_x, color='red', linewidth=2, linestyle='-', label='Errore Coordinata X')
    # SS Assi
    ax1.set_xlabel("t [s]", fontsize=12)
    ax1.set_ylabel("Errore X [m]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best', fontsize=10)
    ax1.tick_params(labelsize=10)
    ax1.legend(loc='best', fontsize=10)

    # NOTE: [1] Errore Y vs Tempo
    ax2 = axes[1]
    ax2.set_facecolor('#f0f0f0')
    ax2.set_title("Evoluzione Errore Coordinata Y Nel Tempo (Implementazione EKF - EKF Isarlab)", fontsize=12)

    # SS Errore Y
    ax2.plot(t, errore_y, color='red', linewidth=2, linestyle='-', label='Errore Coordinata Y')

    # SS Assi
    ax2.set_xlabel("t [s]", fontsize=12)
    ax2.set_ylabel("Errore Y [m]", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best', fontsize=9)

    # NOTE: [2] Errore Theta vs Tempo
    ax3 = axes[2]
    ax3.set_facecolor('#f0f0f0')
    ax3.set_title("Evoluzione Errore Orientamento Theta Nel Tempo", fontsize=12)

    # SS Errore Theta
    ax3.plot(t, erroreTheta, color='red', linewidth=2, linestyle='-', label='Errore Orientamento Theta')

    # SS Assi
    ax3.set_xlabel("t [s]", fontsize=12)
    ax3.set_ylabel("Errore Theta [rad]", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='best', fontsize=9)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()


# AA Confronto Evoluzione Delle Componenti Di Posa (x, y, theta) Tra Due Filtri Diversi
def plotFiltersPoseComparison(percorsoGPS,orientamentiLidar, percorsoStimato, percorsoEKF, titolo, id_fig):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), num=id_fig)
    fig.suptitle(titolo, fontsize=13, fontweight='bold')

    # NOTE: [0] X vs Tempo
    ax1 = axes[0]
    ax1.set_facecolor('#f0f0f0')
    ax1.set_title("Confronto Evoluzione Coordinata X Nel Tempo", fontsize=12)

    # SS GPS
    # Percorso GPS
    ax1.scatter(percorsoGPS[:, 0], percorsoGPS[:, 1], color='orange', s=10, alpha=0.75, marker='p', linewidths=0.5, label='Percorso GPS (Reale)')
    # SS EKF
    # Percorso EKF
    ax1.plot(percorsoStimato[:, 0], percorsoStimato[:, 1], color='blue', linewidth=2, linestyle='-', label='Percorso Stimato (Filtro)')
    # SS EK Isarlab
    # Percorso EKF
    ax1.plot(percorsoEKF[:, 0], percorsoEKF[:, 1], color='green', linewidth=2, linestyle='-', label='Filtro EKF Isarlab')

    # SS Assi
    ax1.set_xlabel("t [s]", fontsize=12)
    ax1.set_ylabel("X [m]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best', fontsize=10)
    ax1.tick_params(labelsize=10)
    ax1.legend(loc='best', fontsize=10)

    # NOTE: [1] Y vs Tempo
    ax2 = axes[1]
    ax2.set_facecolor('#f0f0f0')
    ax2.set_title("Confronto Evoluzione Coordinata Y Nel Tempo", fontsize=12)

    # SS GPS
    # Percorso GPS
    ax2.scatter(percorsoGPS[:, 0], percorsoGPS[:, 2], color='orange', s=10, alpha=0.75, marker='p', linewidths=0.5, label='Percorso GPS (Reale)')
    # SS EKF
    # Percorso EKF
    ax2.plot(percorsoStimato[:, 0], percorsoStimato[:, 2], color='blue', linewidth=2, linestyle='-', label='Filtro EKF')
    # SS EK Isarlab
    # Percorso EKF
    ax2.plot(percorsoEKF[:, 0], percorsoEKF[:, 2], color='green', linewidth=2, linestyle='-',label='Filtro EKF Isarlab')

    # SS Assi
    ax2.set_xlabel("t [s]", fontsize=12)
    ax2.set_ylabel("Y [m]", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best', fontsize=9)

    # NOTE: [2] Theta vs Tempo
    ax3 = axes[2]
    ax3.set_facecolor('#f0f0f0')
    ax3.set_title("Confronto Evoluzione Orientamento Theta Nel Tempo", fontsize=12)

    # SS LIDAR
    # Orientamento LIDAR
    ax3.scatter(orientamentiLidar[:, 0], orientamentiLidar[:, 1], color='orange', s=8, alpha=0.75, marker='p', linewidths=0.5, label='Orientamento (LIDAR)')
    # SS EKF
    # Percorso EKF
    ax3.plot(percorsoStimato[:, 0], percorsoStimato[:, 3], color='blue', linewidth=2, linestyle='-',label='Filtro EKF')
    # SS EK Isarlab
    # Percorso EKF
    ax3.plot(percorsoEKF[:, 0], percorsoEKF[:, 3], color='green', linewidth=2, linestyle='-', label='Filtro EKF Isarlab')

    # SS Assi
    ax3.set_xlabel("t [s]", fontsize=12)
    ax3.set_ylabel("Theta [rad]", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='best', fontsize=9)

    plt.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.08,
        right=0.95,
        hspace=0.4,
        wspace=0.2
    )
    plt.show()
