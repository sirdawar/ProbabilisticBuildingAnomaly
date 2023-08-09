# ProbabilisticBuildingAnomaly
Repo from *Building consumption anomaly detection: A comparative study of two probabilistic approaches* paper
Davor Stjelja, Vladimir Kuzmanovski, Risto Kosonen, Juha Jokisalo 
Aalto University, Granlund Oy, Vaisala Oyj, Nanjing Tech University


Abstract:
This paper investigates the performance of two probabilistic approaches, EnbPI a conformal prediction algorithm and XGBoostLSS, in predicting building energy consumption and in detecting systemic anomalies. The research questions focus on the effectiveness of these algorithms in providing both point and probabilistic predictions and their utility in identifying collective anomalies. Both algorithms showed good point and distribution prediction performance. For example, the observed point prediction had CV-RMSE in the range of 9 to 17\%, outperforming ASHRAE's recommendations. Furthermore, a post-processing stage, the alarm matrix, effectively flags collective, repetitive anomalies, thus shifting focus from conventional point anomalies. The EnbPI model yields higher recall rates, reducing false alarms, while XGBoostLSS excels in precision, minimizing overlooked alarms. While the findings contribute significantly to building energy consumption prediction and anomaly detection, future research could address the methodology's performance in dynamic environments and explore continual learning strategies.
