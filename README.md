# ProbabilisticBuildingAnomaly
Repo from published paper:
*Stjelja, D., Kuzmanovski, V., Kosonen, R., & Jokisalo, J. (2024). Building consumption anomaly detection: A comparative study of two probabilistic approaches. Energy and Buildings, 114249.*
https://doi.org/10.1016/j.enbuild.2024.114249 

Davor Stjelja, Vladimir Kuzmanovski, Risto Kosonen, Juha Jokisalo 

Aalto University, Granlund Oy, Vaisala Oyj, Nanjing Tech University


Abstract:
This paper investigates the performance of two probabilistic approaches, Ensemble batch Prediction Intervals (EnbPI) a conformal prediction approach and XGBoost Location, Scale and Shape (XGBoostLSS), in predicting building energy consumption and in detecting systemic anomalies with proposed alarm matrix. The research questions focus on the effectiveness of these models in providing both point and probabilistic predictions and their utility in identifying collective anomalies. Both models showed good point and distribution prediction performance. For example, the observed point prediction had CV-RMSE in the range of 9 to 17%, outperforming recommendations from ASHRAE. Furthermore, a post-processing stage, the alarm matrix, effectively flags collective, repetitive anomalies, thus shifting focus from conventional point anomalies. The EnbPI-based method yields higher recall rates, with a trade-off of having more false alarms, while XGBoostLSS-based method excels in precision, minimizing overlooked alarms. Moreover, a robustness analysis was carried out to evaluate how these models performed when faced with training datasets containing anomalies. The robustness analysis revealed that the EnbPI-based method was more prone to overfitting, meaning its performance actually improved when the training data included some noise. On the other hand, the XGBoostLSS-based method was more stable, performing well with low levels of noise with performance drop when the noise level was high. While the findings contribute significantly to building energy consumption prediction and anomaly detection, future research could address performance in dynamic environments for the methodology and explore continual learning strategies.
