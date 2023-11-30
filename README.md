# ProbabilisticBuildingAnomaly
Repo from *Building consumption anomaly detection: A comparative study of two probabilistic approaches* paper

Davor Stjelja, Vladimir Kuzmanovski, Risto Kosonen, Juha Jokisalo 

Aalto University, Granlund Oy, Vaisala Oyj, Nanjing Tech University


Abstract:
This paper investigates the performance of two probabilistic approaches, EnbPI a conformal prediction algorithm andXGBoostLSS, in predicting building energy consumption and in detecting systemic anomalies. The research questionsfocus on the effectiveness of these algorithms in providing both point and probabilistic predictions and their utility inidentifying collective anomalies. Both algorithms showed good point and distribution prediction performance. For exam-ple, the observed point prediction had CV-RMSE in the range of 9 to 17%, outperforming ASHRAE’s recommendations.Furthermore, a post-processing stage, the alarm matrix, effectively flags collective, repetitive anomalies, thus shiftingfocus from conventional point anomalies. The EnbPI model yields higher recall rates, with a trade-off of having morefalse alarms, while XGBoostLSS excels in precision, minimizing overlooked alarms. Moreover, a robustness analysiswas carried out to evaluate how these models performed when faced with training datasets containing anomalies. Thisanalysis revealed that the EnbPI model tended to overfit, while the XGBoostLSS demonstrated greater stability in theseconditions. While the findings contribute significantly to building energy consumption prediction and anomaly detection,future research could address the methodology’s performance in dynamic environments and explore continual learningstrategies.
