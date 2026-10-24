# Fixed-candidate retrospective diagnosis

Outer-optimal scores are hindsight references, not achieved predictive performance or blind evidence.

## Own selected-system K cap

| Method | Frozen outer NRMSE | Inner-only choice, revealed outer | Hindsight outer minimum | Within-filter gap | Excluded-by-filter gap | Cells with >=1% headroom |
|---|---:|---:|---:|---:|---:|---:|
| independent20 | 0.889092 | 0.889092 | 0.840939 | 0.000004 | 0.048150 | 37/45 |
| independent25 | 0.887598 | 0.888022 | 0.839387 | 0.000004 | 0.048207 | 37/45 |
| semantic25 | 0.892289 | 0.892383 | 0.832782 | -0.000000 | 0.059508 | 39/45 |
| whole25 | 0.886255 | 0.887320 | 0.831310 | 0.000004 | 0.054941 | 39/45 |
| context25 | 0.886178 | 0.887628 | 0.833105 | 0.000005 | 0.053068 | 38/45 |
| joint25 | 0.918713 | 0.920620 | 0.887837 | NA | NA | 25/45 |

## Common independent25 K cap

| Method | Hindsight outer minimum | Inner-only choice, revealed outer | Hindsight mean K |
|---|---:|---:|---:|
| independent20 | 0.840662 | 0.888479 | 32.488889 |
| independent25 | 0.839387 | 0.888022 | 33.155556 |
| semantic25 | 0.832308 | 0.888398 | 30.088889 |
| whole25 | 0.834092 | 0.884559 | 32.022222 |
| context25 | 0.834654 | 0.886553 | 32.511111 |
| joint25 | 0.889459 | 0.924046 | 21.377778 |

## Every output (active-head own K cap)

Rank rho uses finite candidate pairs; percentile 0 is best. Joint projections are diagnostic views only.

| Method | Dataset/output | Selected NRMSE | Hindsight NRMSE | Selected R2 | Hindsight R2 | Mean rank rho | Mean selected percentile |
|---|---|---:|---:|---:|---:|---:|---:|
| independent20 | air_quality/CO(GT) | 0.390932 | 0.373440 | 0.874182 | 0.889850 | 0.719312 | 0.362941 |
| independent20 | air_quality/NO2(GT) | 0.855036 | 0.728199 | 0.528438 | 0.667182 | 0.236321 | 0.261004 |
| independent20 | air_quality/NOx(GT) | 1.753672 | 1.635583 | 0.339633 | 0.456360 | 0.315088 | 0.305141 |
| independent20 | concrete_slump/Compressive Strength (28-day)(Mpa) | 0.252700 | 0.237250 | 0.933464 | 0.941215 | 0.703673 | 0.212130 |
| independent20 | concrete_slump/FLOW(cm) | 0.834754 | 0.792406 | 0.335796 | 0.386379 | 0.558539 | 0.182638 |
| independent20 | concrete_slump/SLUMP(cm) | 0.902619 | 0.871839 | 0.135515 | 0.194355 | 0.534711 | 0.183113 |
| independent20 | parkinsons/motor_UPDRS | 1.015603 | 1.009475 | -0.020706 | -0.009150 | 0.379669 | 0.194487 |
| independent20 | parkinsons/total_UPDRS | 0.992475 | 0.974769 | -0.008963 | 0.028432 | 0.568368 | 0.284544 |
| independent25 | air_quality/CO(GT) | 0.388799 | 0.373536 | 0.876546 | 0.889776 | 0.694921 | 0.366624 |
| independent25 | air_quality/NO2(GT) | 0.860368 | 0.728235 | 0.521713 | 0.667148 | 0.245014 | 0.291565 |
| independent25 | air_quality/NOx(GT) | 1.742945 | 1.627423 | 0.340644 | 0.460359 | 0.301116 | 0.282313 |
| independent25 | concrete_slump/Compressive Strength (28-day)(Mpa) | 0.252700 | 0.237250 | 0.933464 | 0.941215 | 0.705684 | 0.202443 |
| independent25 | concrete_slump/FLOW(cm) | 0.837264 | 0.792845 | 0.333012 | 0.385903 | 0.610891 | 0.190469 |
| independent25 | concrete_slump/SLUMP(cm) | 0.894185 | 0.867963 | 0.152552 | 0.201542 | 0.575210 | 0.125466 |
| independent25 | parkinsons/motor_UPDRS | 1.015603 | 1.009475 | -0.020706 | -0.009150 | 0.363406 | 0.180814 |
| independent25 | parkinsons/total_UPDRS | 0.992475 | 0.974769 | -0.008963 | 0.028432 | 0.552416 | 0.279642 |
| semantic25 | air_quality/CO(GT) | 0.385005 | 0.366014 | 0.880172 | 0.892877 | 0.546924 | 0.360536 |
| semantic25 | air_quality/NO2(GT) | 0.868705 | 0.716447 | 0.507087 | 0.677904 | 0.154797 | 0.326397 |
| semantic25 | air_quality/NOx(GT) | 1.735992 | 1.608027 | 0.352169 | 0.472659 | 0.281369 | 0.306638 |
| semantic25 | concrete_slump/Compressive Strength (28-day)(Mpa) | 0.254336 | 0.231828 | 0.933157 | 0.944289 | 0.667663 | 0.257107 |
| semantic25 | concrete_slump/FLOW(cm) | 0.820213 | 0.765918 | 0.353636 | 0.436310 | 0.566207 | 0.190245 |
| semantic25 | concrete_slump/SLUMP(cm) | 0.905094 | 0.871372 | 0.131056 | 0.194627 | 0.553014 | 0.145025 |
| semantic25 | parkinsons/motor_UPDRS | 1.038292 | 1.009475 | -0.082601 | -0.009150 | 0.269826 | 0.195131 |
| semantic25 | parkinsons/total_UPDRS | 1.002546 | 0.976059 | -0.029821 | 0.025920 | 0.495995 | 0.264091 |
| whole25 | air_quality/CO(GT) | 0.381662 | 0.368699 | 0.882554 | 0.892018 | 0.712324 | 0.294394 |
| whole25 | air_quality/NO2(GT) | 0.845501 | 0.714143 | 0.534438 | 0.680192 | 0.254402 | 0.256181 |
| whole25 | air_quality/NOx(GT) | 1.735630 | 1.612065 | 0.351282 | 0.470323 | 0.267880 | 0.302226 |
| whole25 | concrete_slump/Compressive Strength (28-day)(Mpa) | 0.251810 | 0.238000 | 0.934073 | 0.940875 | 0.676963 | 0.169973 |
| whole25 | concrete_slump/FLOW(cm) | 0.820584 | 0.769031 | 0.345811 | 0.439161 | 0.578615 | 0.172428 |
| whole25 | concrete_slump/SLUMP(cm) | 0.903908 | 0.858230 | 0.137602 | 0.223571 | 0.522112 | 0.169707 |
| whole25 | parkinsons/motor_UPDRS | 1.027612 | 1.009475 | -0.054241 | -0.009150 | 0.352098 | 0.197930 |
| whole25 | parkinsons/total_UPDRS | 0.997190 | 0.966071 | -0.019486 | 0.046725 | 0.496711 | 0.275721 |
| context25 | air_quality/CO(GT) | 0.393403 | 0.377863 | 0.874854 | 0.886249 | 0.701889 | 0.386151 |
| context25 | air_quality/NO2(GT) | 0.832521 | 0.721054 | 0.554679 | 0.673632 | 0.215881 | 0.286768 |
| context25 | air_quality/NOx(GT) | 1.752213 | 1.627066 | 0.343356 | 0.461924 | 0.294753 | 0.311723 |
| context25 | concrete_slump/Compressive Strength (28-day)(Mpa) | 0.246347 | 0.233728 | 0.936591 | 0.942788 | 0.721249 | 0.176166 |
| context25 | concrete_slump/FLOW(cm) | 0.812604 | 0.764736 | 0.376302 | 0.449102 | 0.577479 | 0.189132 |
| context25 | concrete_slump/SLUMP(cm) | 0.909414 | 0.862331 | 0.123984 | 0.208220 | 0.523372 | 0.214780 |
| context25 | parkinsons/motor_UPDRS | 1.026157 | 1.008020 | -0.050640 | -0.005548 | 0.380097 | 0.201461 |
| context25 | parkinsons/total_UPDRS | 0.993244 | 0.971600 | -0.010985 | 0.034360 | 0.530283 | 0.283937 |
| joint25 | air_quality/CO(GT) | 0.392342 | 0.367889 | 0.878273 | 0.892824 | 0.392116 | 0.270397 |
| joint25 | air_quality/NO2(GT) | 0.841337 | 0.731385 | 0.542947 | 0.662782 | 0.437221 | 0.336801 |
| joint25 | air_quality/NOx(GT) | 1.718772 | 1.627901 | 0.380489 | 0.464903 | 0.256535 | 0.300906 |
| joint25 | concrete_slump/Compressive Strength (28-day)(Mpa) | 0.403759 | 0.361508 | 0.822712 | 0.861162 | 0.620915 | 0.457741 |
| joint25 | concrete_slump/FLOW(cm) | 0.890724 | 0.779233 | 0.245161 | 0.429987 | 0.555552 | 0.292320 |
| joint25 | concrete_slump/SLUMP(cm) | 0.952860 | 0.854531 | 0.046902 | 0.243380 | 0.486508 | 0.333287 |
| joint25 | parkinsons/motor_UPDRS | 1.053983 | 1.009475 | -0.125593 | -0.009150 | 0.590909 | 0.122771 |
| joint25 | parkinsons/total_UPDRS | 0.991763 | 0.976088 | -0.008168 | 0.025811 | 0.438577 | 0.181193 |
