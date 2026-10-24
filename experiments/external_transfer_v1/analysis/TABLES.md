# Frozen external evaluation

Lower NRMSE is better. K applies only to symbolic models.

| Dataset | Method | Test NRMSE | K | Training CPU |
| --- | --- | ---: | ---: | ---: |
| tetouan | independent20 | 0.903822 | 69.80 | 20.016 |
| tetouan | independent25 | 0.904305 | 69.20 | 25.031 |
| tetouan | random25 | 0.906284 | 60.40 | 25.016 |
| tetouan | semantic25 | 0.908798 | 58.40 | 25.016 |
| tetouan | joint25 | 0.859469 | 33.80 | 25.000 |
| tetouan | mean | 1.174248 | n/a | 0.000 |
| tetouan | ridge | 0.916894 | n/a | 0.000 |
| tetouan | extra_trees | 0.830684 | n/a | 0.431 |
| water | independent20 | 0.486589 | 3.00 | 20.016 |
| water | independent25 | 0.486589 | 3.00 | 25.031 |
| water | random25 | 0.486513 | 4.40 | 25.016 |
| water | semantic25 | 0.486012 | 7.20 | 25.016 |
| water | joint25 | 0.486589 | 3.00 | 25.000 |
| water | mean | 0.486589 | n/a | 0.000 |
| water | ridge | 0.488945 | n/a | 0.000 |
| water | extra_trees | 0.490508 | n/a | 0.141 |
| video | independent20 | 0.566642 | 68.20 | 20.000 |
| video | independent25 | 0.564781 | 70.20 | 25.000 |
| video | random25 | 0.539160 | 65.80 | 25.003 |
| video | semantic25 | 0.546899 | 65.40 | 25.000 |
| video | joint25 | 0.536002 | 56.40 | 25.000 |
| video | mean | 1.173187 | n/a | 0.000 |
| video | ridge | 1.021588 | n/a | 0.003 |
| video | extra_trees | 0.462810 | n/a | 0.600 |
| all | independent20 | 0.652351 | 47.00 | 20.010 |
| all | independent25 | 0.651892 | 47.47 | 25.021 |
| all | random25 | 0.643986 | 43.53 | 25.011 |
| all | semantic25 | 0.647236 | 43.67 | 25.010 |
| all | joint25 | 0.627353 | 31.07 | 25.000 |
| all | mean | 0.944675 | n/a | 0.000 |
| all | ridge | 0.809142 | n/a | 0.001 |
| all | extra_trees | 0.594668 | n/a | 0.391 |
