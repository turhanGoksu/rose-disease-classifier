### val (used to choose the epoch and the method)

| run | best epoch | accuracy | macro-F1 | same-source macro-F1 | Black_Spot precision | Black_Spot recall | Healthy recall | missed Black_Spot (FN) | false alarms (FP) | studio accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 15 | 0.997 | 0.996 | 0.994 | 1.000 | 0.988 | 1.000 | 1 | 0 | 1.000 |
| sampler | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 1.000 |
| loss | 10 | 0.997 | 0.996 | 0.994 | 1.000 | 0.988 | 1.000 | 1 | 0 | 1.000 |

### test (evaluated once, not used for any choice)

| run | best epoch | accuracy | macro-F1 | same-source macro-F1 | Black_Spot precision | Black_Spot recall | Healthy recall | missed Black_Spot (FN) | false alarms (FP) | studio accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 15 | 0.993 | 0.991 | 0.987 | 0.987 | 0.987 | 0.996 | 1 | 1 | 1.000 |
| sampler | 8 | 0.987 | 0.983 | 0.974 | 1.000 | 0.949 | 1.000 | 4 | 0 | 1.000 |
| loss | 10 | 0.993 | 0.991 | 0.987 | 1.000 | 0.975 | 1.000 | 2 | 0 | 1.000 |
