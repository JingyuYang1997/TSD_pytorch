## Pytorch Implementation of [Time Series Deconfounder](https://arxiv.org/abs/1902.00450)

This is a tentative **Pytorch** implementation of the paper "Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders" . The codes are adapted from the [Original **TensorFlow** Version](https://github.com/ioanabica/Time-Series-Deconfounder).

Refer to `main_run.py` for running.

In my case in the synthetic experiment, the counfounded prediction is with RMSE value 3.0452 while the deconfounded with 2.6390 (original tensorflow). My pytorch implementation achieves a marginally better result with 2.5902. 