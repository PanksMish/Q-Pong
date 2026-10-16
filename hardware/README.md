# Hardware runs

`scripts/run_hardware.py submit` writes one folder per run:

```
hardware/<run-id>/calibration_snapshot.json   median/best 2Q error, median readout error, timestamp
hardware/<run-id>/<system>/noiseless.jsonl     trained theta* and Ising Hamiltonian of every interval
hardware/<run-id>/<system>/jobs.json           Qiskit Runtime job IDs (Sampler/Estimator, raw/mitigated),
                                               transpiled 2Q-gate count and depth, versions
hardware/<run-id>/<system>/raw_counts.json     raw and mitigated measurement counts per interval
hardware/<run-id>/<system>/replay_*.jsonl      pipeline replay with the hardware counts
hardware/<run-id>/table8.csv                   Table VIII
```

Commit these files after a real run. Folders starting with `dryrun_` come from `FakeKingston`
simulations and must not be reported as hardware results.
