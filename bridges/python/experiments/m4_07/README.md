# Offline evaluation support against interactive simulation (#104)

Everything here is synthetic. It uses #102's queue-reactive days: a simulator with made-up
parameters, whose rows do not react to orders. Those days, the simulator and its parameters
bound what this study can show. The Hyperliquid capture holds no orders of our own, so it has
no actions, behaviour policy or propensities. It enters only through a state coverage
diagnostic, and the study's decision for it is a no-go for learning or evaluating a policy
offline.

- `design.json` fixes everything before any log is generated:
  - the behaviours, the targets and the estimators with their settings;
  - the diagnostics and the decision rule;
  - the stop rules and the coverage diagnostic.
- `decide.py` is the decision rule, in code, committed with the design.
  - It reads diagnostics only. A target's true value is never an input.
  - A go says only that these checks found nothing against reading the estimates for that target on that log.

Every value in the report is labelled:

- `[observed]`: read from a log;
- `[estimated]`: an off-policy estimate under its stated assumption;
- `[simulated]`: an interactive run of the target in the same simulator.

#102 has already evaluated its arms on these holdout days. No estimator, threshold or choice
of this study reads a holdout log before this study's freeze.

Offline policy training is #280, not this study.
