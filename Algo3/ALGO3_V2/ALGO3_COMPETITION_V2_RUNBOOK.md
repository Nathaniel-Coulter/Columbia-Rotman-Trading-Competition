# ALGO3 Competition Engine v2.0 — runbook

**For RIT ALGO3 classroom simulation only.** No real broker or production exchange support.

The engine is deliberately **SHADOW by default**, without POST or DELETE requests. The live, proven path continues to hedge **100%** of every accepted tender. A conditional core/tail variant is included behind **two additional explicit opt-in flags**, because its own-impact and live fill behavior has **not** been calibrated in the simulator.

## Install

Place `algo3_competition_engine_v2.py` at:

`C:\Users\hocke\Desktop\quant_portfolio_scaffold\columbia\algorithmic_trading_class\algo3\pipeline\algo3_competition_engine_v2.py`

Use your existing virtualenv and `requests` package. No new Python dependencies are required for either the competition engine or the offline research companion.

In PowerShell:

```powershell
cd "C:\Users\hocke\Desktop\quant_portfolio_scaffold"
.\.venv\Scripts\Activate
$env:RIT_API_KEY="YOUR_CURRENT_RIT_CLIENT_API_KEY"
```

Run the offline deterministic self-test (**no RIT connection needed**):

```powershell
python ".\columbia\algorithmic_trading_class\algo3\pipeline\algo3_competition_engine_v2.py" --self-test
```

Then run the **read-only preflight** with RIT open and the demo case selected:

```powershell
python ".\columbia\algorithmic_trading_class\algo3\pipeline\algo3_competition_engine_v2.py" --preflight
```

Expected demo settings: `THOR_M / THOR_A`, fees 0.010 / 0.005, ALT rebate 0.0075, max order 25,000, API rate 100/s, execution delay 0 ms, gross/net 200,000 / 100,000, **flat positions and zero open orders**. The engine discovers runtime settings; it does not blindly assume these numbers.

## Shadow research, multiple heats

```powershell
python ".\columbia\algorithmic_trading_class\algo3\pipeline\algo3_competition_engine_v2.py" --mode normal --rounds 10
```

**No trading occurs.** `--rounds 10` means up to ten completed heats. `Ctrl+C` requests graceful stop; completion of all ten heats is not required to make earlier SQLite data useful. A normal shadow session may produce a large database (~1 GB over eight heats at 50-ms books), so maintain disk space.

## Default calibrated live demonstration: one heat

Only after reviewing preflight and mock/unit tests, start from STOPPED and use:

```powershell
python ".\columbia\algorithmic_trading_class\algo3\pipeline\algo3_competition_engine_v2.py" --live --confirmed --mode normal --rounds 1 --max-trades 5
```

The default **NORMAL** trigger is 5¢/share after taker fees on the **entire** tender quantity. The live default does **not** hold an inventory tail. This is the same calibrated booking-gated, reservation-protected, near-immediate 100% hedge as COMP-1.1, with more robust fail-closed order-state inspection.

Do **not** run COMP-1.1, COMP-2.0, calibration harnesses, or manual orders concurrently. The competition and calibration scripts use the same `algo3_write.lock`.

## Experimental conditional live inventory tail (requires separate human review)

**Not recommended for unattended competition or uncalibrated first use.** This is implemented and mock-tested for a controlled later demonstration, but the model selection and fill kinetics are unproven on a live RIT heat.

```powershell
python ".\columbia\algorithmic_trading_class\algo3\pipeline\algo3_competition_engine_v2.py" --live --confirmed --mode normal --rounds 1 --max-trades 1 --experimental-live-tail --acknowledge-uncalibrated-tail
```

When explicitly armed, only tenders with a fresh trigger edge from **5¢ up to (but not including) 10¢** are eligible. The script immediately hedges the other 75%, **waits for confirmed core fills and the mirrored residual position**, then monitors up to **25%**, capped at **25,000 shares**, for at most **8 seconds**. It exits earlier for an edge improvement of +10¢, deterioration of −8¢ (both relative to a prehedge marginal-tail proxy), missing depth, mismatched position, a fine increase, case end, or a near-end time buffer. It then places exposure-reducing orders and reconciles all the way to flat.

On 10¢+ triggers the engine immediately hedges the entire block, **even when experimental tail is armed**. This comes from the eight-heat dataset: most exceptionally large edges decayed, while the 5–under-10¢ cohort's average edge tended to improve. **These outcomes are indicative shadow proxies, not a guarantee of positive tail P&L.**

Tuning options: `--tail-fraction`, `--tail-hold-seconds`, `--tail-stop-dollars`, `--tail-take-dollars`, `--tail-low-edge`, and `--tail-high-edge`. Do not change multiple parameters simultaneously before collecting at least one controlled live-tail execution.

## Offline analysis after a shadow run

Copy `algo3_tournament_research_v2.py` alongside the engine or anywhere convenient. Run:

```powershell
python ".\columbia\algorithmic_trading_class\algo3\pipeline\algo3_tournament_research_v2.py" --db ".\columbia\algorithmic_trading_class\algo3\outputs\YOUR_RUN_DIRECTORY\algo3_competition.sqlite"
```

The companion produces a Markdown research report, CSVs for all tenders and triggers, tail cohorts at each horizon, and causal *option-to-wait* policy simulations. It uses the **Python standard library only**, reads an existing SQLite file, and never connects to RIT.

## Data outputs

Each engine run writes to `algo3\outputs\algo3_competition_<utc_timestamp>-<run_id>\`, including:

- `algo3_competition.sqlite`: L2 books, trader IDs/features, TAS, tenders, trigger-relative tail counterfactuals, all order events, HTTP and position timelines.
- `summary.json`: final run-level accounting, including the authoritative account NLV delta when available.
- `manifest.json`: actual version/CLI/configuration and safety defaults.
- `tenders.csv`, `orders.csv`, `trades.csv` when those row types exist.

Send the entire run folder, compressed, for post-trade analysis. If the engine reports ambiguous writes, incomplete booking, unconfirmed core, fine increases, nonzero ending positions, or an unexpected halt, **do not start another live bot** until RIT positions/open orders are reconciled.

## Validation already performed for v2.0

- Offline self-test: full-depth route/VWAP, tender sign, mirrored limits, reservation quantity, conditional tail opt-in and threshold boundary.
- Full mocked RIT integration: BUY tender / SELL hedge; SELL tender / BUY hedge; conditional BUY and SELL core+tail with confirmed inventory and time exit; read-only shadow with zero writes.
- Failure injection: a market-order POST with a lost HTTP response despite a server-side fill. Engine did not duplicate the order, halted additional writes, and correctly flagged unresolved inventory rather than pretending it was flat.
- Focused offline tests: tail stop, profit target, mirrored-position mismatch, and eligibility.

**Not yet validated:** how the optional held tail affects real RIT fills, market impact, fee/fine accounting, queue dynamics, and late partial executions. Default full hedge is supported by the earlier one-tender calibration, not by the eight shadow heats alone.
