# Interactive Brokers paper-trading setup

Atlas connects to **Trader Workstation (TWS)** or **IB Gateway** through the
optional `ib_insync` package. It is built for paper trading:
`EXECUTION_MODE=live` is rejected at configuration load *and* again by the
pre-trade safety gate.

> Atlas will refuse to trade any account whose identifier does not look like an
> Interactive Brokers paper account (a `DU`/`DF` prefix), regardless of the port
> you connect on.

---

## 1. Get a paper account

1. Open an IBKR account (a funded live account is required before a paper
   account is issued; the paper account itself trades simulated money).
2. In Client Portal: **Settings → Account Settings → Paper Trading Account**.
3. Note the paper username and the account id — it starts with `DU`.

## 2. Install TWS or IB Gateway

* **TWS** — the full trading platform. Easier to see what is happening.
* **IB Gateway** — a minimal API-only application. Lighter, better for
  long-running sessions.

Log in with your **paper** credentials.

## 3. Enable the API

In TWS: **File → Global Configuration → API → Settings**
(IB Gateway: **Configure → Settings → API → Settings**)

- [x] **Enable ActiveX and Socket Clients**
- [ ] **Read-Only API** — must be **unchecked** to submit orders
- **Socket port**: `7497` (TWS paper) or `4002` (Gateway paper)
- **Trusted IPs**: add `127.0.0.1`
- [x] Allow connections from localhost only (unless connecting from Docker — see
      below)

Restart TWS/Gateway after changing these.

### Ports

| Application | Paper | Live |
|---|---|---|
| TWS | **7497** | 7496 |
| IB Gateway | **4002** | 4001 |

Use a paper port. Atlas checks the *account*, not the port, but there is no
reason to point it at a live one.

## 4. Market data

A paper account often has no live data subscription. Atlas requests **delayed**
data by default (`market_data_type: 3` in `configs/execution.yaml`), which works
without a subscription. Set it to `1` if you have live data.

Where no price is available, Atlas **skips that order** rather than guessing.

## 5. Configure Atlas

```bash
pip install -e ".[broker]"
cp .env.example .env
```

Edit `.env`:

```bash
EXECUTION_MODE=dry_run          # keep this until you have run a preview
ATLAS_IBKR_HOST=127.0.0.1
ATLAS_IBKR_PORT=7497
ATLAS_IBKR_CLIENT_ID=17
ATLAS_IBKR_ACCOUNT_ALLOWLIST=DU1234567   # your paper account id
```

The allowlist is **required** before any order can be submitted. Without it the
safety gate blocks submission even in paper mode.

## 6. Verify the connection

```bash
atlas broker-check
```

This connects, reads the account summary and positions, and prints every safety
check. All must pass before Atlas will submit anything.

## 7. Preview orders

```bash
atlas order-preview
```

Nothing is sent. You get the target portfolio, the required trades, the
estimated turnover and the reconciliation status.

No TWS running? Use the in-memory mock broker:

```bash
atlas order-preview --mock
```

## 8. Submit to the paper account

Only when the preview looks right:

```bash
EXECUTION_MODE=paper atlas paper-trade
```

You will be asked to confirm. Add `--yes` for a scheduled run.

## 9. Reconcile

```bash
atlas reconcile
```

Compares broker positions with Atlas's record. A material mismatch **blocks the
next trading cycle** until it is resolved.

---

## Order of checks before submission

1. Kill switch off.
2. `EXECUTION_MODE=paper`.
3. Broker connected.
4. Account identified.
5. Account is a paper account (`DU`/`DF`).
6. Account on the allowlist.
7. Market data fresh.
8. Positions retrieved.
9. Positions reconcile.
10. Risk manager not halted.
11. Per-order: valid price, notional bounds, buying power, trading window, not a
    duplicate.

Any failure blocks submission and is logged with a reason.

---

## Troubleshooting

**"could not connect ... connection refused"**
TWS/Gateway is not running, the API is not enabled, or the port is wrong.

**"couldn't connect to TWS. Confirm that API is enabled"**
*Enable ActiveX and Socket Clients* is unchecked.

**Orders rejected as read-only**
Uncheck **Read-Only API** and restart.

**"client id is already in use"**
Another session holds that id. Change `ATLAS_IBKR_CLIENT_ID`.

**"does not look like a paper account"**
Working as designed. Atlas will not trade a non-paper account.

**"not on the allowlist"**
Set `ATLAS_IBKR_ACCOUNT_ALLOWLIST` to the exact account id.

**No prices**
No market-data subscription. Keep `market_data_type: 3` (delayed).

**"outside the configured trading window"**
Orders are only permitted 09:35–15:55 New York on weekdays. Adjust
`trading_window` in `configs/execution.yaml` if you need a different window.

---

## Docker

TWS/Gateway runs on the **host**, not in a container. `docker-compose.yml`
already maps `host.docker.internal`. You must also allow connections from the
Docker bridge network in the API settings — add the bridge subnet (commonly
`172.17.0.0/16`) to Trusted IPs and untick "localhost only".

---

## What paper trading does not tell you

A paper account fills orders a real market may not, particularly in size or in
stress, and models neither queue position nor latency. Paper results are an
upper bound on real execution quality. See [limitations.md](limitations.md).
