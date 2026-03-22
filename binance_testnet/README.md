Binance Testnet integration plan

Goal: map local strategy to exchange testnet, verify order flow, latency handling, and rate-limit safety.

Steps:
1) Adapter simulation: run strategy through a local simulated orderbook (adapter_sim.py).
2) Mapping: implement an exchange adapter following Binance Testnet REST API semantics (create_order, cancel_order, get_order, fetch_order_book). Keep the adapter behind a thin interface so simulation and real adapter swap easily.
3) Safety: include rate-limit handling, API error retry, nonce/timestamp signing. Use CCXT or direct REST + HMAC signing.
4) Integration tests: dry-run simulated orders -> adapter -> simulated matching engine. Then, after you provide testnet keys, call actual Binance Testnet endpoints in a limited sandbox mode.

Files:
- adapter_sim.py  # simulated REST-like adapter (no network)
- adapter_binance.py (TODO)  # real adapter that will call Binance Testnet (requires keys)
- strategy.py
- runner.py

To start simulation: python3 runner.py

Note: I will not call real exchange APIs without your explicit approval per-request. For Testnet I will still ask for keys, but it's safe to use testnet keys (no real money). For mainnet I will request separate approval before any action.
