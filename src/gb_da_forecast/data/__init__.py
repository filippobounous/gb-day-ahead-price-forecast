"""Data collection from open sources.

Modules:
    elexon    Elexon Insights API (BMRS): demand and wind forecasts, generation mix, imbalance prices,
              market index prices.
    neso      NESO data portal: N2EX day-ahead prices (the target) and embedded wind/solar forecasts.
    external  Everything else: gas price (ONS), bank holidays (gov.uk), continental day-ahead prices
              (energy-charts), weather forecasts (Open-Meteo), solar outturn (PV_Live).

Run ``python -m gb_da_forecast.collect`` to download everything into ``data/raw/``.
"""
