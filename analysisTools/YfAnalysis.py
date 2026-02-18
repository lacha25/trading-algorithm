import contextlib
import io

import pandas as pd
import yfinance as yf


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def f_yf_analysis(ent):
    number_of_elements = 2.0
    signal = 0.0
    has_external_data = False
    dat = yf.Ticker(ent)

    price_target = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            price_target = dat.analyst_price_targets
    except Exception:
        price_target = None

    if isinstance(price_target, dict):
        high = _safe_float(price_target.get("high"))
        low = _safe_float(price_target.get("low"))
        mean = _safe_float(price_target.get("mean"))
        current = _safe_float(price_target.get("current"))
    else:
        high = low = mean = current = None

    if None not in (high, low, mean, current):
        has_external_data = True
        if current > mean and (high - mean) > 0:
            signal -= min((current - mean) / (high - mean), 1.0)
        elif current < mean and (mean - low) > 0:
            signal += min((mean - current) / (mean - low), 1.0)

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            recommandations = dat.recommendations
    except Exception:
        recommandations = None

    if isinstance(recommandations, pd.DataFrame) and not recommandations.empty:
        has_external_data = True
        first_row = recommandations.iloc[0]
        sb = int(first_row.get("strongBuy", 0) or 0)
        buy = int(first_row.get("buy", 0) or 0)
        hold = int(first_row.get("hold", 0) or 0)
        sell = int(first_row.get("sell", 0) or 0)
        ss = int(first_row.get("strongSell", 0) or 0)
        if buy > hold and buy + sb > sell + ss:
            signal = 1.0
        elif sell > hold and sell + ss > buy + sb:
            signal = -1.0

    if not has_external_data:
        return None

    return signal / number_of_elements
