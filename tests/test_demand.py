import numpy as np

from src import config
from src.generators import demand


def test_panel_covers_every_company_and_calendar_day():
    panel = demand.build_demand_panel()
    days = config.trading_days(config.START_DATE, config.END_DATE)
    assert set(panel["ticker"]) == {c.ticker for c in config.COMPANIES}
    assert len(panel) == len(days) * len(config.COMPANIES)


def test_demand_is_strictly_positive():
    panel = demand.build_demand_panel()
    assert (panel["demand"] > 0).all()


def test_shock_is_standardized_per_ticker():
    panel = demand.build_demand_panel()
    for _, grp in panel.groupby("ticker"):
        assert abs(grp["shock"].mean()) < 0.15
        assert 0.7 < grp["shock"].std() < 1.4


def test_deterministic_under_same_seed():
    a = demand.build_demand_panel(seed=7)
    b = demand.build_demand_panel(seed=7)
    assert a.equals(b)
    c = demand.build_demand_panel(seed=8)
    assert not np.allclose(a["demand"], c["demand"])


def test_weekly_seasonality_present():
    panel = demand.build_demand_panel()
    sbux = panel[panel["ticker"] == "SBUX"].copy()
    sbux["dow"] = [d.weekday() for d in sbux["day"]]
    by_dow = sbux.groupby("dow")["demand"].mean()
    assert by_dow.max() / by_dow.min() > 1.1
