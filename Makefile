.PHONY: all bronze silver gold forecast agent test lint clean timetravel drift-demo schema-history cross-version maintenance
UV := uv run

all: bronze silver gold forecast

bronze:      ; $(UV) python -m src.lakehouse.bronze
silver:      ; $(UV) python -m src.lakehouse.silver
gold:        ; $(UV) python -m src.lakehouse.gold
forecast:    ; $(UV) python -m src.forecast.report
agent:       ; $(UV) python -m src.agent.graph
timetravel:  ; $(UV) python -m src.lakehouse.maintenance timetravel
drift-demo:  ; $(UV) python -m src.lakehouse.maintenance drift-demo
schema-history: ; $(UV) python -m src.lakehouse.maintenance schema-history
cross-version:  ; $(UV) python -m src.lakehouse.maintenance cross-version
maintenance: ; $(UV) python -m src.lakehouse.maintenance expire
test:        ; $(UV) pytest -v
lint:        ; $(UV) ruff check src tests
clean:       ; rm -rf warehouse

smoke:
	N_TRANSACTIONS=5000 $(MAKE) all
