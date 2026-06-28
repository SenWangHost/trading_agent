# Trading Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an intraday LangGraph trading agent that analyzes stocks via technical, fundamental, and news signals and executes paper trades through Alpaca.

**Architecture:** A LangGraph `StateGraph` runs every 30 minutes during market hours. A `dispatch` node fans out one `analyze_ticker` node per watchlist ticker (via `Send` API); each node runs all three analyzers sequentially. After all ticker nodes complete (fan-in), the `supervisor` node synthesizes signals into `TradeDecision` objects, and `trade_executor` submits them to the broker.

**Tech Stack:** Python 3.10+, uv, LangGraph 0.2+, langchain-anthropic, claude-sonnet-4-6, robin-stocks, pandas-ta, alpaca-py, APScheduler, Polygon.io (news only), pytest.

## Global Constraints

- Python >= 3.10
- Managed with `uv` (not pip) — use `uv add` to add deps, `uv run pytest` to run tests
- LLM model: `claude-sonnet-4-6`
- Structured output via `.with_structured_output(PydanticModel)` on every LLM call
- All agents must catch exceptions and return a neutral/hold signal rather than raise
- `broker/robinhood.py` and `broker/ibkr.py` are stubs — they must raise `NotImplementedError`, never execute real trades
- Broker instantiation happens only in `scheduler.py` — agents and graph have no direct broker imports except through `graph.py`
- Decision log path: `logs/decisions.jsonl`
- Tests live in `tests/`; one test file per source file

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `pyproject.toml` | Create | uv project config, all dependencies |
| `.env.example` | Create | All required env var names with placeholder values |
| `state.py` | Create | `AgentState` TypedDict + all Pydantic signal/decision models |
| `broker/__init__.py` | Create | Empty |
| `broker/base.py` | Create | `BaseBroker` ABC + `Position` + `OrderResult` Pydantic models |
| `broker/alpaca.py` | Create | `AlpacaBroker` — paper trading via `alpaca-py` |
| `broker/robinhood.py` | Create | `RobinhoodBroker` stub — raises `NotImplementedError` |
| `broker/ibkr.py` | Create | `IBKRBroker` stub — raises `NotImplementedError` |
| `agents/__init__.py` | Create | Empty |
| `agents/technical.py` | Create | `run_technical(state)` — robin_stocks OHLCV + pandas-ta + LLM |
| `agents/fundamental.py` | Create | `run_fundamental(state)` — robin_stocks fundamentals + LLM + daily cache |
| `agents/news.py` | Create | `run_news(state)` — Polygon.io headlines + LLM |
| `agents/supervisor.py` | Create | `run_supervisor(state)` — synthesizes 3 signals → `TradeDecision` |
| `graph.py` | Create | `build_graph(broker)` — assembles and compiles `StateGraph` |
| `scheduler.py` | Create | `start_scheduler(tickers)` — APScheduler + cycle runner + decision logger |
| `main.py` | Create | Entry point: load env, robin_stocks login, start scheduler |
| `tests/test_state.py` | Create | Validates all Pydantic models parse and reject bad input |
| `tests/test_broker.py` | Create | `AlpacaBroker` with mocked `TradingClient`; stub raises |
| `tests/test_technical.py` | Create | `run_technical` with mocked `rh` and fake LLM |
| `tests/test_fundamental.py` | Create | `run_fundamental` cache hit/miss + mocked `rh` and fake LLM |
| `tests/test_news.py` | Create | `run_news` with mocked `requests` and fake LLM |
| `tests/test_supervisor.py` | Create | `run_supervisor` with fixed signals and fake LLM |
| `tests/test_graph.py` | Create | Full graph with stub broker and all agents mocked |

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `agents/__init__.py`
- Create: `broker/__init__.py`
- Create: `tests/__init__.py`
- Create: `logs/.gitkeep`

**Interfaces:**
- Produces: `uv sync` installs all deps; `uv run pytest` runs the test suite

- [ ] **Step 1: Initialise uv project**

```bash
cd /Users/senwang/LLMApp/trading_agent
uv init --no-readme --python 3.10
```

Expected: creates `pyproject.toml` and `.python-version`.

- [ ] **Step 2: Replace pyproject.toml with full dependency spec**

Overwrite `pyproject.toml` with:

```toml
[project]
name = "trading-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "langgraph>=0.2",
    "langchain-anthropic>=0.3",
    "langchain-core>=0.3",
    "robin-stocks>=3.0",
    "pandas>=2.0",
    "pandas-ta>=0.3.14b",
    "requests>=2.31",
    "alpaca-py>=0.29",
    "apscheduler>=3.10",
    "pydantic>=2.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync --extra dev
```

Expected: `.venv/` created, all packages installed with no errors.

- [ ] **Step 4: Create directory structure and empty init files**

```bash
mkdir -p agents broker tests logs
touch agents/__init__.py broker/__init__.py tests/__init__.py logs/.gitkeep
```

- [ ] **Step 5: Create `.env.example`**

```bash
cat > .env.example << 'EOF'
ANTHROPIC_API_KEY=your_anthropic_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ROBINHOOD_USERNAME=your_robinhood_email
ROBINHOOD_PASSWORD=your_robinhood_password
POLYGON_API_KEY=your_polygon_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=optional_langsmith_key
LANGCHAIN_PROJECT=trading-agent
WATCHLIST=AAPL,MSFT,NVDA
EOF
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example agents/__init__.py broker/__init__.py tests/__init__.py logs/.gitkeep
git commit -m "feat: scaffold project with uv, deps, and directory structure"
```

---

### Task 2: State models

**Files:**
- Create: `state.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Produces:
  - `TechnicalSignal(ticker, direction, confidence, rsi, macd_signal, current_price, reasoning)`
  - `FundamentalSignal(ticker, direction, confidence, pe_ratio, pb_ratio, market_cap, week_52_high, week_52_low, reasoning)`
  - `NewsSignal(ticker, direction, confidence, material_event, reasoning)`
  - `TradeDecision(ticker, action, size_pct, rationale)`
  - `AgentState` TypedDict with keys: `tickers`, `current_ticker`, `prices`, `technical_signals`, `fundamental_signals`, `news_signals`, `decisions`, `cycle_timestamp`
  - `_merge_dicts(a, b)` reducer used on all signal dicts

- [ ] **Step 1: Write failing tests**

Create `tests/test_state.py`:

```python
import pytest
from pydantic import ValidationError
from state import (
    TechnicalSignal,
    FundamentalSignal,
    NewsSignal,
    TradeDecision,
    AgentState,
    _merge_dicts,
)


def test_technical_signal_valid():
    s = TechnicalSignal(
        ticker="AAPL",
        direction="bullish",
        confidence=0.8,
        rsi=45.0,
        macd_signal="bullish",
        current_price=195.0,
        reasoning="RSI not overbought, MACD crossed up",
    )
    assert s.ticker == "AAPL"
    assert s.direction == "bullish"


def test_technical_signal_rejects_bad_direction():
    with pytest.raises(ValidationError):
        TechnicalSignal(
            ticker="AAPL",
            direction="up",  # invalid
            confidence=0.8,
            rsi=45.0,
            macd_signal="bullish",
            current_price=195.0,
            reasoning="x",
        )


def test_fundamental_signal_nullable_fields():
    s = FundamentalSignal(
        ticker="AAPL",
        direction="neutral",
        confidence=0.5,
        pe_ratio=None,
        pb_ratio=None,
        market_cap=None,
        week_52_high=None,
        week_52_low=None,
        reasoning="no data",
    )
    assert s.pe_ratio is None


def test_news_signal_valid():
    s = NewsSignal(
        ticker="AAPL",
        direction="bearish",
        confidence=0.7,
        material_event=True,
        reasoning="earnings miss",
    )
    assert s.material_event is True


def test_trade_decision_valid():
    d = TradeDecision(ticker="AAPL", action="buy", size_pct=5.0, rationale="strong signals")
    assert d.action == "buy"


def test_trade_decision_rejects_bad_action():
    with pytest.raises(ValidationError):
        TradeDecision(ticker="AAPL", action="short", size_pct=5.0, rationale="x")


def test_merge_dicts():
    a = {"AAPL": "sig_a"}
    b = {"MSFT": "sig_b"}
    assert _merge_dicts(a, b) == {"AAPL": "sig_a", "MSFT": "sig_b"}


def test_merge_dicts_overwrites_on_collision():
    a = {"AAPL": "old"}
    b = {"AAPL": "new"}
    assert _merge_dicts(a, b) == {"AAPL": "new"}
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'state'`

- [ ] **Step 3: Implement `state.py`**

```python
import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel


class TechnicalSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    rsi: float
    macd_signal: str
    current_price: float
    reasoning: str


class FundamentalSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    pe_ratio: float | None
    pb_ratio: float | None
    market_cap: float | None
    week_52_high: float | None
    week_52_low: float | None
    reasoning: str


class NewsSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    material_event: bool
    reasoning: str


class TradeDecision(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    size_pct: float
    rationale: str


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict):
    tickers: list[str]
    current_ticker: str
    prices: Annotated[dict[str, float], _merge_dicts]
    technical_signals: Annotated[dict[str, TechnicalSignal], _merge_dicts]
    fundamental_signals: Annotated[dict[str, FundamentalSignal], _merge_dicts]
    news_signals: Annotated[dict[str, NewsSignal], _merge_dicts]
    decisions: Annotated[list[TradeDecision], operator.add]
    cycle_timestamp: str
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_state.py -v
```

Expected: 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add state models and AgentState TypedDict"
```

---

### Task 3: Broker abstraction + Alpaca implementation + stubs

**Files:**
- Create: `broker/base.py`
- Create: `broker/alpaca.py`
- Create: `broker/robinhood.py`
- Create: `broker/ibkr.py`
- Create: `tests/test_broker.py`

**Interfaces:**
- Consumes: nothing (no state dependency)
- Produces:
  - `BaseBroker` with methods `get_positions() -> list[Position]`, `place_order(ticker, action, qty) -> OrderResult`, `get_portfolio_value() -> float`
  - `Position(ticker, qty, avg_entry_price, current_price)`
  - `OrderResult(order_id, ticker, action, qty, status)`
  - `AlpacaBroker(api_key, secret_key, paper=True)` — concrete implementation
  - `RobinhoodBroker()` and `IBKRBroker()` — raise `NotImplementedError` on all methods

- [ ] **Step 1: Write failing tests**

Create `tests/test_broker.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from broker.base import Position, OrderResult
from broker.alpaca import AlpacaBroker
from broker.robinhood import RobinhoodBroker
from broker.ibkr import IBKRBroker


@pytest.fixture
def mock_trading_client():
    with patch("broker.alpaca.TradingClient") as MockClient:
        client = MagicMock()
        MockClient.return_value = client

        # get_all_positions returns list of position-like objects
        pos = MagicMock()
        pos.symbol = "AAPL"
        pos.qty = "10"
        pos.avg_entry_price = "190.00"
        pos.current_price = "195.00"
        client.get_all_positions.return_value = [pos]

        # get_account returns account with portfolio_value
        account = MagicMock()
        account.portfolio_value = "100000.00"
        client.get_account.return_value = account

        # submit_order returns an order
        order = MagicMock()
        order.id = "order-123"
        order.status = "accepted"
        client.submit_order.return_value = order

        yield client


def test_alpaca_get_positions(mock_trading_client):
    broker = AlpacaBroker("key", "secret", paper=True)
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"
    assert positions[0].qty == 10.0
    assert positions[0].current_price == 195.0


def test_alpaca_get_portfolio_value(mock_trading_client):
    broker = AlpacaBroker("key", "secret", paper=True)
    assert broker.get_portfolio_value() == 100000.0


def test_alpaca_place_order_buy(mock_trading_client):
    broker = AlpacaBroker("key", "secret", paper=True)
    result = broker.place_order("AAPL", "buy", 5.0)
    assert result.order_id == "order-123"
    assert result.action == "buy"
    assert result.status == "accepted"


def test_alpaca_place_order_sell(mock_trading_client):
    broker = AlpacaBroker("key", "secret", paper=True)
    result = broker.place_order("AAPL", "sell", 5.0)
    assert result.action == "sell"


def test_robinhood_stub_raises():
    broker = RobinhoodBroker()
    with pytest.raises(NotImplementedError):
        broker.get_portfolio_value()


def test_ibkr_stub_raises():
    broker = IBKRBroker()
    with pytest.raises(NotImplementedError):
        broker.place_order("AAPL", "buy", 1.0)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_broker.py -v
```

Expected: `ModuleNotFoundError: No module named 'broker.alpaca'`

- [ ] **Step 3: Implement `broker/base.py`**

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel


class Position(BaseModel):
    ticker: str
    qty: float
    avg_entry_price: float
    current_price: float


class OrderResult(BaseModel):
    order_id: str
    ticker: str
    action: str
    qty: float
    status: str


class BaseBroker(ABC):
    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult: ...

    @abstractmethod
    def get_portfolio_value(self) -> float: ...
```

- [ ] **Step 4: Implement `broker/alpaca.py`**

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from broker.base import BaseBroker, OrderResult, Position


class AlpacaBroker(BaseBroker):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_positions(self) -> list[Position]:
        return [
            Position(
                ticker=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
            )
            for p in self.client.get_all_positions()
        ]

    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult:
        side = OrderSide.BUY if action == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(req)
        return OrderResult(
            order_id=str(order.id),
            ticker=ticker,
            action=action,
            qty=qty,
            status=str(order.status),
        )

    def get_portfolio_value(self) -> float:
        return float(self.client.get_account().portfolio_value)
```

- [ ] **Step 5: Implement `broker/robinhood.py`**

```python
from broker.base import BaseBroker, OrderResult, Position


class RobinhoodBroker(BaseBroker):
    """Live Robinhood broker. NOT ACTIVE — raises NotImplementedError on all methods.
    Implement and activate only after paper trading phase is validated."""

    def get_positions(self) -> list[Position]:
        raise NotImplementedError("RobinhoodBroker is not yet activated.")

    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult:
        raise NotImplementedError("RobinhoodBroker is not yet activated.")

    def get_portfolio_value(self) -> float:
        raise NotImplementedError("RobinhoodBroker is not yet activated.")
```

- [ ] **Step 6: Implement `broker/ibkr.py`**

```python
from broker.base import BaseBroker, OrderResult, Position


class IBKRBroker(BaseBroker):
    """Live IBKR broker. NOT ACTIVE — raises NotImplementedError on all methods.
    Implement and activate only after paper trading phase is validated."""

    def get_positions(self) -> list[Position]:
        raise NotImplementedError("IBKRBroker is not yet activated.")

    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult:
        raise NotImplementedError("IBKRBroker is not yet activated.")

    def get_portfolio_value(self) -> float:
        raise NotImplementedError("IBKRBroker is not yet activated.")
```

- [ ] **Step 7: Run tests to confirm pass**

```bash
uv run pytest tests/test_broker.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 8: Commit**

```bash
git add broker/base.py broker/alpaca.py broker/robinhood.py broker/ibkr.py tests/test_broker.py
git commit -m "feat: add broker abstraction, Alpaca implementation, and live broker stubs"
```

---

### Task 4: Technical analyzer agent

**Files:**
- Create: `agents/technical.py`
- Create: `tests/test_technical.py`

**Interfaces:**
- Consumes: `AgentState` with `current_ticker: str` set
- Produces: `run_technical(state: AgentState) -> dict` returning `{"technical_signals": {ticker: TechnicalSignal}, "prices": {ticker: float}}`

- [ ] **Step 1: Write failing tests**

Create `tests/test_technical.py`:

```python
from unittest.mock import MagicMock, patch
from langchain_core.language_models.fake import FakeListChatModel
from state import AgentState, TechnicalSignal


def _make_state(ticker: str = "AAPL") -> AgentState:
    return AgentState(
        tickers=[ticker],
        current_ticker=ticker,
        prices={},
        technical_signals={},
        fundamental_signals={},
        news_signals={},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
    )


FAKE_HISTORICALS = [
    {"close_price": str(190 + i), "open_price": "189", "high_price": "192",
     "low_price": "188", "volume": "1000000", "begins_at": f"2026-06-28T{9+i//12:02d}:{(i*5)%60:02d}:00Z"}
    for i in range(60)
]


def test_run_technical_returns_signal():
    fake_signal = TechnicalSignal(
        ticker="AAPL",
        direction="bullish",
        confidence=0.75,
        rsi=48.0,
        macd_signal="bullish",
        current_price=195.0,
        reasoning="RSI neutral, MACD bullish crossover",
    )

    with patch("agents.technical.rh") as mock_rh, \
         patch("agents.technical.ChatAnthropic") as MockLLM:
        mock_rh.stocks.get_stock_historicals.return_value = FAKE_HISTORICALS

        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm_instance

        from agents.technical import run_technical
        result = run_technical(_make_state())

    assert "technical_signals" in result
    assert "AAPL" in result["technical_signals"]
    assert result["technical_signals"]["AAPL"].direction == "bullish"
    assert "prices" in result
    assert result["prices"]["AAPL"] > 0


def test_run_technical_returns_neutral_on_api_failure():
    with patch("agents.technical.rh") as mock_rh, \
         patch("agents.technical.ChatAnthropic"):
        mock_rh.stocks.get_stock_historicals.side_effect = Exception("API down")

        from agents.technical import run_technical
        result = run_technical(_make_state())

    signal = result["technical_signals"]["AAPL"]
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert "API down" in signal.reasoning
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_technical.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.technical'`

- [ ] **Step 3: Implement `agents/technical.py`**

```python
import pandas as pd
import pandas_ta as ta
import robin_stocks.robinhood as rh
from langchain_anthropic import ChatAnthropic

from state import AgentState, TechnicalSignal

_llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TechnicalSignal)


def run_technical(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    try:
        historicals = rh.stocks.get_stock_historicals(
            ticker, interval="5minute", span="week", bounds="regular"
        )
        df = _to_dataframe(historicals)
        current_price = float(df.iloc[-1]["close_price"])
        summary = _compute_indicators(ticker, df)
        signal = _llm.invoke(
            f"Analyze these technical indicators for {ticker} and return a structured signal.\n\n{summary}"
        )
    except Exception as e:
        return {
            "technical_signals": {
                ticker: TechnicalSignal(
                    ticker=ticker,
                    direction="neutral",
                    confidence=0.0,
                    rsi=50.0,
                    macd_signal="unknown",
                    current_price=0.0,
                    reasoning=f"Data fetch failed: {e}",
                )
            },
            "prices": {ticker: 0.0},
        }
    return {"technical_signals": {ticker: signal}, "prices": {ticker: current_price}}


def _to_dataframe(historicals: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(historicals)
    for col in ["close_price", "open_price", "high_price", "low_price", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df


def _compute_indicators(ticker: str, df: pd.DataFrame) -> str:
    df.ta.rsi(close="close_price", append=True)
    df.ta.macd(close="close_price", append=True)
    df.ta.sma(close="close_price", length=20, append=True)
    df.ta.sma(close="close_price", length=50, append=True)

    latest = df.iloc[-1]
    price = float(latest["close_price"])
    rsi = float(latest.get("RSI_14", 50))
    macd_line = float(latest.get("MACD_12_26_9", 0))
    macd_sig = float(latest.get("MACDs_12_26_9", 0))
    sma20 = float(latest.get("SMA_20", price))
    sma50 = float(latest.get("SMA_50", price))

    return (
        f"Ticker: {ticker}\n"
        f"Current price: {price:.2f}\n"
        f"RSI(14): {rsi:.2f}\n"
        f"MACD line: {macd_line:.4f}, Signal line: {macd_sig:.4f} "
        f"({'bullish' if macd_line > macd_sig else 'bearish'} crossover)\n"
        f"SMA20: {sma20:.2f} ({'price above' if price > sma20 else 'price below'})\n"
        f"SMA50: {sma50:.2f} ({'price above' if price > sma50 else 'price below'})"
    )
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_technical.py -v
```

Expected: 2 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agents/technical.py tests/test_technical.py
git commit -m "feat: add technical analyzer agent with RSI, MACD, SMA via robin_stocks"
```

---

### Task 5: Fundamental analyzer agent

**Files:**
- Create: `agents/fundamental.py`
- Create: `tests/test_fundamental.py`

**Interfaces:**
- Consumes: `AgentState` with `current_ticker: str` set
- Produces: `run_fundamental(state: AgentState) -> dict` returning `{"fundamental_signals": {ticker: FundamentalSignal}}`
- Internal: `_cache: dict[str, tuple[str, FundamentalSignal]]` module-level cache keyed by `(ticker, date)`

- [ ] **Step 1: Write failing tests**

Create `tests/test_fundamental.py`:

```python
import datetime
from unittest.mock import MagicMock, patch
from state import AgentState, FundamentalSignal


def _make_state(ticker: str = "AAPL") -> AgentState:
    return AgentState(
        tickers=[ticker],
        current_ticker=ticker,
        prices={},
        technical_signals={},
        fundamental_signals={},
        news_signals={},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
    )


FAKE_FUNDAMENTALS = [{
    "pe_ratio": "28.5",
    "pb_ratio": None,
    "market_cap": "3000000000000",
    "high_52_weeks": "220.0",
    "low_52_weeks": "165.0",
    "dividend_yield": "0.005",
    "description": "Apple Inc. designs consumer electronics.",
}]


def test_run_fundamental_returns_signal():
    fake_signal = FundamentalSignal(
        ticker="AAPL",
        direction="bullish",
        confidence=0.65,
        pe_ratio=28.5,
        pb_ratio=None,
        market_cap=3e12,
        week_52_high=220.0,
        week_52_low=165.0,
        reasoning="Reasonable PE, near 52-week high",
    )

    with patch("agents.fundamental.rh") as mock_rh, \
         patch("agents.fundamental.ChatAnthropic") as MockLLM:
        mock_rh.stocks.get_fundamentals.return_value = FAKE_FUNDAMENTALS

        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm_instance

        from agents.fundamental import run_fundamental
        result = run_fundamental(_make_state())

    assert "fundamental_signals" in result
    assert result["fundamental_signals"]["AAPL"].direction == "bullish"


def test_run_fundamental_uses_cache_on_second_call():
    fake_signal = FundamentalSignal(
        ticker="AAPL", direction="neutral", confidence=0.5,
        pe_ratio=None, pb_ratio=None, market_cap=None,
        week_52_high=None, week_52_low=None, reasoning="cached",
    )

    import agents.fundamental as mod
    today = datetime.date.today().isoformat()
    mod._cache["AAPL"] = (today, fake_signal)

    with patch("agents.fundamental.rh") as mock_rh, \
         patch("agents.fundamental.ChatAnthropic"):
        result = mod.run_fundamental(_make_state())
        mock_rh.stocks.get_fundamentals.assert_not_called()

    assert result["fundamental_signals"]["AAPL"].reasoning == "cached"

    # cleanup
    del mod._cache["AAPL"]


def test_run_fundamental_returns_neutral_on_failure():
    with patch("agents.fundamental.rh") as mock_rh, \
         patch("agents.fundamental.ChatAnthropic"):
        mock_rh.stocks.get_fundamentals.side_effect = Exception("timeout")

        from agents.fundamental import run_fundamental
        result = run_fundamental(_make_state())

    signal = result["fundamental_signals"]["AAPL"]
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_fundamental.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.fundamental'`

- [ ] **Step 3: Implement `agents/fundamental.py`**

```python
import datetime

import robin_stocks.robinhood as rh
from langchain_anthropic import ChatAnthropic

from state import AgentState, FundamentalSignal

_llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(FundamentalSignal)
_cache: dict[str, tuple[str, FundamentalSignal]] = {}


def run_fundamental(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    today = datetime.date.today().isoformat()

    if ticker in _cache and _cache[ticker][0] == today:
        return {"fundamental_signals": {ticker: _cache[ticker][1]}}

    try:
        data = rh.stocks.get_fundamentals(ticker)[0]
        summary = _format(ticker, data)
        signal = _llm.invoke(
            f"Analyze these fundamental metrics for {ticker} and return a structured signal.\n\n{summary}"
        )
    except Exception as e:
        signal = FundamentalSignal(
            ticker=ticker,
            direction="neutral",
            confidence=0.0,
            pe_ratio=None,
            pb_ratio=None,
            market_cap=None,
            week_52_high=None,
            week_52_low=None,
            reasoning=f"Data fetch failed: {e}",
        )

    _cache[ticker] = (today, signal)
    return {"fundamental_signals": {ticker: signal}}


def _fmt(data: dict, key: str) -> str:
    val = data.get(key)
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _format(ticker: str, data: dict) -> str:
    return (
        f"Ticker: {ticker}\n"
        f"PE ratio: {_fmt(data, 'pe_ratio')}\n"
        f"PB ratio: {_fmt(data, 'pb_ratio')}\n"
        f"Market cap: {_fmt(data, 'market_cap')}\n"
        f"52-week high: {_fmt(data, 'high_52_weeks')}\n"
        f"52-week low: {_fmt(data, 'low_52_weeks')}\n"
        f"Dividend yield: {_fmt(data, 'dividend_yield')}\n"
        f"Description: {str(data.get('description', ''))[:300]}"
    )
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_fundamental.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agents/fundamental.py tests/test_fundamental.py
git commit -m "feat: add fundamental analyzer agent with daily cache via robin_stocks"
```

---

### Task 6: News/sentiment agent

**Files:**
- Create: `agents/news.py`
- Create: `tests/test_news.py`

**Interfaces:**
- Consumes: `AgentState` with `current_ticker: str` set; `POLYGON_API_KEY` in environment
- Produces: `run_news(state: AgentState) -> dict` returning `{"news_signals": {ticker: NewsSignal}}`

- [ ] **Step 1: Write failing tests**

Create `tests/test_news.py`:

```python
import os
from unittest.mock import MagicMock, patch
from state import AgentState, NewsSignal

os.environ.setdefault("POLYGON_API_KEY", "test_key")


def _make_state(ticker: str = "AAPL") -> AgentState:
    return AgentState(
        tickers=[ticker],
        current_ticker=ticker,
        prices={},
        technical_signals={},
        fundamental_signals={},
        news_signals={},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
    )


FAKE_NEWS_RESPONSE = {
    "results": [
        {"published_utc": "2026-06-28T09:00:00Z", "title": "Apple beats earnings",
         "description": "Apple reported record quarterly revenue."},
        {"published_utc": "2026-06-27T15:00:00Z", "title": "AAPL upgrade to buy",
         "description": "Analyst upgrades Apple to buy with $220 target."},
    ]
}


def test_run_news_returns_signal():
    fake_signal = NewsSignal(
        ticker="AAPL",
        direction="bullish",
        confidence=0.8,
        material_event=True,
        reasoning="Earnings beat is a material positive event",
    )

    with patch("agents.news.requests.get") as mock_get, \
         patch("agents.news.ChatAnthropic") as MockLLM:
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_NEWS_RESPONSE
        mock_get.return_value = mock_resp

        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm_instance

        from agents.news import run_news
        result = run_news(_make_state())

    assert "news_signals" in result
    assert result["news_signals"]["AAPL"].direction == "bullish"
    assert result["news_signals"]["AAPL"].material_event is True


def test_run_news_returns_neutral_on_api_failure():
    with patch("agents.news.requests.get") as mock_get, \
         patch("agents.news.ChatAnthropic"):
        mock_get.side_effect = Exception("connection refused")

        from agents.news import run_news
        result = run_news(_make_state())

    signal = result["news_signals"]["AAPL"]
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert signal.material_event is False
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_news.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.news'`

- [ ] **Step 3: Implement `agents/news.py`**

```python
import os

import requests
from langchain_anthropic import ChatAnthropic

from state import AgentState, NewsSignal

_POLYGON_URL = "https://api.polygon.io/v2/reference/news"
_llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(NewsSignal)


def run_news(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    try:
        articles = _fetch(ticker)
        summary = _format(ticker, articles)
        signal = _llm.invoke(
            f"Analyze these recent news items for {ticker} and return a structured sentiment signal.\n\n{summary}"
        )
    except Exception as e:
        signal = NewsSignal(
            ticker=ticker,
            direction="neutral",
            confidence=0.0,
            material_event=False,
            reasoning=f"News fetch failed: {e}",
        )
    return {"news_signals": {ticker: signal}}


def _fetch(ticker: str, limit: int = 10) -> list[dict]:
    resp = requests.get(
        _POLYGON_URL,
        params={"ticker": ticker, "limit": limit, "apiKey": os.environ["POLYGON_API_KEY"]},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _format(ticker: str, articles: list[dict]) -> str:
    if not articles:
        return f"No recent news found for {ticker}."
    lines = [f"Recent news for {ticker} ({len(articles)} articles):"]
    for a in articles:
        date = a.get("published_utc", "")[:10]
        title = a.get("title", "")
        desc = str(a.get("description", ""))[:150]
        lines.append(f"- [{date}] {title} | {desc}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_news.py -v
```

Expected: 2 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agents/news.py tests/test_news.py
git commit -m "feat: add news/sentiment agent via Polygon.io"
```

---

### Task 7: Supervisor agent

**Files:**
- Create: `agents/supervisor.py`
- Create: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: `AgentState` with `tickers`, `technical_signals`, `fundamental_signals`, `news_signals` populated
- Produces: `run_supervisor(state: AgentState) -> dict` returning `{"decisions": [TradeDecision, ...]}`

- [ ] **Step 1: Write failing tests**

Create `tests/test_supervisor.py`:

```python
from unittest.mock import MagicMock, patch
from state import AgentState, TechnicalSignal, FundamentalSignal, NewsSignal, TradeDecision


def _make_state_with_signals() -> AgentState:
    tech = TechnicalSignal(
        ticker="AAPL", direction="bullish", confidence=0.8,
        rsi=45.0, macd_signal="bullish", current_price=195.0,
        reasoning="RSI neutral, MACD bullish",
    )
    fund = FundamentalSignal(
        ticker="AAPL", direction="bullish", confidence=0.6,
        pe_ratio=28.5, pb_ratio=None, market_cap=3e12,
        week_52_high=220.0, week_52_low=165.0, reasoning="Reasonable PE",
    )
    news = NewsSignal(
        ticker="AAPL", direction="bullish", confidence=0.75,
        material_event=False, reasoning="Positive analyst coverage",
    )
    return AgentState(
        tickers=["AAPL"],
        current_ticker="",
        prices={"AAPL": 195.0},
        technical_signals={"AAPL": tech},
        fundamental_signals={"AAPL": fund},
        news_signals={"AAPL": news},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
    )


def test_run_supervisor_returns_decision():
    fake_decision = TradeDecision(
        ticker="AAPL", action="buy", size_pct=5.0, rationale="All three signals bullish",
    )

    with patch("agents.supervisor.ChatAnthropic") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_decision
        MockLLM.return_value = mock_llm_instance

        from agents.supervisor import run_supervisor
        result = run_supervisor(_make_state_with_signals())

    assert "decisions" in result
    assert len(result["decisions"]) == 1
    assert result["decisions"][0].action == "buy"
    assert result["decisions"][0].ticker == "AAPL"


def test_run_supervisor_holds_on_llm_failure():
    with patch("agents.supervisor.ChatAnthropic") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.side_effect = Exception("LLM error")
        MockLLM.return_value = mock_llm_instance

        from agents.supervisor import run_supervisor
        result = run_supervisor(_make_state_with_signals())

    assert result["decisions"][0].action == "hold"
    assert result["decisions"][0].size_pct == 0.0


def test_run_supervisor_handles_missing_signals():
    state = AgentState(
        tickers=["AAPL"],
        current_ticker="",
        prices={},
        technical_signals={},   # no signals
        fundamental_signals={},
        news_signals={},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
    )
    fake_decision = TradeDecision(
        ticker="AAPL", action="hold", size_pct=0.0, rationale="Insufficient data",
    )

    with patch("agents.supervisor.ChatAnthropic") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_decision
        MockLLM.return_value = mock_llm_instance

        from agents.supervisor import run_supervisor
        result = run_supervisor(state)

    assert result["decisions"][0].action == "hold"
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_supervisor.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.supervisor'`

- [ ] **Step 3: Implement `agents/supervisor.py`**

```python
from langchain_anthropic import ChatAnthropic

from state import AgentState, FundamentalSignal, NewsSignal, TechnicalSignal, TradeDecision

_llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TradeDecision)


def run_supervisor(state: AgentState) -> dict:
    decisions = []
    for ticker in state["tickers"]:
        tech = state["technical_signals"].get(ticker)
        fund = state["fundamental_signals"].get(ticker)
        news = state["news_signals"].get(ticker)
        try:
            decision = _llm.invoke(_build_prompt(ticker, tech, fund, news))
        except Exception:
            decision = TradeDecision(
                ticker=ticker, action="hold", size_pct=0.0, rationale="LLM synthesis failed."
            )
        decisions.append(decision)
    return {"decisions": decisions}


def _build_prompt(
    ticker: str,
    tech: TechnicalSignal | None,
    fund: FundamentalSignal | None,
    news: NewsSignal | None,
) -> str:
    return f"""You are a trading supervisor for a paper trading account.
Given the signals below for {ticker}, decide: buy, sell, or hold.
- size_pct is the percentage of total portfolio to allocate (0–10%).
- Default to hold with size_pct=0 when signals conflict or data is missing.
- Never exceed 10% per position.

Technical signal: {tech.model_dump_json() if tech else 'unavailable'}
Fundamental signal: {fund.model_dump_json() if fund else 'unavailable'}
News signal: {news.model_dump_json() if news else 'unavailable'}"""
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_supervisor.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agents/supervisor.py tests/test_supervisor.py
git commit -m "feat: add supervisor agent that synthesizes signals into trade decisions"
```

---

### Task 8: LangGraph graph assembly

**Files:**
- Create: `graph.py`
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: `BaseBroker` instance
- Produces: `build_graph(broker: BaseBroker)` → compiled LangGraph runnable; call `.invoke(initial_state)` to run one cycle

- [ ] **Step 1: Write failing tests**

Create `tests/test_graph.py`:

```python
from unittest.mock import MagicMock, patch
from state import AgentState, TechnicalSignal, FundamentalSignal, NewsSignal, TradeDecision
from broker.base import BaseBroker, Position, OrderResult


class StubBroker(BaseBroker):
    def __init__(self):
        self.orders = []

    def get_positions(self) -> list[Position]:
        return []

    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult:
        self.orders.append((ticker, action, qty))
        return OrderResult(order_id="stub-1", ticker=ticker, action=action, qty=qty, status="filled")

    def get_portfolio_value(self) -> float:
        return 100_000.0


def _make_signals(ticker: str):
    tech = TechnicalSignal(
        ticker=ticker, direction="bullish", confidence=0.8,
        rsi=45.0, macd_signal="bullish", current_price=195.0, reasoning="ok",
    )
    fund = FundamentalSignal(
        ticker=ticker, direction="bullish", confidence=0.6,
        pe_ratio=28.0, pb_ratio=None, market_cap=3e12,
        week_52_high=220.0, week_52_low=165.0, reasoning="ok",
    )
    news = NewsSignal(
        ticker=ticker, direction="bullish", confidence=0.7,
        material_event=False, reasoning="ok",
    )
    return tech, fund, news


def test_build_graph_returns_runnable():
    from graph import build_graph
    broker = StubBroker()
    g = build_graph(broker)
    assert hasattr(g, "invoke")


def test_graph_full_cycle_places_buy_order():
    tech, fund, news = _make_signals("AAPL")
    buy_decision = TradeDecision(ticker="AAPL", action="buy", size_pct=5.0, rationale="all bullish")

    with patch("agents.technical.rh") as mock_rh, \
         patch("agents.technical.ChatAnthropic") as MockTechLLM, \
         patch("agents.fundamental.rh") as mock_frh, \
         patch("agents.fundamental.ChatAnthropic") as MockFundLLM, \
         patch("agents.news.requests.get") as mock_get, \
         patch("agents.news.ChatAnthropic") as MockNewsLLM, \
         patch("agents.supervisor.ChatAnthropic") as MockSupLLM:

        # Technical
        mock_rh.stocks.get_stock_historicals.return_value = [
            {"close_price": str(190 + i), "open_price": "189", "high_price": "192",
             "low_price": "188", "volume": "1000000", "begins_at": "2026-06-28T10:00:00Z"}
            for i in range(60)
        ]
        _setup_llm(MockTechLLM, tech)

        # Fundamental
        mock_frh.stocks.get_fundamentals.return_value = [{"pe_ratio": "28.0"}]
        _setup_llm(MockFundLLM, fund)

        # News
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"published_utc": "2026-06-28T09:00:00Z",
                                                     "title": "Good news", "description": "Positive"}]}
        mock_get.return_value = mock_resp
        _setup_llm(MockNewsLLM, news)

        # Supervisor
        _setup_llm(MockSupLLM, buy_decision)

        broker = StubBroker()
        from graph import build_graph
        g = build_graph(broker)
        initial_state: AgentState = {
            "tickers": ["AAPL"],
            "current_ticker": "",
            "prices": {},
            "technical_signals": {},
            "fundamental_signals": {},
            "news_signals": {},
            "decisions": [],
            "cycle_timestamp": "2026-06-28T10:00:00",
        }
        result = g.invoke(initial_state)

    assert len(result["decisions"]) == 1
    assert result["decisions"][0].action == "buy"
    assert len(broker.orders) == 1  # order was placed


def _setup_llm(MockLLM, return_value):
    instance = MagicMock()
    instance.with_structured_output.return_value.invoke.return_value = return_value
    MockLLM.return_value = instance
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: `ModuleNotFoundError: No module named 'graph'`

- [ ] **Step 3: Implement `graph.py`**

```python
import math

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.fundamental import run_fundamental
from agents.news import run_news
from agents.supervisor import run_supervisor
from agents.technical import run_technical
from broker.base import BaseBroker
from state import AgentState


def build_graph(broker: BaseBroker):
    builder = StateGraph(AgentState)

    def dispatch_tickers(state: AgentState) -> list[Send]:
        return [
            Send("analyze_ticker", {**state, "current_ticker": ticker})
            for ticker in state["tickers"]
        ]

    def analyze_ticker(state: AgentState) -> dict:
        updates: dict = {}
        updates.update(run_technical(state))
        updates.update(run_fundamental(state))
        updates.update(run_news(state))
        return updates

    def execute_trades(state: AgentState) -> dict:
        try:
            portfolio_value = broker.get_portfolio_value()
        except Exception:
            return {}
        for decision in state["decisions"]:
            if decision.action == "hold" or decision.size_pct <= 0:
                continue
            price = state["prices"].get(decision.ticker, 0.0)
            if price <= 0:
                continue
            qty = math.floor((decision.size_pct / 100) * portfolio_value / price)
            if qty < 1:
                continue
            try:
                broker.place_order(decision.ticker, decision.action, float(qty))
            except Exception:
                pass
        return {}

    builder.add_node("dispatch", lambda state: state)
    builder.add_node("analyze_ticker", analyze_ticker)
    builder.add_node("supervisor", run_supervisor)
    builder.add_node("trade_executor", execute_trades)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", dispatch_tickers)
    builder.add_edge("analyze_ticker", "supervisor")
    builder.add_edge("supervisor", "trade_executor")
    builder.add_edge("trade_executor", END)

    return builder.compile()
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: 2 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: assemble LangGraph StateGraph with Send fan-out per ticker"
```

---

### Task 9: Scheduler, decision logger, and entry point

**Files:**
- Create: `scheduler.py`
- Create: `main.py`

**Interfaces:**
- Consumes: `build_graph(broker)` from `graph.py`; `AlpacaBroker` from `broker/alpaca.py`
- Produces: `start_scheduler(tickers: list[str])` — blocking call that runs cycles on schedule; `main()` — CLI entry point

- [ ] **Step 1: Implement `scheduler.py`**

No unit test for the scheduler itself — APScheduler wiring is not meaningfully testable in isolation. The graph test in Task 8 covers the cycle logic.

```python
import datetime
import json
import os
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from broker.alpaca import AlpacaBroker
from graph import build_graph
from state import AgentState


def run_cycle(tickers: list[str], graph) -> None:
    now = datetime.datetime.utcnow()
    print(f"\n[{now.isoformat()}] Starting cycle for: {', '.join(tickers)}")

    initial_state: AgentState = {
        "tickers": tickers,
        "current_ticker": "",
        "prices": {},
        "technical_signals": {},
        "fundamental_signals": {},
        "news_signals": {},
        "decisions": [],
        "cycle_timestamp": now.isoformat(),
    }

    result = graph.invoke(initial_state)
    _log(result)
    _print_summary(result)


def _log(state: AgentState) -> None:
    log_path = Path("logs/decisions.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": state["cycle_timestamp"],
        "tickers": state["tickers"],
        "technical_signals": {k: v.model_dump() for k, v in state["technical_signals"].items()},
        "fundamental_signals": {k: v.model_dump() for k, v in state["fundamental_signals"].items()},
        "news_signals": {k: v.model_dump() for k, v in state["news_signals"].items()},
        "decisions": [d.model_dump() for d in state["decisions"]],
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _print_summary(state: AgentState) -> None:
    print(f"{'─' * 60}")
    for d in state["decisions"]:
        tech = state["technical_signals"].get(d.ticker)
        fund = state["fundamental_signals"].get(d.ticker)
        news = state["news_signals"].get(d.ticker)
        print(
            f"  {d.ticker:6s} → {d.action.upper():4s} {d.size_pct:.1f}%"
            f" | tech={tech.direction if tech else '?'}"
            f" | fund={fund.direction if fund else '?'}"
            f" | news={news.direction if news else '?'}"
        )
        print(f"         rationale: {d.rationale[:100]}")
    print(f"{'─' * 60}")


def start_scheduler(tickers: list[str]) -> None:
    broker = AlpacaBroker(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )
    graph = build_graph(broker)

    # Run one cycle immediately on startup, then every 30 min during market hours
    run_cycle(tickers, graph)

    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        run_cycle,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/30",
                    timezone="America/New_York"),
        args=[tickers, graph],
    )
    print(f"Scheduler running. Next cycle at market hours (Mon–Fri 9:00–15:30 ET).")
    scheduler.start()
```

- [ ] **Step 2: Implement `main.py`**

```python
import os

from dotenv import load_dotenv
import robin_stocks.robinhood as rh

from scheduler import start_scheduler

load_dotenv()


def main() -> None:
    rh.login(
        username=os.environ["ROBINHOOD_USERNAME"],
        password=os.environ["ROBINHOOD_PASSWORD"],
        store_session=True,
    )
    print("Robinhood authenticated.")

    watchlist = [t.strip() for t in os.environ.get("WATCHLIST", "AAPL,MSFT,NVDA").split(",")]
    start_scheduler(watchlist)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run full test suite to verify nothing broken**

```bash
uv run pytest -v
```

Expected: all previously written tests PASSED (test_state, test_broker, test_technical, test_fundamental, test_news, test_supervisor, test_graph).

- [ ] **Step 4: Commit**

```bash
git add scheduler.py main.py
git commit -m "feat: add APScheduler cycle runner and main entry point"
```

---

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md to reflect the built system**

Replace the contents of `CLAUDE.md` with:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An AI-powered intraday trading agent for US stocks and ETFs. Uses LangGraph to orchestrate parallel per-ticker analysis (technical, fundamental, news/sentiment), synthesizes signals via LLM, and executes paper trades through Alpaca.

## Setup

```bash
uv sync --extra dev       # install all dependencies
cp .env.example .env      # fill in credentials
uv run python main.py     # start the agent
```

## Running Tests

```bash
uv run pytest             # full suite
uv run pytest tests/test_technical.py -v   # single file
```

## Architecture

```
main.py → scheduler.py → graph.py (LangGraph StateGraph)
                               ├── analyze_ticker (per ticker, parallel via Send)
                               │     ├── agents/technical.py   (robin_stocks + pandas-ta)
                               │     ├── agents/fundamental.py (robin_stocks, cached daily)
                               │     └── agents/news.py        (Polygon.io)
                               ├── agents/supervisor.py        (LLM synthesis → TradeDecision)
                               └── trade_executor              (broker abstraction)
                                     └── broker/alpaca.py      (active — paper trading)
```

- **State flows** through `AgentState` (TypedDict in `state.py`). All signals are Pydantic models validated before reaching the supervisor.
- **Parallel fan-out** uses LangGraph's `Send` API: one `analyze_ticker` node per ticker, results merged via `_merge_dicts` reducer.
- **Broker abstraction**: swap execution backend by changing the `AlpacaBroker` instantiation in `scheduler.py`. `broker/robinhood.py` and `broker/ibkr.py` are stubs that raise `NotImplementedError`.
- **Decision log**: every cycle appends to `logs/decisions.jsonl`.
- **Fundamentals cache**: `agents/fundamental.py` caches per-ticker per-day in a module-level dict — Robinhood fundamentals are fetched once daily.

## Data Sources

| Source | Library | What it provides |
|---|---|---|
| Robinhood | `robin_stocks` | Real-time quotes, OHLCV bars, fundamentals |
| Polygon.io | `requests` | News headlines + article summaries |

## Environment Variables

See `.env.example` for all required keys. Critical ones:
- `ROBINHOOD_USERNAME` / `ROBINHOOD_PASSWORD` — robin_stocks auth at startup
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` — paper trading
- `POLYGON_API_KEY` — news agent
- `ANTHROPIC_API_KEY` — LLM calls (model: `claude-sonnet-4-6`)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with final architecture and setup instructions"
```
