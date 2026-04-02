#!/usr/bin/env python3
"""
Risk Metrics — Advanced risk analytics for trading performance evaluation.
12 metrics + viability gate. No external dependencies beyond numpy/scipy.

Research basis:
- Sortino Ratio: Frank Sortino (1994), penalizes only downside
- Calmar Ratio: Terry Young (1991), return per max drawdown
- Omega Ratio: Keating & Shadwick (2002), full distribution
- MAE/MFE: John Sweeney (1996), optimal SL/TP calibration
- SQN: Van Tharp "Trade Your Way to Financial Freedom"
- VaR/CVaR: JP Morgan RiskMetrics (1994)
- Kelly: Kelly (1956), optimal position sizing
- Risk of Ruin: Ralph Vince "Portfolio Management Formulas"

SOLID Principles:
- SRP: Each metric is its own class
- OCP: New metrics added without modifying existing
- DIP: RiskEngine depends on MetricBase interface
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

log = logging.getLogger("risk.metrics")


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Single trade record."""
    symbol: str
    entry_price: float
    exit_price: float
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    side: str = "long"  # "long" or "short"
    quantity: float = 1.0
    commission: float = 0.001  # 0.1%

    @property
    def pnl(self) -> float:
        if self.side == "long":
            gross = (self.exit_price - self.entry_price) * self.quantity
        else:
            gross = (self.entry_price - self.exit_price) * self.quantity
        cost = (self.entry_price + self.exit_price) * self.quantity * self.commission
        return gross - cost

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == "long":
            return (self.exit_price / self.entry_price - 1) * 100 - self.commission * 200
        return (self.entry_price / self.exit_price - 1) * 100 - self.commission * 200

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class TradeList:
    """Collection of trades with basic statistics."""
    trades: List[Trade] = field(default_factory=list)

    def add(self, trade: Trade):
        self.trades.append(trade)

    @property
    def pnls(self) -> np.ndarray:
        return np.array([t.pnl for t in self.trades])

    @property
    def pnl_pcts(self) -> np.ndarray:
        return np.array([t.pnl_pct for t in self.trades])

    @property
    def winners(self) -> List[Trade]:
        return [t for t in self.trades if t.is_winner]

    @property
    def losers(self) -> List[Trade]:
        return [t for t in self.trades if not t.is_winner]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.winners) / len(self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.winners]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [abs(t.pnl) for t in self.losers]
        return np.mean(losses) if losses else 0.0

    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        if not self.trades:
            return 0.0
        return np.mean(self.pnls)

    @property
    def total_pnl(self) -> float:
        return float(np.sum(self.pnls))

    def from_pnl_list(self, pnls: List[float], symbol: str = "TEST"):
        """Create TradeList from simple PnL list."""
        self.trades = []
        for pnl in pnls:
            price = 100.0
            exit_p = price + pnl
            self.trades.append(Trade(symbol=symbol, entry_price=price,
                                      exit_price=exit_p, commission=0))
        return self


# ═══════════════════════════════════════════════════════════════════
# Interface
# ═══════════════════════════════════════════════════════════════════

class MetricBase(ABC):
    """Base interface for risk metrics."""

    @abstractmethod
    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def higher_is_better(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════
# 1. Sortino Ratio — Sortino (1994)
# ═══════════════════════════════════════════════════════════════════

class SortinoRatio(MetricBase):
    """
    Sortino = (Mean Return - MAR) / Downside Deviation.
    Better than Sharpe: only penalizes downside volatility.
    > 2.0 = excellent, > 1.0 = good, < 0 = losing money.
    """

    def __init__(self, mar: float = 0.0, annualization: float = np.sqrt(365)):
        self.mar = mar  # Minimum Acceptable Return
        self.annualization = annualization

    @property
    def name(self) -> str:
        return "sortino"

    @property
    def description(self) -> str:
        return "Return per unit of downside risk"

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if equity_curve is not None and len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
        elif len(trades.trades) > 1:
            returns = trades.pnl_pcts / 100
        else:
            return 0.0

        excess = returns - self.mar
        downside = returns[returns < self.mar]
        if len(downside) == 0:
            return 10.0  # No downside — perfect

        downside_dev = np.sqrt(np.mean(downside ** 2))
        if downside_dev == 0:
            return 10.0

        return float(np.mean(excess) / downside_dev * self.annualization)


# ═══════════════════════════════════════════════════════════════════
# 2. Calmar Ratio — Young (1991)
# ═══════════════════════════════════════════════════════════════════

class CalmarRatio(MetricBase):
    """
    Calmar = CAGR / Max Drawdown.
    Measures return relative to worst-case scenario.
    > 3.0 = excellent, > 1.0 = good.
    """

    @property
    def name(self) -> str:
        return "calmar"

    @property
    def description(self) -> str:
        return "Annualized return per max drawdown"

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if equity_curve is None or len(equity_curve) < 2:
            return 0.0

        # Simple annualized return (avoid CAGR overflow on short periods)
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        n_periods = len(equity_curve)
        if n_periods < 2:
            return 0.0
        # Annualize: assume hourly data
        annual_factor = 365 * 24 / max(n_periods, 1)
        cagr = total_return * annual_factor

        # Max Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown)

        if max_dd == 0:
            return 10.0  # No drawdown

        return float(cagr / max_dd)


# ═══════════════════════════════════════════════════════════════════
# 3. Omega Ratio — Keating & Shadwick (2002)
# ═══════════════════════════════════════════════════════════════════

class OmegaRatio(MetricBase):
    """
    Omega = Sum(gains above threshold) / Sum(losses below threshold).
    Captures the FULL return distribution (not just mean/variance).
    > 1.5 = excellent, > 1.0 = profitable.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "omega"

    @property
    def description(self) -> str:
        return "Probability-weighted gains vs losses"

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if equity_curve is not None and len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
        elif len(trades.trades) > 1:
            returns = trades.pnl_pcts / 100
        else:
            return 1.0

        gains = returns[returns > self.threshold] - self.threshold
        losses = self.threshold - returns[returns <= self.threshold]

        sum_losses = np.sum(losses)
        if sum_losses == 0:
            return 10.0  # No losses

        return float(np.sum(gains) / sum_losses)


# ═══════════════════════════════════════════════════════════════════
# 4. Maximum Adverse Excursion (MAE) — Sweeney (1996)
# ═══════════════════════════════════════════════════════════════════

class MaxAdverseExcursion(MetricBase):
    """
    MAE: worst unrealized loss during each trade.
    Use 95th percentile to calibrate optimal stop-loss.
    Directly improves TrailingStop parameters.
    """

    def __init__(self, percentile: float = 95):
        self.percentile = percentile

    @property
    def name(self) -> str:
        return "mae"

    @property
    def description(self) -> str:
        return "Optimal stop-loss level from trade data"

    @property
    def higher_is_better(self) -> bool:
        return False  # Lower MAE = tighter SL possible

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if not trades.trades:
            return 0.0

        # Approximate MAE from PnL distribution
        losses = [abs(t.pnl_pct) for t in trades.losers]
        if not losses:
            return 0.0

        return float(np.percentile(losses, self.percentile))

    def optimal_stop_loss(self, trades: TradeList) -> float:
        """Return suggested stop-loss percentage."""
        mae = self.compute(trades)
        # Set SL slightly beyond typical adverse excursion
        return round(mae * 1.1, 2)


# ═══════════════════════════════════════════════════════════════════
# 5. Maximum Favorable Excursion (MFE)
# ═══════════════════════════════════════════════════════════════════

class MaxFavorableExcursion(MetricBase):
    """
    MFE: best unrealized profit during each trade.
    Use to calibrate optimal take-profit levels.
    Complements MAE for complete SL/TP optimization.
    """

    def __init__(self, percentile: float = 75):
        self.percentile = percentile

    @property
    def name(self) -> str:
        return "mfe"

    @property
    def description(self) -> str:
        return "Optimal take-profit level from trade data"

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if not trades.trades:
            return 0.0

        gains = [t.pnl_pct for t in trades.winners]
        if not gains:
            return 0.0

        return float(np.percentile(gains, self.percentile))

    def optimal_take_profit(self, trades: TradeList) -> float:
        """Return suggested take-profit percentage."""
        mfe = self.compute(trades)
        return round(mfe * 0.8, 2)  # Take 80% of typical favorable move


# ═══════════════════════════════════════════════════════════════════
# 6. System Quality Number (SQN) — Van Tharp
# ═══════════════════════════════════════════════════════════════════

class SystemQualityNumber(MetricBase):
    """
    SQN = sqrt(N) × (Mean R / StdDev R), where R = R-multiple (PnL/risk).
    < 1.7: Poor system
    1.7-2.5: Average
    2.5-3.0: Good
    3.0-5.0: Excellent
    > 5.0: Superb (Holy Grail)
    """

    @property
    def name(self) -> str:
        return "sqn"

    @property
    def description(self) -> str:
        return "System quality (Van Tharp scale)"

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        pnls = trades.pnls
        if len(pnls) < 10:
            return 0.0

        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)
        if std_pnl == 0:
            return 0.0

        # Use min(N, 100) to prevent over-inflating with many trades
        n = min(len(pnls), 100)
        return float(np.sqrt(n) * mean_pnl / std_pnl)

    def quality_label(self, sqn: float) -> str:
        if sqn < 1.7:
            return "Poor"
        elif sqn < 2.5:
            return "Average"
        elif sqn < 3.0:
            return "Good"
        elif sqn < 5.0:
            return "Excellent"
        return "Superb"


# ═══════════════════════════════════════════════════════════════════
# 7. Profit Factor
# ═══════════════════════════════════════════════════════════════════

class ProfitFactor(MetricBase):
    """
    Profit Factor = Gross Profit / Gross Loss.
    > 2.0 = excellent, > 1.5 = good, < 1.0 = losing system.
    """

    @property
    def name(self) -> str:
        return "profit_factor"

    @property
    def description(self) -> str:
        return "Gross profit divided by gross loss"

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        pnls = trades.pnls
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = abs(np.sum(pnls[pnls < 0]))

        if gross_loss == 0:
            return 10.0 if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)


# ═══════════════════════════════════════════════════════════════════
# 8. Tail Ratio
# ═══════════════════════════════════════════════════════════════════

class TailRatio(MetricBase):
    """
    Tail Ratio = 95th percentile return / abs(5th percentile return).
    Measures asymmetry: > 1.0 means right tail (gains) is fatter than left (losses).
    """

    @property
    def name(self) -> str:
        return "tail_ratio"

    @property
    def description(self) -> str:
        return "Return distribution asymmetry"

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if equity_curve is not None and len(equity_curve) > 10:
            returns = np.diff(equity_curve) / equity_curve[:-1]
        elif len(trades.trades) > 10:
            returns = trades.pnl_pcts / 100
        else:
            return 1.0

        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))

        if left_tail == 0:
            return 10.0 if right_tail > 0 else 1.0

        return float(right_tail / left_tail)


# ═══════════════════════════════════════════════════════════════════
# 9. Value at Risk (VaR) — parametric + historical
# ═══════════════════════════════════════════════════════════════════

class ValueAtRisk(MetricBase):
    """
    VaR: maximum expected loss at given confidence level.
    Parametric: assumes normal distribution (fast but inaccurate for crypto).
    Historical: empirical distribution (better for fat tails).
    """

    def __init__(self, confidence: float = 0.95, method: str = "historical"):
        self.confidence = confidence
        self.method = method

    @property
    def name(self) -> str:
        return "var"

    @property
    def description(self) -> str:
        return f"Maximum expected loss at {self.confidence*100:.0f}% confidence"

    @property
    def higher_is_better(self) -> bool:
        return False  # Lower VaR = less risk

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if equity_curve is not None and len(equity_curve) > 10:
            returns = np.diff(equity_curve) / equity_curve[:-1]
        elif len(trades.trades) > 10:
            returns = trades.pnl_pcts / 100
        else:
            return 0.0

        if self.method == "parametric":
            from scipy.stats import norm
            mu = np.mean(returns)
            sigma = np.std(returns)
            return float(abs(mu + sigma * norm.ppf(1 - self.confidence)))
        else:
            # Historical: percentile of actual returns
            return float(abs(np.percentile(returns, (1 - self.confidence) * 100)))


# ═══════════════════════════════════════════════════════════════════
# 10. CVaR / Expected Shortfall
# ═══════════════════════════════════════════════════════════════════

class ExpectedShortfall(MetricBase):
    """
    CVaR: average loss BEYOND the VaR threshold.
    More informative than VaR for crypto's fat-tailed distributions.
    Shows the expected loss in the worst-case scenarios.
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    @property
    def name(self) -> str:
        return "cvar"

    @property
    def description(self) -> str:
        return f"Average loss beyond {self.confidence*100:.0f}% VaR"

    @property
    def higher_is_better(self) -> bool:
        return False

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if equity_curve is not None and len(equity_curve) > 10:
            returns = np.diff(equity_curve) / equity_curve[:-1]
        elif len(trades.trades) > 10:
            returns = trades.pnl_pcts / 100
        else:
            return 0.0

        var_threshold = np.percentile(returns, (1 - self.confidence) * 100)
        tail_losses = returns[returns <= var_threshold]

        if len(tail_losses) == 0:
            return 0.0

        return float(abs(np.mean(tail_losses)))


# ═══════════════════════════════════════════════════════════════════
# 11. Risk of Ruin — Ralph Vince
# ═══════════════════════════════════════════════════════════════════

class RiskOfRuin(MetricBase):
    """
    Probability of account destruction.
    RoR = ((1-edge)/(1+edge))^units
    where edge = win_rate * payoff_ratio - (1-win_rate).
    < 1% = excellent, < 5% = acceptable, > 10% = dangerous.
    """

    def __init__(self, ruin_level: float = 0.5):
        self.ruin_level = ruin_level  # 50% of account = ruin

    @property
    def name(self) -> str:
        return "risk_of_ruin"

    @property
    def description(self) -> str:
        return f"Probability of losing {self.ruin_level*100:.0f}% of account"

    @property
    def higher_is_better(self) -> bool:
        return False

    def compute(self, trades: TradeList, equity_curve: Optional[np.ndarray] = None) -> float:
        if len(trades.trades) < 10:
            return 50.0  # Unknown = assume high risk

        wr = trades.win_rate
        avg_w = trades.avg_win
        avg_l = trades.avg_loss

        if avg_l == 0:
            return 0.0  # No losses = no ruin risk

        payoff = avg_w / avg_l
        edge = wr * payoff - (1 - wr)

        if edge <= 0:
            return 100.0  # Negative edge = certain ruin

        # Units to ruin (approximate)
        units = int(1 / (edge * 0.01))  # How many trades to ruin at 1% risk
        units = max(1, min(units, 1000))

        # RoR formula
        if edge >= 1:
            return 0.0

        ror = ((1 - edge) / (1 + edge)) ** units
        return float(min(ror * 100, 100))


# ═══════════════════════════════════════════════════════════════════
# 12. Trade Statistics (comprehensive summary)
# ═══════════════════════════════════════════════════════════════════

class TradeStatistics:
    """Comprehensive trade statistics — not a metric, but a data aggregator."""

    def compute(self, trades: TradeList) -> Dict:
        if not trades.trades:
            return {"n_trades": 0}

        pnls = trades.pnls

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)

        return {
            "n_trades": len(trades.trades),
            "win_rate": round(trades.win_rate * 100, 1),
            "total_pnl": round(trades.total_pnl, 4),
            "avg_win": round(trades.avg_win, 4),
            "avg_loss": round(trades.avg_loss, 4),
            "expectancy": round(trades.expectancy, 4),
            "payoff_ratio": round(trades.avg_win / max(trades.avg_loss, 0.001), 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "largest_win": round(float(np.max(pnls)), 4) if len(pnls) > 0 else 0,
            "largest_loss": round(float(np.min(pnls)), 4) if len(pnls) > 0 else 0,
            "avg_trade": round(float(np.mean(pnls)), 4),
            "std_trade": round(float(np.std(pnls)), 4),
        }


# ═══════════════════════════════════════════════════════════════════
# Viability Gate — Should this system go live?
# ═══════════════════════════════════════════════════════════════════

class ViabilityGate:
    """
    Decision gate: is this trading system viable for deployment?
    All conditions must be met.
    Similar to PBO gate in daily_retrain_v3.py but for system-level.
    """

    def __init__(self, min_sqn: float = 1.7, min_profit_factor: float = 1.2,
                 max_risk_of_ruin: float = 10.0, min_trades: int = 30):
        self.min_sqn = min_sqn
        self.min_profit_factor = min_profit_factor
        self.max_risk_of_ruin = max_risk_of_ruin
        self.min_trades = min_trades

    def evaluate(self, trades: TradeList,
                 equity_curve: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
        """Returns (is_viable, list_of_reasons_if_not)."""
        reasons = []

        if len(trades.trades) < self.min_trades:
            reasons.append(f"Insufficient trades: {len(trades.trades)} < {self.min_trades}")

        sqn = SystemQualityNumber().compute(trades)
        if sqn < self.min_sqn:
            reasons.append(f"SQN too low: {sqn:.2f} < {self.min_sqn}")

        pf = ProfitFactor().compute(trades)
        if pf < self.min_profit_factor:
            reasons.append(f"Profit Factor too low: {pf:.2f} < {self.min_profit_factor}")

        ror = RiskOfRuin().compute(trades)
        if ror > self.max_risk_of_ruin:
            reasons.append(f"Risk of Ruin too high: {ror:.1f}% > {self.max_risk_of_ruin}%")

        if trades.expectancy <= 0:
            reasons.append(f"Negative expectancy: {trades.expectancy:.4f}")

        return len(reasons) == 0, reasons


# ═══════════════════════════════════════════════════════════════════
# Facade — RiskEngine
# ═══════════════════════════════════════════════════════════════════

class RiskEngine:
    """Facade for all risk metrics computation."""

    def __init__(self):
        self.metrics: List[MetricBase] = [
            SortinoRatio(),
            CalmarRatio(),
            OmegaRatio(),
            MaxAdverseExcursion(),
            MaxFavorableExcursion(),
            SystemQualityNumber(),
            ProfitFactor(),
            TailRatio(),
            ValueAtRisk(),
            ExpectedShortfall(),
            RiskOfRuin(),
        ]
        self.trade_stats = TradeStatistics()
        self.viability_gate = ViabilityGate()

    def compute_all(self, trades: TradeList,
                     equity_curve: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}
        for metric in self.metrics:
            try:
                results[metric.name] = round(metric.compute(trades, equity_curve), 4)
            except Exception as e:
                log.warning(f"Metric {metric.name} failed: {e}")
                results[metric.name] = 0.0
        return results

    def full_report(self, trades: TradeList,
                     equity_curve: Optional[np.ndarray] = None) -> Dict:
        """Generate comprehensive risk report."""
        metrics = self.compute_all(trades, equity_curve)
        stats = self.trade_stats.compute(trades)
        viable, reasons = self.viability_gate.evaluate(trades, equity_curve)

        sqn_val = metrics.get("sqn", 0)
        sqn_label = SystemQualityNumber().quality_label(sqn_val)

        # MAE/MFE optimal SL/TP
        mae = MaxAdverseExcursion()
        mfe = MaxFavorableExcursion()
        optimal_sl = mae.optimal_stop_loss(trades)
        optimal_tp = mfe.optimal_take_profit(trades)

        return {
            "metrics": metrics,
            "statistics": stats,
            "system_quality": sqn_label,
            "is_viable": viable,
            "viability_reasons": reasons,
            "optimal_stop_loss_pct": optimal_sl,
            "optimal_take_profit_pct": optimal_tp,
        }

    def dashboard_summary(self, trades: TradeList,
                           equity_curve: Optional[np.ndarray] = None) -> str:
        """Format for dashboard display."""
        report = self.full_report(trades, equity_curve)
        m = report["metrics"]
        s = report["statistics"]

        lines = [
            f"Trades: {s.get('n_trades', 0)} | WR: {s.get('win_rate', 0)}%",
            f"Sortino: {m.get('sortino', 0):.2f} | Calmar: {m.get('calmar', 0):.2f}",
            f"Omega: {m.get('omega', 0):.2f} | PF: {m.get('profit_factor', 0):.2f}",
            f"SQN: {m.get('sqn', 0):.2f} ({report['system_quality']})",
            f"VaR(95%): {m.get('var', 0)*100:.2f}% | CVaR: {m.get('cvar', 0)*100:.2f}%",
            f"RoR: {m.get('risk_of_ruin', 0):.1f}%",
            f"Optimal SL: {report['optimal_stop_loss_pct']}% | TP: {report['optimal_take_profit_pct']}%",
            f"Viable: {'YES' if report['is_viable'] else 'NO — ' + '; '.join(report['viability_reasons'])}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("RISK METRICS TEST — 12 metrics + viability gate")
    print("=" * 70)

    # Generate realistic trade data
    np.random.seed(42)
    n_trades = 100

    # Slightly profitable system: 55% win rate, 1.3:1 payoff
    trades = TradeList()
    for i in range(n_trades):
        is_win = np.random.random() < 0.55
        if is_win:
            pnl_pct = np.random.uniform(0.5, 5.0)
        else:
            pnl_pct = -np.random.uniform(0.3, 3.5)

        entry = 50000 + np.random.randn() * 1000
        exit_p = entry * (1 + pnl_pct / 100)
        trades.add(Trade(symbol="BTCUSDT", entry_price=entry, exit_price=exit_p,
                          commission=0.001))

    # Generate equity curve
    equity = [100.0]
    for t in trades.trades:
        equity.append(equity[-1] * (1 + t.pnl_pct / 100))
    equity_curve = np.array(equity)

    # Run engine
    engine = RiskEngine()
    report = engine.full_report(trades, equity_curve)

    print(f"\n  Trade Statistics:")
    for k, v in report["statistics"].items():
        print(f"    {k}: {v}")

    print(f"\n  Risk Metrics:")
    for k, v in report["metrics"].items():
        direction = "↑" if any(m.name == k and m.higher_is_better for m in engine.metrics) else "↓"
        print(f"    {k}: {v:.4f} {direction}")

    print(f"\n  System Quality: {report['system_quality']}")
    print(f"  Viable: {report['is_viable']}")
    if not report['is_viable']:
        print(f"  Reasons: {report['viability_reasons']}")

    print(f"\n  Optimal Stop-Loss: {report['optimal_stop_loss_pct']}%")
    print(f"  Optimal Take-Profit: {report['optimal_take_profit_pct']}%")

    print(f"\n  Dashboard Summary:")
    print(engine.dashboard_summary(trades, equity_curve))

    print(f"\n  Total lines: {sum(1 for _ in open(__file__))}")
    print("=" * 70)
