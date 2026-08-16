# engine/__init__.py
# Clean public API re-exports for the engine package.

from .metrics import summary
from .backtest import run_backtest

from .strategies import (
    Strategy,
    EqStrategy,
    RiskParityStrategy,
    MVPStrategy,
    Momentum6mStrategy,
    Trend6mStrategy,
    LLMStrategy,
    SharpeWeightedStrategy,
    SortinoWeightedStrategy,
)

from .statsig import (
    build_comparison_table,
    build_significance_table,  # wrapper alias kept for GUI compatibility
    hac_t_test_mean_diff,
    wilcoxon_signed_rank_p,
    jackknife_sharpe_z,
    moving_block_bootstrap_p,
    diebold_mariano,
    reality_check_df,
    spa_df,
)

__all__ = [
    # Core
    "summary",
    "run_backtest",
    # Strategies
    "Strategy",
    "EqStrategy",
    "RiskParityStrategy",
    "MVPStrategy",
    "Momentum12mStrategy",
    "Momentum6mStrategy",
    "Trend10mStrategy",
    "Trend6mStrategy",
    "LLMStrategy",
    "SharpeWeightedStrategy",
    "SortinoWeightedStrategy",
    # Significance / tests
    "build_comparison_table",
    "build_significance_table",
    "hac_t_test_mean_diff",
    "wilcoxon_signed_rank_p",
    "jackknife_sharpe_z",
    "moving_block_bootstrap_p",
    "diebold_mariano",
    "reality_check_df",
    "spa_df",
]
