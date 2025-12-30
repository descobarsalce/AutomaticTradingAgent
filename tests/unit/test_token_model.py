import pytest

torch = pytest.importorskip("torch")

from src.core.market_token_model import MarketTokenModel
from src.core.risk_budget_manager import RiskBudgetManager
from src.core.token_schema import TokenBatch, TokenSchema


def _build_batch(schema: TokenSchema, batch_size: int = 2) -> TokenBatch:
    global_tokens = torch.randn(batch_size, *schema.global_shape)
    symbol_tokens = torch.randn(batch_size, *schema.symbol_shape)
    drift_features = torch.randn(batch_size, schema.global_feature_dim)
    return TokenBatch(global_tokens=global_tokens, symbol_tokens=symbol_tokens, drift_features=drift_features)


def test_forward_outputs_and_gate():
    schema = TokenSchema(global_feature_dim=8, symbol_feature_dim=6, num_symbols=3)
    model = MarketTokenModel(schema, d_model=16, nhead=2, num_layers=1)
    batch = _build_batch(schema)

    outputs = model(batch)

    assert "drift_gate" in outputs
    gate = outputs["drift_gate"].detach()
    assert torch.all(gate >= 0) and torch.all(gate <= 1)
    assert outputs["return_quantiles"].shape[-1] == 3
    assert outputs["idiosyncratic_vol"].shape[1] == schema.num_symbols


def test_loss_computation_matches_shapes():
    schema = TokenSchema(global_feature_dim=4, symbol_feature_dim=5, num_symbols=2)
    model = MarketTokenModel(schema, d_model=12, nhead=2, num_layers=1)
    batch = _build_batch(schema)
    outputs = model(batch)

    targets = {
        "realized_volatility": torch.zeros_like(outputs["realized_volatility"]),
        "dispersion": torch.zeros_like(outputs["dispersion"]),
        "correlation_proxy": torch.zeros_like(outputs["correlation_proxy"]),
        "liquidity_regime": torch.zeros(outputs["liquidity_regime"].shape[0], dtype=torch.long),
        "return_quantiles": torch.zeros_like(outputs["return_quantiles"]),
        "idiosyncratic_vol": torch.zeros_like(outputs["idiosyncratic_vol"]),
    }

    total, loss_dict = model.compute_losses(outputs, targets)
    assert total.item() >= 0
    assert set(loss_dict.keys()) == {
        "contrastive",
        "realized_vol",
        "dispersion",
        "correlation_proxy",
        "liq_regime",
        "quantiles",
        "idio_vol",
    }


def test_risk_budget_routing_and_allocation():
    schema = TokenSchema(global_feature_dim=4, symbol_feature_dim=4, num_symbols=3)
    manager = RiskBudgetManager()
    model = MarketTokenModel(schema, d_model=10, nhead=2, num_layers=1)
    model.attach_risk_budget_manager(manager)

    batch = _build_batch(schema, batch_size=4)
    model(batch)

    budgets = manager.compute_budgets()
    assert len(budgets) == schema.num_symbols
    assert torch.isclose(torch.tensor(sum(budgets.values())), torch.tensor(manager.config.total_risk_budget))
