"""The committed condition configs are exactly what the condition table generates."""
from agents.common import paths
from agents.provider.cost_estimators import ESTIMATOR_MODES
from agents.provider.policies import POLICIES
from experiments import generate_conditions as G


def test_configs_and_table_match_the_generator():
    base = G.load(G.BASE)
    for row in G.CONDITIONS:
        assert (paths.CONDITIONS_DIR / f"{row[0]}.yaml").read_text() == G.to_yaml(G.render(base, row)), row[0]
    assert G.TABLE.read_text() == G.table()
    committed = {p.stem for p in paths.CONDITIONS_DIR.glob("*.yaml")} - {"base"}
    assert committed == {row[0] for row in G.CONDITIONS}


def test_every_strategy_is_runnable():
    for name, (_, behaviour) in G.STRATEGIES.items():
        if behaviour["reasoning_mode"] == "heuristic":
            assert behaviour["heuristic_strategy"] in POLICIES, name
        assert behaviour["cost_estimator"] in ESTIMATOR_MODES, name


def test_secrets_are_placeholders():
    """Keys are substituted from the environment at run time, never written into a config."""
    text = G.BASE.read_text()
    assert '"${BLOCKCHAIN_PRIVATE_KEY}"' in text and '"${BLOCKCHAIN_CONSUMER_PRIVATE_KEY}"' in text
