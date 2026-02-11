"""palaestrAI environment — optional dependency."""

try:
    from provider_sim.env.environment import ProviderEnvironment
except ImportError:
    pass

try:
    from provider_sim.env.objectives import AttackerObjective, DefenderObjective
except ImportError:
    pass
