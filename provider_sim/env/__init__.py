"""palaestrAI environment — optional dependency."""

try:
    from provider_sim.env.environment import ProviderEnvironment
except ImportError:
    pass
