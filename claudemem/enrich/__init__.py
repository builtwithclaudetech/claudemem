"""claudemem.enrich — the only model-touching layer (L3).

Package marker only; no eager submodule imports (tech-design §6.3). Reachable
only from write-flow dispatch and the SessionEnd reflection path (architecture §4).
"""
