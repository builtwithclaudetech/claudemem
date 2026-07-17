"""claudemem — stateless, file-based memory system for Claude Code.

This package intentionally performs NO eager submodule imports (tech-design
§6.3, architecture §4.2). Importing ``claudemem`` must stay thin to protect the
SC-2 cold-start budget and the read-path / enrich firewall (SC-6, C-17).
Submodules are imported lazily by ``claudemem.cli`` at dispatch time.
"""

__version__ = "0.1.0"
