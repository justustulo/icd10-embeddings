"""Downstream feature interfaces that consume the trained embeddings.

Everything here runs AFTER the embeddings exist (Phase 2). These modules shape
the code/member vectors into features for the ML models you build separately;
they do not train the embeddings themselves.
"""
