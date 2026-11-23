"""Stall attribution: engine stall ledger against ncu warp-stall sampling.

Design: dev/design/stall-attribution.md.  Units are cycles per warp;
classes are the ncu warp-stall classes; regions are ``loop`` (top-level
loops of the traced launch that do work) and ``outside``.
"""
