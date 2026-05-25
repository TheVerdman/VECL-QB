# Trust Anchor Contract

Trust roots are exogenous. They come from regulatory credentials, governance approval, verified operational records, human experts, system policy, or test fixtures.

Sleep-cycle consolidation may update learned trust state inside bounded rules, but it must not create a `TrustAnchor`.

Expired or revoked anchors contribute zero root trust. v0 combines multiple active anchors with capped max:

```text
root_trust(source) = min(1.0, max(active_anchor.trust_value))
```
