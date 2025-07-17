# Archived ADRs

This folder contains archived ADRs. This keeps ADRs under `ADRs/cartesian/` valid at all time. Because old decisions might still be useful (e.g. to link to in currently valid ADRs), we keep them around in this clearly marked folder.

ADRs can end up here in two cases:

## Superseding ADR

Example: Feature F replaces both previous features E and D.

Process:

- Move ADRs of features D and E in this folder change their status to "deprecated".
- Create a new ADR for feature F.
- Link from ADRs of features D and E to ADR of feature F (at the top)
- Link from ADR of feature F to ADRs of features D and E (e.g. in the references section at the bottom)

Rationale:

All ADRs under `ADRs/cartesian/` should always be valid and up-to-date. Linking the old ADRs highlights old decisions and thus combats "back and forth decision making".

## Not applicable anymore

Example: Feature F is deprecated and removed after a grace period.

Process:

- Move ADR of feature F in this folder and change its status to "deprecated".
- Check for references (to the ADR of feature F) and update them accordingly.

Rationale:

All ADRs under `ADRs/cartesian/` are always valid. Revisiting e.g. the decision to deprecate a feature should be easily accessible. Restoring an ADR from the git history is way more complicated than looking in the `archived/` folder.

## GDPs

GDP is short for _GT4Py Development Proposal_ and was the predecessor of ADRs. We keep [these documents](./GDPs/) for reference only.
