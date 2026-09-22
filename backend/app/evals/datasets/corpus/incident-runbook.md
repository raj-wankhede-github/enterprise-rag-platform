# Production Incident Runbook

Reference OPS-2025-02.

## Declaring an Incident

Any engineer may declare an incident. Declaring early is always preferred to waiting for
certainty. Declare in the incident channel with the severity and a one-line summary.

## Roles

An incident has three roles. The incident commander coordinates and makes decisions. The
communications lead handles updates to stakeholders. The operations lead makes the changes. One
person may hold two roles on a small incident, but the incident commander must never also be the
operations lead.

## Escalation Timeline

| Elapsed time | Action |
| --- | --- |
| 0 minutes | Declare, assign incident commander |
| 15 minutes | Page the on-call engineering manager |
| 30 minutes | Notify the head of engineering |
| 60 minutes | Notify the executive team and prepare customer communication |

## Post-Incident Review

A written review is required for every S1 and S2 incident within five working days. Reviews are
blameless and focus on contributing conditions rather than individual error.
