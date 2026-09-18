# Context

Glossary for the hero-car matting pipeline. Terms only — no implementation detail.

## Hero Car

The single vehicle a photograph is *about*. In a dealer photo the frame usually contains
several vehicles; exactly one of them is the subject. A vehicle qualifies as the Hero Car
by being large in frame, **fully contained within the frame**, and near its centre.
Largest-by-area alone is not sufficient: a vehicle cropped by the frame edge can occupy
more pixels than the subject while plainly not being it.

There is at most one Hero Car per image, and there may be none.

## Candidate

Any vehicle the detector finds, before selection. Every Candidate carries the three
measurements that decide the Hero Car — see [[Completeness]], Area Fraction, Centrality —
plus the detector's own confidence. Candidates that lose are still recorded, so a
selection can be audited after the fact.

Note that *vehicle* here spans saloons, hatchbacks, SUVs, pickups and vans; the detector's
own class names (`car` / `truck` / `bus`) are an implementation artefact, not a domain
distinction.

## Completeness

How much of a Candidate lies inside the frame. A vehicle touching no frame edge is fully
complete; completeness falls as more of its outline runs off an edge. This is the term
that encodes "full car" and it outranks size in selection.

## Alpha Matte

The per-pixel opacity of the Hero Car, continuous from 0 to 1 — not a binary mask. Partial
values along the silhouette are what distinguish matting from segmentation, and are the
reason antennae, wing mirrors and wheel spokes survive extraction.

The Alpha Matte is the **canonical artefact**: the Cutout and the Composite are both
derived from it and can be regenerated at will.

## Glass Treatment

Whether car windows are opaque or transparent in a derived image. Two readings coexist:

- **Matte glass** — the model's own alpha, so glass is partially transparent and a new
  background shows through the cabin.
- **Solid glass** — enclosed interior regions forced opaque, so the car reads as a solid
  object and its real reflections are preserved.

Both are legitimate; they are presentation choices applied *to* the Alpha Matte, never
baked into it.

## Captured Shadow vs Contact Shadow

The **Captured Shadow** is the real shadow and floor reflection present in the source
photograph. It belongs to the background, not to the Hero Car, and is excluded from the
Alpha Matte.

The **Contact Shadow** is synthesised at composite time from the bottom of the Alpha
Matte. It exists so that a batch of extractions looks consistent regardless of the
lighting of the room each photograph was taken in.

## Flag

A machine-raised note that a result deserves human review — an ambiguous Hero Car, a
Hero Car that looks cropped, an image with no vehicle at all. A Flag never changes the
outcome; it only marks it. A flagged image costs a few seconds of attention, whereas a
silently wrong extraction propagates.
