# Reference resolution

The target journal requires a DOI for every reference. A DOI taken from a search
snippet is worse than none, because it resolves and the reader assumes it is
right. `convgap.bib` queries Crossref by title and accepts a record only when
the returned title is close enough to the one requested; anything else is
reported UNRESOLVED for a human to settle.

Two filters are load-bearing:

- **Preprints are rejected, not down-ranked.** Crossref indexes the SSRN or NBER
  version separately, and it often scores an identical title match. A reference
  list for a peer-reviewed submission cites the published article.
- **The year is the issue year, not the first online posting.** Crossref's
  `issued` field carries the earliest recorded date, which for an online-first
  journal is a year or two before the issue a reader would cite.

Both mattered. Without the first, three references would have cited working
papers. Without the second, `Replicating Anomalies` would have been dated 2018
rather than 2020.
