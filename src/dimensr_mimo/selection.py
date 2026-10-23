"""Bounded candidate selection with optional coverage of individual outputs."""


def shortlist_indices(scores, sizes, limit=12, per_output=0, start_output=0):
    """Return candidate indices, with stable size-tiebroken score rankings.

    ``scores`` has shape (candidates, outputs); zero scores contribute no
    per-output candidates. With ``per_output=0``, rank by summed scores only.
    Otherwise interleave each output's original positive-score top K, taking
    one rank per round and rotating the first output by ``start_output``.
    Duplicates consume no slots, and do not extend an output beyond its top K.
    Fill unused slots from the joint ranking. Inputs are validated by callers.
    """
    if limit <= 0:
        return []
    joint = sorted(range(len(scores)), key=lambda i: (-sum(scores[i]), sizes[i]))
    if not per_output:
        return joint[:limit]

    n_outputs = scores.shape[1]
    output_rankings = [
        sorted((i for i in range(len(scores)) if scores[i, j] > 0),
               key=lambda i: (-scores[i, j], sizes[i]))[:per_output]
        for j in range(n_outputs)
    ]
    selected, seen = [], set()
    for rank in range(per_output):
        for offset in range(n_outputs):
            output = (start_output + offset) % n_outputs
            ranking = output_rankings[output]
            if rank < len(ranking) and ranking[rank] not in seen:
                index = ranking[rank]
                selected.append(index)
                seen.add(index)
                if len(selected) == limit:
                    return selected
    for index in joint:
        if index not in seen:
            selected.append(index)
            if len(selected) == limit:
                break
    return selected
