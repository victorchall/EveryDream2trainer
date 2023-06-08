import copy
from typing import List

def first_fit_decreasing(input_items: List[List], batch_size: int, filler_items: List=[]) -> List:
    """
    Given as input a list of lists, batch the items so that as much as possible the members of each of the original
    lists end up in the same batch. Pad out too-short batches by taking items from the filler_items list, if available.

    @return flattened list of all items in input_items and filler_items, arranged such that, as much as possible, items
        that are in the same input list end up in the same batch.
    """

    def sort_by_length(items: List[List]) -> List[List]:
        return sorted(items, key=lambda x: len(x))

    remaining = input_items
    output = []
    while remaining:
        remaining = sort_by_length(remaining)
        longest = remaining.pop()
        if len(longest) == 0:
            continue
        if len(longest) >= batch_size:
            output.append(longest[0:batch_size])
            del longest[0:batch_size]
            if len(longest)>0:
                remaining.append(longest)
        else:
            # need to build this chunk by combining multiple
            combined = longest
            while True:
                fill_length = batch_size - len(combined)
                if fill_length == 0:
                    break

                if len(remaining) == 0 and len(filler_items) == 0:
                    break

                from_filler_bucket = filler_items[0:fill_length]
                if len(from_filler_bucket) > 0:
                    del filler_items[0:fill_length]
                    combined.extend(from_filler_bucket)
                    continue

                filler = next((r for r in remaining if len(r) <= fill_length), None)
                if filler is not None:
                    remaining.remove(filler)
                    combined.extend(filler)
                else:
                    # steal from the next longest
                    next_longest = remaining.pop()
                    combined.extend(next_longest[0:fill_length])
                    del next_longest[0:fill_length]
                    if len(next_longest) > 0:
                        remaining.append(next_longest)
            output.append(combined)

    output.append(filler_items)
    return [i for o in output for i in o]




