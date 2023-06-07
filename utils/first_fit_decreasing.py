from typing import List

def first_fit_decreasing(input_list: List[List], batch_size: int) -> List:
    """
    Given as input a list of lists, batch the items so that as much as possible the members of each of the original
    lists end up in the same batch.

    @return a list of batches
    """

    def sort_by_length(items: List[List]):
        return items.sort(key=lambda x: len(x), reverse=True)

    remaining = list(input_list)
    while remaining:
        remaining = sort_by_length(remaining)
