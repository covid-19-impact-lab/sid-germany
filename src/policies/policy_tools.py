"""Tools to work with policy dictionaries without accidentally modifying them."""
import itertools


def filter_dictionary(function, dictionary):
    """Filter a dictionary by conditions on keys

    Args:
        function (callable): Function that takes one argument and returns True or False.
        dictionary (dict): Dictionary to be filtered.

    Returns:
        dict: Filtered dictionary

    Examples:
        >>> filter_dictionary(lambda x: "bla" in x, {"bla": 1, "blubb": 2})
        {'bla': 1}

    """
    keys_to_keep = set(filter(function, dictionary))
    out = {key: val for key, val in dictionary.items() if key in keys_to_keep}
    return out


def update_dictionary(dictionary, other):
    """Create a copy of dictionary and update it with other.

    Args:
        dictionary (dict)
        other (dict)

    Returns:
        dict: The updated dictionary


    Examples:
        # make sure input is not changed
        >>> first = {"a": 1, "b": 2}
        >>> updated = update_dictionary(first, {"c": 3})
        >>> assert first == {"a": 1, "b": 2}

        # make sure update works
        >>> update_dictionary({"a": 1, "b": 2}, {"c": 3})
        {'a': 1, 'b': 2, 'c': 3}

    """
    return {**dictionary, **other}


def combine_dictionaries(dictionaries):
    """Combine a list of non-overlapping dictionaries into one.

    Args:
        dictionaries (list): List of dictionaries.

    Returns:
        dict: The combined dictionary.


    Examples:
        >>> combine_dictionaries([{"a": 1}, {"b": 2}])
        {'a': 1, 'b': 2}

    """

    if not isinstance(dictionaries, list):
        raise ValueError("dictionaries must be a list.")

    key_sets = [set(d) for d in dictionaries]

    for first, second in itertools.combinations(key_sets, 2):
        intersection = first.intersection(second)
        if intersection:
            raise ValueError(
                f"The following keys are in more than one dictionary: {intersection}"
            )

    combined = {}
    for d in dictionaries:
        combined = {**combined, **d}

    return combined
