"""Tools to work with policy dictionaries without accidentally modifying them."""
import itertools

import pandas as pd


def filter_dictionary(function, dictionary, by="keys"):
    """Filter a dictionary by conditions on keys or values.

    Args:
        function (callable): Function that takes one argument and returns True or False.
        dictionary (dict): Dictionary to be filtered.

    Returns:
        dict: Filtered dictionary

    Examples:
        >>> filter_dictionary(lambda x: "bla" in x, {"bla": 1, "blubb": 2})
        {'bla': 1}
        >>> filter_dictionary(lambda x: x <= 1, {"bla": 1, "blubb": 2}, by="values")
        {'bla': 1}

    """
    if by == "keys":
        keys_to_keep = set(filter(function, dictionary))
        out = {key: val for key, val in dictionary.items() if key in keys_to_keep}
    elif by == "values":
        out = {}
        for key, val in dictionary.items():
            if function(val):
                out[key] = val
    else:
        raise ValueError(f"by must be 'keys' or 'values', not {by}")

    return out


def shorten_policies(policies, start_date=None, end_date=None):
    """Shorten policies to only go from start_date to end_date.

    Args:
        policies (dict): policies dictionary with "start" and "end" as keys.
        start_date (pd.Timestamp or str)
        end_date (pd.Timestamp or str)

    Returns:
        dict: reduced policies only including policies that are active between
            start_date and end_date. Kept policies are cropped to not start
            before start_date and to not end after end_date.

    """
    start_date = pd.Timestamp.min if start_date is None else pd.Timestamp(start_date)
    end_date = pd.Timestamp.max if end_date is None else pd.Timestamp(end_date)

    converted = {}

    # convert all dates to pd.Timestamps
    for name, pol in policies.items():
        transformed_dates = {
            "end": pd.Timestamp(pol["end"]),
            "start": pd.Timestamp(pol["start"]),
        }
        converted[name] = update_dictionary(pol, transformed_dates)

    short = filter_dictionary(
        lambda x: x["end"] >= start_date and x["start"] <= end_date,
        converted,
        by="values",
    )

    short_adjusted = {}

    for name, pol in short.items():
        adjusted_dates = {
            "start": max(start_date, pol["start"]),
            "end": min(end_date, pol["end"]),
        }
        short_adjusted[name] = update_dictionary(pol, adjusted_dates)

    return short_adjusted


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
    if isinstance(dictionaries, dict):
        combined = dictionaries
    elif isinstance(dictionaries, list):
        if len(dictionaries) == 1:
            combined = dictionaries[0]
        else:
            key_sets = [set(d) for d in dictionaries]

            for first, second in itertools.combinations(key_sets, 2):
                intersection = first.intersection(second)
                if intersection:
                    raise ValueError(
                        f"The following keys occur more than once: {intersection}"
                    )

            combined = {}
            for d in dictionaries:
                combined = {**combined, **d}

    else:
        raise ValueError("'dictionaries' must be a dict or list of dicts.")

    return combined


def split_policies(
    policies, split_date, start_date=None, end_date=None, suffixes=("first", "second")
):
    """Split a policy dictionary and reduce it to start and end dates.

    The split date is included in the second dictionary. To make it possible that
    split dictionaries can be combined again to obtain a policy dictionary that
    is equivalent to the original one, we add suffixes to all keys that occur in
    both resulting dictionaries.

    Args:
        policies (dict): See :ref:`policies`.
        split_date(pandas.Timestamp or str): The start date of the second dictionary.
        start_date (pandas.Timestamp or str): The start date of the first dictionary.
        end_date (pandas.Timestamp or str): The end date of the second dictionary.

    Returns:
        tuple: Tuple with two non-overlapping policy dictionaries.

    """
    raw_first = shorten_policies(
        policies=policies,
        start_date=start_date,
        end_date=pd.Timestamp(split_date) - pd.Timedelta(days=1),
    )
    raw_second = shorten_policies(
        policies=policies, start_date=split_date, end_date=end_date
    )

    duplicates = set(raw_first).intersection(raw_second)

    first = _rename_duplicates(raw_first, duplicates, suffixes[0])
    second = _rename_duplicates(raw_second, duplicates, suffixes[1])

    return first, second


def _rename_duplicates(policies, duplicates, suffix):
    new = {}
    for key, val in policies.items():
        if key in duplicates:
            new[f"{key}_{suffix}"] = val
        else:
            new[key] = val
    return new


def remove_work_policies(policies):
    """Return reduced policy dicts where the work policies have been removed."""
    return filter_dictionary(lambda x: "work" not in x, policies)


def remove_educ_policies(policies):
    """Return reduced policy dicts where the educ policies have been removed."""
    return filter_dictionary(lambda x: "educ" not in x, policies)


def remove_other_policies(policies):
    """Return reduced policy dicts where the other policies have been removed."""
    return filter_dictionary(lambda x: "other" not in x, policies)


def remove_school_policies(policies):
    """Return reduced policy dicts where the school policies have been removed."""
    return filter_dictionary(lambda x: "school" not in x or "preschool" in x, policies)


def remove_young_educ_policies(policies):
    """Return reduced policy dicts where the young educ policies have been removed."""
    return filter_dictionary(
        lambda x: "preschool" not in x and "nursery" not in x, policies
    )
