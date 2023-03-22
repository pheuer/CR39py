"""
Utilities for making command line interfaces (CLIs)

@author: pvheu
"""

__all__ = [
    "_cli_input",
]


def _cli_input(
    mode="alphanumeric list", always_pass=None, valid_keys=None, valid_values=None
):
    """
    Collects CLI input from the user: continues asking until a valid
    input has been provided. Input modes:

    'numeric'
        Single number

    'alphanumeric list'
        List of alphanumeric characters, separated by commas

    'key:value list' -> e.g. xmin:10, ymin:-1
        Alternating alpha and numeric key value pairs, comma separated
        Letters are only acceptable in the values if they are 'none'


    always_pass : list of str, optional
        Strings to always accept, regardless of the mode chosen. Usually control
        words like 'end' or 'help'. If not provided, a default list will be
        used.


    valid_keys : list of str, optional
        List of valid keys for key:value list input. Default is to accept
        any key. Keys should all be lowercase


    """
    yesno = set("yn")
    integers = set("1234567890+-")
    floats = integers.union(".e")
    alphas = set("abcdefghijklmnopqrstuvwxyz_-")

    # If the input matches one of these
    if always_pass is None:
        always_pass = ["help", "end"]
    elif always_pass is False:
        always_pass = []

    while True:
        x = str(input(">"))

        if x in always_pass:
            return x

        if mode == "integer":
            if set(x).issubset(integers):
                return int(x)

        elif mode == "float":
            if set(x).issubset(floats):
                return float(x)

        elif mode == "alpha-integer":
            if set(x).issubset(integers.union(alphas)):
                return x

        elif mode == "yn":
            if set(x).issubset(yesno):
                return x

        elif mode == "alpha-integer list":
            split = x.split(",")
            split = [s.strip() for s in split]
            # Discard empty strings
            split = [s for s in split if s != ""]
            if all([set(s).issubset(alphas.union(integers)) for s in split]):
                return split

        elif mode == "key:value list":
            split = x.split(",")
            split = [s.split(":") for s in split]

            # Verify that we have a list of at least one pair, and only pairs
            if all([len(s) == 2 for s in split]) and len(split) > 0:
                # Discard empty strings
                split = [s for s in split if (s[0] != "" and s[1] != "")]

                # Transform any 'none' values into None
                # Strip any other values
                for i, s in enumerate(split):
                    if str(s[1]).lower() == "none":
                        split[i][1] = None
                    else:
                        split[i][1] = s[1].strip()

                # Strip keys, convert to lowercase
                for i in range(len(split)):
                    split[i][0] = split[i][0].strip().lower()

                # Test that values are in the correct sets
                test1 = all(
                    [
                        (
                            (set(s[0].strip()).issubset(alphas))
                            and (s[1] is None or set(s[1]).issubset(floats))
                        )
                        for s in split
                    ]
                )

                # Test that all keys are in the allowed list (if set)
                if valid_keys is not None:
                    test2 = all(s[0] in valid_keys for s in split)
                else:
                    test2 = True

                # Convert any non-None values into floats
                for i, s in enumerate(split):
                    if s[1] is not None:
                        split[i][1] = float(s[1])

                if all([test1, test2]):
                    return {str(s[0].strip()): s[1] for s in split}

                if not test2:
                    print("Key not recognized")

        else:
            raise ValueError("Invalid Mode")


if __name__ == "__main__":
    x = _cli_input(mode="alpha-integer list")
    print(x)
