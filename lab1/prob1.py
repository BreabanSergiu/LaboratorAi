
#input string - String
#complexitate - O(n*log n)
def function(string):
    strings = string.split(" ")
    strings.sort(reverse=True)
    return strings[0]


def test():
    assert(function("ana are mere") == "mere")
    assert(function("ana are mere si pere") == "si")


test()