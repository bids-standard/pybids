"""" Miscellanous utilities """
import re

def convertJSON(j):
    """ Recursively convert CamelCase keys to snake_case.
    From: https://stackoverflow.com/questions/17156078/converting-identifier-naming-between-camelcase-and-underscores-during-json-seria
    """
    def convert(s):
        a = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return a.sub(r'_\1', s).lower()

    def convertArray(a):
        newArr = []
        for i in a:
            if isinstance(i,list):
                newArr.append(convertArray(i))
            elif isinstance(i, dict):
                newArr.append(convertJSON(i))
            else:
                newArr.append(i)
        return newArr

    out = {}
    for k in j:
        newK = convert(k)
        if isinstance(j[k],dict):
            out[newK] = convertJSON(j[k])
        elif isinstance(j[k],list):
            out[newK] = convertArray(j[k])
        else:
            out[newK] = j[k]

    return out
