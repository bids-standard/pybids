import re

def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. '''
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def matches_entities(obj, entities, strict=False):
    ''' Checks whether an object's entities match the input. '''
    if strict and set(obj.entities.keys()) != set(entities.keys()):
        return False

    comm_ents = list(set(obj.entities.keys()) & set(entities.keys()))
    for k in comm_ents:
        current = obj.entities[k]
        target = entities[k]
        if isinstance(target, (list, tuple)):
            if current not in target:
                return False
        elif current != target:
            return False
    return True

def convert_JSON(j):
    """ Recursively convert CamelCase keys to snake_case.
    From: https://stackoverflow.com/questions/17156078/converting-identifier-naming-between-camelcase-and-underscores-during-json-seria
    """

    def camel_to_snake(s):
        a = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return a.sub(r'_\1', s).lower()

    def convertArray(a):
        newArr = []
        for i in a:
            if isinstance(i,list):
                newArr.append(convertArray(i))
            elif isinstance(i, dict):
                newArr.append(convert_JSON(i))
            else:
                newArr.append(i)
        return newArr

    out = {}
    for k, value in j.items():
        newK = camel_to_snake(k)

        if isinstance(value, dict):
            out[newK] = convert_JSON(value)
        elif isinstance(value, list):
            out[newK] = convertArray(value)
        else:
            out[newK] = value

    return out
