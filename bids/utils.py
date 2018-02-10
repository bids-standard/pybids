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
