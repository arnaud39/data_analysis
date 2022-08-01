from typing import Set


def dynamic_remover(
    set_: set, el_: set = None, remove_: set = set(), root=False
) -> Set[frozenset]:
    # recusitivity, get a list of frozenset of size 1
    if not el_:
        el_ = {next(iter(next(iter(set_))))}
        root = True

    following = [k for k in set_ if k.intersection(el_)]
    if not (following):
        return []

    to_delete = set(map(lambda x: x.difference(el_), following))

    # if there is nothing new to delete
    if to_delete.intersection(remove_) == to_delete:
        if not root:
            return to_delete
        else:
            return to_delete.difference({frozenset(el_)})
    to_delete = to_delete.union(remove_)

    # delete their correlated associated features too, get a list of list of frozen set
    to_delete_recursive = list(
        map(lambda x: dynamic_remover(set_, el_=x, remove_=to_delete), list(to_delete))
    )
    # [[fr(a), fr(b)], [fr(a), fr(c)]] -> [fr(a), fr(b), fr(c)]
    # shall return a list of frozenset
    if not root:
        return to_delete.union(*[set(i) for i in to_delete_recursive])
    else:
        return to_delete.union(*[set(i) for i in to_delete_recursive]).difference(
            {frozenset(el_)}
        )


def remove_features(set_: set):
    to_delete = set()
    while set_:
        next_ = dynamic_remover(set_)
        to_delete = to_delete.union(next_)
        set_ = {i for i in set_ if not i.intersection(set().union(*list(next_)))}
    return to_delete