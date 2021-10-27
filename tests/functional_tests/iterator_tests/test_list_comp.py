from functional.iterator.runtime import offset


V2E = offset("V2E", 3)


def test_list_comp():
    testee = [v for v in V2E]
    assert testee[0][1] == 0
    assert testee[1][1] == 1
    assert testee[2][1] == 2
