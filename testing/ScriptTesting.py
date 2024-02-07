import filecmp

def test_scripttesting():
    # compare files that are written by an outside script
    assert filecmp.cmp('log.txt', 'testing/gold/scripttesting/log.txt')
    assert filecmp.cmp('browse.txt', 'testing/gold/scripttesting/browse.txt')
    assert filecmp.cmp('query.txt', 'testing/gold/scripttesting/query.txt')
