import filecmp

def test_scripttesting():
    # compare files that are written by an outside script
    assert filecmp.cmp('log.txt', 'testing/gold/scripttest/log.txt')
    assert filecmp.cmp('browse.txt', 'testing/gold/scripttest/browse.txt')
    assert filecmp.cmp('query.txt', 'testing/gold/scripttest/query.txt')
