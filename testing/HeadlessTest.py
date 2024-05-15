import filecmp

def test_headless():
    # this is comparing the output of a script that has been run in the test
    assert filecmp.cmp('TableWriteHeadlessExample.csv', 'testing/gold/TableWriteHeadlessExample.csv')
