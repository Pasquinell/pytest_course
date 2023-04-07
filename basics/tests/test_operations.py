import pytest
from basics.operations import add, substract

def test_add():
    assert add(1, 2) == 3
    #assert add(0, 5) == 5
    #assert add(-1, 1) == 0

def test_substract():
    assert substract(1, 2) == -1
    assert substract(6, 5) == 1