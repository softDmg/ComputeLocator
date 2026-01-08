from typing import List, Any

def ml_method(input: Any) -> Any:
    a = 1
    for i in range(1, 10000000):
        a = (a + a )% 1000000
    # pass
    return a