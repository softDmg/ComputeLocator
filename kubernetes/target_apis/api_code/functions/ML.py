from typing import List, Any

def ml_method(input: int) -> Any:
    a = 1
    for i in range(1, input): # 10000000
        a = (a + a )% 1000000
    return a
