import random
import math

# amount = 10^4, B = 1/T

def generate_random_matrix(L: int):
    """Генерирует двумерный массив L x L, заполненный случайными числами 1 или -1."""
    return [[random.choice([1, -1]) for _ in range(L)] for _ in range(L)]


def sweep(arr: list[list[int]], L: int, B: float) -> None:    #функция одного sweep
    def energy_changes(arr: list[list[int]], L: int, r_i: int, r_j: int) -> int:
        sum = 0
        sum += arr[(r_i - 1) % L][r_j]
        sum += arr[(r_i + 1) % L][r_j]
        sum += arr[r_i][(r_j - 1) % L]
        sum += arr[r_i][(r_j + 1) % L]    
        return 2 * arr[r_i][r_j] * sum
    
    for _ in range(L * L):
        r_i = random.randrange(L)
        r_j = random.randrange(L)
        dH = energy_changes(arr, L, r_i, r_j)   # вычисляю изменение энергии
        r = random.uniform(0, 1)    #r [0, 1]
        if r < math.exp(-B * dH):   # переворачиваю спин с нужной вероятностью
            arr[r_i][r_j] *= -1


def Monte_Carlo_sweeps(L: int, B: float, amount: int) -> list[list[int]]: #функция 10^4 шагов Монте-Карло
    arr = generate_random_matrix(L)
    for _ in range(amount):
        sweep(arr, L, B)
    return arr      #возвращает матрицу после 10^4 шагов
        
