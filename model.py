import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class IsingModel:

    def __init__(self, L: int) -> None:
        self.size = L # размер решётки
        self.spins = [[random.choice([1, -1]) for _ in range(L)] for _ in range(L)] # матрица спинов

    def get_spin(self, i: int, j: int) -> int:
        """Узнать знак спина с учётом замкнутости решётки в клетке (i, j)"""
        return self.spins[i % self.size][j % self.size]

    def flip_spin(self, i: int, j: int) -> None:
        """Перевернуть спин в клетке (i, j)"""
        self.spins[i % self.size][j % self.size] *= -1

    def calc_delta_H(self, i: int, j: int) -> int:
        """Вычислить изменение энергии в случае переворота спина (i, j)"""
        neighbors = (
                self.get_spin(i + 1, j)
                + self.get_spin(i - 1, j)
                + self.get_spin(i, j + 1)
                + self.get_spin(i, j - 1)
        )
        return 2 * self.get_spin(i, j) * neighbors

    def calc_energy(self) -> int:
        """Вычислить полную энергию решётки"""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                cur_s = self.get_spin(i, j)
                energy -= cur_s * self.get_spin(i + 1, j)
                energy -= cur_s * self.get_spin(i, j + 1)

        return energy

    def calc_magnetization(self) -> int:
        """Вычислить полную намагниченность решётки"""
        return sum(sum(row) for row in self.spins)
    
    
class MetropolisSampler:

    def __init__(
            self,
            model: IsingModel,
            beta: float,
            *,
            thermalization_sweeps: int = 10000,
            num_samples: int = 100
    ) -> None:
        self.ising_model = model
        self.beta = beta
        self.thermalization_sweeps = thermalization_sweeps # время релаксации
        self.num_samples = num_samples # число итераций в эксперименте

    def one_sweep(self) -> None:
        """Выполнить один полный проход обновлений в алгоритме Метрополиса"""
        L = self.ising_model.size
        for _ in range(L * L):
            r_i = random.randrange(L)
            r_j = random.randrange(L)
            dH = self.ising_model.calc_delta_H(r_i, r_j)
            if random.random() <= math.exp(-self.beta * dH):
                self.ising_model.flip_spin(r_i, r_j)

    def experiment(self) -> tuple[float, float, float]:
        """Провести эксперимент с заданными L и T"""
        L = self.ising_model.size

        for _ in range(self.thermalization_sweeps):
            self.one_sweep()

        sum_E, sum_M, sum_abs_M = 0, 0, 0
        for _ in range(self.num_samples):
            self.one_sweep()

            sum_E += self.ising_model.calc_energy()

            cur_magnetization = self.ising_model.calc_magnetization()
            sum_M += cur_magnetization
            sum_abs_M += abs(cur_magnetization)

        return (
            sum_E / (L ** 2 * self.num_samples),
            sum_M / (L ** 2 * self.num_samples),
            sum_abs_M / (L ** 2 * self.num_samples)
        )
        

T = 2.269
random_T = [random.uniform(T - 0.3, T + 0.3) for _ in range(10)]
# эксперимент для таблицы 8 x 8
energy_L_8, magnetization_L_8, abs_magnetization_L_8 = [], [], []
for t in random_T:
    model = IsingModel(8)
    sampler = MetropolisSampler(model, 1 / t)
    res = sampler.experiment()
    energy_L_8.append(res[0])
    magnetization_L_8.append(res[1])
    abs_magnetization_L_8.append(res[2])
    
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3)

matrix = np.array(model.spins)

ax_matrix = fig.add_subplot(gs[0])
ax_matrix.imshow(matrix, cmap='gray', interpolation='nearest')

ax_matrix.set_xticks([i - 0.5 for i in range(len(matrix[0]) + 1)], minor=True)
ax_matrix.set_yticks([i - 0.5 for i in range(len(matrix) + 1)], minor=True)
ax_matrix.grid(which='minor', color='black', linestyle='-', linewidth=1)
ax_matrix.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

# Заголовок НАД матрицей
ax_matrix.set_title('матрица спинов', fontsize=14, pad=15, fontweight='bold')

ax_plots = fig.add_subplot(gs[1])

# Сортируем данные для правильного отображения
sorted_data = sorted(zip(random_T, energy_L_8, magnetization_L_8, abs_magnetization_L_8))
sorted_T, sorted_E, sorted_M, sorted_abs_M = zip(*sorted_data)

ax_plots.plot(sorted_T, sorted_E, label='энергия', linewidth=2)
ax_plots.plot(sorted_T, sorted_M, label='намагниченность', linewidth=2)
ax_plots.plot(sorted_T, sorted_abs_M, label='абсолютная намагниченность', linewidth=2)

ax_plots.set_xlabel('Температура (T)', fontsize=10)
ax_plots.set_ylabel('Значение', fontsize=10)
ax_plots.set_title('Зависимости от температуры', fontsize=12, pad=10)
ax_plots.legend(fontsize=9, frameon=True)
ax_plots.grid(True, alpha=0.3, linestyle='--')

# Оптимизация отступов
plt.tight_layout()
plt.show()