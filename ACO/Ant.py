import random
import numpy as np
from threading import *

Seed = 0.000001


class Ant(Thread):
    def __init__(self, ant, tauinit, alpha, beta, rho, phi, q0, cons, hor, ver, heu, mem, colony):
        Thread.__init__(self)
        self.ID = ant
        self.tauinit = tauinit  # Valoarea initiala a feromonului
        self.alpha = alpha  # Coeficientul feromonului
        self.beta = beta  # Coeficientul euristicii
        self.rho = rho  # Coeficientul de evaporare a feromonilor
        self.phi = phi  # Coeficientul de dezintegrare a feromonilor
        self.q0 = q0  # Gradul de explorare
        self.cons = cons  # Numarul de pasi de constructie
        self.hor = hor  # Numărul de compartimentări orizontale
        self.ver = ver  # Numărul de compartimentări verticale
        self.heuristic_matrix = heu  # Euristica informationala
        self.memory = mem  # Memoria furnicilor (Numărul de poziții ale ultimilor pixeli vizitați)
        self.colony = colony

        self.pheromone_matrix = np.ndarray(shape=self.heuristic_matrix.shape, dtype=float)
        self.pheromone_matrix.fill(self.tauinit)
        self.max_heuristic_index = self.ID - 1
        sorted_array = np.argsort(np.ravel(self.heuristic_matrix))[::-1]
        unravel_indices = (np.unravel_index(i, self.heuristic_matrix.shape) for i in sorted_array)
        self.heuristic_sorted_indices = [j for j in unravel_indices]
        # Valoarea maximă în matricea euristică (Folosită pentru normalizarea în matricea euristică)
        self.V_max = np.max(self.heuristic_matrix)
        self.mutex = Lock()
        self.boundary = {
            'min': (0, 0),
            'max': (self.heuristic_matrix.shape[0] - 1, self.heuristic_matrix.shape[1] - 1)
        }
        # noinspection PyTypeChecker
        self.ants = np.empty(shape=(self.hor, self.ver, self.ID), dtype=[('position', 'i8', 2), ('boundary', 'i8', (2, 2)), ('visited', object), ('pheromone', np.float64, 1)])

    # Procesul de start al threadingului
    def start(self) -> None:
        for cons in range(0, self.cons):  # Pentru numărul de pași de construcție
            self.mutex.acquire()
            # index_x și index_y reprezintă partiția, iar index_z reprezintă ant
            for (index_x, index_y, index_z), ant in np.ndenumerate(self.colony.ants):  # Pentru toare furnicile
                self.__chooseAndMove(ant=ant, index_x=index_x, index_y=index_y, index_z=index_z)
                self.__updateLocalPheromone(position=ant['position'], index_x=index_x, index_y=index_y, index_z=index_z)
                # time.sleep(1000)
            self.mutex.release()

    # Faza de adaptarea locala
    def __updateLocalPheromone(self, position, index_x, index_y, index_z):
        self.colony.pheromone_matrix[position[0]][position[1]] = pheromone = \
            (1 - self.phi) * self.__pheromone(position=position) + self.phi * self.tauinit

        # Lăsați furnica să-și amintească cantitatea de feromoni actualizată de furnica de index_x, index_y, index_z
        self.__saveAntPheromone(pheromone=pheromone, index_x=index_x, index_y=index_y, index_z=index_z)

    def __chooseAndMove(self, ant, index_x, index_y, index_z):
        # Obțineți următoarea poziție pentru a fi vizitată
        new_pos = self.__selectNextPixel(ant=ant, index_x=index_x, index_y=index_y, index_z=index_z)

        # Adăugați o nouă poziție la toate pozițiile vizitate dacă nu a fost vizitată anterior de nicio furnică
        if new_pos not in self.colony.all_visited_positions:
            self.colony.all_visited_positions.append(tuple(new_pos))

        # Actualizați poziția curentă și pozițiile vizitate
        self.colony.ants['position'][index_x][index_y][index_z] = list(new_pos)
        self.colony.ants['visited'][index_x][index_y][index_z].append(tuple(new_pos))

    # Selectarea urmatorului pixel
    def __selectNextPixel(self, ant, index_x, index_y, index_z):
        # posturi ale vecinilor permisi
        unvisited_neighbors = self.__unvisitedNeighbors(ant)
        # Probabilitate random
        q = random.random()
        # Calculați numărătorul pentru fiecare vecin nevizitat
        numerators = [
            pow(self.__pheromone(position=neighbor), self.alpha) *
            pow(self.__heuristic(position=neighbor), self.beta)
            for neighbor in unvisited_neighbors]

        # Aplicați ACS (regulă proporțională pseudoaleatoare)
        try:
            if q <= self.q0:  # Explorare
                return unvisited_neighbors[np.argmax(numerators)]
            else:  ## Exploatarea (probabilitatea de tranziție)
                denominator = sum(numerators)
                p_values = [float(num) / denominator for num in numerators]
                return unvisited_neighbors[np.argmax(p_values)]
        except ValueError:
            print("Empty sequence to be handled")
            return self.__moveToNewPosition(index_x=index_x, index_y=index_y, index_z=index_z)

    def __unvisitedNeighbors(self, ant):
        curr_pos = ant['position']  # Pozitia curenta a furnicii
        x_index_min = -1
        x_index_max = 1

        y_index_min = -1
        y_index_max = 1
        # Dacă poziția curentă nu este în colțul sau marginea imaginii # (Lasă furnicile să-și treacă granițele)
        if curr_pos[0] <= self.boundary['min'][0]:
            x_index_min = 0

        if curr_pos[1] <= self.boundary['min'][1]:
            y_index_min = 0

        if curr_pos[0] >= self.boundary['max'][0]:
            x_index_max = 0

        if curr_pos[1] >= self.boundary['max'][1]:
            y_index_max = 0

        # Indici ai vecinilor permisi
        neighbor_index = [(i, j)
                          for i in range(x_index_min, x_index_max + 1)
                          for j in range(y_index_min, y_index_max + 1)
                          if not (i == 0 and j == 0)]

        # Adăugați fiecare index la poziția curentă (oferă pozițiile permise vecinilor) și reveniți
        return [tuple(map(sum, zip(curr_pos, index)))
                for index in neighbor_index
                if tuple(map(sum, zip(curr_pos, index))) not in ant['visited'][-self.memory:]]
        # Dacă verificați pentru a vă asigura că poziția permisă nu este în memoria furnicii

    def __moveToNewPosition(self, index_x, index_y, index_z):
        # Goliți pozițiile vizitate
        del self.colony.ants['visited'][index_x][index_y][index_z][:]
        # Creșteți indicele euristic maxim
        self.colony.max_heuristic_index += 1
        # Mutați furnica într-o nouă poziție unde valoarea euristică este mare (folosind self.max_heuristic_index)
        self.colony.ants['position'][index_x][index_y][index_z] = \
            list(self.heuristic_sorted_indices[self.max_heuristic_index])
        # Marcați pozițiile inițiale ca fiind vizitate pentru fiecare furnică
        self.colony.ants['visited'][index_x][index_y][index_z] = [
            tuple(self.colony.ants['position'][index_x][index_y][index_z])]

    def __saveAntPheromone(self, pheromone, index_x, index_y, index_z):
        self.colony.ants['pheromone'][index_x][index_y][index_z] += pheromone

    def __pheromone(self, position):
        return self.pheromone_matrix[position[0]][position[1]]

    def __modifyAntPositions(self, iter):
        avg_pheromone = np.average(self.ID['pheromone'])
        print("Avg. Pheromone: "+str(avg_pheromone))

        # Pentru fiecare furnica
        for (index_x, index_y, index_z), ant in np.ndenumerate(self.ID):
            # Dacă feromonul depus este mai mic decât media feromonilor depus
            if ant['pheromone'] < avg_pheromone:
                self.__moveToNewPosition(index_x=index_x, index_y=index_y, index_z=index_z)

    def __heuristic(self, position):
        return float(self.heuristic_matrix[position[0]][position[1]]) / self.V_max
