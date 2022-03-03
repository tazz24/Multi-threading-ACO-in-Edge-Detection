import numpy as np
import cv2
import os
import time
import random
from threading import Lock, Condition

from ACO.Ant import Ant

Seed = 0.000001
MAIN_DIRECTORY = 'C:/Users/user/Desktop/LICENTA/Practic/RESULTS'
SUB_DIRECTORY_IMG = 'C:/Users/user/Desktop/LICENTA/Practic/RESULTS/IMG'


class PartitionBasedAcs(Ant):
    def __init__(self, nants, tauinit, alpha, beta, rho, phi, q0, iters, cons, hor, ver, heu, mem):
        super(PartitionBasedAcs, self).__init__(nants, tauinit, alpha, beta, rho, phi, q0, cons, hor, ver, heu, mem, self)
        self.nants = nants  # Numar de furnici
        self.tauinit = tauinit  # Valoarea initiala a feromonului
        self.alpha = alpha  # Coeficientul feromonului
        self.beta = beta  # Coeficientul euristicii
        self.rho = rho  # Coeficientul de evaporare a feromonilor
        self.phi = phi  # Coeficientul de dezintegrare a feromonilor
        self.q0 = q0  # Gradul de explorare
        self.iter = iters  # Numarul de iteratii
        self.cons = cons  # Numarul de pasi de constructie
        self.hor = hor  # Numărul de compartimentări orizontale
        self.ver = ver  # Numărul de compartimentări verticale
        self.heuristic_matrix = heu  # Euristica informationala
        self.memory = mem  # Memoria furnicilor (Numărul de poziții ale ultimilor pixeli vizitați)

        # Valoarea maximă în matricea euristică (Folosită pentru normalizarea în matricea euristică)
        self.V_max = np.max(self.heuristic_matrix)


        self.cv = Condition()
        # Initializarea matricei feromonului
        self.pheromone_matrix = np.ndarray(shape=self.heuristic_matrix.shape, dtype=float)
        self.pheromone_matrix.fill(self.tauinit)  # Setați valoarea inițială a feromonilor la tauinit
        print("Initialized pheromone matrix")

        self.image_matrix = np.zeros(shape=self.heuristic_matrix.shape, dtype=float)

        # Toate pozitiile vizitate sunt mentinute
        self.all_visited_positions = []

        # Granita imaginii
        self.boundary = {
            'min': (0, 0),
            'max': (self.heuristic_matrix.shape[0] - 1, self.heuristic_matrix.shape[1] - 1)
        }
        # Pixelii in fiecare segment
        self.segment = {
            'hor': ((self.heuristic_matrix.shape[0]) / self.hor),
            'ver': ((self.heuristic_matrix.shape[1]) / self.ver)
        }
        # Creați o structură de date furnici
        self.ants = np.empty(shape=(self.hor, self.ver, self.nants),
                             dtype=[('position', 'i8', 2),
                                    ('boundary', 'i8', (2, 2)),
                                    ('visited', object),
                                    ('pheromone', np.float64, 1)])
        # Setați SEED pentru aleatoriu
        random.seed(Seed)

        # Matricea imaginii
        img_mat = np.loadtxt(fname="C:/Users/user/Desktop/LICENTA/Practic/gray_values", dtype=np.uint8)

        # Poziții de inițiere a furnicilor
        self.ants_init_positions = []

        # Setați pozițiile și limitele inițiale ale furnicilor (poziția furnicii este o coordonată aleatorie în interiorul graniței)
        for (index_x, index_y, index_z), ant in np.ndenumerate(self.ants):
            self.ants['boundary'][index_x][index_y][index_z] = [
                [index_x * self.segment['hor'],
                 index_y * self.segment['ver']],
                [index_x * self.segment['hor'] + self.segment['hor'] - 1,
                 index_y * self.segment['ver'] + self.segment['ver'] - 1]
            ]

            # Poziționați furnicile pe cele mai înalte valori ale euristicii
            array = self.__arrayWithBoundary(array=self.heuristic_matrix, boundary=ant['boundary'])
            positions = self.__nMaxPos(array=array, count=self.nants)

            self.ants['position'][index_x][index_y][index_z] = list(
                map(sum, zip(ant['boundary'][0], positions[index_z])))

            # Marcați pozițiile inițiale ca fiind vizitate pentru fiecare furnică
            self.ants['visited'][index_x][index_y][index_z] = [tuple(self.ants['position'][index_x][index_y][index_z])]

            # Salvați furnicile în pozițiile init
            self.ants_init_positions.append(tuple(self.ants['position'][index_x][index_y][index_z]))

            # Inițializați feromonul depus de fiecare furnică (inițial zero)
            self.ants['pheromone'][index_x][index_y][index_z] = 0.0

            # Marcați pozițiile inițiale ale furnicilor
            cv2.circle(img_mat, tuple(ant['position'])[::-1], 2, (0, 0, 0))

        # Salvați indexul valorii euristice maxime
        self.max_heuristic_index = self.nants - 1

        # Poziții euristice în scăderea valorii euristice
        sorted_array = np.argsort(np.ravel(self.heuristic_matrix))[::-1]
        unravel_indices = (np.unravel_index(i, self.heuristic_matrix.shape) for i in sorted_array)
        self.heuristic_sorted_indices = [j for j in unravel_indices]

        print("Initialized ants")

        # Crearea directorului unde se va salva rezultatele
        if not os.path.exists(MAIN_DIRECTORY):
            os.mkdir(MAIN_DIRECTORY)
        if not os.path.exists(SUB_DIRECTORY_IMG):
            os.mkdir(SUB_DIRECTORY_IMG)

        # Afisarea imaginii initializre
        cv2.imshow("Init positions", img_mat)
        cv2.imwrite(os.path.join(SUB_DIRECTORY_IMG, "Initialization.png"), img_mat)
        cv2.waitKey(500)

    # Calcularea timpului
    def __current_time_milli(self):
        return int(round(time.time() * 1000))

    def __daemonActions(self, iter):
        self.__displayResults(iter=iter)
        # Procedura de schimbare a furnicii
        self.__modifyAntPositions(iter=iter)
        self.__resetAntPheromone()

    # Granita matricii
    def __arrayWithBoundary(self, array, boundary):
        return array[boundary[0][0]:boundary[1][0]+1, boundary[0][1]:boundary[1][1]+1]

    # Numarul maxim de pozitiii
    def __nMaxPos(self, array, count):
        ravel_indices = np.argsort(np.ravel(array))[-count:]
        unravel_indices = (np.unravel_index(i, array.shape) for i in ravel_indices)
        return [i for i in unravel_indices]

    def __displayResults(self, iter):
        lock = Lock()
        lock.acquire()
        # Salvarea matricii feromnului intr-un fisier txt
        # noinspection PyTypeChecker
        np.savetxt('C:/Users/user/Desktop/LICENTA/Practic/pheromone_matrix', self.pheromone_matrix, fmt='%5f', newline='\n')

        # Valorile minime si maxime folosite in matricea feromonilor
        ph_min, ph_max = self.pheromone_matrix.min(), self.pheromone_matrix.max()

        # Diferenta dintre minim si maxim
        diff = abs(ph_max - ph_min)

        # Pe fiecare poziție vizitată aplicați valoarea gri din matricea imaginii în funcție de cantitatea de feromon
        for position in self.all_visited_positions:
            self.image_matrix[position[0]][position[1]] = int(
                round(abs((self.__pheromone(position=position) - ph_min) / diff) * 255))

        # Salvați matricea imaginii într-un fișier text
        # noinspection PyTypeChecker
        np.savetxt('C:/Users/user/Desktop/LICENTA/Practic/image_matrix', self.image_matrix, fmt='%d', newline='\n')

        # Încărcați matricea imaginii (formatul uint8 care este acceptat de opencv)
        img_mat = np.loadtxt(fname="C:/Users/user/Desktop/LICENTA/Practic/image_matrix", dtype=np.uint8)

        # Metoda de prag
        org, img_mat = cv2.threshold(img_mat, 0, 255, cv2.THRESH_BINARY_INV)

        # Salvarea imaginii in directorul dat
        cv2.imwrite(os.path.join(SUB_DIRECTORY_IMG, "Iteration" + str(iter) + ".png"), img_mat)
        self.cv.acquire()
        self.cv.notifyAll()
        time.sleep(0.5)
        self.cv.release()

        # Display image
        cv2.imshow('Edge', img_mat)
        cv2.waitKey(500)
        lock.release()

    # Determinarea euristicii informationale
    def __heuristic(self, position):
        return float(self.heuristic_matrix[position[0]][position[1]]) / self.V_max

    # Atribuirea unei pozitii aleatoare a unei furnici
    def __randomPosition(self, *boundary):
        return random.randint(boundary[0][0], boundary[1][0] - 1), random.randint(boundary[0][1], boundary[1][1] - 1)

    # Resetarea feromonului
    def __resetAntPheromone(self):
        self.ants['pheromone'] = 0.0

    #   Pozitia feromonului
    def __pheromone(self, position):
        return self.pheromone_matrix[position[0]][position[1]]

    # Crearea threadurilor
    def create_ants(self):
        ants = []
        for i in range(0, self.nants):
            ant = Ant(i, self.tauinit, self.alpha, self.beta, self.rho, self.phi, self.q0, self.cons, self.hor,
                      self.ver, self.heuristic_matrix, self.memory, self)
            ants.append(ant)
        return ants

    def run(self):
        print("Running algorithm ...")
        elapsed_time_array = []  # Stocarea timpului la fiecare iteratie
        self.antss = self.create_ants()
        for iter in range(0, self.iter):  # Pentru fiecare iteratie
            for iss in self.antss:        # Rularea threadurilor(furnicile) din clasa Ant
                iss.start()
            print("Iteration: " + str(iter + 1))
            strt_time_millis = self.__current_time_milli()  # Calculate iteration start time
            self.cv.acquire()
            print("START: "+str(strt_time_millis))
            lock = Lock()
            lock.acquire()

            # Adaptarea globala a feromonului
            self.__updateGlobalPheromone()
            lock.release()
            # Analiza timpului
            end_time_millis = self.__current_time_milli()  # Calcularea orei de încheiere a iterației
            self.cv.release()
            print("END: "+str(end_time_millis))
            elapsed_time = end_time_millis - strt_time_millis
            elapsed_time_array.append(elapsed_time)
            print("ELAPSED TIME: "+str(elapsed_time))  # Timpul scurs de depanare

            # Do daemon actions
            self.__daemonActions(iter + 1)
        print("FINISHED running ACS")
        cv2.waitKey(0)

        # Procesul de apdaptare globala
    def __updateGlobalPheromone(self):
        visited_positions = set(
            [w for x in self.ants['visited'] for y in x for z in y for w in z])
        # Oferă un set de toate pozițiile în care feromonul este depus la ultima iterație
        heuristic_values = [
            [
                [
                    [
                        self.__heuristic(position=position)
                        for position in ind_tour
                    ]
                    for ind_tour in all_visited_tours
                ]
                for all_visited_tours in ant['visited']
            ]
            for ant in self.ants
        ]

        delta_tau = [
            [
                [
                    np.average(heu_ant)
                    for heu_ant in ant
                ]
                for ant in n_ants
            ]
            for n_ants in heuristic_values
        ]

        for position in visited_positions:
            self.pheromone_matrix[position[0]][position[1]] = (1 - self.rho) * self.__pheromone(position=position)

            delta_tau_total = 0

            for (index_x, index_y, index_z), ant in np.ndenumerate(self.ants):
                if position in ant['visited']:
                    delta_tau_ant = delta_tau[index_x][index_y][index_z]
                    delta_tau_total += delta_tau_ant

                    # Lasă furnica să-și amintească cantitatea de feromoni actualizată de furnica de index_x, index_y, index_z
                    self.__saveAntPheromone(pheromone=delta_tau_ant, index_x=index_x, index_y=index_y, index_z=index_z)

            self.pheromone_matrix[position[0]][position[1]] += self.rho * delta_tau_total

        # Poziții care nu sunt în memoria vreunei furnici (poziții uitate)
        forgot_positions = set(self.all_visited_positions).difference(visited_positions)

        # Reduceți feromonul pe pozițiile uitate dacă este mai mult decât tau init
        for position in forgot_positions:
            temp_pheromone_decay = (1 - self.rho) * self.__pheromone(position=position)
            if temp_pheromone_decay > self.tauinit:
                self.pheromone_matrix[position[0]][position[1]] = temp_pheromone_decay
            else:
                self.pheromone_matrix[position[0]][position[1]] = self.tauinit
                self.all_visited_positions.remove(tuple(position))

    def __modifyAntPositions(self, iter):
        avg_pheromone = np.average(self.ants['pheromone'])
        print("Avg. Pheromone: "+str(avg_pheromone))

        # Pentru fiecare furnica
        for (index_x, index_y, index_z), ant in np.ndenumerate(self.ants):
            # Dacă feromonul depus este mai mic decât media feromonului depus
            if ant['pheromone'] < avg_pheromone:
                del self.ants['visited'][index_x][index_y][index_z][:]
                # Creșteți indicele euristic maxim
                self.max_heuristic_index += 1
            # Mutați furnica într-o nouă poziție în care valoarea euristică este mare(folosind self.max_heuristic_index)
                self.ants['position'][index_x][index_y][index_z] = \
                    list(self.heuristic_sorted_indices[self.max_heuristic_index])
                # Marcați pozițiile inițiale ca fiind vizitate pentru fiecare furnică
                self.ants['visited'][index_x][index_y][index_z] = [
                    tuple(self.ants['position'][index_x][index_y][index_z])]

    # Salvarea feromonului al furnicii
    def __saveAntPheromone(self, pheromone, index_x, index_y, index_z):
        self.ants['pheromone'][index_x][index_y][index_z] += pheromone
