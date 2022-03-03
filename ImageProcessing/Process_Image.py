import numpy as np
import cv2 as cv


class ProcessImage:
    def __init__(self, file_path):
        self.file_path = ''.join(file_path)     # calea catre imagine
        N = [0, 1]
        S = [0, -1]
        E = [1, 0]
        V = [-1, 0]
        NE = [1, 1]
        SE = [1, -1]
        SV = [-1, -1]
        NV = [-1, 1]
        self.coords_heu = [[N, S], [E, V], [SE, NV], [NE, SV]]  # coordonatele euristicii

    # calcularea euristicii
    def ParsareIntensitati(self):
        print("Parsing image.......START")
        img = cv.imread(self.file_path, 0)
        self.matrix_intesify = np.array(img, dtype='int64')
        # Inițializarea matricei de intensitate
        self.x_max_index, self.y_max_index = self.matrix_intesify.shape
        print("Parsing image .....................DONE")
        # noinspection PyTypeChecker
        np.savetxt('C:/Users/user/Desktop/LICENTA/Practic/gray_values', self.matrix_intesify, fmt='%d', newline='\n')
        print("Image intensity matrix wrote to gray_values")
        # Inițializarea euristică a matricei
        self.heuristic_matrix = np.zeros(shape=(self.x_max_index, self.y_max_index), dtype='int64')
        print("Calculating heuristic matrix.....START")
        for (x_index, y_index), intensity in np.ndenumerate(self.matrix_intesify):
            self.CalculateHeuristic(x_index, y_index)
        print("Calculating heuristic matrix.....DONE")
        # noinspection PyTypeChecker
        np.savetxt('C:/Users/user/Desktop/LICENTA/Practic/heuristic_matrix', self.heuristic_matrix, fmt='%d', newline='\n')
        print("Heuristic matrix wrote to heuristic_matrix.txt")
        return self.heuristic_matrix

    # Metoda de vecinate 8-conectivitate
    def CalculateHeuristic(self, x_index, y_index):
        V_c = 0
        for index, coords in enumerate(self.coords_heu):
            if ((0 <= x_index + coords[0][0] < self.x_max_index) and
                    (0 <= y_index + coords[0][1] < self.y_max_index) and
                    (0 <= x_index + coords[1][0] < self.x_max_index) and
                    (0 <= y_index + coords[1][1] < self.y_max_index)):
                V_c = V_c + abs(self.matrix_intesify[x_index + coords[0][0]][y_index + coords[0][1]] -
                                self.matrix_intesify[x_index + coords[1][0]][y_index + coords[1][1]])

        self.heuristic_matrix[x_index][y_index] = V_c
