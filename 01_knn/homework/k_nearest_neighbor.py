import numpy as np

"""
Credits: the original code belongs to Stanford CS231n course assignment1. Source link: http://cs231n.github.io/assignments2019/assignment1/
"""


class KNearestNeighbor:
    """a kNN classifier with L2 distance"""

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
              y[i] is the label for X[i].

        ------------------------------------------------------

        Обучение классификатора. Для алгоритма k-ближайших соседей это просто
        запоминание обучающих данных.

        Входные данные:
        - X: Массив numpy формы (num_train, D), содержащий обучающие данные,
          состоящий из num_train выборок, каждая из которых имеет размерность D.
        - y: Массив numpy формы (N,), содержащий обучающие метки, где
          y[i] — метка для X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
              of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].

        ------------------------------------------------------

        Предсказывает метки для тестовых данных с помощью этого классификатора.

        Входные данные:
        - X: Массив numpy формы (num_test, D), содержащий тестовые данные,
          из которых num_test выборок имеют размерность D.
        - k: Количество ближайших соседей, голосующих за предсказанные метки.
        - num_loops: Определяет, какую реализацию использовать для вычисления расстояний
          между обучающими и тестовыми точками.

        Возвращает:
        - y: Массив numpy формы (num_test,), содержащий предсказанные метки для
          тестовых данных, где y[i] — предсказанная метка для тестовой точки X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.

        ------------------------------------------------------

        Вычисляет расстояние между каждой тестовой точкой в X и каждой обучающей точкой
        в self.X_train, используя вложенный цикл по обучающим данным и
        тестовым данным.

        Входные данные:
        - X: Массив numpy формы (num_test, D), содержащий тестовые данные.

        Возвращает:
        - dists: Массив numpy формы (num_test, num_train), где dists[i, j]
          — евклидово расстояние между i-й тестовой точкой и j-й обучающей
          точкой.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################

                ###########################################################################
                # TODO:                                                                   #
                # Вычислите расстояние l2 между i-й тестовой точкой и j-й                 #
                # обучающей точкой и сохраните результат в dists[i, j]. Вам не следует    #
                # использовать цикл по измерениям, а также использовать np.linalg.norm(). #
                ###########################################################################

                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                diff = X[i] - self.X_train[j]
                dists[i, j] = np.sqrt(np.sum(diff ** 2))
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops

        ------------------------------------------------------

        Вычислите расстояние между каждой тестовой точкой в X и каждой обучающей точкой
        в self.X_train, используя один цикл по тестовым данным.

        Входные/выходные данные: То же, что и compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            ########################################################################
            # TODO:                                                                #
            # Compute the l2 distance between the ith test point and all training  #
            # points, and store the result in dists[i, :].                         #
            # Do not use np.linalg.norm().                                         #
            ########################################################################

            ########################################################################
            # TODO:                                                                #
            # Вычислить расстояние l2 между i-й тестовой точкой и всеми обучающими #
            # точками и сохранить результат в dists[i, :].                         #
            # Не использовать np.linalg.norm().                                    #
            ########################################################################

            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            diff = X[i] - self.X_train
            dists[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops

        ------------------------------------------------------

        Вычисляет расстояние между каждой тестовой точкой в X и каждой обучающей точкой
        в self.X_train без явных циклов.

        Входные/выходные данные: То же, что и в compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################

        #######################################################################################
        # TODO:                                                                               #
        # Вычислите расстояние l2 между всеми тестовыми точками и всеми обучающими            #
        # точками без использования каких-либо явных циклов и сохраните результат в           #
        # dists.                                                                              #
        #                                                                                     #
        # Вам следует реализовать эту функцию, используя только базовые операции с массивами; #
        # в частности, вам не следует использовать функции из scipy,                          #
        # а также np.linalg.norm().                                                           #
        #                                                                                     #
        # ПОДСКАЗКА: Попробуйте сформулировать расстояние l2, используя умножение матриц      #
        #            и две широковещательные суммы.                                           #
        #######################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        X_sum_of_square = np.sum(X ** 2, axis=1)
        X_train_sum_of_square = np.sum(self.X_train ** 2, axis=1)
        scalar_product = X @ self.X_train.T
        square_of_dists = X_sum_of_square[:, None] + X_train_sum_of_square[None, :] - 2 * scalar_product
        dists = np.sqrt(np.maximum(square_of_dists, 0.0))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].

        ------------------------------------------------------

        Дана матрица расстояний между тестовыми и обучающими точками.
        Предскажите метку для каждой тестовой точки.

        Входные данные:
        - dists: массив numpy формы (num_test, num_train), где dists[i, j]
          задает расстояние между i-й тестовой точкой и j-й обучающей точкой.

        Возвращает:
        - y: массив numpy формы (num_test,), содержащий предсказанные метки для
          тестовых данных, где y[i] — предсказанная метка для тестовой точки X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################

            # Список длиной k, хранящий метки k ближайших соседей к
            # i-й тестовой точке.
            ########################################################################
            # TODO:                                                                #
            # Используйте матрицу расстояний, чтобы найти k ближайших соседей i-й  #
            # тестовой точки, и используйте self.y_train, чтобы найти метки этих   #
            # соседей. Сохраните эти метки в closest_y.                            #
            # Подсказка: найдите функцию numpy.argsort.                            #
            ########################################################################

            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            idx = np.argpartition(dists[i], k)[:k]
            closest_y = self.y_train[idx]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################

            ############################################################################
            # TODO:                                                                    #
            # Теперь, когда вы нашли метки k ближайших соседей, вам                    #
            # нужно найти наиболее часто встречающуюся метку в списке closest_y меток. #
            # Сохраните эту метку в y_pred[i]. В случае совпадения выбирайте меньшую   #
            # метку.                                                                   #
            ############################################################################

            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            closest_y, frequencies_y = np.unique(closest_y, return_counts=True)
            max_frequence_y = frequencies_y.max()
            y_pred[i] = closest_y[frequencies_y == max_frequence_y].min()
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred
