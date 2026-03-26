# Вихідний код компʼютерної програми «Модифікована нейро-нечітка мережа Ванга-Менделя»
# Комп’ютерна програма для класифікації дефектів лопатей вітрових турбін, що реалізує 
# нейро-нечіткий алгоритм Ванга–Менделя та забезпечує аналіз технічного стану вітроенергетичних 
# установок за обмежених обсягів навчальних даних.

# Дата остаточного завершення роботи над програмою: 26.03.2026 
# Дата публікації програми у вільному доступі на платформі GitHub: 26.03.2026

# Перемішування та нормалізація вхідних даних
def process_input_dataframe(dataframe):
    dataframe.apply(pd.to_numeric, errors='coerce')
    dataframe.dropna(inplace=True)

    return dataframe.sample(frac=1)

# Нормалізація масиву елементів
def normalize_array(array, min_value, max_value):
    return (array - min_value) / (max_value - min_value)


# Обрахунок меж нормалізації по стовпцях для структури даних DataFrame
def calculate_normalization_boundaries_for_columns(dataframe, target_columns):
    boundaries = {}

    for column_name in target_columns:
        boundaries[column_name] = {
            'min': get_column_min(dataframe, column_name),
            'max': get_column_max(dataframe, column_name)
        }

    return boundaries


# Нормалізація по стовпцях структури даних DataFrame
def normalize_dataframe(dataframe, normalization_boundaries_dict):
    for column_name, boundaries_dict in normalization_boundaries_dict.items():
        min_value = boundaries_dict['min']
        max_value = boundaries_dict['max']

        column_arr = np.array(dataframe[column_name])

        dataframe[column_name] = normalize_array(column_arr, min_value, max_value)

# Пошук мінімального значення за назвою стовпця у структурі даних DataFrame
def get_column_min(dataframe, column_name):
    return dataframe[column_name].min()

# Пошук мінімального значення за назвою стовпця у структурі даних DataFrame
def get_column_min(dataframe, column_name):
    return dataframe[column_name].min()

# Пошук максимального значення за назвою стовпця у структурі даних DataFrame
def get_column_max(dataframe, column_name):
    return dataframe[column_name].max()

# Відображення характеристик та прикладу даних, записаних у структурі даних DataFrame
def display_dataframe_info(dataframe):
    print('****** General Information ******')
    display(dataframe.info())
    display(dataframe.describe())

    print('****** Data Sample ******')
    display(dataframe.head())

# Обчислення значення дельти Кронекера
def kronecker_delta(i, j):
    return 1 if i == j else 0

# Клас для реалізації навчання та використання Шару Фазифікації 
class FuzzifyingLayer:
    def __init__(self, membership_functions_count, input_vector_length, c_initial):
        self.membership_functions_count = membership_functions_count
        self.input_vector_length = input_vector_length
        self.c = []
        self.s = (membership_functions_count - 1) ** -1

        for i in range(self.input_vector_length):
            self.c.append(c_initial)

    def fuzzify(self, input_vector):
        memberships_matrix = []

        for i in range(self.input_vector_length):
            x = input_vector[i]
            x_memberships = []

            for j in range(self.membership_functions_count):
                membership = self.calculate_membership_for_x(x, i, j)
                x_memberships.append(membership)

            memberships_matrix.append(x_memberships)

        return memberships_matrix

    def calculate_membership_for_x(self, x, i, j):
        c, s = self.c[i][j], self.s

        membership = math.exp(- (((x - c) ** 2) / (2 * (s ** 2))))

        return membership

    def train(self, v_c, v_s, predicted_value, actual_values, weights, input_vector):
        c_copy = np.empty_like(self.c)

        for r in range(len(self.c)):
            for l in range(len(self.c[r])):
                c_copy[r][l] = self.calculate_new_c(r, l, weights, v_c, predicted_value, actual_values, input_vector)

        self.c = c_copy

    def calculate_new_c(self, r, j, weights, v_c, predicted_value, actual_value, input_vector):
        memberships_matrix = self.fuzzify(input_vector)

        return self.c[r][j] - v_c * self.calculate_de_to_dc(r, j, weights, predicted_value, actual_value, input_vector,
                                                            memberships_matrix)

    def calculate_de_to_dc(self, r, j, weights, predicted_value, actual_value, input_vector, memberships_matrix):
        error_derivative = np.sum(predicted_value - actual_value)

        weighted_outputs = 0

        for l in range(len(weights)):
            weighted_outputs += weights[l] * self.calculate_dy_to_dc(l, r, j, input_vector, memberships_matrix)

        return error_derivative * weighted_outputs

    def calculate_dy_to_dc(self, l, r, j, input_vector, memberships_matrix):
        m = self.calculate_m(memberships_matrix)
        t = self.calculate_t(l, memberships_matrix)

        first_multiplier = (kronecker_delta(l, r) * m - t) / (m ** 2)
        second_multiplier = self.calculate_t_excluding(r, j, memberships_matrix)
        third_multiplier = self.calculate_dm_to_dc(r, j, input_vector)

        dy_to_dc = first_multiplier * second_multiplier * third_multiplier

        return dy_to_dc

    def calculate_dm_to_dc(self, r, j, input_vector):
        x = input_vector[r]
        c = self.c[r][j]
        s = self.s

        return (math.exp(- (((x - c) ** 2) / 2 * (s ** 2))) * (x - c)) / (self.s ** 2)

    def calculate_m(self, memberships_matrix):
        result = 0

        for p in range(self.membership_functions_count):
            result += self.calculate_t(p, memberships_matrix)

        return result

    def calculate_t(self, l, memberships_matrix):
        result = 1

        for i in range(self.input_vector_length):
            result *= memberships_matrix[i][l]

        return result

    def calculate_t_excluding(self, r, j, memberships_matrix):
        result = 1

        for i in range(self.input_vector_length):
            if i != j:
                result *= memberships_matrix[i][r]

        return result

# Клас для реалізації Шару Агрегування 
class AggregatingLayer:
    def __init__(self, membership_functions_count, input_vector_length):
        self.membership_functions_count = membership_functions_count
        self.input_vector_length = input_vector_length

    def aggregate(self, memberships_matrix):
        aggregated_memberships = []

        for i in range(self.membership_functions_count):
            product = 1

            for j in range(self.input_vector_length):
                product *= memberships_matrix[j][i]

            aggregated_memberships.append(product)

        return aggregated_memberships

# Клас для реалізації навчання та використання Лінійного Шару
class LinearLayer:
    def __init__(self, membership_functions_count, input_vector_length):
        self.membership_functions_count = membership_functions_count
        self.input_vector_length = input_vector_length
        self.weights = np.random.uniform(low=-1, high=1, size=self.membership_functions_count).tolist()

    def process(self, aggregated_memberships):
        weighted_aggregated_memberships_sum = 0
        aggregated_memberships_sum = 0

        for i in range(self.membership_functions_count):
            weighted_aggregated_memberships_sum += self.weights[i] * aggregated_memberships[i]
            aggregated_memberships_sum += aggregated_memberships[i]

        return {
            'f1': weighted_aggregated_memberships_sum,
            'f2': aggregated_memberships_sum,
        }

    def train(self, input_data, expected_output_data, plot_weights=False):
        p_vs = []

        for input_vector in input_data:
            p_v = self.calculate_p_vs(input_vector)
            p_vs.append(p_v)

        pseudo_inverse = np.linalg.pinv(p_vs)
        new_weights = np.matmul(pseudo_inverse, expected_output_data)

        if plot_weights:
            plt.clf()
            plt.plot(self.weights, color='r', label='Previous Weights')
            plt.plot(new_weights, color='g', label='New Weights')
            plt.legend()
            plt.show()

        self.weights = new_weights

    def calculate_p_vs(self, memberships_matrix):
        values = []
        p_vs = []

        for i in range(self.membership_functions_count):
            value = 1

            for j in range(self.input_vector_length):
                value *= memberships_matrix[j][i]

            values.append(value)

        for i in range(len(values)):
            numerator = values[i]
            denominator = 0

            for j in range(len(values)):
                denominator += values[j]

            p_vs.append(numerator / denominator)

        return p_vs

class DefuzzificationLayer:
    def diffuzify(self, fs_dict):
        return np.round(fs_dict['f1'] / fs_dict['f2'], 0

# Клас для обʼєднання шарів у єдину мережу; навчання та використання навченної мережі
class WangMendelFuzzyNeuralNetwork:
    def __init__(self, membership_functions_count, input_vector_length, c_initial, s_initial):
        self.fuzzifying_layer = FuzzifyingLayer(membership_functions_count, input_vector_length, c_initial, s_initial)
        self.aggregating_layer = AggregatingLayer(membership_functions_count, input_vector_length)
        self.linear_layer = LinearLayer(membership_functions_count, input_vector_length)
        self.defuzzification_layer = DefuzzificationLayer()

    def predict_exact(self, input_vector):
        return np.round(self.predict(input_vector), 0)

    def predict(self, input_vector):
        fuzzifying_layer_output = self.fuzzifying_layer.fuzzify(input_vector)
        aggregating_layer_output = self.aggregating_layer.aggregate(fuzzifying_layer_output)
        linear_layer_output = self.linear_layer.process(aggregating_layer_output)
        defuzzification_layer_output = self.defuzzification_layer.diffuzify(linear_layer_output)

        return defuzzification_layer_output

    def predict_for_matrix(self, input_data):
        predictions = []

        for input_vector in input_data:
            prediction = self.predict(input_vector)
            predictions.append(prediction)

        return predictions

    def predict_and_validate(self, input_data, expected_output_data, detailed_output=False):
        predicted_output_data = self.predict_for_matrix(input_data)
        predicted_output_data_rounded = []

        total_predictions = len(predicted_output_data)
        correct_predictions = 0

        for i in range(total_predictions):
            predicted_value = predicted_output_data[i]
            predicted_value_rounded = np.round(predicted_value, 0)
            expected_value = expected_output_data[i]
            predicted_output_data_rounded.append(predicted_value_rounded)


            is_predicted_equal_to_expected = predicted_value_rounded == expected_value

            if is_predicted_equal_to_expected:
                correct_predictions += 1

            if detailed_output:
                print(
                    f'Predicted: {predicted_value}; Actual: {expected_value}; {'✅' if is_predicted_equal_to_expected else '❌'}')

        fig, ax = plt.subplots()
        ax.scatter(expected_output_data, predicted_output_data, c='black')
        ax.set(xlabel='Actual defect class')
        ax.set(ylabel='Predicted defect class')
        ax.set(title='Actual and Predicted defect class')
        ax.axline((0.0, 0.0), slope=1, color='red')
        ax.set_aspect(1.0)
        plt.show()

        mse = (np.square(predicted_output_data - expected_output_data)).mean()
        accuracy = np.round(correct_predictions / total_predictions * 100, 0)

        print(f'MSE: {mse}')
        print(f'Accuracy: {accuracy}')

        return [mse, accuracy]

    def train(self, epochs, training_data_df, c_nu=0.5, s_nu=0.5, max_epochs_with_no_mse_change=2):
        print(f'Started training for {epochs} epochs', end='\n')

        input_data = np.array(training_data_df.iloc[:, :-1].values, dtype=float)
        expected_output_data = np.array(training_data_df.iloc[:, -1].values, dtype=float)

        start_time = time.time()

        current_mse = 1
        epochs_without_mse_change = 0
        completed_epochs = 0

        training_vectors_count = len(input_data)

        for i in trange(epochs, desc='Epochs progress'):
            print(f'Epoch {i + 1} started', end='\n'

            self.train_linear_layer(input_data, expected_output_data)

            for j in trange(20, desc='Data passes per epoch progress', leave=False):
                predicted_data = self.predict_for_matrix(input_data)

                for k in range(training_vectors_count):
                    self.fuzzifying_layer.train(
                        c_nu,
                        s_nu,
                        predicted_data,
                        expected_output_data,
                        self.linear_layer.weights,
                        input_data[k]
                    )

            [mse, accuracy] = self.predict_and_validate(input_data, expected_output_data, detailed_output=True)

            previous_mse = current_mse
            current_mse = mse
            mse_change = previous_mse - current_mse

            if mse_change == 0:
                epochs_without_mse_change += 1

            if epochs_without_mse_change == max_epochs_with_no_mse_change:
                print(f'MSE hasn\’t changed during last {max_epochs_with_no_mse_change} epochs; Training ended',
                      end='\n')
                break

            completed_epochs += 1
            print(f'Epoch {i + 1} completed', end='\n')

        end_time = time.time()
        elapsed_time = np.round(end_time - start_time, 0)

        print(f'Finished training in {elapsed_time}s in {completed_epochs} epochs', end='\n')

    def train_linear_layer(self, input_data, expected_output_data):
        input_data_fuzzified = []

        for input_vector in input_data:
            fuzzified_vector = self.fuzzifying_layer.fuzzify(input_vector)
            input_data_fuzzified.append(fuzzified_vector)

        self.linear_layer.train(input_data_fuzzified, expected_output_data)

    def save_to_file(self, file_name):
        c = self.fuzzifying_layer.c
        s = self.fuzzifying_layer.s
        weights = self.linear_layer.weights

        np.savez(f"{file_name}.npz", c=c, s=s, weights=weights)

    def initialize_from_file(self, file_name):
        file = np.load(f"{file_name}.npz")

        self.fuzzifying_layer.c = file['c']
        self.fuzzifying_layer.s = file['s']
        self.linear_layer.weights = file['weights']



# Навчання мережі
class_column_name = 'Type'

min_class = get_column_min(train_df, class_column_name)
max_class = get_column_max(train_df, class_column_name)

membership_functions_count = 30

input_vector_length = len(np.array(train_df.iloc[:, :-1].values, dtype=float)[0])

c_initial = np.linspace(min_class, max_class, membership_functions_count).tolist()
s_initial = np.full(membership_functions_count, 0.2).tolist()

wangMendelFuzzyNeuralNetwork = WangMendelFuzzyNeuralNetwork(membership_functions_count, input_vector_length, c_initial,
                                                            s_initial)

wangMendelFuzzyNeuralNetwork.train(10, train_df, 0.5, 0.5, 2)

# Ініціалізація мережі з файлу та використання її
class_column_name = 'Type'

min_class = get_column_min(train_df, class_column_name)
max_class = get_column_max(train_df, class_column_name)

membership_functions_count = 30

c_initial = np.linspace(min_class, max_class, membership_functions_count).tolist()
s_initial = np.full(membership_functions_count, 0.2).tolist()

wangMendelFuzzyNeuralNetwork1 = WangMendelFuzzyNeuralNetwork(30, 4, c_initial,
                                                             s_initial)
wangMendelFuzzyNeuralNetwork1.initialize_from_file('models/regular/model_91')

input_data = np.array(test_df.iloc[:, :-1].values, dtype=float)
expected_output_data = np.array(test_df.iloc[:, -1].values, dtype=float)

