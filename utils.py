import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class dataloader:

    def __init__(self, dataset='extended', norm_method='min_max', val_fold=1, crop=None, inv_thresh=0.4, custom_data=None, verbose=0):
        # Arguments:
        # dataset = 'balanced' or 'extended'
        # norm method = 'L2', 'min_max', 'max', or None
        # fold = between 1 and 5, indicates how validation set is split from training
        # crop = tuple of (start, finish) or None
        # inv_thresh = between 0 and 1 value of involvement we're interested in; can be used in conjunction with ~custom~
        # custom = 'benign', 'cancer', 'high_inv'
        # verbose = 0 for silent, 1 for summary, 2 for summary + progress

        # Initialize
        if (verbose == 2):
            print("Initializing...")
        self.data_train = None
        self.label_train = None
        self.inv_train = None
        self.data_val = None
        self.label_val = None
        self.inv_val = None
        self.data_test = None
        self.label_test = None
        self.inv_test = None

        # Load data
        if (verbose == 2):
            print("Loading data...")
        if (dataset == 'balanced'):
            self.parse_BK('BK_RF_P1_90.mat', val_fold)
        elif (dataset == 'extended'):
            self.parse_BK('BK_RF_P1_90-ext.mat', val_fold)
        else:
            raise ValueError('Unexpected value for parameter ~dataset~ (expected ~balanced~ or ~extended~)')

        # Extract dataset stats
        nc_train_benign = len([label for label in self.label_train if label == 0])
        nc_train_cancer = len([label for label in self.label_train if label == 1])
        nc_train_hi_inv_cancer = len([inv for inv in self.inv_train if inv >= inv_thresh])
        nc_val_benign = len([label for label in self.label_val if label == 0])
        nc_val_cancer = len([label for label in self.label_val if label == 1])
        nc_val_hi_inv_cancer = len([inv for inv in self.inv_val if inv >= inv_thresh])
        nc_test_benign = len([label for label in self.label_test if label == 0])
        nc_test_cancer = len([label for label in self.label_test if label == 1])
        nc_test_hi_inv_cancer = len([inv for inv in self.inv_test if inv >= inv_thresh])

        # Crop data
        if (crop is not None):
            if (verbose == 2):
                print("Cropping data...")
            self.data_train = self.crop_data(self.data_train, crop)
            self.data_val = self.crop_data(self.data_val, crop)
            self.data_test = self.crop_data(self.data_test, crop)

        # Apply normalization
        if (verbose == 2):
            print("Applying normalization...")
        self.normalize(norm_method)

        # Format data
        if (verbose == 2):
            print("Formatting data...")
        self.data_train, self.inv_train, self.label_train = self.format_data(self.data_train, self.inv_train, self.label_train)
        self.data_val, self.inv_val, self.label_val = self.format_data(self.data_val, self.inv_val, self.label_val)
        self.data_test, self.inv_test, self.label_test = self.format_data(self.data_test, self.inv_test, self.label_test)

        # Get custom data sets
        if (custom_data == 'benign'):
            self.data_train, self.inv_train, self.label_train = self.byLabel(self.data_train, self.label_train, 0, self.inv_train)
            self.data_val, self.inv_val, self.label_val = self.byLabel(self.data_val, self.label_val, 0, self.inv_val)
            self.data_test, self.inv_test, self.label_test = self.byLabel(self.data_test, self.label_test, 0, self.inv_test)
            nc_train_cancer = 0
            nc_train_hi_inv_cancer = 0
            nc_val_cancer = 0
            nc_val_hi_inv_cancer = 0
            nc_test_cancer = 0
            nc_test_hi_inv_cancer = 0
        elif (custom_data == 'cancer'):
            self.data_train, self.inv_train, self.label_train = self.byLabel(self.data_train, self.label_train, 1, self.inv_train)
            self.data_val, self.inv_val, self.label_val = self.byLabel(self.data_val, self.label_val, 1, self.inv_val)
            self.data_test, self.inv_test, self.label_test = self.byLabel(self.data_test, self.label_test, 1, self.inv_test)
            nc_train_benign = 0
            nc_val_benign = 0
            nc_test_benign = 0
        elif (custom_data == 'hi_inv'):
            self.data_train, self.inv_train, self.label_train = self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.label_train)
            self.data_val, self.inv_val, self.label_val = self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.label_val)
            self.data_test, self.inv_test, self.label_test = self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.label_test)
            nc_train_benign = 0
            nc_train_cancer = nc_train_hi_inv_cancer
            nc_val_benign = 0
            nc_val_cancer = nc_val_hi_inv_cancer
            nc_test_benign = 0
            nc_test_cancer = nc_test_hi_inv_cancer
        elif (custom_data == 'benign+hi_inv'):
            dT, iT, lT = self.byLabel(self.data_train, self.label_train, 0, self.inv_train)
            dT = np.append(dT, self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.label_train)[0], axis=0)
            iT = np.append(iT, self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.label_train)[1])
            lT = np.append(lT, self.byInv(self.data_train, self.inv_train, inv_thresh, 'gt', self.label_train)[2])
            self.data_train = dT
            self.inv_train = iT
            self.label_train = lT
            dV, iV, lV = self.byLabel(self.data_val, self.label_val, 0, self.inv_val)
            dV = np.append(dV, self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.label_val)[0], axis=0)
            iV = np.append(iV, self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.label_val)[1])
            lV = np.append(lV, self.byInv(self.data_val, self.inv_val, inv_thresh, 'gt', self.label_val)[2])
            self.data_val = dV
            self.inv_val = iV
            self.label_val = lV
            dS, iS, lS = self.byLabel(self.data_test, self.label_test, 0, self.inv_test)
            dS = np.append(dS, self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.label_test)[0], axis=0)
            iS = np.append(iS, self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.label_test)[1])
            lS = np.append(lS, self.byInv(self.data_test, self.inv_test, inv_thresh, 'gt', self.label_test)[2])
            self.data_test = dS
            self.inv_test = iS
            self.label_test = lS
            nc_train_cancer = nc_train_hi_inv_cancer
            nc_val_cancer = nc_val_hi_inv_cancer
            nc_test_cancer = nc_test_hi_inv_cancer
        else:
            pass

        # Print a summary
        if (verbose != 0):

            print("------------")

            if (dataset == 'balanced'):
                print("Dataset:                BK_RF_P1_90.mat")
            else:
                print("Dataset:                BK_RF_P1_90-ext.mat")

            print("Validation fold:        " + str(val_fold))

            print(" --> Training:          " + str(nc_train_benign+nc_train_cancer) + " total cores (" + str(len(self.data_train)) + " data points)")
            print("                        " + str(nc_train_benign) + " benign cores (" + str(len(self.byLabel(self.data_train, self.label_train, 0))) +  " data points)")
            print("                        " + str(nc_train_cancer) + " cancer cores (" + str(len(self.byLabel(self.data_train, self.label_train, 1))) +  " data points)")
            print("                        " + str(nc_train_hi_inv_cancer) + " high involvement (>=" + str(inv_thresh) + ") cancer cores (" + str(len(self.byInv(self.data_train, self.inv_train, inv_thresh, cond='gt'))) +  " data points)")

            print(" --> Validation:        " + str(nc_val_benign+nc_val_cancer) + " total cores (" + str(len(self.data_val)) + " data points)")
            print("                        " + str(nc_val_benign) + " benign cores (" + str(len(self.byLabel(self.data_val, self.label_val, 0))) +  " data points)")
            print("                        " + str(nc_val_cancer) + " cancer cores (" + str(len(self.byLabel(self.data_val, self.label_val, 1))) +  " data points)")
            print("                        " + str(nc_val_hi_inv_cancer) + " high involvement (>=" + str(inv_thresh) + ") cancer cores (" + str(len(self.byInv(self.data_val, self.inv_val, inv_thresh, cond='gt'))) +  " data points)")

            print(" --> Testing:           " + str(nc_test_benign+nc_test_cancer) + " total cores (" + str(len(self.data_test)) + " data points)")
            print("                        " + str(nc_test_benign) + " benign cores (" + str(len(self.byLabel(self.data_test, self.label_test, 0))) +  " data points)")
            print("                        " + str(nc_test_cancer) + " cancer cores (" + str(len(self.byLabel(self.data_test, self.label_test, 1))) +  " data points)")
            print("                        " + str(nc_test_hi_inv_cancer) + " high involvement (>=" + str(inv_thresh) + ") cancer cores (" + str(len(self.byInv(self.data_test, self.inv_test, inv_thresh, cond='gt'))) +  " data points)")

            print("Normalization method:   " + str(norm_method))

            if (crop is not None):
                print("Time-series cropping:   Frames " + str(crop[0]) + " to " + str(crop[1]))
            else:
                print("Time-series cropping:   None")

            print("------------")

        # Reshape data
        self.data_train = np.reshape(self.data_train, (self.data_train.shape[0], self.data_train.shape[1], 1))
        self.data_val = np.reshape(self.data_val, (self.data_val.shape[0], self.data_val.shape[1], 1))
        self.data_test = np.reshape(self.data_test, (self.data_test.shape[0], self.data_test.shape[1], 1))

    def parse_BK(self, dataset, fold):
        # Parses matlab datasets and populates training, validation, and test sets

        # Extract data
        inputdata = hdf5storage.loadmat(dataset)
        data = inputdata['data'][0]
        label = inputdata['label'][0]
        patient_id = inputdata['PatientId'][0]
        inv = inputdata['inv'][0]

        # Allocate train vs test
        test_id = [5, 6, 18, 45, 37, 30, 12, 14, 24, 46, 52, 58, 79, 71]
        test_idx = [True if pid in test_id else False for pid in patient_id]
        train_idx = [not tr for tr in test_idx]

        # Allocate train vs val
        if (fold == 1):
            fold1_id = [3, 4, 8, 21, 25, 29, 64, 80]
            train_idx = [False if pid in fold1_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold1_id else False for pid in patient_id]
        elif (fold == 2):
            fold2_id = [7, 11, 26, 40, 47, 57, 72, 87]
            train_idx = [False if pid in fold2_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold2_id else False for pid in patient_id]
        elif (fold == 3):
            fold3_id = [13, 19, 27, 59, 60, 65, 81, 85]
            train_idx = [False if pid in fold3_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold3_id else False for pid in patient_id]
        elif (fold == 4):
            fold4_id = [22, 23, 38, 39, 48, 68, 74, 82]
            train_idx = [False if pid in fold4_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold4_id else False for pid in patient_id]
        else:
            fold5_id = [2, 10, 28, 42, 66, 70, 76, 89, 90]
            train_idx = [False if pid in fold5_id else tr_idx for pid, tr_idx in zip(patient_id, train_idx)]
            val_idx = [True if pid in fold5_id else False for pid in patient_id]

        # Separate train vs test set
        self.data_train = data[train_idx]
        self.label_train = label[train_idx]
        self.inv_train = inv[train_idx]
        self.data_val = data[val_idx]
        self.label_val = label[val_idx]
        self.inv_val = inv[val_idx]
        self.data_test = data[test_idx]
        self.label_test = label[test_idx]
        self.inv_test = inv[test_idx]

    def normalize(self, norm_method):

        if (norm_method == 'L2'):
            self.data_train = self.L2_norm(self.data_train)
            self.data_val = self.L2_norm(self.data_val)
            self.data_test = self.L2_norm(self.data_test)
        elif (norm_method == 'min_max'):
            self.data_train = self.min_max_norm(self.data_train)
            self.data_val = self.min_max_norm(self.data_val)
            self.data_test = self.min_max_norm(self.data_test)
        elif (norm_method == 'min_max_per_core'):
            self.data_train = self.min_max_per_core_norm(self.data_train)
            self.data_val = self.min_max_per_core_norm(self.data_val)
            self.data_test = self.min_max_per_core_norm(self.data_test)
        elif (norm_method == 'max'):
            self.data_train = self.max_norm(self.data_train)
            self.data_val = self.max_norm(self.data_val)
            self.data_test = self.max_norm(self.data_test)
        else:
            pass

    @staticmethod
    def L2_norm(data):
        # Performs L2 norm on data
        # Input and output both length = num_cores list of num_data_points by num_timesteps ndarrays

        if (data is None):
            return None

        out = []

        for i in range(len(data)):
            mean_fea = np.mean(data[i], axis=0, keepdims=True) + 1e-6
            std_fea = np.std(data[i], axis=0, keepdims=True) + 1e-6
            out.append(np.divide(data[i] - mean_fea, std_fea))

        return out

    @staticmethod
    def min_max_norm(data):
        # Performs min-max norm on data
        # Input and output both length = num_cores list of num_data_points by num_timesteps ndarrays

        if (data is None):
            return None

        out = []

        for i in range(len(data)):
            scaler = MinMaxScaler(feature_range=(0,1))
            scaler = scaler.fit(data[i].T)
            normed = scaler.transform(data[i].T)
            out.append(np.transpose(normed))

        return out

    @staticmethod
    def min_max_per_core_norm(data):
        # Performs min-max norm on each core (instead of each data point)
        # Input and output both length = num_cores list of num_data_points by num_timesteps ndarrays

        if (data is None):
            return None

        out = []

        for i in range(len(data)):
            minimum = np.min(data[i])
            maximum = np.max(data[i])
            out.append((data[i] - minimum) / (maximum - minimum))

        return out

    @staticmethod
    def max_norm(data):
        # Performs max norm on data (i.e. divide data by max amplitude)
        # Input and output both length = num_cores list of num_data_points by num_timesteps ndarrays

        if (data is None):
            return None

        out = []

        for i in range(len(data)):
            max_fea = np.max(data[i])
            out.append(np.divide(data[i], max_fea))

        return out

    @staticmethod
    def format_data(data, inv, label):
        # Outputs:
        # X (ndarray): total_data_points by num_timesteps
        # I (ndarray): total_data_points by 1
        # Y (ndarray): total_data_points by 1

        if (data is None or inv is None or label is None):
            return None, None, None

        X = np.concatenate(data, axis=0)
        I = np.array([])
        for i in range(len(inv)):
            I = np.append(I, inv[i]*np.ones(data[i].shape[0]))
        Y = np.array([])
        for i in range(len(label)):
            Y = np.append(Y, label[i]*np.ones(data[i].shape[0]))

        return X, I, Y

    @staticmethod
    def crop_data(data, crop):

        if (crop is None or data is None):
            return None

        # Check that cropping fits timeseries boundaries
        num_timestamps = data[0].shape[1]
        if (crop[0] < 0 or crop[1] > num_timestamps):
            raise ValueError('Cropping tuple out of range')

        out = []

        # Iterate through and crop
        for i in range(len(data)):
            out.append(data[i][:,crop[0]:crop[1]])

        return out

    @staticmethod
    def byLabel(X, Y, label, I=None):
        # returns only rows of X with specified label
        # if all of data, inv, and label are provided, returns all three for entries with specified label

        if (X is None or Y is None):
            return None

        mask = Y == label

        if (I is not None):
            return X[mask], I[mask], Y[mask]

        return X[mask]

    @staticmethod
    def byInv(X, I, inv, cond, Y=None):
        # returns only rows of X with involvement satisfying condition
        # if all of data, inv, and label are provided, returns all three for entries with the condition

        if (X is None or I is None):
            return None

        if (cond == 'gt'):
            mask = I >= inv
        else:
            mask = I <= inv

        if (Y is not None):
            return X[mask], I[mask], Y[mask]

        return X[mask]

class visualizer:

    def __init__(self, model, history, dataset, norm_method, val_fold, crop, inv_thresh):

        # Copy over model
        self.model = model
        self.history = history

        # Create test data
        self.benigns = dataloader(dataset=dataset,
                                  norm_method=norm_method,
                                  val_fold=val_fold,
                                  crop=crop,
                                  inv_thresh=inv_thresh,
                                  custom_data='benign',
                                  verbose=0).data_test
        self.cancers = dataloader(dataset=dataset,
                                  norm_method=norm_method,
                                  val_fold=val_fold,
                                  crop=crop,
                                  inv_thresh=inv_thresh,
                                  custom_data='hi_inv',
                                  verbose=0).data_test


    def training_curve(self):
        # Plot training curve

        print("Plotting training curve...")
        plt.close()
        plt.plot(self.history['val_loss'], label='validation')
        plt.plot(self.history['loss'], label='training')
        plt.legend()
        plt.savefig('training-curve.png')

    def benign_cancer_examples(self, num_examples=5):
        # Plot input-output examples for benign and cancer

        print("Plotting input-output examples for benign...")
        for i in range(num_examples):
            sample = np.reshape(self.benigns[np.random.randint(len(self.benigns)),:,0], (1, -1, 1))
            out_ex = np.reshape(self.model.predict(sample), (-1, 1))
            in_ex = np.reshape(sample, (-1, 1))
            loss = self.model.evaluate(x=sample, y=sample, batch_size=None, verbose=0)
            plt.close()
            plt.plot(out_ex, label='output')
            plt.plot(in_ex, label='input')
            plt.suptitle(str(loss))
            plt.legend()
            fname = "input-output-benign-" + str(i) + ".png"
            plt.savefig(fname)

        print("Plotting input-output examples for cancer...")
        for i in range(num_examples):
            sample = np.reshape(self.cancers[np.random.randint(len(self.cancers)),:,0], (1, -1, 1))
            out_ex = np.reshape(self.model.predict(sample), (-1, 1))
            in_ex = np.reshape(sample, (-1, 1))
            loss = self.model.evaluate(x=sample, y=sample, batch_size=None, verbose=0)
            plt.close()
            plt.plot(out_ex, label='output')
            plt.plot(in_ex, label='input')
            plt.suptitle(str(loss))
            plt.legend()
            fname = "input-output-cancer-" + str(i) + ".png"
            plt.savefig(fname)

    def error_distribution(self, max_samples=10000, hist_range=None):
        # Plot error distributions

        length = min((len(self.benigns), len(self.cancers), max_samples))
        print("Calculating error distributions for " + str(length) + " samples of benign and cancer...")
        benign_samples = self.benigns[0:length,:,:]
        cancer_samples = self.cancers[0:length,:,:]
        benign_loss_vec = []
        cancer_loss_vec = []
        for i in range(length):
            benign_signal = np.reshape(benign_samples[i,:,0], (1, -1, 1))
            cancer_signal = np.reshape(cancer_samples[i,:,0], (1, -1, 1))
            benign_loss_vec.append(self.model.evaluate(x=benign_signal, y=benign_signal, batch_size=None, verbose=0)) # Source of error: y = [decoder output, clustering]
            cancer_loss_vec.append(self.model.evaluate(x=cancer_signal, y=cancer_signal, batch_size=None, verbose=0))
        plt.close()
        plt.hist(benign_loss_vec, bins=1000, alpha=0.5, range=hist_range, label='benign')
        plt.hist(cancer_loss_vec, bins=1000, alpha=0.5, range=hist_range, label='hi_inv')
        plt.legend()
        plt.savefig('benign-cancer-loss-distrib.png')

