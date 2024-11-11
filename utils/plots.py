import matplotlib.pyplot as plt

def plot_error(error_list, ylabel='Error'):
    plt.plot(error_list)
    plt.ylabel(ylabel)
    plt.show()

def plot_loss(train_list, test_list):
    plt.subplot(1, 2, 1)
    plt.plot(train_list)
    plt.ylabel('Train Error')
    plt.subplot(1, 2, 2)
    plt.plot(test_list)
    plt.ylabel('Test Error')
    plt.show()
