import numpy as np
import math
import matplotlib.pyplot as plt
from train import *

from os import path



# time_sgm, time_sgm_T = TimesToEpsilonAccuracy( dataset, n_tests, n_sample, epsilons, stepsizes, maxiter ,compute_every)
#
# Performs a number of experiments to determine how sensitive problems of type
# problem_type are to initial stepsizes. Returns three tensors, each of size
#
#  n_tests -by- length(epsilons) -by- length(stepsizes)
#
# where the entry [i,j,k] in the tensor corresponds to the number of iterations
# required by the given method, during experiment i, to achieve accuracy
# epsilons[j] using initial stepsize stepsizes[k].
#
# The currently supported methods are SGM (Stochastic Gradient Method),
# Truncated, Truncated-Adagrad, and Adam.
#
# The input data is as follows:
#
#  dataset - currently  from {MNIST,CIFAR10,Stanford_dogs}
#  dataset_size - size of dataset
#  optimizer - optimizer type ('sgd','truncated', 'adam', 'trunc_adagrad')
#  n_tests - number of experiments to run
#  epsilons - Different accuracies to test the time required to achieve these accuracies
#  stepsizes - list of stepsizes to use
#  maxepoch - maximum number of epochs
#  batch size - batch size for learning
#  compute_every - calculates the loss every 'compute_every' batches
#  use_cuda - specifies whether to use gpu for learning
def TimesToEpsilonAccuracy(model_name, dataset, dataset_size, optimizer, loss, n_tests, epsilons,
                           stepsizes, maxepoch, batch_size, use_cuda, dir_path):
    epsilons_acc = epsilons
    num_stepsizes = len(stepsizes)
    num_epsilons = len(epsilons_acc)
    times_to_eps_sgm_acc = np.zeros((n_tests, num_epsilons, num_stepsizes))
    best_acc = np.zeros((n_tests, num_stepsizes))

    opt_name = get_label(optimizer)
    for test_ind in range(0, n_tests, 1):
        # Generate Data
        print('test_ind = ' + str(test_ind))
        for i in range(0, len(stepsizes), 1):
            init_stepsize = stepsizes[i]
            print('init_step = ' + str(init_stepsize))

            file_acc_name = dir_path+opt_name+'_acc_step='+str(init_stepsize)+'_test'+str(test_ind)+'.npy'
            if path.isfile(file_acc_name) is False:
                train_losses, train_accuracies, val_losses, val_accuracies = NN_optimize_fast(model_name, dataset,
                                                                                              optimizer,
                                                                                              dataset_size,  loss,
                                                                                              maxepoch, init_stepsize,
                                                                                              batch_size, use_cuda)

                np.save(file_acc_name, val_accuracies)
                np.save(dir_path+opt_name+'_train_acc_step='+str(init_stepsize)+'_test'+str(test_ind)+'.npy', train_accuracies)
            else:
                val_accuracies = np.load(file_acc_name)

            if len(val_accuracies) > maxepoch:
                val_accuracies = val_accuracies[:maxepoch]


            best_acc[test_ind, i] = np.max(val_accuracies)
            for j in range(0, len(epsilons_acc), 1):
                eps_val_acc = epsilons_acc[j]
                ind_sgm_acc = maxepoch
                if np.max(val_accuracies) >= eps_val_acc:
                    ind_sgm_acc = np.min(np.where(val_accuracies >= eps_val_acc)[0])
                times_to_eps_sgm_acc[test_ind, j, i] = ind_sgm_acc

    return times_to_eps_sgm_acc, best_acc




# plots all the functions in lst_plots as a function of the corresponding x in lst_x
# lst_x - list of x axis numbers
# lst_plots - list of plots to be plotted as a function of x
# lst_labels - labels for the plots
# x_label - x axis label
# y_label - y axis label
# title - title
# x_log_scale - if True, plots the x in log scale
# y_log_scale - if True, plots the y in log scale
# path_file - a path for saving the plot
def plot_figures(lst_x, lst_plots, lst_labels, x_label, y_label, title, x_log_scale, y_log_scale, path_file):
    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(0,len(lst_plots),1):
        ax.plot(lst_x[i], lst_plots[i], '-o', label=lst_labels[i])
    if x_log_scale:
        ax.set_xscale('log')
    if y_log_scale:
        ax.set_yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    fig.savefig(path_file, bbox_inches='tight')


# PlotTimesToAccuracy(lst_time, lst_labels, stepsizes, epsilons, dir_path)
# lst_time - list of times for different methods
# lst_labels - list of methods names whose result to be displayed
# mode - should be from {'Loss', 'Accuracy'}
# Given the two tensors with given stepsize and epsilon accuracy parameters,
# plots the time required for each method to achieve epsilon accuracy, and saves it in dir_path.
def PlotTimesToAccuracy(lst_time, lst_labels, stepsizes, epsilons, dir_path):
    lst_medians = {}
    for t in range(len(lst_time)):
        median = np.zeros((len(stepsizes), len(epsilons)))
        time_opt = lst_time[t]
        for i in range(0, len(stepsizes), 1):
            for j in range(0, len(epsilons), 1):
                median[i, j] = np.median(time_opt[:, j, i]);

        lst_medians[t] = median

    # plot the results
    for j in range(0, len(epsilons), 1):
        eps = epsilons[j]
        plot_figures(lst_x = [stepsizes for i in range(len(lst_time))],
                     lst_plots=[lst_medians[i][:,j] for i in range(len(lst_time))],
                     lst_labels=lst_labels, x_label=r'$\alpha_1$', y_label = 'number of epochs',
                     title=str('Time to accuracy') + ' = ' + str(eps), x_log_scale = True, y_log_scale=False,
                     path_file=dir_path + str('accuracy')+str(j)+'.pdf')

# PlotMinLoss(lst_min_loss, lst_labels, stepsizes, dir_path)
# lst_min_loss - list of min loss for different methods
# lst_labels - list of methods names whose result to be displayed
# Given the two tensors with given stepsize and epsilon accuracy parameters,
# plots the best accuracy achieved for each step size after X epochs and saves it in dir_path
def PlotBestAccuracy(lst_min_loss, lst_labels, stepsizes, dir_path):
    lst_medians = {}
    for t in range(len(lst_labels)):
        median = np.zeros((len(stepsizes)))
        min_loss_opt = lst_min_loss[t]
        for i in range(0, len(stepsizes), 1):
            median[i] = np.median(min_loss_opt[:, i]);

        lst_medians[t] = median
    plot_figures(lst_x = [stepsizes for i in range(len(lst_min_loss))],
                 lst_plots=[lst_medians[i] for i in range(len(lst_min_loss))],
                 lst_labels=lst_labels, x_label=r'$\alpha_1$', y_label='Maximal_Accuracy',
                 title=str('Max Accuracy'), x_log_scale=True,  y_log_scale=False,
                 path_file=dir_path + 'max_accuracy.pdf')


if __name__ == '__main__':
    dataset = 'Stanford_dogs' # select from {'CIFAR10', 'MNIST', 'Stanford_dogs' }
    dataset_size = -1 #indicates full dataset size
    optimizers_check = ['sgd', 'truncated', 'adam', 'trunc_adagrad']
    loss = 'nn.CrossEntropyLoss'
    n_tests = 1
    stepsizes = np.logspace(-3, math.log(1000.0, 10), num=2, endpoint=True, base=10.0)

    dir_path = 'plots/' + dataset + '/'
    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        os.mkdir(dir_path + 'np_files')

    if dataset == 'MNIST':
        model_name = 'ResNet18_ELU'
        maxepoch = 25
        batch_size = 64
        epsilons = [0.99, 0.97, 0.95]
    if dataset == 'Stanford_dogs':
        model_name = 'pretrained_vgg16_ELU'
        maxepoch = 2 #30
        batch_size = 64
        epsilons = [0.7, 0.67, 0.65, 0.6, 0.5]
    elif dataset == 'CIFAR10':
        model_name = 'ResNet18_ELU' 
        maxepoch = 150
        batch_size = 100
        epsilons = [0.89, 0.85, 0.8]

    use_cuda = True

	
    dir_np_files = dir_path + 'np_files/'

    time_acc = {}
    max_acc = {}
    labels = []

    print("experiment on " + dataset)
    for i in range(len(optimizers_check)):
        opt = optimizers_check[i]
        print(opt)
        print()
        opt_label = opt
        labels.append(opt_label);
        time_acc[opt_label], max_acc[opt_label] = TimesToEpsilonAccuracy(model_name, dataset,
                                                                            dataset_size, opt,
                                                                            loss, n_tests,
                                                                            epsilons, stepsizes,
                                                                            maxepoch, batch_size,
                                                                            use_cuda, dir_np_files)

    # plot the figures for the number of iterations required to get certain accuracy
    labels_plots = [get_label(labels[i]) for i in range(len(labels))]
    PlotTimesToAccuracy([time_acc[labels[i]] for i in range(len(labels))], labels_plots, stepsizes, epsilons, dir_path)

    # plot the figures of Maximal Accuracy
    PlotBestAccuracy([max_acc[labels[i]] for i in range(len(labels))], labels_plots, stepsizes, dir_path)

