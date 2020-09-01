import platform

if 'Linux' in platform.platform():
    path = '/home/riccardo/Documenti/ExplainingPairwiseLearning/code/'
else:
    path = '/Users/riccardo/Documents/ExplainingPairwiseLearning/code/'

path_dataset = path + 'dataset/'
path_eval = path + 'neigh_eval/'
path_discr = path + 'discr/'
path_syht_dataset = path + 'synth_datasets/'
path_ctgan_eval = path + 'ctgan_eval/'
path_cgan_images = path + 'images'
path_clf = path + 'clf/'

ts_methods = ['grabocka', 'random', 'random_rnd', 'learning', 'dash', 'dash_rnd', 'dash_md', 'dash_rnd_md']

ts_datasets = ['gunpoint', 'italypower', 'arrowhead', 'ecg200', 'phalanges', 'electricdevices']

tab_datasets = ['wdbc', 'diabetes', 'ctg', 'ionoshpere', 'parkinsons', 'sonar', 'vehicle']

txt_datasets = ['20newsgroups', 'imdb']

img_datasets = ['mnist', 'fashion_mnist', 'cifar10']

