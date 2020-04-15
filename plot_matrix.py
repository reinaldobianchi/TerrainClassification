import matplotlib.pyplot as plt
import numpy as np

real_field_order = ['Blanket','A. Grass','Rubber','Carpet','MDF','Tile']
sim_field_order  = ['Terrain 1','Terrain 2','Terrain 3','Terrain 4','Terrain 5','Terrain 6']


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


"""[REAL - ONE VELOCITY]:"""
#  Training Number of Walking Steps: 600
#  Validation Number of Walking Steps: 100
#  Test Number of Walking Steps: 100



conf_mat =   np.array([[150,   0,   0,   0,   0,   0],
 [  0, 146,   4,   0,   0,   0],
 [  0,   3, 146,   1,   0,   0],
 [  0,   0,   0, 150,   0,   0],
 [  0,   0,   0,   0, 150,   0],
 [  0,   1,   1,   0,   0, 148]])
plot_confusion_matrix(conf_mat, classes=real_field_order,normalize=True, title="Confusion Matrix: Real Robot")
plt.show()



"""[SIM - VARYING VELOCITY]:"""
#  Training Number of Walking Steps: 600
#  Validation Number of Walking Steps: 200
#  Test Number of Walking Steps: 200
conf_mat =    np.array([[149 ,  0 ,  0 ,  0 ,  0  , 1],
 [  0, 150,   0,   0,   0,   0],
 [  0,   0, 150,   0,   0,   0],
 [  1,   0,   0, 149,   0,   0],
 [  0,   0,   0,   0, 150,   0],
 [  0,   0,   1,   0,   0, 149]])
plot_confusion_matrix(conf_mat, classes=sim_field_order,normalize=True, title="Confusion Matrix: Simulated Robot")
plt.show()