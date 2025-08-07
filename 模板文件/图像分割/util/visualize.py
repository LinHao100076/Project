import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_to_compare(compare_param:list, x_label=None, y_label=None, title=None):
    '''
    plot figure to compare train and test
    :param compare_param: list[dict]
        example: dict = {'X':[1, 2, 3], 'Y':[1, 2, 3]}
    :param x_label:
    :param y_label:
    :param title:
    :return: None
    '''
    for i in range(len(compare_param)):
        df = pd.DataFrame(compare_param[i])
        sns.lineplot(x=x_label, y=y_label, data=df)
    plt.show()

