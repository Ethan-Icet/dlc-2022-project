from torchinfo import summary
from sklearn.metrics import classification_report
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn')
FIGSIZE = (10, 5)
DISPLAY_ARGS = {'show': True, 'close': False, 'path': None}
SAVE_ARGS = {'show': False, 'close': True}


def total_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params


def plot_info_cycle(info_cycle: dict, show: bool = True, close: bool = False, path: str = None):
    figures = []

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(info_cycle['train']['loss'], label='train')
    plt.plot(info_cycle['test']['loss'], '--', label='test')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    if path is not None:
        plt.savefig(os.path.join(path, 'loss.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(info_cycle['train']['acc_classes'], label='train')
    plt.plot(info_cycle['test']['acc_classes'], '--', label='test')
    plt.title('Model class accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_classes.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(info_cycle['train']['acc_leq'], label='train')
    plt.plot(info_cycle['test']['acc_leq'], '--', label='test')
    plt.title('Model target accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_leq.png'))
    if show:
        plt.show()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(info_cycle['train']['acc_naive'], label='train')
    plt.plot(info_cycle['test']['acc_naive'], '--', label='test')
    plt.title('Model target ''naive'' accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_naive.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    return figures


def plot_info(means: dict = None, stds: dict = None, show: bool = False, close: bool = False, path: str = None):
    figures = []

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(means['train']['loss'], label='train')
    plt.plot(means['test']['loss'], '--', label='test')
    plt.fill_between(range(len(means['train']['loss'])), means['train']['loss'] - stds['train']['loss'],
                     means['train']['loss'] + stds['train']['loss'], alpha=0.2)
    plt.fill_between(range(len(means['test']['loss'])), means['test']['loss'] - stds['test']['loss'],
                     means['test']['loss'] + stds['test']['loss'], alpha=0.2)
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    if path is not None:
        plt.savefig(os.path.join(path, 'loss.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(means['train']['acc_classes'], label='train')
    plt.plot(means['test']['acc_classes'], '--', label='test')
    plt.fill_between(range(len(means['train']['acc_classes'])),
                     means['train']['acc_classes'] - stds['train']['acc_classes'],
                     means['train']['acc_classes'] + stds['train']['acc_classes'], alpha=0.2)
    plt.fill_between(range(len(means['test']['acc_classes'])),
                     means['test']['acc_classes'] - stds['test']['acc_classes'],
                     means['test']['acc_classes'] + stds['test']['acc_classes'], alpha=0.2)
    plt.title('Model class accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_classes.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(means['train']['acc_leq'], label='train')
    plt.plot(means['test']['acc_leq'], '--', label='test')
    plt.fill_between(range(len(means['train']['acc_leq'])), means['train']['acc_leq'] - stds['train']['acc_leq'],
                     means['train']['acc_leq'] + stds['train']['acc_leq'], alpha=0.2)
    plt.fill_between(range(len(means['test']['acc_leq'])), means['test']['acc_leq'] - stds['test']['acc_leq'],
                     means['test']['acc_leq'] + stds['test']['acc_leq'], alpha=0.2)
    plt.title('Model target accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_leq.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(means['train']['acc_naive'], label='train')
    plt.plot(means['test']['acc_naive'], '--', label='test')
    plt.fill_between(range(len(means['train']['acc_naive'])), means['train']['acc_naive'] - stds['train']['acc_naive'],
                     means['train']['acc_naive'] + stds['train']['acc_naive'], alpha=0.2)
    plt.fill_between(range(len(means['test']['acc_naive'])), means['test']['acc_naive'] - stds['test']['acc_naive'],
                     means['test']['acc_naive'] + stds['test']['acc_naive'], alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_naive.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    return figures


def plot_stats(stats: dict, show: bool = True, close: bool = False, path: str = None):
    figures = []

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(stats['train']['loss']['mean'], label='train')
    plt.plot(stats['test']['loss']['mean'], '--', label='test')
    plt.fill_between(range(len(stats['train']['loss']['mean'])),
                     stats['train']['loss']['mean'] - stats['train']['loss']['std'],
                     stats['train']['loss']['mean'] + stats['train']['loss']['std'], alpha=0.2)
    plt.fill_between(range(len(stats['test']['loss']['mean'])),
                     stats['test']['loss']['mean'] - stats['test']['loss']['std'],
                     stats['test']['loss']['mean'] + stats['test']['loss']['std'], alpha=0.2)
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    if path is not None:
        plt.savefig(os.path.join(path, 'loss.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(stats['train']['acc_classes']['mean'], label='train')
    plt.plot(stats['test']['acc_classes']['mean'], '--', label='test')
    plt.fill_between(range(len(stats['train']['acc_classes']['mean'])),
                     stats['train']['acc_classes']['mean'] - stats['train']['acc_classes']['std'],
                     stats['train']['acc_classes']['mean'] + stats['train']['acc_classes']['std'], alpha=0.2)
    plt.fill_between(range(len(stats['test']['acc_classes']['mean'])),
                     stats['test']['acc_classes']['mean'] - stats['test']['acc_classes']['std'],
                     stats['test']['acc_classes']['mean'] + stats['test']['acc_classes']['std'], alpha=0.2)
    plt.title('Model class accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_classes.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(stats['train']['acc_leq']['mean'], label='train')
    plt.plot(stats['test']['acc_leq']['mean'], '--', label='test')
    plt.fill_between(range(len(stats['train']['acc_leq']['mean'])),
                     stats['train']['acc_leq']['mean'] - stats['train']['acc_leq']['std'],
                     stats['train']['acc_leq']['mean'] + stats['train']['acc_leq']['std'], alpha=0.2)
    plt.fill_between(range(len(stats['test']['acc_leq']['mean'])),
                     stats['test']['acc_leq']['mean'] - stats['test']['acc_leq']['std'],
                     stats['test']['acc_leq']['mean'] + stats['test']['acc_leq']['std'], alpha=0.2)
    plt.title('Model target accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_leq.png'))
    if show:
        plt.show()
    if close:
        plt.close()

    fig = plt.figure(figsize=FIGSIZE)
    figures.append(fig)
    plt.plot(stats['train']['acc_naive']['mean'], label='train')
    plt.plot(stats['test']['acc_naive']['mean'], '--', label='test')
    plt.fill_between(range(len(stats['train']['acc_naive']['mean'])),
                     stats['train']['acc_naive']['mean'] - stats['train']['acc_naive']['std'],
                     stats['train']['acc_naive']['mean'] + stats['train']['acc_naive']['std'], alpha=0.2)
    plt.fill_between(range(len(stats['test']['acc_naive']['mean'])),
                     stats['test']['acc_naive']['mean'] - stats['test']['acc_naive']['std'],
                     stats['test']['acc_naive']['mean'] + stats['test']['acc_naive']['std'], alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if path is not None:
        plt.savefig(os.path.join(path, 'acc_naive.png'))
    if show:
        plt.show()
    if close:
        plt.close()
